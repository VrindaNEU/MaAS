# graph.py (QA-adapted, safe prompt injection)
import torch
import maas.ext.maas.scripts.optimized.SquAD.train.template.prompt as prompt_custom
import maas.ext.maas.scripts.optimized.SquAD.train.template.operator as operator
from maas.ext.maas.scripts.optimized.SquAD.train.template.operator_registry import operator_mapping, operator_names
from maas.provider.llm_provider_registry import create_llm_instance
from maas.utils.cost_manager import CostManager
from maas.logs import logger
from typing import Any, Dict, List, Optional


class Workflow:
    def __init__(
        self,
        name: str,
        llm_config,
        dataset,
        controller: torch.nn.Module,
        operator_embeddings,
    ) -> None:
        self.name = name
        self.dataset = dataset
        self.llm = create_llm_instance(llm_config)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.llm.cost_manager = CostManager()

        # Basic operators
        self.custom = operator.Generate(self.llm)
        self.sc_ensemble = operator.ScEnsemble(self.llm)

        self.controller = controller.to(self.device)
        self.operator_embeddings = operator_embeddings.to(self.device)
        self.selection_operator_instances = {
            operator_name: operator_mapping[operator_name](self.llm)
            for operator_name in operator_names
        }
        self.selection_operator_names = operator_names

    def _extract_answer_text(self, op_result: Any) -> str:
        """
        Robustly extract a string answer from various operator return formats.
        """
        if op_result is None:
            return ""

        if isinstance(op_result, str):
            return op_result

        if isinstance(op_result, dict):
            # preferred extractive keys
            for key in ("final_answer", "response", "solution", "refined_answer", "answer"):
                if key in op_result:
                    val = op_result[key]
                    if isinstance(val, dict):
                        nested = self._extract_answer_text(val)
                        if nested:
                            return nested
                        continue
                    if isinstance(val, str):
                        return val

            # response could be a list
            if "response" in op_result and isinstance(op_result["response"], list):
                for item in op_result["response"]:
                    text = self._extract_answer_text(item)
                    if text:
                        return text

        return ""

    def _safe_inject(self, template: str, **kwargs) -> str:
        """
        Safely inject only the exact placeholders we expect into the prompt TEMPLATE.
        This avoids errors when TEMPLATE contains other braces/JSON examples.
        Allowed placeholders: context, question, instruction, solutions
        """
        if template is None:
            return ""
        out = template
        # Only replace known placeholders to avoid touching example braces
        if "context" in kwargs:
            out = out.replace("{context}", kwargs["context"])
        if "question" in kwargs:
            out = out.replace("{question}", kwargs["question"])
        if "instruction" in kwargs:
            out = out.replace("{instruction}", kwargs["instruction"])
        if "solutions" in kwargs:
            out = out.replace("{solutions}", kwargs["solutions"])
        return out

    async def __call__(self, problem: str):
        """
        problem: expected to contain "Context: ...\n\nQuestion: ..." or at least the context.
        Returns: final_solution (str), total_cost (float), sum_log_prob
        """
        # Try to extract context and question from problem string if not provided separately
        context = problem
        question = ""
        try:
            # heuristic split: "Context: ...\n\nQuestion: ..."
            if "\n\nQuestion:" in problem:
                parts = problem.split("\n\nQuestion:", 1)
                context = parts[0].replace("Context:", "").strip()
                question = parts[1].strip()
            elif "\nQuestion:" in problem:
                parts = problem.split("\nQuestion:", 1)
                context = parts[0].replace("Context:", "").strip()
                question = parts[1].strip()
            else:
                # fallback: use entire problem as context
                context = problem.strip()
                question = ""
        except Exception:
            context = problem
            question = ""

        # controller returns log_probs_layers, selected_names_layers
        log_probs_layers, selected_names_layers = self.controller.forward(
            problem, self.operator_embeddings, self.selection_operator_names
        )

        current_solution = ""
        solutions: List[str] = []
        sum_log_prob = 0.0

        # --- Initial Generate (use safe injection) ---
        try:
            gen_prompt = self._safe_inject(
                prompt_custom.GENERATE_QA_ANSWER_PROMPT,
                context=context,
                question=question,
            )
            initial = await self.custom(input=problem, instruction=gen_prompt)
        except Exception as e:
            logger.exception("Initial Generate operator failed: %s", e)
            initial = {}

        initial_answer = self._extract_answer_text(initial)
        solutions.append(initial_answer)
        current_solution = initial_answer

        # --- Iterate through operator layers selected by controller ---
        for layer_idx, selected_names in enumerate(selected_names_layers):
            for op_name in selected_names:
                selected_operator = self.selection_operator_instances.get(op_name)
                new_solution = current_solution

                try:
                    if op_name in ["Generate", "GenerateCoT"]:
                        # Use detailed prompt; safe-inject context/question/instruction
                        detailed_prompt = self._safe_inject(
                            prompt_custom.DETAILED_SOLUTION_PROMPT,
                            context=context,
                            question=question,
                            instruction="Provide a detailed extractive answer (short)."
                        )
                        result = await selected_operator(input=problem, instruction=detailed_prompt)
                        new_solution = self._extract_answer_text(result)
                        solutions.append(new_solution)

                    elif op_name == "SelfRefine":
                        # SelfRefine operator builds its own prompt internally; pass inputs directly
                        result = await selected_operator(input=problem, solution=current_solution)
                        new_solution = self._extract_answer_text(result)
                        solutions.append(new_solution)

                    elif op_name == "ScEnsemble":
                        # Do not rely on SC_ENSEMBLE_PROMPT existing — call operator directly.
                        result = await selected_operator(solutions=solutions, question=question)
                        new_solution = self._extract_answer_text(result)
                        # reset candidates to only the chosen one
                        solutions = [new_solution] if new_solution else []

                    else:
                        # Unknown operator -> skip safely
                        logger.info("Unknown operator selected: %s. Skipping.", op_name)
                        new_solution = current_solution

                except Exception as e:
                    logger.exception("Operator %s failed with error: %s", op_name, e)
                    new_solution = current_solution

                current_solution = new_solution

            # accumulate layer log probability if available
            try:
                sum_log_prob += log_probs_layers[layer_idx]
            except Exception:
                logger.debug("Could not add layer log prob for layer %d", layer_idx)

        # --- Final ensemble if multiple candidate solutions remain ---
        final_solution = current_solution
        if len(solutions) > 1:
            try:
                ensemble_out = await self.sc_ensemble(solutions=solutions, question=question)
                final_text = self._extract_answer_text(ensemble_out)
                if final_text:
                    final_solution = final_text
            except Exception as e:
                logger.exception("Final ScEnsemble failed: %s", e)

        return final_solution, self.llm.cost_manager.total_cost, sum_log_prob
