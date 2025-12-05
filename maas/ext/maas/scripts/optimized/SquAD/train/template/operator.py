# operator.py (QA-adapted)
from typing import List, Optional

# NOTE: changed imports to point to QA templates (you'll create these next)
from maas.ext.maas.scripts.optimized.SquAD.train.template.operator_an import *
from maas.ext.maas.scripts.optimized.SquAD.train.template.op_prompt import *
from maas.actions.action_node import ActionNode
from maas.llm import LLM
from maas.logs import logger
import asyncio


class Operator:
    def __init__(self, llm: LLM, name: str):
        self.name = name
        self.llm = llm

    def __call__(self, *args, **kwargs):
        raise NotImplementedError

    async def _fill_node(self, op_class, prompt: str, mode: Optional[str] = None, **extra_kwargs):
        """
        Generic helper to create an ActionNode from a pydantic op_class and fill it.
        Expects 'prompt' to already contain both context and question formatted as desired.
        """
        fill_kwargs = {"context": prompt, "llm": self.llm}
        if mode:
            fill_kwargs["mode"] = mode
        fill_kwargs.update(extra_kwargs)
        node = await ActionNode.from_pydantic(op_class).fill(**fill_kwargs)
        # return a plain dict representation for downstream code
        return {
            "response": node.instruct_content.final_answer,
            "start": node.instruct_content.start_char,
            "end": node.instruct_content.end_char,
            "analysis": node.instruct_content.analysis,
            "confidence": node.instruct_content.confidence,
        }



class Generate(Operator):
    """
    Generate an answer given input (context + question) and an instruction string (high-level instruction).
    For SQuAD-style QA, `input` should be the concatenation of context + question, e.g.:
      "Context: ...\n\nQuestion: ..."
    """
    def __init__(self, llm: LLM, name: str = "Generate"):
        super().__init__(llm, name)

    async def __call__(self, input: str, instruction: str):
        prompt = instruction + "\n\n" + input
        response = await self._fill_node(GenerateOp, prompt, mode="single_fill")
        return response

class GenerateCoT(Operator):
    def __init__(self, llm: LLM, name: str = "GenerateCoT"):
        super().__init__(llm, name)

    async def __call__(self, input, instruction):
        # Use the same pattern as Generate: prepend the instruction and then the input string.
        # This avoids depending on specific {context}/{question} placeholders inside the prompt.
        prompt = instruction + "\n\n" + input
        response = await self._fill_node(GenerateOp, prompt, mode="single_fill")
        return response


class MultiGenerateCoT(Operator):
    def __init__(self, llm: LLM, name: str = "MultiGenerateCoT"):
        super().__init__(llm, name)

    async def __call__(self, input, instruction):
        # Same approach: three independent generations using the instruction + input format.
        prompt = instruction + "\n\n" + input
        response1 = await self._fill_node(GenerateOp, prompt, mode="single_fill")
        response2 = await self._fill_node(GenerateOp, prompt, mode="single_fill")
        response3 = await self._fill_node(GenerateOp, prompt, mode="single_fill")

        return {"response": [response1, response2, response3]}


class ScEnsemble(Operator):
    """
    Simple selection ensemble for multiple-candidate answers.
    `solutions` is a list of answer strings; `question` is the original question (and optionally context).
    Expects SC_ENSEMBLE_PROMPT to instruct the model to return a letter (A/B/C...) indicating the best solution.
    """
    def __init__(self, llm: LLM, name: str = "ScEnsemble"):
        super().__init__(llm, name)

    async def __call__(self, solutions: List[str], question: str):
        answer_mapping = {}
        solution_text = ""
        for index, solution in enumerate(solutions):
            label = chr(65 + index)
            answer_mapping[label] = index
            solution_text += f"{label}: {str(solution)}\n\n"

        prompt = SC_ENSEMBLE_PROMPT.format_map({
            "question": question,
            "context": context if 'context' in locals() else "",
            "solutions": solution_text
        })


        response = await self._fill_node(ScEnsembleOp, prompt, mode="xml_fill")

        # Expect the node to provide a "solution_letter" field (or similar).
        answer_letter = response.get("solution_letter", "")
        if not isinstance(answer_letter, str):
            answer_letter = str(answer_letter)

        answer_letter = answer_letter.strip().upper()
        if answer_letter in answer_mapping:
            return {"response": solutions[answer_mapping[answer_letter]]}
        else:
            # fallback: if model couldn't pick, return top candidate (first)
            logger.info("ScEnsemble: couldn't parse letter '%s', returning first candidate", answer_letter)
            return {"response": solutions[0] if solutions else ""}


class SelfRefine(Operator):
    """
    Ask the model to refine an existing answer given the context/question and the current solution.
    """
    def __init__(self, llm: LLM, name: str = "SelfRefine"):
        super().__init__(llm, name)

    async def __call__(self, input: str, solution: str):
        """
        `input` should contain context + question text.
        `solution` is the model's previous answer text.
        """
        prompt = SELFREFINE_PROMPT.format(input=input, solution=solution)
        response = await self._fill_node(SelfRefineOp, prompt, mode="single_fill")
        return response


class EarlyStop(Operator):
    """
    Simple early-stop operator for flow control.
    """
    def __init__(self, llm: LLM, name: str = "EarlyStop"):
        super().__init__(llm, name)

    async def __call__(self):
        # Return a structure that the rest of the system can use to stop further generation.
        return {"stop": True}
