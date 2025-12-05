import asyncio
import time
import torch
import os
import numpy as np
from typing import List, Literal

from pydantic import BaseModel, Field
from maas.ext.maas.scripts.evaluator import DatasetType
from maas.ext.maas.scripts.optimizer_utils.data_utils import DataUtils
from maas.ext.maas.scripts.optimizer_utils.experience_utils import ExperienceUtils
from maas.ext.maas.scripts.optimizer_utils.evaluation_utils import EvaluationUtils
from maas.ext.maas.scripts.optimizer_utils.graph_utils import GraphUtils
from maas.logs import logger
from maas.ext.maas.models.utils import get_sentence_embedding
from maas.ext.maas.models.controller import MultiLayerController

# -----------------------------
# Question Types for MaAS
# -----------------------------
QuestionType = Literal["math", "code", "qa"]
OptimizerType = Literal["Graph", "Test"]

class GraphOptimize(BaseModel):
    modification: str = Field(default="", description="Graph modification text")
    graph: str = Field(default="", description="New graph content")
    prompt: str = Field(default="", description="Prompt text")


# =====================================================
#                   OPTIMIZER CORE (QA version)
# =====================================================
class Optimizer:
    """
    The optimizer orchestrates:
      - loading dataset
      - training controller operator-selection policy
      - running evaluation loops using QA extractive metrics (EM/F1)
    """

    def __init__(
        self,
        dataset: DatasetType,
        question_type: QuestionType,
        opt_llm_config,
        exec_llm_config,
        operators: List,
        sample: int,
        optimized_path: str = None,
        round: int = 1,
        batch_size: int = 4,
        lr: float = 0.01,
        is_textgrad: bool = False,
    ) -> None:

        # -----------------------------
        # Basic config
        # -----------------------------
        self.optimize_llm_config = opt_llm_config
        self.execute_llm_config = exec_llm_config

        self.dataset = dataset
        self.type = question_type  # must match 'qa'
        self.graph = None
        self.operators = operators

        # Path for optimization artifacts
        if optimized_path is None:
            optimized_path = os.path.join(".", "optimized")

        self.root_path = os.path.join(optimized_path, str(self.dataset))

        # Training settings
        self.sample = sample
        self.top_scores = []
        self.round = round
        self.batch_size = batch_size
        self.lr = lr
        self.is_textgrad = is_textgrad

        # -----------------------------
        # Utils
        # -----------------------------
        self.graph_utils = GraphUtils(self.root_path)
        self.data_utils = DataUtils(self.root_path)
        self.experience_utils = ExperienceUtils(self.root_path)
        self.evaluation_utils = EvaluationUtils(self.root_path)

        # -----------------------------
        # Controller model
        # -----------------------------
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.controller = MultiLayerController(device=self.device).to(self.device)
        self.optimizer = torch.optim.Adam(self.controller.parameters(), lr=self.lr)


    # =====================================================
    #                     ENTRY POINT
    # =====================================================
    def optimize(self, mode: OptimizerType = "Graph"):
        """
        mode == "Graph" → full training
        mode == "Test"  → load controller checkpoint & compute test score
        """
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        if mode == "Test":
            score = loop.run_until_complete(self.test())
            return score

        retry_count = 0
        max_retries = 1

        while retry_count < max_retries:
            try:
                score = loop.run_until_complete(self._optimize_graph_maas())
                break
            except Exception as e:
                retry_count += 1
                logger.error(f"[Optimizer] Error: {e}. Retry {retry_count}/{max_retries}")

                if retry_count == max_retries:
                    logger.error("Max retries reached — continuing.")
                    score = None

                time.sleep(5 * retry_count)

        logger.info(f"Final Score for round {self.round}: {score}")
        return score


    # =====================================================
    #             TRAINING (QA-compatible)
    # =====================================================
    async def _optimize_graph_maas(self):
        graph_path = os.path.join(self.root_path, "train")

        # Load existing results
        data = self.data_utils.load_results(graph_path)

        # ---------------------------------------------
        # Load operator descriptions → embeddings
        # ---------------------------------------------
        operator_descriptions = self.graph_utils.load_operators_description_maas(self.operators)
        precomputed_operator_embeddings = torch.stack(
            [get_sentence_embedding(d) for d in operator_descriptions]
        ).to(self.device)

        # ---------------------------------------------
        # Create round directory
        # ---------------------------------------------
        directory = self.graph_utils.create_round_directory(graph_path, self.round)
        logger.info(f"[Round Directory] {directory}")

        # Load graph structure defined by user
        self.graph = self.graph_utils.load_graph_maas(graph_path)

        # ---------------------------------------------
        # QA execution parameters → VERY IMPORTANT
        # ---------------------------------------------
        extra_exec_params = {
            "max_tokens": 64,
            "response_format": "plain_text",
            "instructions": (
                "Provide a short extractive answer copied exactly from the context. "
                "Do not explain. Do not paraphrase."
            )
        }

        # Params passed into evaluator
        params = {
            "operator_embeddings": precomputed_operator_embeddings,
            "controller": self.controller,
            "execute_llm_config": self.execute_llm_config,
            "dataset": self.dataset,
            "optimizer": self.optimizer,
            "sample": self.sample,
            "is_textgrad": self.is_textgrad,
            **extra_exec_params
        }

        # ---------------------------------------------
        # Run evaluation (EM/F1 scoring)
        # ---------------------------------------------
        avg_score = await self.evaluation_utils.evaluate_graph_maas(
            self, directory, data, initial=False, params=params
        )

        # ---------------------------------------------
        # Ensure results are written
        # ---------------------------------------------
        try:
            json_file_path = self.data_utils.get_results_file_path(graph_path)
            current_results = self.data_utils.load_results(graph_path)

            need_append = True
            if current_results:
                last_round = current_results[-1].get("round", None)
                if last_round == self.round:
                    need_append = False

            if need_append:
                new_data = self.data_utils.create_result_data(
                    round=self.round,
                    score=avg_score,
                    avg_cost=0.0,
                    total_cost=0.0,
                    token=0
                )
                current_results.append(new_data)
                self.data_utils.save_results(json_file_path, current_results)
                logger.info("[Results] New round result appended.")
            else:
                logger.info("[Results] Already saved previously.")

        except Exception as e:
            logger.error(f"[Results] Failed to save: {e}")

        return avg_score


    # =====================================================
    #            TEST / FINAL EVALUATION
    # =====================================================
    async def test(self):
        """
        Test phase after training — loads controller weights
        and computes final QA scores using evaluate_graph_test_maas().
        """
        graph_path = os.path.join(self.root_path, "test")

        json_file_path = self.data_utils.get_results_file_path(graph_path)
        data = self.data_utils.load_results(graph_path)

        # Operator embeddings
        operator_descriptions = self.graph_utils.load_operators_description_maas(self.operators)
        precomputed_operator_embeddings = torch.stack(
            [get_sentence_embedding(x) for x in operator_descriptions]
        ).to(self.device)

        # Load graph
        self.graph = self.graph_utils.load_graph_maas(graph_path)

        directory = self.graph_utils.create_round_directory(graph_path, self.round)

        # -----------------------------
        # Load trained controller
        # -----------------------------
        pth_path = os.path.join(self.root_path, "train")
        pth_directory = self.graph_utils.create_round_directory(pth_path, self.round)

        controller_filename = f"{self.dataset}_controller_sample{self.sample}.pth"
        controller_path = os.path.join(pth_directory, controller_filename)
        logger.info(f"[Checkpoint] Looking at: {controller_path}")

        if not os.path.exists(controller_path):
            raise FileNotFoundError(f"Controller checkpoint not found: {controller_path}")

        checkpoint = torch.load(controller_path, map_location=self.device)
        self.controller.load_state_dict(checkpoint)
        self.controller.eval()
        logger.info("[Checkpoint] Loaded successfully!")

        # -----------------------------
        # QA execution params
        # -----------------------------
        extra_exec_params = {
            "max_tokens": 64,
            "response_format": "plain_text",
            "instructions": "Provide a short extractive span from the context only.",
        }

        params = {
            "operator_embeddings": precomputed_operator_embeddings,
            "controller": self.controller,
            "execute_llm_config": self.execute_llm_config,
            "dataset": self.dataset,
            "optimizer": self.optimizer,
            "sample": self.sample,
            "is_textgrad": False,
            **extra_exec_params
        }

        # Run evaluation
        score = await self.evaluation_utils.evaluate_graph_test_maas(
            self, directory, is_test=True, params=params
        )

        new_data = self.data_utils.create_result_data(self.round, score)
        data.append(new_data)

        # Save results
        try:
            self.data_utils.save_results(json_file_path, data)
            logger.info("[Test Results] Saved successfully.")
        except Exception as e:
            logger.error(f"[Test Results] Save failed: {e}")

        return score
