import asyncio
from typing import Callable, List, Tuple
import torch

from maas.ext.maas.benchmark.benchmark import BaseBenchmark
from maas.logs import logger

class SQuADBenchmark(BaseBenchmark):

    def __init__(self,
                 name: str,
                 file_path: str,
                 log_path: str,
                 batch_size: int,
                 controller: torch.nn.Module,
                 operator_embeddings: List[List[float]],
                 optimizer: torch.optim.Optimizer):
        super().__init__(name, file_path, log_path, batch_size,
                         controller, operator_embeddings, optimizer)

    def extract_model_answer(self, text: str) -> str:
        # SQuAD answers are normally short spans
        return text.strip()

    def calculate_score(self, expected_output: str, prediction: str) -> Tuple[int, str]:
        expected = expected_output.strip().lower()
        predicted = prediction.strip().lower()

        # simple exact match (you can replace with F1 score later)
        score = int(expected == predicted)
        return score, prediction

    async def evaluate_problem(self, problem: dict, graph: Callable):
        input_text = problem["context"] + "\n" + problem["question"]
        expected_output = problem["answer"]

        try:
            output, cost, logprob = await asyncio.wait_for(
                graph(input_text),
                timeout=1500
            )

            score, extracted_output = self.calculate_score(expected_output, output)

            if score == 0:
                self.log_mismatch(
                    input_text,
                    expected_output,
                    output,
                    extracted_output
                )

            return input_text, output, expected_output, score, cost, logprob

        except Exception as e:
            logger.info(f"Skipping sample due to error: {e}")
            return input_text, str(e), expected_output, 0.0, 0.0, torch.tensor(0.0, device=self.device)

    def get_result_columns(self):
        return ["question", "prediction", "expected_output", "score", "cost", "logprob"]
