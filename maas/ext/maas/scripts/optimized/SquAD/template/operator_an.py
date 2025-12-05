import re
from typing import Dict

# import the main operator classes from the operator module (should exist in your repo)
from maas.ext.maas.scripts.optimized.SquAD.train.template.operator import (
    Generate,
    ScEnsemble,
    SelfRefine,
    EarlyStop,
)

# --- Lightweight QA Test operator ---
class Test:
    """
    Simple QA test operator for quick correctness checks.
    Compares predicted answer (string) to gold answer (string) using a normalization routine.
    Returns a dict: {'correct': bool, 'predicted': str, 'gold': str, 'pred_norm': str, 'gold_norm': str}
    """

    def __init__(self, llm=None, name: str = "Test"):
        # keep signature similar to other operators; LLM not used here
        self.name = name
        self.llm = llm

    @staticmethod
    def _normalize(text: str) -> str:
        if text is None:
            return ""
        # basic SQuAD-ish normalization:
        # lowercase, remove punctuation, remove articles, collapse whitespace
        text = text.lower().strip()
        # remove punctuation (keep alphanumerics and spaces)
        text = re.sub(r"[^a-z0-9\s]", " ", text)
        # remove articles
        text = re.sub(r"\b(a|an|the)\b", " ", text)
        # collapse whitespace
        text = re.sub(r"\s+", " ", text).strip()
        return text

    async def __call__(self, predicted: str, gold: str):
        pred_norm = self._normalize(predicted)
        gold_norm = self._normalize(gold)
        correct = pred_norm == gold_norm
        return {
            "correct": correct,
            "predicted": predicted,
            "gold": gold,
            "pred_norm": pred_norm,
            "gold_norm": gold_norm,
        }

# export mapping - keeps same shape as other operator modules
operator_mapping: Dict[str, object] = {
    "Generate": Generate,
    "ScEnsemble": ScEnsemble,
    "SelfRefine": SelfRefine,
    "EarlyStop": EarlyStop,
    "Test": Test,
}

operator_names = list(operator_mapping.keys())
