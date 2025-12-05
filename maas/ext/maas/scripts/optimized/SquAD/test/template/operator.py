import concurrent
import sys
import traceback
from typing import List

from tenacity import retry, stop_after_attempt, wait_fixed

from maas.actions.action_node import ActionNode
from maas.llm import LLM
from maas.logs import logger
import asyncio


# ===============================
# SQuAD QA Operators
# ===============================

class Operator:
    def __init__(self, llm: LLM, name: str):
        self.name = name
        self.llm = llm

    def __call__(self, *args, **kwargs):
        raise NotImplementedError

    async def _fill_node(self, op_class, prompt, mode=None, **extra_kwargs):
        fill_kwargs = {"context": prompt, "llm": self.llm}
        if mode:
            fill_kwargs["mode"] = mode
        fill_kwargs.update(extra_kwargs)
        node = await ActionNode.from_pydantic(op_class).fill(**fill_kwargs)
        return node.instruct_content.model_dump()


# ===============================
# SIMPLE LLM ANSWER GENERATION
# ===============================

# ---- SQuAD prompt template ----
QA_PROMPT = """
You are a question answering AI. 
Answer the question using ONLY the information from the context.

Context:
{context}

Question:
{question}

Answer:
"""

REFINE_PROMPT = """
You are improving a question answering result.

Context:
{context}

Question:
{question}

Previous Answer:
{answer}

Provide a BETTER and more accurate answer based strictly on the context.
Return only the answer text.
"""


class GenerateAnswer(Operator):
    """Generate a QA answer from context + question."""

    def __init__(self, llm: LLM, name="GenerateAnswer"):
        super().__init__(llm, name)

    async def __call__(self, context: str, question: str):
        prompt = QA_PROMPT.format(context=context, question=question)
        response = await self._fill_node(GenerateOp, prompt, mode="single_fill")
        return response


class SelfRefineAnswer(Operator):
    """Refine an existing answer."""

    def __init__(self, llm: LLM, name="SelfRefineAnswer"):
        super().__init__(llm, name)

    async def __call__(self, context: str, question: str, answer: str):
        prompt = REFINE_PROMPT.format(context=context, question=question, answer=answer)
        response = await self._fill_node(SelfRefineOp, prompt, mode="single_fill")
        return response


class EnsembleSelect(Operator):
    """Choose the best answer from multiple candidate LLM answers."""

    def __init__(self, llm: LLM, name="EnsembleSelect"):
        super().__init__(llm, name)

    async def __call__(self, candidates: List[str], context: str, question: str):
        # string frequency voting
        freq = {}
        for ans in candidates:
            ans = ans.strip()
            freq[ans] = freq.get(ans, 0) + 1

        # pick most frequent
        best = sorted(freq.items(), key=lambda x: -x[1])[0][0]

        return {"response": best}


# ===============================
# DEPRECATED OPERATORS (NO USE IN SQuAD)
# ===============================

class GenerateCoT(Operator):
    async def __call__(self, *args, **kwargs):
        raise NotImplementedError("CoT not required for SQuAD QA.")

class MultiGenerateCoT(Operator):
    async def __call__(self, *args, **kwargs):
        raise NotImplementedError("Multi-CoT not required for SQuAD QA.")

class Programmer(Operator):
    async def __call__(self, *args, **kwargs):
        raise NotImplementedError("Programmer not used for QA tasks.")

class EarlyStop(Operator):
    async def __call__(self):
        return NotImplementedError
