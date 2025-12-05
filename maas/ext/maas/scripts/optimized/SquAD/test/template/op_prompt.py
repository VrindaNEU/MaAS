# op_prompt.py
# Prompts tuned for SQuAD-style extractive QA (no chain-of-thought, no LaTeX, no extra formatting)

# -----------------------
# Strict extractive QA prompt
# -----------------------
QA_PROMPT = """
You are a precise question-answering system. Given a CONTEXT and a QUESTION, return ONLY the exact short phrase
from the CONTEXT that answers the QUESTION. Do NOT explain, do NOT reason, do NOT add examples, do NOT use LaTeX,
do NOT use Markdown, and do NOT add any additional text. Return exactly the answer and nothing else (one short phrase).
"""

# -----------------------
# Short refinement prompt
# -----------------------
REFINE_PROMPT = """
You are a short-answer refiner. Given a CONTEXT, a QUESTION, and a PREVIOUS_ANSWER, return only a corrected or improved
short answer phrase that is supported directly by the CONTEXT. Do NOT explain your reasoning. Return exactly the answer
and nothing else.
"""

# -----------------------
# Ensemble selection prompt (choose the best short answer)
# -----------------------
# This prompt is used when multiple candidate answers are available.
# Ask the model to choose the best one by index or to return the best candidate verbatim.
ENSEMBLE_PROMPT = """
You are performing answer selection. Given a CONTEXT, a QUESTION, and a list of candidate short answers, choose
the single candidate that is best supported by the CONTEXT and that answers the QUESTION. Return ONLY the chosen
candidate verbatim (no commentary, no extra text). If multiple candidates are identical, return that common phrase.
"""

# -----------------------
# Minimal instruction for operator pipelines that expect a prompt template
# -----------------------
# Template that operators can use to build a full message/prompt with context & question inserted.
QA_PROMPT_TEMPLATE = """
Context:
{context}

Question:
{question}

Instruction:
Return ONLY the exact short phrase from the Context that answers the Question. NO explanation, NO formatting, NO LaTeX.
"""

# -----------------------
# Utility short prompts for system messages or LLM request extras
# -----------------------
SYSTEM_QA_INSTRUCTION = "Answer concisely. Return the final answer only. No explanations."
