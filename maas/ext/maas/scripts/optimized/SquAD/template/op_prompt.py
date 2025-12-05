SC_ENSEMBLE_PROMPT = """
Given the following question and the candidate answers generated for it, evaluate which candidate is the most likely correct answer grounded in the provided context.

Question: {problem}

Candidate answers:
{solutions}

In the 'thought' field, provide a short explanation (1-2 sentences) of how you compared the candidates (grounding, coverage, direct quoting). In the 'solution_letter' field, output only the single letter ID (A, B, C, etc.) corresponding to the selected best answer. Do not include any additional text in 'solution_letter'.
"""

SELFREFINE_PROMPT = """
You are a QA assistant specialized in refining short factual answers.

Question:
{problem}

Current answer:
{solution}

Instruction:
1. Verify the answer is strictly supported by the context.
2. If unsupported, attempt to correct using only the context.
3. Make the final answer concise (one sentence) and factual.
Return only the improved answer text (no extra commentary).
"""

GENERATE_QA_PROMPT = """
You are a question answering assistant. Use ONLY the provided context to answer the question.

Context:
{input}

Question:
{instruction}

Provide a short, direct answer taken from the context. If the answer is not present, respond with "unanswerable".
"""

# Utility prompt (detailed generation if needed)
DETAILED_QA_PROMPT = """
You are a QA assistant. Given the context and the question, produce a concise, evidence-grounded answer.
1. If the exact answer can be copied from the passage, return that span or a short paraphrase.
2. If the answer is not present, return "unanswerable".
3. Keep the final answer 1-2 sentences max.
"""
