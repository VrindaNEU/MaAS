# prompt.py for SQuAD

GENERATE_SOLUTION_PROMPT = """
You are a knowledgeable assistant. Based on the following passage, answer the question accurately and concisely.

Passage:
{context}

Question:
{question}

Instructions:
- Provide the answer directly, without quoting the entire passage.
- Keep the answer brief and precise.
- If the answer is not explicitly stated, provide the most probable inference.
"""

REFINE_ANSWER_PROMPT = """
You are an expert QA assistant. Given the original passage, the question, and a candidate answer, refine the answer for clarity, accuracy, and conciseness.

Passage:
{context}

Question:
{question}

Candidate answer:
{answer}

Instructions:
- Ensure the answer is correct according to the passage.
- Make it clear and concise.
- Do not add unrelated information.
"""

SC_ENSEMBLE_PROMPT = """
Multiple answers have been generated for the following question:

Question:
{question}

Generated answers:
{solutions}

Instructions:
- Identify the answer that is most consistent across the generated solutions.
- Provide only the final answer.
- Do not include additional explanations.
"""
