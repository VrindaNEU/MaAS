GENERATE_QA_ANSWER_PROMPT = """
You are given a question and a context paragraph.
Provide a concise, accurate answer in one or two sentences.
If the answer is not present in the passage, reply "unanswerable".
"""

REFINE_QA_ANSWER_PROMPT = """
Improve the answer based on the question and context. 
Ensure it is grounded strictly in the passage and is concise.
"""

DETAILED_QA_PROMPT = """
Provide a high-quality, evidence-grounded answer to the question using the context.
1. Quote only necessary supporting text.
2. Keep the final answer short and factual.
"""
