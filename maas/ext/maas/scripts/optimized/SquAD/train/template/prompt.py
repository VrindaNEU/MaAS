# prompt.py
"""
QA / SQuAD-style prompts for MaAS operators (extractive).
Prompts instruct the model to RETURN A SINGLE-LINE JSON ONLY with the exact fields listed.
"""

GENERATE_QA_ANSWER_PROMPT = """
You are given a passage (CONTEXT) and a QUESTION. Produce a precise, extractive answer by selecting the minimal text span from the CONTEXT that answers the QUESTION.
Return your output AS A SINGLE LINE JSON object and nothing else, with keys:
  - "final_answer": the answer string (exactly as it appears in the context). If the answer cannot be found in the context, set "final_answer": "".
  - "start_char": the character index (0-based) in the context where final_answer begins, or -1 if unknown/unavailable.
  - "end_char": the character index (0-based, exclusive) in the context where final_answer ends, or -1 if unknown/unavailable.
  - "analysis": optional short string (internal reasoning / steps) — may be omitted or null.

Example (single-line JSON only):
{"final_answer":"September 1876","start_char":45,"end_char":60,"analysis":null}

Use the provided CONTEXT verbatim for any evidence; do not invent facts not present in the CONTEXT.
"""

DETAILED_SOLUTION_PROMPT = """
You are given a passage (CONTEXT) and a QUESTION. Provide a detailed extractive response that contains:
  - A concise 'final_answer' (exact substring) and explicit character offsets.
  - A short 'analysis' (1-3 sentences) referencing the evidence in the context.

Return a SINGLE-LINE JSON object with keys:
  - "final_answer": short answer string (or empty string "")
  - "start_char": start char index in context or -1
  - "end_char": end char index (exclusive) in context or -1
  - "analysis": 1-3 sentence explanation tying the evidence to the answer
  - "confidence": optional numeric estimate 0.0-1.0 (optional)

Example:
{"final_answer":"a copper statue of Christ","start_char":73,"end_char":99,"analysis":"The sentence explicitly states that a copper statue of Christ stands in front of the Main Building.","confidence":0.95}
Make the JSON valid. Do not include any additional commentary or markdown.
"""

REFINE_ANSWER_PROMPT = """
Refinement prompt: You are given:
  - CONTEXT (passage),
  - QUESTION,
  - an existing candidate answer string (CANDIDATE_ANSWER).
Your task:
  - If CANDIDATE_ANSWER is an exact contiguous substring of CONTEXT and answers the QUESTION, return it unchanged (same JSON format).
  - If it is not an exact substring or is incorrect, find the best exact substring in CONTEXT and return that.
Return a SINGLE-LINE JSON object with keys:
  - "final_answer": exact substring (or "")
  - "start_char": integer start index or -1
  - "end_char": integer end index (exclusive) or -1
  - "analysis": optional short note
If no answer is extractable, set final_answer "" and start_char/end_char -1 and include analysis "no extractive span found".
"""

UNANSWERABLE_POLICY_PROMPT = """
If a question cannot be answered using only the provided CONTEXT, return:
{"final_answer":"","start_char":-1,"end_char":-1,"analysis":"No supporting text in the provided context."}
Do NOT attempt to hallucinate external knowledge.
"""
