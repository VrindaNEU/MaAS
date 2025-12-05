# op_prompt.py
"""
Prompts adapted for extractive (SQuAD-style) QA.
Each prompt instructs the model to output a single JSON object only (no extra text).
Fields returned must be exact substrings of the provided context when valid.
"""

SC_ENSEMBLE_PROMPT = """
Given the following input (JSON):
Question: {question}
Context: {context}
Candidate solutions (labeled):
{solutions}

Task:
1) Identify which candidate solution is the best extractive answer for the Question.
   - Prefer candidates that appear **exactly** as substrings in the Context.
   - If multiple candidates are exact substrings, choose the one that occurs most frequently among the candidates.
   - If none of the candidates is an exact substring, choose the candidate that best matches a contiguous substring in Context and return that exact substring.
2) Output **only one** JSON object (no surrounding commentary) with these fields:
   - "solution_letter": single uppercase letter corresponding to the chosen candidate (e.g., "A").
   - "solution": the exact answer string as it appears in Context (must be a contiguous substring).
   - "start_char": integer start index (0-based) of the substring in Context.
   - "end_char": integer end index (exclusive) of the substring in Context.

Example output:
{"solution_letter": "B", "solution": "Saint Bernadette Soubirous", "start_char": 157, "end_char": 183}
"""

GENERATE_PROMPT = """
Task:
You are given a Context and a Question. Produce an extractive answer by locating the exact span in the Context that best answers the Question.

Input:
Context:
{context}

Question:
{question}

Output requirements:
- Return **only** a single JSON object (no extra text) containing:
  - "final_answer": the exact substring from Context that answers the Question.
  - "start_char": integer start index (0-based) of the substring in Context.
  - "end_char": integer end index (exclusive) of the substring in Context.
- If no exact substring in Context answers the question, return:
  {"final_answer": "", "start_char": -1, "end_char": -1}

Example:
Input Context: "The Scholastic magazine began as a one-page journal in September 1876."
Question: "When did the Scholastic Magazine begin publishing?"
Output:
{"final_answer":"September 1876","start_char":45,"end_char":60}
"""

GENERATE_COT_PROMPT = """
Chain-of-thought allowed (for internal analysis) but **final output must still be extractive**.

Input:
Context:
{context}

Question:
{question}

Instruction:
1) Optionally produce step-by-step analysis of how you locate the span (this will be captured in the "analysis" field).
2) The final answer must be an exact substring of Context.
3) Output a single JSON object (no extra text) with:
   - "final_answer": the exact answer string (substring from Context).
   - "start_char": integer 0-based start index of the answer in Context.
   - "end_char": integer end index (exclusive).
   - "analysis": an optional short string describing the steps you used to find the answer (may include internal reasoning).

If no extractive answer can be found, return:
{"final_answer": "", "start_char": -1, "end_char": -1, "analysis": "no extractive span found"}
"""

SELFREFINE_PROMPT = """
You are given Context, Question, and a candidate answer (candidate_answer).
Task:
- Verify whether candidate_answer is an exact contiguous substring of Context and whether it correctly answers the Question.
- If it is an exact substring and answers the question, return it unchanged with start/end offsets.
- If it is not an exact substring, search Context for the best exact substring that answers the question and return that span.
- If Context contains no suitable extractive span, return final_answer empty and start/end set to -1.

Input (fill placeholders):
Context:
{context}

Question:
{question}

Candidate answer:
{candidate_answer}

Output:
Return a single valid JSON object only (no extra text) with fields:
- "refined_answer": exact substring from Context or "".
- "start_char": integer start (0-based) or -1.
- "end_char": integer end (exclusive) or -1.
- "valid": true if refined_answer is non-empty, otherwise false.
- "note": optional short note only if valid is false (e.g., "no extractive span found").

Example valid output:
{"refined_answer":"a copper statue of Christ","start_char":73,"end_char":99,"valid":true}
"""
