# convert_squad_to_maas_fixed.py
import json
import argparse
import os
import uuid

def convert_squad_file(input_path, output_path, split="train", filter_unanswerable=True):
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    out_dir = os.path.dirname(output_path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    count_in = 0
    count_out = 0
    with open(output_path, "w", encoding="utf-8") as outf:
        for article in data.get("data", []):
            for para in article.get("paragraphs", []):
                context = para.get("context", "").strip()
                for qa in para.get("qas", []):
                    qid = qa.get("id") or str(uuid.uuid4())
                    question = qa.get("question", "").strip()
                    answers = qa.get("answers", [])
                    is_impossible = qa.get("is_impossible", False) or qa.get("unanswerable", False)
                    count_in += 1

                    if filter_unanswerable and is_impossible:
                        continue

                    answer_text = ""
                    if answers:
                        for a in answers:
                            if isinstance(a, dict) and a.get("text", "").strip():
                                answer_text = a.get("text").strip()
                                break
                    if not answer_text:
                        continue

                    # Produce both context/question keys and the input field (optional)
                    obj = {
                        "id": qid,
                        "context": context,             # <-- REQUIRED BY MAAS
                        "question": question,           # <-- REQUIRED BY MAAS
                        "input": f"Context: {context}\n\nQuestion: {question}",
                        "answer": answer_text,
                        "metadata": {
                            "dataset": "SQuAD",
                            "split": split
                        }
                    }

                    outf.write(json.dumps(obj, ensure_ascii=False) + "\n")
                    count_out += 1

    print(f"Read {count_in} QA pairs from {input_path}. Wrote {count_out} entries to {output_path}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to SQuAD json file (train/dev).")
    parser.add_argument("--output", required=True, help="Path to output JSONL file.")
    parser.add_argument("--split", default="train", help="Split name to record in metadata.")
    parser.add_argument("--filter_unanswerable", action="store_true", help="Filter unanswerable examples (SQuAD v2).")
    args = parser.parse_args()

    convert_squad_file(args.input, args.output, split=args.split, filter_unanswerable=args.filter_unanswerable)
