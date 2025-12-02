# scripts/prepare_asdiv_for_maas.py
# Usage: python scripts/prepare_asdiv_for_maas.py --out_dir maas/ext/maas/data --split test

import argparse
from datasets import load_dataset
import json
import os
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--out_dir", default="maas/ext/maas/data")
parser.add_argument("--split", default="test")   # or 'train'
parser.add_argument("--variant", default="yimingzhang/asdiv")
args = parser.parse_args()

# load dataset from Hugging Face
ds = load_dataset(args.variant)

# choose split: some variants use 'train','test'
if args.split not in ds:
    # fallback to 'validation' or 'train' if test absent
    if "test" in ds:
        split = "test"
    elif "validation" in ds:
        split = "validation"
    else:
        split = "train"
else:
    split = args.split

data = ds[split]

os.makedirs(args.out_dir, exist_ok=True)
out_path = os.path.join(args.out_dir, f"asdiv_{split}.jsonl")

def to_maas_item(i, item):
    # ASDiv / HuggingFace rows often contain: 'id' (or index), 'question', 'answer' (a string) or 'formula'
    q = item.get("question") or item.get("problem") or item.get("text") or item.get("input") or ""
    # many ASDiv variants store 'answer' under 'answer' or 'output' or compute from 'equation'
    a = item.get("answer") or item.get("label") or item.get("output") or ""
    # some ASDiv variants include a 'chain' or 'solution' field
    solution = item.get("chain") or item.get("solution") or item.get("explanation") or None

    # If answer is missing but chain contains final result in tags, you may need to parse; for now keep as string.
    return {
        "id": int(item.get("id", i)),
        "question": q,
        "answer": str(a).strip(),
        "solution": solution
    }

with open(out_path, "w", encoding="utf-8") as fo:
    for i, it in enumerate(tqdm(data)):
        obj = to_maas_item(i, it)
        fo.write(json.dumps(obj, ensure_ascii=False) + "\n")

print("Wrote:", out_path)
