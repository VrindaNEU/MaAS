import json

input_file = "maas\\ext\\maas\\data\\squad_train.jsonl"
output_file = "maas\\ext\\maas\\data\\squad_train_subset.jsonl"
num_examples = 80

with open(input_file, "r", encoding="utf-8") as f:
    lines = [json.loads(line) for line in f]

subset = lines[:num_examples]

with open(output_file, "w", encoding="utf-8") as f:
    for item in subset:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

print(f"Wrote {len(subset)} examples to {output_file}")
