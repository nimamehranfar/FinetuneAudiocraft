import json
import random

# Paths
input_jsonl = "TrimmedPersianMusic/manifest_finetune.jsonl"
train_jsonl = "TrimmedPersianMusic/manifest_finetune_train.jsonl"
valid_jsonl = "TrimmedPersianMusic/manifest_finetune_valid.jsonl"
test_jsonl  = "TrimmedPersianMusic/manifest_finetune_test.jsonl"

# Load all lines
with open(input_jsonl, "r", encoding="utf-8") as f:
    data = [json.loads(line) for line in f]

random.seed(72)
random.shuffle(data)
n = len(data)
n_train = int(0.948 * n)
n_valid = int(0.05 * n)
n_test  = n - n_train - n_valid  # Whatever is left

train_data = data[:n_train]
valid_data = data[n_train:n_train+n_valid]
test_data  = data[n_train+n_valid:]

with open(train_jsonl, "w", encoding="utf-8") as f:
    for item in train_data:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

with open(valid_jsonl, "w", encoding="utf-8") as f:
    for item in valid_data:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

with open(test_jsonl, "w", encoding="utf-8") as f:
    for item in test_data:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

print(f"Train: {len(train_data)}\nValid: {len(valid_data)}\nTest: {len(test_data)}")
