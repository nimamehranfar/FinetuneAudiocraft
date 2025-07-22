import pandas as pd
import numpy as np
import os
from collections import defaultdict

# PARAMETERS
eval_csv = 'FSD50K.ground_truth/eval_filtered.csv'
vocab_csv = 'FSD50K.ground_truth/vocabulary.csv'
target_count = 600
out_txt = 'FSD50K.ground_truth/selected_eval_subset.txt'
random_seed = 72

# --- 1. Load vocabulary ---
vocab_df = pd.read_csv(vocab_csv, sep=',', header=None)
print(vocab_df.head())
colnames = vocab_df.columns.tolist()
print("Column names:", colnames)
mid_to_class = dict(zip(vocab_df[colnames[2]], vocab_df[colnames[1]]))

all_vocab_mids = set(mid_to_class.keys())
all_vocab_classes = set(mid_to_class.values())

# --- 2. Load eval.csv ---
eval_df = pd.read_csv(eval_csv, sep=',')
file_to_labels = {}
file_to_mids = {}
mid_to_files = defaultdict(set)

for idx, row in eval_df.iterrows():
    fname = str(row['fname'])
    labels = row['labels'].split(',')
    mids = row['mids'].split(',') if 'mids' in row and isinstance(row['mids'], str) else []
    file_to_labels[fname] = labels
    file_to_mids[fname] = mids
    for mid in mids:
        mid_to_files[mid].add(fname)

# --- 3. Ensure each vocabulary class is present at least once ---
selected_files = set()
np.random.seed(random_seed)
for mid in all_vocab_mids:
    files = list(mid_to_files.get(mid, []))
    if files:
        selected_files.add(np.random.choice(files))

print(f"Step 1: Picked {len(selected_files)} files to cover all {len(all_vocab_mids)} MIDs/classes.")

# --- 4. Randomly sample the rest (to reach target_count) ---
remaining_needed = target_count - len(selected_files)
all_files = set(file_to_labels.keys())
remaining_files = list(all_files - selected_files)
extra_files = np.random.choice(remaining_files, size=remaining_needed, replace=False)
selected_files.update(extra_files)

selected_files = sorted(selected_files)
print(f"Step 2: Final selected set is {len(selected_files)} files.")

# --- 5. Write filenames to text file ---
with open(out_txt, "w") as f:
    for fname in selected_files:
        f.write(f"{fname}\n")

print(f"Selected file names written to {out_txt}.")

# --- 6. Optional: Check which classes are present in the selection ---
selected_mids = set()
for fname in selected_files:
    selected_mids.update(file_to_mids.get(fname, []))
selected_classnames = {mid_to_class[mid] for mid in selected_mids if mid in mid_to_class}
print(f"Selection covers {len(selected_classnames)}/{len(all_vocab_classes)} unique classes from vocabulary.")
