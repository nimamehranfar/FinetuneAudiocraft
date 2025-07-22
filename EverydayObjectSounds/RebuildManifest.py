import os
import pandas as pd
import csv

# Settings
audio_dir = "musicgen_outputs"
eval_csv = "FSD50K.ground_truth/eval.csv"
manifest_path = os.path.join(audio_dir, "manifest.tsv")

# Load eval.csv
eval_df = pd.read_csv(eval_csv, sep=',')
fname_to_labels = dict(zip(eval_df['fname'].astype(str), eval_df['labels']))

# Helper: extract base name (original id) from generated file name
def get_base_and_desc_seed(filename):
    # E.g., "100030_classical_seed72.wav" -> ("100030", "classical", "72")
    name = filename.rsplit('.', 1)[0]
    base, desc, seed = name.rsplit('_', 2)
    return base, desc, seed

# Gather all generated files
generated_files = [f for f in os.listdir(audio_dir) if f.endswith('.wav') and not f.startswith('.')]

# Build manifest rows
rows = []
for fname in generated_files:
    try:
        base, desc, seed = get_base_and_desc_seed(fname)
        label = fname_to_labels.get(base, "UNKNOWN")
        rows.append(f"{fname}\t{desc}\t{label}")
    except Exception as e:
        print(f"Skipped {fname}: {e}")

# Write manifest
with open(manifest_path, "w") as f:
    f.write("filename\tdescription\tlabels\n")
    for row in rows:
        f.write(row + "\n")

print(f"Rebuilt manifest with {len(rows)} entries at {manifest_path}.")


audio_dir = "musicgen_outputs"
eval_csv = "FSD50K.ground_truth/eval.csv"
manifest_path = os.path.join(audio_dir, "manifest.csv")  # .csv

eval_df = pd.read_csv(eval_csv, sep=',')
fname_to_labels = dict(zip(eval_df['fname'].astype(str), eval_df['labels']))

def get_base_and_desc_seed(filename):
    # E.g., "100030_classical_seed72.wav" -> ("100030", "classical", "72")
    name = filename.rsplit('.', 1)[0]
    base, desc, seed = name.rsplit('_', 2)
    return base, desc, seed

generated_files = [f for f in os.listdir(audio_dir) if f.endswith('.wav') and not f.startswith('.')]

with open(manifest_path, "w", newline='', encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["filename", "description", "labels"])
    for fname in generated_files:
        try:
            base, desc, seed = get_base_and_desc_seed(fname)
            label = fname_to_labels.get(base, "UNKNOWN")
            writer.writerow([fname, desc, label])
        except Exception as e:
            print(f"Skipped {fname}: {e}")

print(f"Rebuilt manifest with {len(generated_files)} entries at {manifest_path}.")


def tags_to_natural(tags):
    tag_list = [t.replace('_', ' ').strip() for t in tags.split(',') if t.strip()]
    if not tag_list:
        return ""
    if len(tag_list) == 1:
        return tag_list[0]
    return ", ".join(tag_list[:-1]) + " and " + tag_list[-1]

in_path = "musicgen_outputs/manifest.tsv"  # or your manifest location
out_path = "musicgen_outputs/musicgen_2x_manifest.csv"

df = pd.read_csv(in_path, sep='\t')
with open(out_path, "w", encoding="utf-8") as f:
    f.write("text,audio\n")
    for i, row in df.iterrows():
        genre = row['description'].capitalize()
        tags = row['labels']
        tags_natural = tags_to_natural(tags)
        # Row 1: Comma-separated
        prompt_comma = f"{genre} {tags}"
        audio_path = f"musicgen_outputs/{row['filename']}"
        f.write(f"\"{prompt_comma}\",{audio_path}\n")
        # Row 2: Natural language
        if tags_natural:
            prompt_nl = f"A {genre} track with sounds of {tags_natural}."
        else:
            prompt_nl = f"A {genre} track."
        f.write(f"\"{prompt_nl}\",{audio_path}\n")

print(f"New 2x manifest written to {out_path}. Rows: {df.shape[0]*2}")
