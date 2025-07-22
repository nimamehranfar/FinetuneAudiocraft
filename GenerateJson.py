import os
import json

# Paths
trimmed_dir = "TrimmedPersianMusic"
label_path = r"S:\PersianMusic\Music\Dastgah\Label.txt"
class_path = r"S:\PersianMusic\Music\Dastgah\Class.txt"
# output_json_dir = os.path.join(trimmed_dir, "metadata")
output_json_dir = "TrimmedPersianMusic"
os.makedirs(output_json_dir, exist_ok=True)

# Persian dastgah
persian_instruments = "Tar, Setar, Santur, Kamancheh, Ney, Tombak, Oud, Violin, Daf"
moods = ["deep feelings, sorrow, poetry, storytelling, uplifting"]

# Load class mapping
class_map = {}
with open(class_path, encoding="utf-8") as f:
    lines = [line.strip() for line in f if line.strip()]
    for i in range(0, len(lines), 2):
        class_name = lines[i]
        class_idx = lines[i+1]
        class_map[class_idx] = class_name

# Load label mapping
label_map = {}
with open(label_path, encoding="utf-8") as f:
    lines = [line.strip() for line in f if line.strip()]
    for i in range(0, len(lines), 2):
        file_base = lines[i]
        class_idx = lines[i+1]
        label_map[file_base] = class_idx

# Loop over trimmed audio files
for fname in os.listdir(trimmed_dir):
    if not fname.endswith('.wav'):
        continue

    # Extract file base from trimmed name: first 6 chars
    file_base = fname[:6]
    # Defensive: handle missing mapping
    class_idx = label_map.get(file_base, None)
    if class_idx is None:
        print(f"WARNING: No class for {file_base} ({fname}), skipping.")
        continue

    class_name = class_map.get(class_idx, "Unknown")

    # Prepare fields
    json_dict = {
        "key": "",
        "artist": "Nima",
        "sample_rate": 32000,
        "file_extension": "wav",
        "description": f"a persian dastgah track of {class_name}",
        "keywords": f"persian dastgah {class_name}",
        "duration": 8.0,
        "bpm": "",
        "genre": "persian dastgah",
        "title": fname[:-4],
        "name": fname[:-4],
        "instrument": persian_instruments,
        "moods": moods
    }

    # Write JSON with the full file name as the json name
    json_path = os.path.join(output_json_dir, f"{fname[:-4]}.json")
    with open(json_path, "w", encoding="utf-8") as jf:
        json.dump(json_dict, jf, indent=2, ensure_ascii=False)

print("Done! Metadata JSONs saved to:", output_json_dir)
