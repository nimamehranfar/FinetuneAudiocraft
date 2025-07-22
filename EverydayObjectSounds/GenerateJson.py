import pandas as pd
import os
import json

def tags_to_natural(tags):
    tag_list = [t.replace('_', ' ').strip() for t in tags.split(',') if t.strip()]
    if not tag_list:
        return ""
    if len(tag_list) == 1:
        return tag_list[0]
    return ", ".join(tag_list[:-1]) + " and " + tag_list[-1]

csv_path = "musicgen_train_manifest.csv"
df = pd.read_csv(csv_path)

output_folder = "musicgen_outputs"

for _, row in df.iterrows():
    audio_path = row["audio"]  # e.g., musicgen_outputs/100030_classical_seed72.wav
    basename = os.path.splitext(os.path.basename(audio_path))[0]
    json_path = os.path.join(output_folder, f"{basename}.json")

    text = row["text"].strip()

    # Detect if already in natural language (starts with 'A ' or 'An ')
    if text.lower().startswith(('a ', 'an ')):
        description = text
    else:
        # Assume "Genre tag1,tag2,..."
        if " " in text:
            genre, tags = text.split(" ", 1)
            description = f"A {genre.capitalize()} track"
            natural = tags_to_natural(tags)
            if natural:
                description += f" with sounds of {natural}."
            else:
                description += "."
        else:
            description = f"A {text.capitalize()} track."

    genre = text.split(" ")[0].capitalize() if " " in text else text.capitalize()
    keywords = text

    info = {
        "key": "",
        "artist": "Voyager I",
        "sample_rate": 32000,
        "file_extension": "wav",
        "description": description,
        "keywords": keywords,
        "duration": 7.0,
        "bpm": "",
        "genre": genre,
        "title": basename,
        "name": basename,
        "instrument": "Mix",
        "moods": [],
    }
    with open(json_path, "w", encoding="utf-8") as jf:
        json.dump(info, jf, ensure_ascii=False, indent=2)
    print(f"Wrote {json_path}")

print("All JSONs written!")
