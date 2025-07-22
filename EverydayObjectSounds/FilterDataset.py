import pandas as pd
import os

# === PATHS ===
eval_csv_path = "FSD50K.ground_truth/eval.csv"
vocab_csv_path = "FSD50K.ground_truth/vocabulary.csv"
filtered_csv_path = "FSD50K.ground_truth/eval_filtered.csv"
audio_folder = "FSD50K.eval_audio"  # change this if your folder is elsewhere

# === LOAD VOCABULARY ===
vocab_df = pd.read_csv(vocab_csv_path, sep="\t", header=None, names=["index", "label", "mid"])
mid_to_label = dict(zip(vocab_df["mid"], vocab_df["label"]))

# === LOAD EVAL CSV ===
eval_df = pd.read_csv(eval_csv_path, sep=",")
eval_df.columns = eval_df.columns.str.strip()

# Display columns for debug
print("Columns in eval_df:", eval_df.columns.tolist())
print(eval_df.head())

# === EXACT UNWANTED LABELS (as in vocabulary.csv) ===
unwanted_labels = [
    "Speech",
    "Conversation",
    "Male_speech_and_man_speaking",
    "Female_speech_and_woman_speaking",
    "Child_speech_and_kid_speaking",
    "Singing",
    "Male_singing",
    "Female_singing",
    "Giggle",
    "Chuckle_and_chortle",
    "Crying_and_sobbing",
    "Cry",
    "Gasp",
    "Shout",
    "Yell",
    "Whispering",
    "Cough",
    "Sneeze",
    "Breathing",
    "Sniff",
    "Burping_and_eructation",
    "Fart",
    "Applause",
    "Crowd",
    "Cheering",
    "Explosion",
    "Siren",
    "Boom",
    "Car",
    "Bus",
    "Truck",
    "Train",
    "Subway_and_metro_and_underground",
    "Traffic_noise_and_roadway_noise",
    "Vehicle",
    "Motorcycle",
    "Aircraft",
    "Fixed-wing_aircraft_and_airplane",
    "Boat_and_Water_vehicle",
    "Music",
    "Musical_instrument",
    "Accordion",
    "Acoustic_guitar",
    "Bass_drum",
    "Bass_guitar",
    "Bell",
    "Bowed_string_instrument",
    "Brass_instrument",
    "Church_bell",
    "Clapping",
    "Crash_cymbal",
    "Cymbal",
    "Drum",
    "Drum_kit",
    "Electric_guitar",
    "Finger_snapping",
    "Glockenspiel",
    "Gong",
    "Harmonica",
    "Harp",
    "Hi-hat",
    "Keyboard_(musical)",
    "Mallet_percussion",
    "Marimba_and_xylophone",
    "Organ",
    "Percussion",
    "Piano",
    "Plucked_string_instrument",
    "Snare_drum",
    "Tabla",
    "Tambourine",
    "Trumpet",
    "Wind_instrument_and_woodwind_instrument",
    "Strum",
    "Scratching_(performance_technique)",
    "Rattle_(instrument)",
    "Laughter",
    "Toilet_flush",
    "Screaming"

]

# Lowercase for matching
unwanted_labels_lower = set([label.lower() for label in unwanted_labels])

# === FILTER FUNCTION ===
def has_unwanted_class(row):
    # Check labels
    labels = str(row["labels"]).split(",") if pd.notnull(row["labels"]) else []
    for label in labels:
        label_clean = label.strip().lower()
        if label_clean in unwanted_labels_lower:
            return True

    # Check mids via vocabulary
    mids = str(row["mids"]).split(",") if pd.notnull(row["mids"]) else []
    for mid in mids:
        mid = mid.strip()
        if mid in mid_to_label:
            vocab_label = mid_to_label[mid].strip().lower()
            if vocab_label in unwanted_labels_lower:
                return True

    return False

# === FILTER ===
filtered_df = eval_df[~eval_df.apply(has_unwanted_class, axis=1)]

# === SAVE FILTERED CSV ===
filtered_df.to_csv(filtered_csv_path, sep=",", index=False, quoting=1)

print(f"Filtering complete. Kept {len(filtered_df)} of {len(eval_df)} rows.")
print(f"Filtered CSV saved to {filtered_csv_path}")

# === DELETE UNWANTED AUDIO FILES ===
# IMPORTANT: only run this if you're sure you want to delete files!
# Uncomment below to activate file deletion:

kept_files = set(filtered_df["fname"].astype(str) + ".wav")
deleted = 0
for f in os.listdir(audio_folder):
    if f.endswith(".wav") and f not in kept_files:
        os.remove(os.path.join(audio_folder, f))
        deleted += 1
        print(f"Deleted {f}")
print(f"Deleted {deleted} files from {audio_folder}")
