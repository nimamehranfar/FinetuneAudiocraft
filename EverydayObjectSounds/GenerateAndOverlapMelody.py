import gc
import os
import torch
import torchaudio
import pandas as pd
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write
import time
from datetime import timedelta

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)
torch.cuda.empty_cache()

# Load eval.csv for labels
eval_df = pd.read_csv('FSD50K.ground_truth/eval.csv', sep=',')
fname_to_labels = dict(zip(eval_df['fname'].astype(str), eval_df['labels']))

# Parameters
genre_descriptions = ["pop", "jazz", "classical"]
seeds = [72]
output_seconds = 7.0

model = MusicGen.get_pretrained("facebook/musicgen-melody", device=device)
model.set_generation_params(duration=output_seconds)

os.makedirs("./musicgen_outputs", exist_ok=True)

# Read selected file list
with open("FSD50K.ground_truth/selected_eval_subset.txt") as f:
    selected_names = set(line.strip() for line in f)

sound_files = [
    f for f in os.listdir("FSD50K.eval_audio")
    if f.endswith(".wav") and os.path.splitext(f)[0] in selected_names
]
sound_files = ["FSD50K.eval_audio/" + fn for fn in sound_files]
# sound_files = ["FSD50K.eval_audio/" + fn for fn in os.listdir("FSD50K.eval_audio") if fn.endswith(".wav")]

def overlay_first_n(orig, gen, sample_rate, seconds=7.0):
    """
    Overlay the first `seconds` of original with the first `seconds` of generated.
    Output is also exactly `seconds` long.
    """
    length_n = int(sample_rate * seconds)
    # Take/crop or pad both to n seconds
    orig_n = orig[..., :length_n]
    gen_n = gen[..., :length_n]

    # Pad if original is too short
    if orig_n.shape[-1] < length_n:
        pad = torch.zeros(orig_n.shape[0], length_n - orig_n.shape[-1])
        orig_n = torch.cat([orig_n, pad], dim=-1)
    if gen_n.shape[-1] < length_n:
        pad = torch.zeros(gen_n.shape[0], length_n - gen_n.shape[-1])
        gen_n = torch.cat([gen_n, pad], dim=-1)

    # Overlay: sum and normalize if necessary
    mixed = orig_n + gen_n
    max_val = mixed.abs().max()
    if max_val > 1:
        mixed = mixed / max_val
    return mixed.clamp(-1, 1)

# Optional: clear manifest file
with open("musicgen_outputs/manifest.tsv", "w") as f:
    f.write("filename\tdescription\tlabels\n")

total_files = len(sound_files) * len(genre_descriptions) * len(seeds)
processed = 0
skipped = 0
start_time = time.time()
last_time = start_time

for fn in sound_files:
    base = os.path.splitext(os.path.basename(fn))[0]
    if base not in fname_to_labels:
        print(f"{base} not found in eval.csv, skipping.")
        continue
    audio, sr = torchaudio.load(fn)
    # Mono
    if audio.shape[0] > 1:
        audio = audio.mean(dim=0, keepdim=True)
    # Resample
    if sr != 32000:
        audio = torchaudio.functional.resample(audio, sr, 32000)
        sr = 32000

    melody_batch = audio.unsqueeze(0).to(device)  # (1, 1, T)

    for desc in genre_descriptions:
        for seed in seeds:
            out_name = f"{base}_{desc}_seed{seed}.wav"
            out_path = os.path.join("musicgen_outputs", out_name)
            if os.path.exists(out_path):
                print(f"Already exists, skipping: {out_path}")
                skipped += 1
                continue  # Skip this combination if already done

            torch.manual_seed(seed)
            if device == "cuda":
                torch.cuda.manual_seed_all(seed)
            try:
                wavs = model.generate_with_chroma([desc], melody_batch, sr)

                for i, gen in enumerate(wavs):
                    mixed = overlay_first_n(audio, gen.cpu(), sr, seconds=output_seconds)
                    out_name = f"{base}_{desc}_seed{seed}"
                    out_path = os.path.join("musicgen_outputs", out_name)
                    audio_write(out_path, mixed, model.sample_rate, strategy="loudness", loudness_compressor=True)
                    print("Saved:", out_path)

                    # Save manifest entry
                    with open("musicgen_outputs/manifest.tsv", "a") as f:
                        f.write(f"{out_name}\t{desc}\t{fname_to_labels[base]}\n")

                    processed += 1

                    # Print timing and ETA every 10 files
                    if processed % 10 == 0 or processed == total_files:
                        now = time.time()
                        elapsed = now - start_time
                        avg_per_file = elapsed / processed
                        remaining = total_files - skipped - processed
                        eta = avg_per_file * remaining
                        print(
                            f"\nProcessed: {processed + skipped}/{total_files} | "
                            f"Elapsed: {str(timedelta(seconds=int(elapsed)))} | "
                            f"ETA: {str(timedelta(seconds=int(eta)))}\n"
                        )


                    # Every 100 outputs, extra cleanup
                    if (processed + skipped) % 100 == 0:
                        print(f"Cleaning up at file {processed + skipped}")
                        gc.collect()
                        torch.cuda.empty_cache()

            except RuntimeError as e:
                print(f"CUDA or other error for {base}, desc={desc}, seed={seed}: {e}")
                torch.cuda.empty_cache()
                continue
