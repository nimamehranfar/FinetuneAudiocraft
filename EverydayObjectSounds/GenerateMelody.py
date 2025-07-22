import os
import torch
import torchaudio
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

model = MusicGen.get_pretrained("facebook/musicgen-melody", device=device)
model.set_generation_params(duration=8.0)  # Generate 8 seconds

os.makedirs("./musicgen_outputs", exist_ok=True)

sound_files = ["selected_sounds/" + fn for fn in os.listdir("selected_sounds") if fn.endswith(".wav")]

# Define your 3 genre descriptions
genre_descriptions = ["pop", "jazz", "classical"]
seeds = [76]

for fn in sound_files:
    audio, sr = torchaudio.load(fn)
    print(f"Loaded: {fn} | Sample rate: {sr}")

    # Convert to mono if stereo
    if audio.shape[0] > 1:
        audio = audio.mean(dim=0, keepdim=True)

    # Resample to 32000 Hz (model requirement)
    if sr != 32000:
        audio = torchaudio.functional.resample(audio, sr, 32000)
        sr = 32000

    # Expand the audio batch to match descriptions length
    melody_batch = audio.unsqueeze(0).to(device)

    # Loop through each (seed, description) pair
    for desc, seed in zip(genre_descriptions, seeds):
        # Set the seed for reproducibility
        torch.manual_seed(seed)
        if device == "cuda":
            torch.cuda.manual_seed_all(seed)

        # Generate with chroma (melody conditioning)
        wavs = model.generate_with_chroma([desc], melody_batch, sr)

        # Save each generated sample
        for i, wav in enumerate(wavs):
            out_path = f"./musicgen_outputs/{os.path.basename(fn)[:-4]}_{desc}_seed{seed}.wav"
            audio_write(out_path, wav.cpu(), model.sample_rate, strategy="loudness", loudness_compressor=True)
            print("Saved:", out_path)
