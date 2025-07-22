import os
import torch
import torchaudio
import numpy as np
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write

# Settings
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

checkpoint_path = "checkpoint.th"

ckpt = torch.load(checkpoint_path, map_location=device)
model = MusicGen.get_pretrained("facebook/musicgen-medium", device=device)
# model.lm.load_state_dict(ckpt["model"])

model.set_generation_params(duration=7.0)

# Config
descriptions = ["A jazz track with sounds of Glass Shatter"]

os.makedirs("musicgen_continuation_outputs", exist_ok=True)

sound_files = ["musicgen_outputs/20133_jazz_seed72.wav"]
# sound_files = ["musicgen_outputs/" + fn for fn in os.listdir("musicgen_outputs") if fn.endswith(".wav")]


def normalize_audio(audio_data):
    audio_data = audio_data.astype(np.float32)
    max_value = np.max(np.abs(audio_data))
    audio_data /= max_value
    return audio_data


for fn in sound_files:
    print(f"Processing: {fn}")
    audio, sr = torchaudio.load(fn)
    audio = normalize_audio(audio.numpy())  # (channels, time)
    audio = torch.from_numpy(audio).t()  # (time, channels)

    # Safeguard: skip audio that’s physically too short
    min_samples_required = 1024  # to be safe
    if audio.shape[0] < min_samples_required:
        print(f"Skipping {fn}: too short ({audio.shape[0]} samples).")
        continue

    sample_length = audio.shape[0] / sr

    gen_duration = max(sample_length + 0.5, 14.0)
    model.set_generation_params(duration=gen_duration)


    # Generate continuations for each genre
    for i, desc in enumerate(descriptions):
        # torch.manual_seed(42 + i)
        # print(f"trimmed shape before generate_continuation: {trimmed.shape}")
        mono_trimmed = audio.t()

        print(f"Converted to mono: shape {mono_trimmed.shape}")
        print(f"Audio length in samples: {mono_trimmed.shape[1]}")

        out = model.generate_continuation(
            prompt=mono_trimmed.to(device),
            prompt_sample_rate=sr,
            descriptions=[desc],
            progress=True,
            return_tokens=False
        )
        out_path = f"musicgen_continuation_outputs/{os.path.basename(fn)[:-4]}_{desc}.wav"
        out_audio = out[0].cpu()

        # prompt_duration_sec = audio.shape[0] / sr
        # prompt_samples = int(model.sample_rate * prompt_duration_sec)
        # generated_only = out_audio[:, prompt_samples:]  # Slice out the prompt part
        generated_only = out_audio  # Slice out the prompt part

        max_val = generated_only.abs().max()

        if max_val > 1:
            generated_only = generated_only / max_val
            print(f"  normalized max: {generated_only.abs().max().item()}")

        # Apply clamp as last line of defense
        generated_only = generated_only.clamp(-1, 1)
        audio_write(out_path, generated_only, model.sample_rate, strategy="loudness",loudness_headroom_db=16,loudness_compressor=True,add_suffix=False)


        print(f"Saved: {out_path}")
