import os
import torch
import torchaudio
import numpy as np
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write
from audiocraft.utils.autocast import TorchAutocast
import json

# Settings
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

orig_init = MusicGen.__init__
def safe_init(self, name, compression_model, lm, max_duration=None):
    # Normal init up to setting max_duration
    if max_duration is None and hasattr(lm, 'cfg'):
        max_duration = 20
        # max_duration = lm.cfg.dataset.segment_duration
    elif max_duration is None:
        raise ValueError("You must provide max_duration when building directly MusicGen")
    self.name = name
    self.compression_model = compression_model
    self.lm = lm
    self.max_duration: float = max_duration
    self.device = next(iter(lm.parameters())).device
    self.generation_params: dict = {}
    # Use a duration slightly less than max_duration to avoid assertion error
    safe_duration = max(0.5, self.max_duration - 0.1)
    # Avoid calling set_generation_params with a duration that is too high
    self.set_generation_params(duration=safe_duration)
    self._progress_callback = None
    if self.device.type == 'cpu':
        self.autocast = TorchAutocast(enabled=False)
    else:
        self.autocast = TorchAutocast(
            enabled=True, device_type=self.device.type, dtype=torch.float16
        )

MusicGen.__init__ = safe_init

model = MusicGen.get_pretrained("my_finetune_model", device=device)
model.set_generation_params(duration=8.0)

# Config
# descriptions = ["A jazz track with sounds of Glass Shatter"]
seeds = [72,216,1381]
trim_start = 0.0
trim_end = 4.0
maximum_size = 8
min_required_length = 4  # seconds — below this we skip trimming

os.makedirs("finetune_continuation_outputs", exist_ok=True)

sound_files = ['TrimmedPersianMusic/000467_8_16.wav','TrimmedPersianMusic/000333_64_72.wav','TrimmedPersianMusic/000635_32_40.wav','TrimmedPersianMusic/000806_40_48.wav','TrimmedPersianMusic/000953_32_40.wav','TrimmedPersianMusic/000770_32_40.wav','TrimmedPersianMusic/000379_24_32.wav']
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

    # If audio is too short to trim, skip trimming
    if sample_length < 4.0:
        print(f"Audio too short to trim (< 4.0s), using full sample.")
        trimmed = audio
    else:
        trim_start_sec = min(trim_start, sample_length - 0.5)
        trim_end_sec = min(trim_end, sample_length - 0.5)

        if trim_start_sec + trim_end_sec >= sample_length:
            trim_start_sec = trim_end_sec = (sample_length - 0.5) / 2

        start_idx = int(sr * trim_start_sec)
        end_idx = int(sr * (sample_length - trim_end_sec))
        trimmed = audio[start_idx:end_idx]

    # trimmed = audio
    trimmed_length = trimmed.shape[0] / sr
    if trimmed_length > maximum_size:
        cut = int(sr * (trimmed_length - maximum_size))
        trimmed = trimmed[cut:]

    if trimmed.shape[0] < min_samples_required:
        print(f"Trimmed audio in {fn} too short after all adjustments. Skipping.")
        continue

    gen_duration = max(trimmed_length + 0.5, 8)
    model.set_generation_params(duration=gen_duration)

    description=[]

    json_fn = fn.replace(".wav", ".json")
    if not os.path.exists(json_fn):
        print(f"Warning: JSON metadata not found for {fn} (expected {json_fn})")
        description = []
    else:
        with open(json_fn, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        # This can be a string, or you can modify to combine with keywords, genre, etc.
        # For multiple descriptions, you could use a list: [metadata["description"], metadata["keywords"], ...]
        description = [metadata.get("description", "")]

    if not description or description[0] == "":
        print(f"No valid description found for {fn}, skipping.")
        continue

    # Generate continuations for each genre
    for i, desc in enumerate(description):
        for seed in seeds:
            torch.manual_seed(seed)
            if device == "cuda":
                torch.cuda.manual_seed_all(seed)

            # print(f"trimmed shape before generate_continuation: {trimmed.shape}")
            # mono_trimmed = audio.t()
            mono_trimmed = trimmed.t()

            print(f"Converted to mono: shape {mono_trimmed.shape}")
            print(f"Audio length in samples: {mono_trimmed.shape[1]}")

            out = model.generate_continuation(
                prompt=mono_trimmed.to(device),
                prompt_sample_rate=sr,
                descriptions=[desc],
                progress=True,
                return_tokens=False
            )
            out_path = f"finetune_continuation_outputs/{os.path.basename(fn)[:-4]}_{desc}_{seed}"
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
            audio_write(out_path, generated_only, model.sample_rate, strategy="loudness",loudness_headroom_db=16,loudness_compressor=True)

            print(f"Saved: {out_path}")
