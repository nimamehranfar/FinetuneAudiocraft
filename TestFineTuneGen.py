import torch
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write

device = "cuda" if torch.cuda.is_available() else "cpu"

# checkpoint_path = f"S:/020819d8/checkpoint_5.th"
# ckpt = torch.load(checkpoint_path, map_location=device)
# # print("Top-level keys:", ckpt.keys())
#
# model = MusicGen.get_pretrained("facebook/musicgen-small", device=device)
# model.lm.load_state_dict(ckpt["model"])


from audiocraft.utils.autocast import TorchAutocast
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

# 5. Now you can generate audio!
model.set_generation_params(duration=8.0)
descriptions=["a persian dastgah track of chahargah"]


# Generate continuations for each genre
for i, desc in enumerate(descriptions):
    outputs = model.generate(descriptions=descriptions)


    out_path = f"finetune_musicgen_outputs/{desc}"
    audio_write(out_path, outputs[0].cpu(), model.sample_rate, strategy="loudness", loudness_compressor=True)

print("Generation complete.")
