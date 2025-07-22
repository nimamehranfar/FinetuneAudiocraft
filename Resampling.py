import os
import json
from pydub import AudioSegment

input_dir = f"C:/Users/mehra/IdeaProjects/FinetunePersianMusic/TrimmedPersianMusic"
output_dir = "TrimmedPersianMusic"
os.makedirs(output_dir, exist_ok=True)

for fname in os.listdir(input_dir):
    if fname.endswith('.wav'):
        audio = AudioSegment.from_wav(os.path.join(input_dir, fname))
        audio = audio.set_channels(1)
        audio = audio.set_frame_rate(32000)
        out_path = os.path.join(output_dir, fname)
        audio.export(out_path, format="wav")
        print("Exported:", out_path)
print("Done!")


# meta_dir = "TrimmedPersianMusic"
#
# for fname in os.listdir(meta_dir):
#     if fname.endswith('.json'):
#         path = os.path.join(meta_dir, fname)
#         with open(path, "r", encoding="utf-8") as f:
#             meta = json.load(f)
#         meta["sample_rate"] = 32000
#         with open(path, "w", encoding="utf-8") as f:
#             json.dump(meta, f, ensure_ascii=False, separators=(',', ':'))
# print("All metadata updated to 32000 Hz sample rate!")


