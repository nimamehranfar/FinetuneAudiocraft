import os
from pydub import AudioSegment

# Path to SoundSamples and Label.txt
sound_samples_dir = "PersianMusic/Data/Music/SoundSamples"
label_file = "PersianMusic/Music/Avaz/Label.txt"

# # Get all Avaz file bases from label.txt (every even line)
# with open(label_file, encoding='utf-8') as f:
#     lines = [line.strip() for line in f if line.strip()]
#
# for i in range(0, len(lines), 2):
#     file_base = lines[i]
#     # Avaz audio file is {file_base}.mp3.wav
#     fname = f"{file_base}.mp3.wav"
#     file_path = os.path.join(sound_samples_dir, fname)
#     if os.path.exists(file_path):
#         os.remove(file_path)
#         print("Deleted:", fname)
#     else:
#         print("File not found:", fname)
#
# print("Done! All Avaz audio files are deleted from SoundSamples.")


# for fname in os.listdir(sound_samples_dir):
#     # Look for .mp3.wav files
#     if fname.endswith('.mp3.wav'):
#         src = os.path.join(sound_samples_dir, fname)
#         # Remove the .mp3 part
#         new_fname = fname.replace('.mp3.wav', '.wav')
#         dst = os.path.join(sound_samples_dir, new_fname)
#         os.rename(src, dst)
#         print(f"Renamed: {fname} -> {new_fname}")
#
# print("Done! All files are now *.wav without .mp3 in the name.")


input_dir = f"S:/PersianMusic/Data/Music/SoundSamples"
output_dir = "TrimmedPersianMusic"
os.makedirs(output_dir, exist_ok=True)

# Parameters
MIN_LEN = 8  # seconds
MAX_LEN = 8  # seconds
MAX_TOTAL = 8255

# Get files sorted by size (ascending)
files = [f for f in os.listdir(input_dir) if f.endswith('.wav')]
files = sorted(files, key=lambda x: os.path.getsize(os.path.join(input_dir, x)))

counter = 0

for fname in files:
    audio_path = os.path.join(input_dir, fname)
    audio = AudioSegment.from_wav(audio_path)
    total_len_sec = len(audio) / 1000.0
    start = 0
    while start < total_len_sec and counter < MAX_TOTAL:
        end = min(start + MAX_LEN, total_len_sec)
        seg_len = end - start
        if seg_len < MIN_LEN:
            break  # skip too short section
        trimmed = audio[start * 1000 : end * 1000]
        trimmed_name = f"{os.path.splitext(fname)[0]}_{int(start)}_{int(end)}.wav"
        trimmed_path = os.path.join(output_dir, trimmed_name)
        trimmed.export(trimmed_path, format="wav")
        print(f"Saved {trimmed_name}")
        counter += 1
        start += MAX_LEN
        if counter >= MAX_TOTAL:
            break
    if counter >= MAX_TOTAL:
        break

print(f"Done! {counter} trimmed audio files between 8 and 8 seconds saved to {output_dir}")
