import os
import glob
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt

DATASET_DIR = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "dataset"
)

for label in sorted(os.listdir(DATASET_DIR)):
    folder = os.path.join(DATASET_DIR, label)

    if not os.path.isdir(folder):
        continue

    files = glob.glob(os.path.join(folder, "*.wav"))

    if not files:
        continue

    print(f"\n{label}: {len(files)} files")

    for file in files[:2]:
        audio, sr = sf.read(file)

        print(
            os.path.basename(file),
            f"| sample rate={sr}",
            f"| channels={1 if audio.ndim == 1 else audio.shape[1]}",
            f"| duration={len(audio)/sr:.2f}s",
            f"| peak={np.max(np.abs(audio)):.4f}"
        )

    audio, sr = sf.read(files[0])

    plt.figure(figsize=(10, 3))
    time = np.arange(len(audio)) / sr

    plt.plot(time, audio)
    plt.title(f"{label} - {os.path.basename(files[0])}")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.tight_layout()
    plt.show()