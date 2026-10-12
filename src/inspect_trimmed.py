import os
import glob
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt

PROJECT_DIR = os.path.dirname(os.path.dirname(__file__))

ORIGINAL_DIR = os.path.join(
    PROJECT_DIR,
    "dataset"
)

TRIMMED_DIR = os.path.join(
    PROJECT_DIR,
    "trimmed_dataset"
)

CLASSES = [
    "light_on",
    "light_off",
    "fan_on",
    "fan_off",
    "unknown"
]


def duration(path):
    audio, sr = sf.read(path)
    return len(audio) / sr


for label in CLASSES:

    files = glob.glob(
        os.path.join(
            TRIMMED_DIR,
            label,
            "*.wav"
        )
    )

    files.sort(
        key=duration
    )

    print(f"\n{label} shortest files:")

    for file in files[:3]:

        filename = os.path.basename(file)

        original = os.path.join(
            ORIGINAL_DIR,
            label,
            filename
        )

        trimmed = file

        original_audio, sr = sf.read(
            original
        )

        trimmed_audio, _ = sf.read(
            trimmed
        )

        print(
            f"{filename}: "
            f"{len(original_audio)/sr:.2f}s -> "
            f"{len(trimmed_audio)/sr:.2f}s"
        )

        # Plot
        plt.figure(figsize=(10, 4))

        t1 = np.arange(
            len(original_audio)
        ) / sr

        t2 = np.arange(
            len(trimmed_audio)
        ) / sr

        plt.plot(
            t1,
            original_audio,
            label="Original"
        )

        plt.plot(
            t2,
            trimmed_audio,
            label="Trimmed"
        )

        plt.title(
            f"{label} - {filename}"
        )

        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")

        plt.legend()
        plt.tight_layout()
        plt.show()