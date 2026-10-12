import os
import glob
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt

DATASET_DIR = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "dataset"
)

SAMPLE_RATE = 16000
N_MFCC = 13
N_FFT = 512
HOP_LENGTH = 160
WIN_LENGTH = 400

classes = [
    "light_on",
    "light_off",
    "fan_on",
    "fan_off",
    "unknown",
    "silence"
]

for label in classes:

    files = glob.glob(
        os.path.join(DATASET_DIR, label, "*.wav")
    )

    if not files:
        print(f"No files found for {label}")
        continue

    file = files[0]

    audio, sr = librosa.load(
        file,
        sr=SAMPLE_RATE,
        mono=True
    )

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sr,
        n_mfcc=N_MFCC,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        win_length=WIN_LENGTH
    )

    print(
        f"{label}: "
        f"audio={audio.shape}, "
        f"MFCC={mfcc.shape}"
    )

    plt.figure(figsize=(10, 4))

    librosa.display.specshow(
        mfcc,
        x_axis="time",
        sr=sr,
        hop_length=HOP_LENGTH
    )

    plt.colorbar()
    plt.title(f"MFCC - {label}")
    plt.xlabel("Time (s)")
    plt.ylabel("MFCC Coefficient")
    plt.tight_layout()
    plt.show()