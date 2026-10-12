import os
import glob
import numpy as np
import librosa.display
import matplotlib.pyplot as plt

PROJECT_DIR = os.path.dirname(os.path.dirname(__file__))

FEATURE_DIR = os.path.join(
    PROJECT_DIR,
    "logmel_features"
)

file = glob.glob(
    os.path.join(
        FEATURE_DIR,
        "speaker_1",
        "light_on",
        "*.npy"
    )
)[0]

mel = np.load(file)

plt.figure(figsize=(10, 4))

librosa.display.specshow(
    mel,
    x_axis="time",
    y_axis="mel",
    sr=16000,
    hop_length=160
)

plt.colorbar(
    format="%+2.0f dB"
)

plt.title(
    "Log-Mel Spectrogram - LIGHT_ON"
)

plt.tight_layout()
plt.show()

print("Shape:", mel.shape)