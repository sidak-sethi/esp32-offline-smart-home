import os
import glob
import numpy as np
import librosa

DATASET_DIR = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "dataset"
)

FEATURE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "features"
)

SAMPLE_RATE = 16000
N_MFCC = 13
N_FFT = 512
HOP_LENGTH = 160
WIN_LENGTH = 400

CLASSES = [
    "light_on",
    "light_off",
    "fan_on",
    "fan_off",
    "unknown",
    "silence"
]

os.makedirs(FEATURE_DIR, exist_ok=True)

for label in CLASSES:

    input_folder = os.path.join(DATASET_DIR, label)
    output_folder = os.path.join(FEATURE_DIR, label)

    os.makedirs(output_folder, exist_ok=True)

    files = glob.glob(
        os.path.join(input_folder, "*.wav")
    )

    print(f"{label}: {len(files)} files")

    for file in files:

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

        filename = os.path.splitext(
            os.path.basename(file)
        )[0]

        output_file = os.path.join(
            output_folder,
            filename + ".npy"
        )

        np.save(output_file, mfcc)

print("\nFeature extraction complete.")