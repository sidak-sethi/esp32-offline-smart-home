import os
import glob
import numpy as np
import librosa

PROJECT_DIR = os.path.dirname(os.path.dirname(__file__))

INPUT_DIR = os.path.join(
    PROJECT_DIR,
    "trimmed_speaker_dataset"
)

OUTPUT_DIR = os.path.join(
    PROJECT_DIR,
    "logmel_features_fixed"
)

SAMPLE_RATE = 16000
N_FFT = 512
HOP_LENGTH = 160
WIN_LENGTH = 400
N_MELS = 32

TARGET_FRAMES = 201
PAD_VALUE = -80.0

COMMANDS = [
    "light_on",
    "light_off",
    "fan_on",
    "fan_off"
]

os.makedirs(OUTPUT_DIR, exist_ok=True)


for speaker in sorted(os.listdir(INPUT_DIR)):

    speaker_dir = os.path.join(
        INPUT_DIR,
        speaker
    )

    if not os.path.isdir(speaker_dir):
        continue

    for command in COMMANDS:

        input_dir = os.path.join(
            speaker_dir,
            command
        )

        output_dir = os.path.join(
            OUTPUT_DIR,
            speaker,
            command
        )

        os.makedirs(
            output_dir,
            exist_ok=True
        )

        files = glob.glob(
            os.path.join(input_dir, "*.wav")
        )

        for file in files:

            audio, sr = librosa.load(
                file,
                sr=SAMPLE_RATE,
                mono=True
            )

            mel = librosa.feature.melspectrogram(
                y=audio,
                sr=sr,
                n_fft=N_FFT,
                hop_length=HOP_LENGTH,
                win_length=WIN_LENGTH,
                n_mels=N_MELS,
                power=2.0
            )

            log_mel = librosa.power_to_db(
                mel,
                ref=np.max
            )

            # Fixed-size matrix
            fixed = np.full(
                (N_MELS, TARGET_FRAMES),
                PAD_VALUE,
                dtype=np.float32
            )

            frames = min(
                log_mel.shape[1],
                TARGET_FRAMES
            )

            fixed[:, :frames] = log_mel[:, :frames]

            filename = os.path.splitext(
                os.path.basename(file)
            )[0]

            output_file = os.path.join(
                output_dir,
                filename + ".npy"
            )

            np.save(
                output_file,
                fixed
            )

print("Fixed-size Log-Mel extraction complete.")
print(
    f"Feature shape: "
    f"({N_MELS}, {TARGET_FRAMES})"
)