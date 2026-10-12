import os
import glob
import numpy as np
import librosa

PROJECT_DIR = os.path.dirname(os.path.dirname(__file__))

INPUT_DIR = os.path.join(
    PROJECT_DIR,
    "speaker_dataset"
)

OUTPUT_DIR = os.path.join(
    PROJECT_DIR,
    "speaker_features"
)

SAMPLE_RATE = 16000
N_MFCC = 13
N_FFT = 512
HOP_LENGTH = 160
WIN_LENGTH = 400

COMMANDS = [
    "light_on",
    "light_off",
    "fan_on",
    "fan_off"
]

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


for speaker in sorted(
    os.listdir(INPUT_DIR)
):

    speaker_input = os.path.join(
        INPUT_DIR,
        speaker
    )

    if not os.path.isdir(speaker_input):
        continue

    for command in COMMANDS:

        command_input = os.path.join(
            speaker_input,
            command
        )

        command_output = os.path.join(
            OUTPUT_DIR,
            speaker,
            command
        )

        os.makedirs(
            command_output,
            exist_ok=True
        )

        files = glob.glob(
            os.path.join(
                command_input,
                "*.wav"
            )
        )

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
                command_output,
                filename + ".npy"
            )

            np.save(
                output_file,
                mfcc
            )

print("Speaker feature extraction complete.")