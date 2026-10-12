import os
import glob
import numpy as np
import soundfile as sf
import librosa

PROJECT_DIR = os.path.dirname(os.path.dirname(__file__))

INPUT_DIR = os.path.join(
    PROJECT_DIR,
    "dataset"
)

OUTPUT_DIR = os.path.join(
    PROJECT_DIR,
    "trimmed_dataset"
)

SAMPLE_RATE = 16000

# Silence threshold
TOP_DB = 35

# Keep a little audio after detected speech
PADDING_MS = 150
PADDING_SAMPLES = int(
    SAMPLE_RATE * PADDING_MS / 1000
)

COMMAND_CLASSES = [
    "light_on",
    "light_off",
    "fan_on",
    "fan_off",
    "unknown"
]

# Silence files should NOT be trimmed.
# They are useful as real background/no-command examples.
ALL_CLASSES = COMMAND_CLASSES + ["silence"]


def trim_audio(audio):

    # Detect non-silent region
    trimmed, indices = librosa.effects.trim(
        audio,
        top_db=TOP_DB
    )

    start = indices[0]
    end = indices[1]

    # Keep a small amount of trailing context
    end = min(
        end + PADDING_SAMPLES,
        len(audio)
    )

    return audio[start:end], start, end


for label in ALL_CLASSES:

    input_folder = os.path.join(
        INPUT_DIR,
        label
    )

    output_folder = os.path.join(
        OUTPUT_DIR,
        label
    )

    os.makedirs(
        output_folder,
        exist_ok=True
    )

    files = glob.glob(
        os.path.join(
            input_folder,
            "*.wav"
        )
    )

    print(f"\n{label}: {len(files)} files")

    for file in files:

        audio, sr = sf.read(file)

        # Make sure mono
        if audio.ndim > 1:
            audio = np.mean(
                audio,
                axis=1
            )

        if label == "silence":

            # Keep silence/background recordings unchanged
            trimmed_audio = audio
            start = 0
            end = len(audio)

        else:

            trimmed_audio, start, end = trim_audio(
                audio
            )

        output_file = os.path.join(
            output_folder,
            os.path.basename(file)
        )

        sf.write(
            output_file,
            trimmed_audio,
            sr
        )

        original_duration = len(audio) / sr
        new_duration = len(trimmed_audio) / sr

        print(
            f"{os.path.basename(file)}: "
            f"{original_duration:.2f}s → "
            f"{new_duration:.2f}s"
        )

print("\nTrimming complete.")