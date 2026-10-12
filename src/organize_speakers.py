import os
import glob
import shutil
import re

PROJECT_DIR = os.path.dirname(os.path.dirname(__file__))
DATASET_DIR = os.path.join(PROJECT_DIR, "dataset")
SPEAKER_DIR = os.path.join(PROJECT_DIR, "speaker_dataset")

COMMANDS = [
    "light_on",
    "light_off",
    "fan_on",
    "fan_off"
]


def get_number(filename):
    """
    Extract number from filenames such as:
    light_on_0017.wav
    """
    match = re.search(r"_(\d+)\.wav$", filename)
    return int(match.group(1)) if match else 0


for command in COMMANDS:

    files = glob.glob(
        os.path.join(DATASET_DIR, command, "*.wav")
    )

    files.sort(key=get_number)

    if len(files) < 40:
        print(
            f"WARNING: {command} has only "
            f"{len(files)} files"
        )
        continue

    for speaker_index in range(4):

        speaker = f"speaker_{speaker_index + 1}"

        output_dir = os.path.join(
            SPEAKER_DIR,
            speaker,
            command
        )

        os.makedirs(
            output_dir,
            exist_ok=True
        )

        start = speaker_index * 10
        end = start + 10

        speaker_files = files[start:end]

        for file in speaker_files:

            destination = os.path.join(
                output_dir,
                os.path.basename(file)
            )

            shutil.copy2(
                file,
                destination
            )

        print(
            f"{command} -> {speaker}: "
            f"{len(speaker_files)} files"
        )


print("\nSpeaker organization complete.")