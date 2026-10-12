import os
import glob
import shutil
import re

PROJECT_DIR = os.path.dirname(os.path.dirname(__file__))

INPUT_DIR = os.path.join(
    PROJECT_DIR,
    "trimmed_dataset"
)

OUTPUT_DIR = os.path.join(
    PROJECT_DIR,
    "trimmed_speaker_dataset"
)

COMMANDS = [
    "light_on",
    "light_off",
    "fan_on",
    "fan_off"
]


def get_number(filename):
    match = re.search(r"_(\d+)\.wav$", filename)
    return int(match.group(1)) if match else 0


for command in COMMANDS:

    files = glob.glob(
        os.path.join(INPUT_DIR, command, "*.wav")
    )

    files.sort(key=get_number)

    if len(files) < 40:
        print(
            f"WARNING: {command} has "
            f"{len(files)} files"
        )
        continue

    for speaker_idx in range(4):

        speaker = f"speaker_{speaker_idx + 1}"

        output_dir = os.path.join(
            OUTPUT_DIR,
            speaker,
            command
        )

        os.makedirs(
            output_dir,
            exist_ok=True
        )

        start = speaker_idx * 10
        end = start + 10

        for file in files[start:end]:

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
            f"{len(files[start:end])} files"
        )

print("\nDone.")