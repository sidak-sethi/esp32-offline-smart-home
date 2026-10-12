import os
import time
import numpy as np
import sounddevice as sd
import soundfile as sf
from scipy.signal import resample_poly

DEVICE = 9
INPUT_SR = 48000
TARGET_SR = 16000
CHANNELS = 1
DURATION = 2.0

CLASSES = [
    "light_on",
    "light_off",
    "fan_on",
    "fan_off",
    "unknown",
    "silence"
]

BASE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "dataset")


def record_audio():
    audio = sd.rec(
        int(INPUT_SR * DURATION),
        samplerate=INPUT_SR,
        channels=CHANNELS,
        dtype="float32",
        device=DEVICE
    )

    sd.wait()

    audio = np.squeeze(audio)

    # 48 kHz -> 16 kHz
    audio = resample_poly(audio, TARGET_SR, INPUT_SR)

    # Normalize
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio / peak * 0.95

    return audio.astype(np.float32)


def save_audio(audio, label, number):
    folder = os.path.join(BASE_DIR, label)
    os.makedirs(folder, exist_ok=True)

    filename = os.path.join(
        folder,
        f"{label}_{number:04d}.wav"
    )

    sf.write(filename, audio, TARGET_SR)
    return filename


def main():
    print("\nTinyML Voice Dataset Recorder")
    print("=" * 35)

    for i, label in enumerate(CLASSES, 1):
        print(f"{i}. {label}")

    choice = input("\nSelect class number: ").strip()

    if not choice.isdigit() or not 1 <= int(choice) <= len(CLASSES):
        print("Invalid choice.")
        return

    label = CLASSES[int(choice) - 1]

    try:
        count = int(input("Number of recordings: "))
    except ValueError:
        print("Invalid number.")
        return

    folder = os.path.join(BASE_DIR, label)
    os.makedirs(folder, exist_ok=True)

    existing = [
        f for f in os.listdir(folder)
        if f.lower().endswith(".wav")
    ]

    start_number = len(existing) + 1

    print(f"\nRecording class: {label}")
    print("Press Ctrl+C to stop.\n")

    for i in range(count):
        input(f"Press ENTER for sample {i + 1}/{count}...")

        print("Get ready...")
        time.sleep(0.5)

        print("Recording!")
        audio = record_audio()

        filename = save_audio(
            audio,
            label,
            start_number + i
        )

        print(f"Saved: {filename}\n")

    print("Done.")


if __name__ == "__main__":
    main()