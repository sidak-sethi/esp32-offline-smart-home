#!/usr/bin/env python3

"""
Offline voice control:
laptop mic -> Vosk (local) -> command detection -> USB serial -> ESP32

The ESP32 decides whether a command is executed
(only when the PIR says someone is in the room).

Commands:
    1 = light on
    2 = light off
    3 = fan on
    4 = fan off
"""

import argparse
import array
import json
import math
import os
import queue
import sys
import threading
import time

import serial
import serial.tools.list_ports
import sounddevice as sd
from vosk import KaldiRecognizer, Model, SetLogLevel


# ============================================================
# CONFIGURATION
# ============================================================

SAMPLE_RATE = 16000
BLOCK_SIZE = 4000

# Ignore the exact same command if it repeats within this time.
DEBOUNCE_SECONDS = 1.0


# ============================================================
# COMMANDS
# ============================================================

COMMANDS = {
    "1": "LIGHT ON",
    "2": "LIGHT OFF",
    "3": "FAN ON",
    "4": "FAN OFF",
}


# ============================================================
# PHRASES
# ============================================================
#
# The important change:
# These phrases no longer need to match the entire sentence.
#
# Example:
#   "please turn the light on"
#
# contains:
#   "turn the light on"
#
# therefore -> command 1
#

PHRASES = {
    # LIGHT ON
    "light on": "1",
    "lights on": "1",
    "turn on the light": "1",
    "turn the light on": "1",

    # LIGHT OFF
    "light off": "2",
    "lights off": "2",
    "turn off the light": "2",
    "turn the light off": "2",
    "laid off": "2",

    # FAN ON
    "fan on": "3",
    "turn on the fan": "3",
    "turn the fan on": "3",

    # FAN OFF
    "fan off": "4",
    "turn off the fan": "4",
    "turn the fan off": "4",
    "fan of":"4",
}


# ============================================================
# TERMINAL HELPERS
# ============================================================

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


if os.name == "nt":
    os.system("")


GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
CYAN = "\033[96m"
DIM = "\033[2m"
BOLD = "\033[1m"
RESET = "\033[0m"

CLEAR_LINE = "\r\033[K"


# ============================================================
# ROOM STATE
# ============================================================

# None  = unknown
# True  = occupied
# False = empty

room = {
    "occupied": None
}


# ============================================================
# COMMAND DETECTION
# ============================================================

def detect_command(text):
    """
    Detect a known command anywhere inside the recognized
    sentence.

    Example:

        "please turn the light on"

    becomes:

        "turn the light on" -> "1"

    Returns:
        command code ("1", "2", "3", "4")
        or None if no command was found.
    """

    if not text:
        return None

    text = text.lower().strip()

    # Longer phrases first.
    #
    # This makes sure that:
    #
    # "turn the light on"
    #
    # gets priority over:
    #
    # "light on"
    #
    # when both could technically match.

    phrases = sorted(
        PHRASES.items(),
        key=lambda item: len(item[0]),
        reverse=True
    )

    for phrase, code in phrases:
        if phrase in text:
            return code

    return None


# ============================================================
# BEEP
# ============================================================

def beep():
    try:
        if os.name == "nt":
            import winsound
            winsound.Beep(1000, 150)
        else:
            sys.stdout.write("\a")
            sys.stdout.flush()
    except Exception:
        pass


# ============================================================
# MICROPHONE LEVEL
# ============================================================

def level_bar(rms, width=12):
    frac = min(
        1.0,
        math.sqrt(rms / 8000.0)
    ) if rms > 0 else 0

    filled = int(frac * width)

    return (
        "█" * filled +
        "░" * (width - filled)
    )


def block_rms(data):
    samples = array.array("h")
    samples.frombytes(data)

    if not samples:
        return 0

    return math.sqrt(
        sum(s * s for s in samples) /
        len(samples)
    )


# ============================================================
# SERIAL PORTS
# ============================================================

def list_ports():

    ports = list(
        serial.tools.list_ports.comports()
    )

    if not ports:
        print(
            "No serial ports found. "
            "Is the ESP32 plugged in "
            "(and the driver installed)?"
        )

        return

    for p in ports:
        print(
            f"  {p.device:<14} "
            f"{p.description}"
        )


def open_serial(port):

    try:

        ser = serial.Serial(
            port,
            115200,
            timeout=0.5,
            write_timeout=1
        )

    except serial.SerialException as e:

        print(
            f"{RED}Could not open {port}: "
            f"{e}{RESET}"
        )

        print("Available ports:")

        list_ports()

        sys.exit(1)

    # ESP32 often resets when serial opens.
    time.sleep(2)

    return ser


# ============================================================
# ESP32 SERIAL READER
# ============================================================

def esp32_reader(ser, stop):
    """
    Print everything the ESP32 sends and track
    room occupancy.
    """

    while not stop.is_set():

        try:

            line = (
                ser.readline()
                .decode(errors="ignore")
                .strip()
            )

        except (
            serial.SerialException,
            OSError
        ):

            print(
                f"{CLEAR_LINE}"
                f"{RED}"
                "Lost connection to the ESP32."
                f"{RESET}"
            )

            return

        if not line:
            continue


        # ----------------------------------------------------
        # PRESENCE ACTIVE
        # ----------------------------------------------------

        if line.startswith(
            "[PRESENCE] ACTIVE"
        ):

            room["occupied"] = True

            color = GREEN + BOLD


        # ----------------------------------------------------
        # PRESENCE INACTIVE
        # ----------------------------------------------------

        elif line.startswith(
            "[PRESENCE] INACTIVE"
        ):

            room["occupied"] = False

            color = YELLOW + BOLD


        # ----------------------------------------------------
        # COMMAND SUCCESS
        # ----------------------------------------------------

        elif line.startswith(
            "[CMD] OK"
        ):

            color = GREEN + BOLD


        # ----------------------------------------------------
        # COMMAND IGNORED
        # ----------------------------------------------------

        elif line.startswith(
            "[CMD] IGNORED"
        ):

            color = YELLOW + BOLD


        # ----------------------------------------------------
        # STATE
        # ----------------------------------------------------

        elif line.startswith(
            "[STATE]"
        ):

            if "room=OCCUPIED" in line:
                room["occupied"] = True

            elif "room=EMPTY" in line:
                room["occupied"] = False

            color = CYAN


        else:

            color = CYAN


        print(
            f"{CLEAR_LINE}"
            f"{color}"
            f"ESP32 | {line}"
            f"{RESET}"
        )


# ============================================================
# MAIN
# ============================================================

def main():

    ap = argparse.ArgumentParser(
        description="Offline voice control for ESP32"
    )

    ap.add_argument(
        "--port",
        help="serial port of the ESP32, "
             "e.g. COM4 or /dev/ttyUSB0"
    )

    ap.add_argument(
        "--model",
        default="vosk-model-small-en-us-0.15",
        help="path to the Vosk model folder"
    )

    ap.add_argument(
        "--device",
        help="microphone device index or name "
             "(see --list-devices)"
    )

    ap.add_argument(
        "--list-ports",
        action="store_true",
        help="show serial ports and exit"
    )

    ap.add_argument(
        "--list-devices",
        action="store_true",
        help="show audio devices and exit"
    )

    ap.add_argument(
        "--no-serial",
        action="store_true",
        help="dry run: do not talk to the ESP32"
    )

    ap.add_argument(
        "--no-beep",
        action="store_true",
        help="disable the beep on recognition"
    )

    args = ap.parse_args()


    # ========================================================
    # LIST SERIAL PORTS
    # ========================================================

    if args.list_ports:
        list_ports()
        return


    # ========================================================
    # LIST AUDIO DEVICES
    # ========================================================

    if args.list_devices:

        print(
            sd.query_devices()
        )

        return


    # ========================================================
    # SERIAL VALIDATION
    # ========================================================

    if (
        not args.no_serial
        and not args.port
    ):

        print(
            f"{RED}"
            "Please give the ESP32 port with "
            "--port (or use --no-serial to test)."
            f"{RESET}"
        )

        print("Available ports:")

        list_ports()

        sys.exit(1)


    # ========================================================
    # VOSK MODEL VALIDATION
    # ========================================================

    if not os.path.isdir(args.model):

        print(
            f"{RED}"
            f"Model folder not found: {args.model}"
            f"{RESET}"
        )

        print(
            "Unzip the Vosk model so that this "
            "folder contains 'am', 'conf', "
            "'graph' directly."
        )

        print(
            "Or pass the path with --model."
        )

        sys.exit(1)


    # ========================================================
    # AUDIO DEVICE
    # ========================================================

    device = args.device

    if (
        device is not None
        and device.isdigit()
    ):

        device = int(device)


    # ========================================================
    # SERIAL CONNECTION
    # ========================================================

    ser = (
        None
        if args.no_serial
        else open_serial(args.port)
    )


    stop = threading.Event()


    if ser is not None:

        threading.Thread(
            target=esp32_reader,
            args=(ser, stop),
            daemon=True
        ).start()

        # Ask ESP32 for current state.
        ser.write(b"?")


    # ========================================================
    # LOAD VOSK
    # ========================================================

    SetLogLevel(-1)

    print(
        "Loading speech model "
        "(offline, runs on this laptop)..."
    )

    model = Model(args.model)


    # ========================================================
    # VOSK RECOGNIZER
    # ========================================================
    #
    # IMPORTANT:
    #
    # We intentionally DO NOT pass a grammar here.
    #
    # Previously:
    #
    #     KaldiRecognizer(model, SAMPLE_RATE, grammar)
    #
    # restricted Vosk to our command phrases.
    #
    # Now Vosk can recognize the whole sentence, and
    # detect_command() searches that sentence for commands.
    #

    rec = KaldiRecognizer(
        model,
        SAMPLE_RATE
    )


    # ========================================================
    # AUDIO QUEUE
    # ========================================================

    audio_q = queue.Queue()


    def callback(
        indata,
        frames,
        t,
        status
    ):

        audio_q.put(
            bytes(indata)
        )


    # ========================================================
    # COMMAND DEBOUNCE
    # ========================================================

    last_code = None
    last_time = 0.0

    last_partial = ""


    # ========================================================
    # READY
    # ========================================================

    print(
        f"\n{BOLD}"
        "Ready."
        f"{RESET} "
        f"Say: {CYAN}"
        "light on · light off · "
        "fan on · fan off"
        f"{RESET}"
    )

    print(
        f"{DIM}"
        "Commands can appear anywhere "
        "inside a sentence."
        f"{RESET}"
    )

    print(
        f"{DIM}"
        "Commands only work while the PIR "
        "sensor says someone is in the room."
        f"{RESET}"
    )

    print(
        f"{DIM}"
        f"Mode: "
        f"{'DRY RUN (no ESP32)' if ser is None else 'connected to ' + args.port}"
        "   Ctrl+C to quit"
        f"{RESET}\n"
    )


    # ========================================================
    # MICROPHONE LOOP
    # ========================================================

    try:

        with sd.RawInputStream(
            samplerate=SAMPLE_RATE,
            blocksize=BLOCK_SIZE,
            dtype="int16",
            channels=1,
            device=device,
            callback=callback
        ):

            while True:

                data = audio_q.get()

                rms = block_rms(data)


                # =================================================
                # COMPLETE SPEECH RESULT
                # =================================================

                if rec.AcceptWaveform(data):

                    result = json.loads(
                        rec.Result()
                    )

                    text = result.get(
                        "text",
                        ""
                    ).strip()

                    last_partial = ""


                    # -------------------------------------------------
                    # DETECT COMMAND ANYWHERE IN SENTENCE
                    # -------------------------------------------------

                    code = detect_command(text)


                    if code:

                        now = time.time()


                        # ------------------------------------------------
                        # DEBOUNCE
                        # ------------------------------------------------

                        if (
                            code == last_code
                            and
                            now - last_time
                            < DEBOUNCE_SECONDS
                        ):

                            continue


                        last_code = code
                        last_time = now


                        label = COMMANDS[code]


                        # ------------------------------------------------
                        # SEND TO ESP32
                        # ------------------------------------------------

                        if ser is not None:

                            try:

                                ser.write(
                                    code.encode()
                                )

                            except serial.SerialException as e:

                                print(
                                    f"{CLEAR_LINE}"
                                    f"{RED}"
                                    f"Serial error: {e}"
                                    f"{RESET}"
                                )

                                continue


                        # ------------------------------------------------
                        # BEEP
                        # ------------------------------------------------

                        if not args.no_beep:
                            beep()


                        # ------------------------------------------------
                        # DISPLAY
                        # ------------------------------------------------

                        stamp = time.strftime(
                            "%H:%M:%S"
                        )

                        print(
                            f"{CLEAR_LINE}"
                            f"{GREEN}{BOLD}"
                            f"✔ COMMAND DETECTED: {label}"
                            f"{RESET}"
                            f"  {DIM}[{stamp}]{RESET}"
                        )


                        if ser is None:

                            print(
                                f'   heard: "{text}" '
                                f"(dry run, nothing sent)\n"
                            )

                        else:

                            print(
                                f'   heard: "{text}"'
                                f"  ->  sent code {code}"
                            )


                            if room["occupied"] is False:

                                print(
                                    f"   {YELLOW}"
                                    "room looks empty, "
                                    "the ESP32 will ignore this"
                                    f"{RESET}"
                                )


                            print(
                                f"   {DIM}"
                                "waiting for the ESP32 "
                                "result below..."
                                f"{RESET}"
                            )


                    # -------------------------------------------------
                    # NO COMMAND
                    # -------------------------------------------------

                    elif (
                        text
                        and
                        text != "[unk]"
                    ):

                        print(
                            f"{CLEAR_LINE}"
                            f"{YELLOW}"
                            f'✘ No command found: "{text}"'
                            f"{RESET}\n"
                        )


                    elif text == "[unk]":

                        print(
                            f"{CLEAR_LINE}"
                            f"{YELLOW}"
                            "✘ Speech heard but not "
                            "recognized"
                            f"{RESET}\n"
                        )


                # =================================================
                # PARTIAL SPEECH
                # =================================================

                else:

                    partial = json.loads(
                        rec.PartialResult()
                    ).get(
                        "partial",
                        ""
                    ).strip()


                    if partial:

                        last_partial = partial


                        occ = room["occupied"]


                        if occ:
                            room_txt = "room: OCCUPIED"

                        elif occ is False:
                            room_txt = "room: EMPTY"

                        else:
                            room_txt = "room: ?"


                        room_col = (
                            GREEN
                            if occ
                            else YELLOW
                        )


                        sys.stdout.write(
                            f"{CLEAR_LINE}"
                            f"{DIM}"
                            f"mic [{level_bar(rms)}]"
                            f"{RESET} "
                            f"{room_col}"
                            f"{room_txt}"
                            f"{RESET}  "
                            f"{CYAN}"
                            f"{last_partial}"
                            f"{RESET}"
                        )

                        sys.stdout.flush()


    # ========================================================
    # EXIT
    # ========================================================

    except KeyboardInterrupt:

        print(
            f"{CLEAR_LINE}\nStopped."
        )


    except sd.PortAudioError as e:

        print(
            f"{CLEAR_LINE}"
            f"{RED}"
            f"Microphone error: {e}"
            f"{RESET}"
        )

        print(
            "Run with --list-devices "
            "and pick one using --device <index>."
        )


    finally:

        stop.set()

        if ser is not None:

            time.sleep(0.6)

            ser.close()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()