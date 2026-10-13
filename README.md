# ESP-32 Based Offline Smart Home System

**Team**
- Sidak Singh — 24BCE1460
- Akshit Madathil — 24BCE5304
- Sarthak Singh — 24BCE5455
- Sanath Garg — 24BCE1526

## Overview

This project implements an offline, voice-controlled smart-home prototype using an ESP32 and a lightweight Convolutional Neural Network (CNN). The CNN is trained on a laptop and then converted/deployed as a TinyML model for inference on the ESP32. During operation, voice commands are classified locally; the system does not need an internet connection or cloud speech-recognition service.

The system supports four command classes:

- `light_on`
- `light_off`
- `fan_on`
- `fan_off`

A PIR motion sensor adds an occupancy condition: commands are acted on only when the room is considered occupied. For the classroom prototype, an LED represents the light and a buzzer represents the fan.

## Key Features

- Offline voice-command classification on the ESP32 using TinyML
- CNN training performed on a laptop, followed by model deployment to the microcontroller
- Four output classes for light/fan ON/OFF actions
- PIR-based occupancy gating
- LED and buzzer used as safe, low-voltage demonstration outputs
- Evaluation using a held-out test set and a confusion matrix

## System Architecture

```text
                  TRAINING PHASE
  Google Speech Commands dataset / prepared audio
                         |
                         v
              Data preparation and labels
                         |
                         v
                CNN training on laptop
                         |
                         v
               Model conversion for TinyML
                         |
                         v
              Deploy model to ESP32 firmware

                  INFERENCE PHASE
  Microphone --> Audio preprocessing --> CNN/TinyML inference
                                             |
                                             v
                                  Predicted command class
                                             |
                      PIR occupancy check --+
                                             |
                                             v
                                 ESP32 GPIO control
                                   /           \
                                  v             v
                           LED (light demo)  Buzzer (fan demo)
```

**Figure 1.** Training and on-device inference workflow.

## Hardware

| Component | Purpose |
|---|---|
| ESP32 development board | Runs the TinyML model and controls the outputs |
| Microphone module | Captures audio for command recognition |
| PIR motion sensor | Detects motion for occupancy gating |
| LED with suitable current-limiting resistor | Demonstrates the light output |
| Buzzer | Demonstrates the fan output |
| Breadboard and jumper wires | Prototype wiring |
| USB power/data cable | Power and firmware upload |
| Relay module (if installed for actuator testing) | Interface for switching suitable external loads |

### Known GPIO assignments

| Signal | ESP32 pin | Notes |
|---|---:|---|
| PIR sensor output | GPIO 13 | Used to determine room presence |
| Light output / demo LED | GPIO 26 | Confirm the pin against the current firmware and wiring |
| Fan output / demo buzzer | GPIO 27 | Confirm the pin against the current firmware and wiring |
| Microphone signals | As configured in firmware | Use the exact microphone interface and pin mapping defined in the model/board code; do not assume a pinout from this README |

> **Hardware note:** The LED and buzzer are demonstration substitutes for a light and fan. Do not connect mains-powered loads directly to ESP32 GPIO pins. Use a suitably rated relay/driver, proper isolation, and qualified supervision for any mains-voltage installation.

## Dataset and Model Training

The project uses Google's **Speech Commands** dataset, introduced by Pete Warden for limited-vocabulary speech and keyword-spotting research [1]. For this project, the dataset's `yes` and `no` samples were repurposed and labelled as `on` and `off` for the command-learning pipeline. The prepared data was used to train a CNN on a laptop; the resulting model was then converted for TinyML deployment on the ESP32.

The deployed model is intended to classify the four application command classes (`light_on`, `light_off`, `fan_on`, and `fan_off`). Ensure that the training-data preparation script, class mapping, and the firmware's output-index mapping use the same label order. Changing a dataset label does not change the spoken waveform itself, so the evaluation should use the same preprocessing and label-mapping pipeline as the deployed model.

### High-level workflow

1. Download and prepare the Speech Commands audio data.
2. Prepare the class labels and preprocessing pipeline used by the project.
3. Split the prepared samples into training, validation, and test sets without leakage between sets.
4. Train the CNN on the laptop and select the model using validation performance.
5. Evaluate the selected model on the held-out test set and generate its confusion matrix.
6. Convert/export the trained model into a format supported by the TinyML runtime used in the firmware.
7. Integrate the model and matching input/output preprocessing in the ESP32 firmware.
8. Connect the microphone, PIR sensor, and demo outputs, then verify each command and the occupancy condition.

## Operation

1. Power the ESP32 and microphone.
2. The microphone captures the spoken command and the on-device TinyML pipeline predicts a command class.
3. The PIR sensor determines whether the room is occupied. Motion refreshes the presence timer; the firmware uses a 60-second timeout after the last detected motion.
4. If the room is occupied, the ESP32 applies the recognised action to the relevant output.
5. The LED demonstrates `light_on` / `light_off`; the buzzer demonstrates `fan_on` / `fan_off`.

The device is designed to operate locally without Wi-Fi, internet access, or a cloud speech service. The PIR condition can reject commands while the room is considered empty.

## Results

The supplied four-class confusion matrix contains **1,500 test samples**. Its diagonal entries sum to 1,435 correct predictions:

| Actual class | Predicted `light_on` | Predicted `light_off` | Predicted `fan_on` | Predicted `fan_off` | Total |
|---|---:|---:|---:|---:|---:|
| `light_on` | 356 | 8 | 7 | 4 | 375 |
| `light_off` | 6 | 361 | 5 | 3 | 375 |
| `fan_on` | 4 | 6 | 358 | 7 | 375 |
| `fan_off` | 5 | 4 | 6 | 360 | 375 |
| **Total** | **371** | **379** | **376** | **374** | **1,500** |

Calculated accuracy from the displayed cell counts:

\[
\text{Accuracy} = \frac{356+361+358+360}{1500}\times100 = 95.67\%
\]

**Important:** The original confusion-matrix image is titled “Accuracy: 95.00%,” but the values shown in its cells calculate to **95.67%**. This README reports the value computed from the cell counts. If the 95.00% title came from another run or calculation, regenerate the plot from the final test predictions before publication so the title and matrix agree.

## Limitations

- Performance may vary with microphone quality, distance, speaker, accent, and background noise.
- The model's deployed input shape, audio preprocessing, class-index order, and quantisation settings must match those used during training.
- The reported test result is meaningful only if the test set was held out from model training and tuning.
- The PIR sensor detects motion, not a person's continuous presence. Its timeout and placement affect occupancy behaviour.
- An LED and buzzer demonstrate the control logic but do not, by themselves, validate operation with real household appliances.

## Future Improvements

- Add a larger and more diverse set of labelled recordings for each complete command class.
- Test across multiple speakers, distances, rooms, and noise conditions.
- Measure inference latency, model size, and RAM/flash usage on the ESP32.
- Add feedback for recognised commands and rejected commands.
- Integrate appropriately rated relay drivers for real appliances after electrical-safety review.

## Demonstration Video

**Video link:** `[Paste Google Drive demonstration link here]`

Suggested filename: `SidakSingh_ESP32_Offline_Smart_Home_System`

## References

[1] P. Warden, “Speech Commands: A Dataset for Limited-Vocabulary Speech Recognition,” *arXiv preprint arXiv:1804.03209*, Apr. 2018. [Online]. Available: https://arxiv.org/abs/1804.03209

[2] TensorFlow, “Simple audio recognition: Recognizing keywords,” *TensorFlow Tutorials*. [Online]. Available: https://www.tensorflow.org/tutorials/audio/simple_audio

[3] TensorFlow, “TensorFlow Lite for Microcontrollers.” [Online]. Available: https://www.tensorflow.org/lite/microcontrollers
