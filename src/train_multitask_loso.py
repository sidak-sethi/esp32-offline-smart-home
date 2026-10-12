import os
import glob
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix


# ============================================================
# Paths
# ============================================================

PROJECT_DIR = os.path.dirname(os.path.dirname(__file__))

FEATURE_DIR = os.path.join(
    PROJECT_DIR,
    "logmel_features_fixed"
)


# ============================================================
# Command definitions
# ============================================================

COMMANDS = [
    "light_on",
    "light_off",
    "fan_on",
    "fan_off"
]

SPEAKERS = [
    "speaker_1",
    "speaker_2",
    "speaker_3",
    "speaker_4"
]


# Device labels
# LIGHT = 0
# FAN   = 1

DEVICE_LABEL = {
    "light_on": 0,
    "light_off": 0,
    "fan_on": 1,
    "fan_off": 1
}


# Action labels
# ON  = 0
# OFF = 1

ACTION_LABEL = {
    "light_on": 0,
    "light_off": 1,
    "fan_on": 0,
    "fan_off": 1
}


# Command labels, useful for stratification
COMMAND_LABEL = {
    "light_on": 0,
    "light_off": 1,
    "fan_on": 2,
    "fan_off": 3
}


# ============================================================
# Load one speaker
# ============================================================

def load_speaker(speaker):

    X = []
    device_y = []
    action_y = []
    command_y = []

    for command in COMMANDS:

        folder = os.path.join(
            FEATURE_DIR,
            speaker,
            command
        )

        files = glob.glob(
            os.path.join(folder, "*.npy")
        )

        for file in files:

            feature = np.load(file)

            X.append(feature)

            device_y.append(
                DEVICE_LABEL[command]
            )

            action_y.append(
                ACTION_LABEL[command]
            )

            command_y.append(
                COMMAND_LABEL[command]
            )

    return (
        np.array(X, dtype=np.float32),
        np.array(device_y, dtype=np.int32),
        np.array(action_y, dtype=np.int32),
        np.array(command_y, dtype=np.int32)
    )


# ============================================================
# Build multi-head model
# ============================================================

def build_model(input_shape):

    inputs = tf.keras.Input(
        shape=input_shape,
        name="audio_features"
    )

    # Shared feature extractor
    x = tf.keras.layers.Conv2D(
        8,
        (3, 3),
        padding="same",
        activation="relu"
    )(inputs)

    x = tf.keras.layers.MaxPooling2D(
        (2, 2)
    )(x)

    x = tf.keras.layers.Conv2D(
        16,
        (3, 3),
        padding="same",
        activation="relu"
    )(x)

    x = tf.keras.layers.MaxPooling2D(
        (2, 2)
    )(x)

    x = tf.keras.layers.Flatten()(x)

    x = tf.keras.layers.Dense(
        32,
        activation="relu"
    )(x)

    x = tf.keras.layers.Dropout(
        0.2
    )(x)

    # Two task-specific heads

    device_output = tf.keras.layers.Dense(
        2,
        activation="softmax",
        name="device"
    )(x)

    action_output = tf.keras.layers.Dense(
        2,
        activation="softmax",
        name="action"
    )(x)

    model = tf.keras.Model(
        inputs=inputs,
        outputs={
            "device": device_output,
            "action": action_output
        }
    )

    model.compile(
        optimizer="adam",

        loss={
            "device":
                "sparse_categorical_crossentropy",

            "action":
                "sparse_categorical_crossentropy"
        },

        metrics={
            "device": "accuracy",
            "action": "accuracy"
        }
    )

    return model


# ============================================================
# LOSO
# ============================================================

all_true_command = []
all_pred_command = []

all_true_device = []
all_pred_device = []

all_true_action = []
all_pred_action = []

fold_command_accuracy = []
fold_device_accuracy = []
fold_action_accuracy = []


for test_speaker in SPEAKERS:

    print("\n" + "=" * 65)
    print(f"TEST SPEAKER: {test_speaker}")
    print("=" * 65)

    train_speakers = [
        s for s in SPEAKERS
        if s != test_speaker
    ]

    # --------------------------------------------------------
    # Load training speakers
    # --------------------------------------------------------

    X_train = []
    device_train = []
    action_train = []
    command_train = []

    for speaker in train_speakers:

        (
            X_s,
            device_s,
            action_s,
            command_s
        ) = load_speaker(speaker)

        X_train.append(X_s)
        device_train.append(device_s)
        action_train.append(action_s)
        command_train.append(command_s)

    X_train = np.concatenate(X_train)
    device_train = np.concatenate(device_train)
    action_train = np.concatenate(action_train)
    command_train = np.concatenate(command_train)

    # --------------------------------------------------------
    # Load held-out speaker
    # --------------------------------------------------------

    (
        X_test,
        device_test,
        action_test,
        command_test
    ) = load_speaker(test_speaker)

    print(
        "Training samples:",
        len(X_train)
    )

    print(
        "Test samples:",
        len(X_test)
    )

    # Add CNN channel
    X_train = X_train[..., np.newaxis]
    X_test = X_test[..., np.newaxis]

    # --------------------------------------------------------
    # Validation split
    # --------------------------------------------------------

    (
        X_train,
        X_val,
        device_train,
        device_val,
        action_train,
        action_val
    ) = train_test_split(
        X_train,
        device_train,
        action_train,
        test_size=0.20,
        random_state=42,
        stratify=command_train
    )

    # --------------------------------------------------------
    # Normalize using training set only
    # --------------------------------------------------------

    mean = np.mean(
        X_train,
        axis=(0, 1, 2),
        keepdims=True
    )

    std = np.std(
        X_train,
        axis=(0, 1, 2),
        keepdims=True
    )

    std = np.maximum(
        std,
        1e-6
    )

    X_train = (
        X_train - mean
    ) / std

    X_val = (
        X_val - mean
    ) / std

    X_test = (
        X_test - mean
    ) / std

    # --------------------------------------------------------
    # Fresh model
    # --------------------------------------------------------

    tf.keras.backend.clear_session()

    model = build_model(
        X_train.shape[1:]
    )

    model.summary()

    callbacks = [

        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=10,
            restore_best_weights=True
        ),

        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=4,
            min_lr=1e-6
        )
    ]

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    model.fit(

        X_train,

        {
            "device": device_train,
            "action": action_train
        },

        validation_data=(

            X_val,

            {
                "device": device_val,
                "action": action_val
            }
        ),

        epochs=80,

        batch_size=8,

        callbacks=callbacks,

        verbose=1
    )

    # --------------------------------------------------------
    # Predict
    # --------------------------------------------------------

    predictions = model.predict(
        X_test,
        verbose=0
    )

    device_pred = np.argmax(
        predictions["device"],
        axis=1
    )

    action_pred = np.argmax(
        predictions["action"],
        axis=1
    )

    # --------------------------------------------------------
    # Convert two predictions back to command
    # --------------------------------------------------------

    command_pred = []

    for d, a in zip(
        device_pred,
        action_pred
    ):

        if d == 0 and a == 0:
            command_pred.append(0)  # light_on

        elif d == 0 and a == 1:
            command_pred.append(1)  # light_off

        elif d == 1 and a == 0:
            command_pred.append(2)  # fan_on

        else:
            command_pred.append(3)  # fan_off

    command_pred = np.array(
        command_pred
    )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    device_acc = accuracy_score(
        # Convert command labels to devices
        np.array([
            0 if c < 2 else 1
            for c in command_test
        ]),
        device_pred
    )

    action_acc = accuracy_score(
        np.array([
            0 if c in [0, 2] else 1
            for c in command_test
        ]),
        action_pred
    )

    command_acc = accuracy_score(
        command_test,
        command_pred
    )

    fold_device_accuracy.append(
        device_acc
    )

    fold_action_accuracy.append(
        action_acc
    )

    fold_command_accuracy.append(
        command_acc
    )

    print(
        f"\nDevice accuracy: "
        f"{device_acc * 100:.2f}%"
    )

    print(
        f"Action accuracy: "
        f"{action_acc * 100:.2f}%"
    )

    print(
        f"Exact command accuracy: "
        f"{command_acc * 100:.2f}%"
    )

    # Store global
    all_true_command.extend(command_test)
    all_pred_command.extend(command_pred)

    all_true_device.extend(
        np.array([
            0 if c < 2 else 1
            for c in command_test
        ])
    )

    all_pred_device.extend(
        device_pred
    )

    all_true_action.extend(
        np.array([
            0 if c in [0, 2] else 1
            for c in command_test
        ])
    )

    all_pred_action.extend(
        action_pred
    )


# ============================================================
# Final results
# ============================================================

overall_command = accuracy_score(
    all_true_command,
    all_pred_command
)

overall_device = accuracy_score(
    all_true_device,
    all_pred_device
)

overall_action = accuracy_score(
    all_true_action,
    all_pred_action
)


print("\n" + "=" * 65)
print("MULTI-TASK LOSO RESULTS")
print("=" * 65)

for i, speaker in enumerate(SPEAKERS):

    print(
        f"{speaker}: "
        f"device={fold_device_accuracy[i] * 100:.2f}% | "
        f"action={fold_action_accuracy[i] * 100:.2f}% | "
        f"command={fold_command_accuracy[i] * 100:.2f}%"
    )

print(
    f"\nOverall device accuracy: "
    f"{overall_device * 100:.2f}%"
)

print(
    f"Overall action accuracy: "
    f"{overall_action * 100:.2f}%"
)

print(
    f"Overall exact command accuracy: "
    f"{overall_command * 100:.2f}%"
)


# ============================================================
# Exact command confusion matrix
# ============================================================

cm = confusion_matrix(
    all_true_command,
    all_pred_command,
    labels=[0, 1, 2, 3]
)

print("\nCommand Confusion Matrix:")
print(cm)


plt.figure(figsize=(7, 6))

plt.imshow(
    cm,
    interpolation="nearest"
)

plt.xticks(
    range(4),
    COMMANDS,
    rotation=45
)

plt.yticks(
    range(4),
    COMMANDS
)

plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.title(
    "Multi-Task Exact Command Confusion Matrix"
)

for i in range(4):
    for j in range(4):

        plt.text(
            j,
            i,
            cm[i, j],
            ha="center",
            va="center"
        )

plt.tight_layout()
plt.show()