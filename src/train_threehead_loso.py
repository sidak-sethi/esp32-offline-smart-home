import os
import glob
import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt


# ============================================================
# Configuration
# ============================================================

PROJECT_DIR = os.path.dirname(os.path.dirname(__file__))

FEATURE_DIR = os.path.join(
    PROJECT_DIR,
    "logmel_features_fixed"
)

CLASSES = [
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

# DEVICE:
# LIGHT = 0
# FAN   = 1
DEVICE_LABEL = {
    "light_on": 0,
    "light_off": 0,
    "fan_on": 1,
    "fan_off": 1
}

# ACTION:
# ON  = 0
# OFF = 1
ACTION_LABEL = {
    "light_on": 0,
    "light_off": 1,
    "fan_on": 0,
    "fan_off": 1
}

# COMMAND:
COMMAND_LABEL = {
    "light_on": 0,
    "light_off": 1,
    "fan_on": 2,
    "fan_off": 3
}


# ============================================================
# Load data for one speaker
# ============================================================

def load_speaker(speaker):

    X = []
    device_y = []
    action_y = []
    command_y = []

    for command in CLASSES:

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
# Build three-head model
# ============================================================

def build_model(input_shape):

    inputs = tf.keras.Input(
        shape=input_shape,
        name="audio_features"
    )

    # ----------------------------
    # Shared feature extractor
    # ----------------------------

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

    # ----------------------------
    # DEVICE HEAD
    # ----------------------------

    device_output = tf.keras.layers.Dense(
        2,
        activation="softmax",
        name="device"
    )(x)

    # ----------------------------
    # ACTION HEAD
    # ----------------------------

    action_output = tf.keras.layers.Dense(
        2,
        activation="softmax",
        name="action"
    )(x)

    # ----------------------------
    # COMMAND HEAD
    # ----------------------------

    command_output = tf.keras.layers.Dense(
        4,
        activation="softmax",
        name="command"
    )(x)

    model = tf.keras.Model(
        inputs=inputs,
        outputs={
            "device": device_output,
            "action": action_output,
            "command": command_output
        }
    )

    model.compile(

        optimizer="adam",

        loss={
            "device":
                "sparse_categorical_crossentropy",

            "action":
                "sparse_categorical_crossentropy",

            "command":
                "sparse_categorical_crossentropy"
        },

        # The exact command is our main task.
        # Auxiliary heads help the shared representation.
        loss_weights={
            "device": 0.5,
            "action": 0.5,
            "command": 1.0
        },

        metrics={
            "device": "accuracy",
            "action": "accuracy",
            "command": "accuracy"
        }
    )

    return model


# ============================================================
# Global results
# ============================================================

all_true_command = []
all_pred_command = []

all_true_device = []
all_pred_device = []

all_true_action = []
all_pred_action = []

fold_device_acc = []
fold_action_acc = []
fold_command_acc = []


# ============================================================
# Leave-One-Speaker-Out
# ============================================================

for test_speaker in SPEAKERS:

    print("\n" + "=" * 70)
    print(
        f"TEST SPEAKER: {test_speaker}"
    )
    print("=" * 70)

    train_speakers = [
        s for s in SPEAKERS
        if s != test_speaker
    ]

    print(
        "Training speakers:",
        train_speakers
    )

    # --------------------------------------------------------
    # Load training data
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

    # --------------------------------------------------------
    # CNN channel
    # --------------------------------------------------------

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
        action_val,
        command_train,
        command_val
    ) = train_test_split(

        X_train,

        device_train,

        action_train,

        command_train,

        test_size=0.20,

        random_state=42,

        stratify=command_train
    )

    # --------------------------------------------------------
    # Normalize using TRAINING SET ONLY
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
    # Build fresh model
    # --------------------------------------------------------

    tf.keras.backend.clear_session()

    model = build_model(
        X_train.shape[1:]
    )

    print("\nModel:")
    model.summary()

    callbacks = [

        tf.keras.callbacks.EarlyStopping(
            monitor="val_command_accuracy",
            mode="max",
            patience=12,
            restore_best_weights=True
        ),

        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_command_accuracy",
            mode="max",
            factor=0.5,
            patience=5,
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
            "action": action_train,
            "command": command_train
        },

        validation_data=(

            X_val,

            {
                "device": device_val,
                "action": action_val,
                "command": command_val
            }
        ),

        epochs=100,

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

    command_pred = np.argmax(
        predictions["command"],
        axis=1
    )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    device_acc = accuracy_score(
        device_test,
        device_pred
    )

    action_acc = accuracy_score(
        action_test,
        action_pred
    )

    command_acc = accuracy_score(
        command_test,
        command_pred
    )

    fold_device_acc.append(
        device_acc
    )

    fold_action_acc.append(
        action_acc
    )

    fold_command_acc.append(
        command_acc
    )

    print("\nFold results:")

    print(
        f"Device : {device_acc * 100:.2f}%"
    )

    print(
        f"Action : {action_acc * 100:.2f}%"
    )

    print(
        f"Command: {command_acc * 100:.2f}%"
    )

    # --------------------------------------------------------
    # Save global predictions
    # --------------------------------------------------------

    all_true_device.extend(
        device_test
    )

    all_pred_device.extend(
        device_pred
    )

    all_true_action.extend(
        action_test
    )

    all_pred_action.extend(
        action_pred
    )

    all_true_command.extend(
        command_test
    )

    all_pred_command.extend(
        command_pred
    )

    # --------------------------------------------------------
    # Fold command confusion matrix
    # --------------------------------------------------------

    cm = confusion_matrix(
        command_test,
        command_pred,
        labels=[0, 1, 2, 3]
    )

    print("\nCommand confusion matrix:")
    print(cm)

    plt.figure(figsize=(7, 6))

    plt.imshow(
        cm,
        interpolation="nearest"
    )

    plt.xticks(
        range(4),
        CLASSES,
        rotation=45
    )

    plt.yticks(
        range(4),
        CLASSES
    )

    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    plt.title(
        f"Three-Head LOSO - {test_speaker}"
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


# ============================================================
# Overall results
# ============================================================

overall_device = accuracy_score(
    all_true_device,
    all_pred_device
)

overall_action = accuracy_score(
    all_true_action,
    all_pred_action
)

overall_command = accuracy_score(
    all_true_command,
    all_pred_command
)


print("\n" + "=" * 70)
print("THREE-HEAD LOSO FINAL RESULTS")
print("=" * 70)

for i, speaker in enumerate(SPEAKERS):

    print(
        f"{speaker}: "
        f"device={fold_device_acc[i] * 100:.2f}% | "
        f"action={fold_action_acc[i] * 100:.2f}% | "
        f"command={fold_command_acc[i] * 100:.2f}%"
    )

print("\nOverall:")

print(
    f"Device accuracy: "
    f"{overall_device * 100:.2f}%"
)

print(
    f"Action accuracy: "
    f"{overall_action * 100:.2f}%"
)

print(
    f"Exact command accuracy: "
    f"{overall_command * 100:.2f}%"
)

print(
    f"\nMean command fold accuracy: "
    f"{np.mean(fold_command_acc) * 100:.2f}%"
)

print(
    f"Command fold std: "
    f"{np.std(fold_command_acc) * 100:.2f}%"
)


# ============================================================
# Overall command confusion matrix
# ============================================================

overall_cm = confusion_matrix(
    all_true_command,
    all_pred_command,
    labels=[0, 1, 2, 3]
)

print(
    "\nOverall Command Confusion Matrix:"
)

print(overall_cm)


plt.figure(figsize=(7, 6))

plt.imshow(
    overall_cm,
    interpolation="nearest"
)

plt.xticks(
    range(4),
    CLASSES,
    rotation=45
)

plt.yticks(
    range(4),
    CLASSES
)

plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.title(
    "Three-Head Overall Command Confusion Matrix"
)

for i in range(4):
    for j in range(4):

        plt.text(
            j,
            i,
            overall_cm[i, j],
            ha="center",
            va="center"
        )

plt.tight_layout()
plt.show()