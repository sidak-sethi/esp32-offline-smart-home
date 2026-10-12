import os
import glob
import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt


# ============================================================
# Paths
# ============================================================

PROJECT_DIR = os.path.dirname(os.path.dirname(__file__))

FEATURE_DIR = os.path.join(
    PROJECT_DIR,
    "logmel_features_fixed"
)


# ============================================================
# Labels
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

DEVICE_LABEL = {
    "light_on": 0,   # LIGHT
    "light_off": 0,
    "fan_on": 1,     # FAN
    "fan_off": 1
}

ACTION_LABEL = {
    "light_on": 0,   # ON
    "light_off": 1,  # OFF
    "fan_on": 0,
    "fan_off": 1
}

COMMAND_LABEL = {
    "light_on": 0,
    "light_off": 1,
    "fan_on": 2,
    "fan_off": 3
}


# ============================================================
# Data loading
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
# SpecAugment-style layer
# Applied ONLY during training
# ============================================================

class AudioAugmentation(tf.keras.layers.Layer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs, training=None):

        if not training:
            return inputs

        x = inputs

        # ----------------------------------------------------
        # Small random feature noise
        # ----------------------------------------------------

        noise = tf.random.normal(
            tf.shape(x),
            mean=0.0,
            stddev=0.025
        )

        x = x + noise

        # ----------------------------------------------------
        # Frequency masking
        # Input shape: [B, 32, 201, 1]
        # ----------------------------------------------------

        batch_size = tf.shape(x)[0]
        n_mels = tf.shape(x)[1]
        n_frames = tf.shape(x)[2]

        freq_width = tf.random.uniform(
            [],
            minval=1,
            maxval=5,
            dtype=tf.int32
        )

        freq_start = tf.random.uniform(
            [],
            minval=0,
            maxval=tf.maximum(
                1,
                n_mels - freq_width
            ),
            dtype=tf.int32
        )

        freq_mask = tf.ones(
            [batch_size, n_mels, n_frames, 1],
            dtype=x.dtype
        )

        mask_region = tf.pad(
            tf.zeros(
                [batch_size, freq_width, n_frames, 1],
                dtype=x.dtype
            ),
            [
                [0, 0],
                [freq_start, n_mels - freq_start - freq_width],
                [0, 0],
                [0, 0]
            ],
            constant_values=1
        )

        freq_mask = mask_region

        x = x * freq_mask

        # ----------------------------------------------------
        # Time masking
        # ----------------------------------------------------

        time_width = tf.random.uniform(
            [],
            minval=4,
            maxval=16,
            dtype=tf.int32
        )

        time_start = tf.random.uniform(
            [],
            minval=0,
            maxval=tf.maximum(
                1,
                n_frames - time_width
            ),
            dtype=tf.int32
        )

        time_mask = tf.pad(
            tf.zeros(
                [batch_size, n_mels, time_width, 1],
                dtype=x.dtype
            ),
            [
                [0, 0],
                [0, 0],
                [time_start, n_frames - time_start - time_width],
                [0, 0]
            ],
            constant_values=1
        )

        x = x * time_mask

        return x


# ============================================================
# Build two-head model
# ============================================================

def build_model(input_shape):

    inputs = tf.keras.Input(
        shape=input_shape,
        name="logmel"
    )

    # Augmentation happens only when model.fit() is training.
    x = AudioAugmentation(
        name="audio_augmentation"
    )(inputs)

    # --------------------------------------------------------
    # Shared CNN
    # --------------------------------------------------------

    x = tf.keras.layers.Conv2D(
        8,
        (3, 3),
        padding="same",
        activation="relu"
    )(x)

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

    # --------------------------------------------------------
    # Device head
    # --------------------------------------------------------

    device_output = tf.keras.layers.Dense(
        2,
        activation="softmax",
        name="device"
    )(x)

    # --------------------------------------------------------
    # Action head
    # --------------------------------------------------------

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
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=0.001
        ),

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
# Result storage
# ============================================================

all_true_command = []
all_pred_command = []

fold_device_acc = []
fold_action_acc = []
fold_command_acc = []


# ============================================================
# LOSO
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

    # --------------------------------------------------------
    # Training data
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
    # Held-out speaker
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
    # Add CNN channel dimension
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
    # Normalize using training data only
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

    callbacks = [

        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=12,
            restore_best_weights=True
        ),

        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
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
            "action": action_train
        },

        validation_data=(

            X_val,

            {
                "device": device_val,
                "action": action_val
            }
        ),

        epochs=100,

        batch_size=8,

        callbacks=callbacks,

        verbose=1
    )

    # --------------------------------------------------------
    # Predict held-out speaker
    # Augmentation automatically turns OFF
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
    # Combine device + action
    # --------------------------------------------------------

    command_pred = []

    for device, action in zip(
        device_pred,
        action_pred
    ):

        if device == 0 and action == 0:
            command_pred.append(0)

        elif device == 0 and action == 1:
            command_pred.append(1)

        elif device == 1 and action == 0:
            command_pred.append(2)

        else:
            command_pred.append(3)

    command_pred = np.array(
        command_pred
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
    # Global results
    # --------------------------------------------------------

    all_true_command.extend(
        command_test
    )

    all_pred_command.extend(
        command_pred
    )

    # --------------------------------------------------------
    # Fold confusion matrix
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
        f"Augmented Multi-Task LOSO - {test_speaker}"
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
# Final results
# ============================================================

overall_command = accuracy_score(
    all_true_command,
    all_pred_command
)

print("\n" + "=" * 70)
print("AUGMENTED MULTI-TASK LOSO RESULTS")
print("=" * 70)

for i, speaker in enumerate(SPEAKERS):

    print(
        f"{speaker}: "
        f"device={fold_device_acc[i] * 100:.2f}% | "
        f"action={fold_action_acc[i] * 100:.2f}% | "
        f"command={fold_command_acc[i] * 100:.2f}%"
    )

print(
    f"\nOverall exact command accuracy: "
    f"{overall_command * 100:.2f}%"
)

print(
    f"Mean command fold accuracy: "
    f"{np.mean(fold_command_acc) * 100:.2f}%"
)

print(
    f"Command fold std: "
    f"{np.std(fold_command_acc) * 100:.2f}%"
)


# ============================================================
# Overall confusion matrix
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
    "Augmented Multi-Task Overall Confusion Matrix"
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