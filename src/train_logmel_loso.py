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
# Classes
# ============================================================

CLASSES = [
    "light_on",
    "light_off",
    "fan_on",
    "fan_off"
]

LABELS = {
    "light_on": 0,
    "light_off": 1,
    "fan_on": 2,
    "fan_off": 3
}

SPEAKERS = [
    "speaker_1",
    "speaker_2",
    "speaker_3",
    "speaker_4"
]


# ============================================================
# Load speaker data
# ============================================================

def load_speaker(speaker):

    X = []
    y = []

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
            y.append(LABELS[command])

    return (
        np.array(X, dtype=np.float32),
        np.array(y, dtype=np.int32)
    )


# ============================================================
# CNN
# Keep architecture comparable to MFCC baseline
# ============================================================

def build_model(input_shape):

    model = tf.keras.Sequential([

        tf.keras.layers.Input(
            shape=input_shape
        ),

        tf.keras.layers.Conv2D(
            8,
            (3, 3),
            activation="relu",
            padding="same"
        ),

        tf.keras.layers.MaxPooling2D(
            (2, 2)
        ),

        tf.keras.layers.Conv2D(
            16,
            (3, 3),
            activation="relu",
            padding="same"
        ),

        tf.keras.layers.MaxPooling2D(
            (2, 2)
        ),

        tf.keras.layers.Flatten(),

        tf.keras.layers.Dense(
            32,
            activation="relu"
        ),

        tf.keras.layers.Dropout(0.2),

        tf.keras.layers.Dense(
            4,
            activation="softmax"
        )
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


# ============================================================
# LOSO experiment
# ============================================================

all_true = []
all_pred = []

fold_accuracies = []


for test_speaker in SPEAKERS:

    print("\n" + "=" * 60)
    print(f"TEST SPEAKER: {test_speaker}")
    print("=" * 60)

    train_speakers = [
        s for s in SPEAKERS
        if s != test_speaker
    ]

    X_train = []
    y_train = []

    for speaker in train_speakers:

        X_s, y_s = load_speaker(speaker)

        X_train.append(X_s)
        y_train.append(y_s)

    X_train = np.concatenate(X_train)
    y_train = np.concatenate(y_train)

    X_test, y_test = load_speaker(
        test_speaker
    )

    print("Training samples:", len(X_train))
    print("Test samples:", len(X_test))

    # Add CNN channel
    X_train = X_train[..., np.newaxis]
    X_test = X_test[..., np.newaxis]

    # Validation split
    (
        X_train,
        X_val,
        y_train,
        y_val
    ) = train_test_split(
        X_train,
        y_train,
        test_size=0.20,
        random_state=42,
        stratify=y_train
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

    std = np.maximum(std, 1e-6)

    X_train = (X_train - mean) / std
    X_val = (X_val - mean) / std
    X_test = (X_test - mean) / std

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
        y_train,
        validation_data=(X_val, y_val),
        epochs=80,
        batch_size=8,
        callbacks=callbacks,
        verbose=1
    )

    # --------------------------------------------------------
    # Test
    # --------------------------------------------------------

    probabilities = model.predict(
        X_test,
        verbose=0
    )

    y_pred = np.argmax(
        probabilities,
        axis=1
    )

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    fold_accuracies.append(
        accuracy
    )

    print(
        f"\n{test_speaker} accuracy: "
        f"{accuracy * 100:.2f}%"
    )

    cm = confusion_matrix(
        y_test,
        y_pred,
        labels=[0, 1, 2, 3]
    )

    print("\nConfusion Matrix:")
    print(cm)

    all_true.extend(y_test)
    all_pred.extend(y_pred)

    # Plot
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
        f"Log-Mel LOSO - Test {test_speaker}"
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

overall_accuracy = accuracy_score(
    all_true,
    all_pred
)

print("\n" + "=" * 60)
print("LOG-MEL LOSO RESULTS")
print("=" * 60)

for speaker, accuracy in zip(
    SPEAKERS,
    fold_accuracies
):

    print(
        f"{speaker}: "
        f"{accuracy * 100:.2f}%"
    )

print(
    f"\nOverall LOSO accuracy: "
    f"{overall_accuracy * 100:.2f}%"
)

print(
    f"Mean fold accuracy: "
    f"{np.mean(fold_accuracies) * 100:.2f}%"
)

print(
    f"Std deviation: "
    f"{np.std(fold_accuracies) * 100:.2f}%"
)


# ============================================================
# Overall confusion matrix
# ============================================================

overall_cm = confusion_matrix(
    all_true,
    all_pred,
    labels=[0, 1, 2, 3]
)

print("\nOverall Confusion Matrix:")
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
    "Overall Log-Mel Speaker-Independent Confusion Matrix"
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