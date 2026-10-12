import os
import glob
import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt


# ============================================================
# Paths
# ============================================================

PROJECT_DIR = os.path.dirname(os.path.dirname(__file__))

FEATURE_DIR = os.path.join(
    PROJECT_DIR,
    "features"
)

MODEL_DIR = os.path.join(
    PROJECT_DIR,
    "models"
)

os.makedirs(MODEL_DIR, exist_ok=True)


# ============================================================
# Classes
# ============================================================

CLASSES = [
    "light_on",
    "light_off",
    "fan_on",
    "fan_off",
    "no_command"
]

# unknown and silence are merged into the same class
CLASS_FOLDERS = {
    "light_on": 0,
    "light_off": 1,
    "fan_on": 2,
    "fan_off": 3,
    "unknown": 4,
    "silence": 4
}


# ============================================================
# Load features
# ============================================================

X = []
y = []

for folder, label in CLASS_FOLDERS.items():

    folder_path = os.path.join(
        FEATURE_DIR,
        folder
    )

    files = glob.glob(
        os.path.join(folder_path, "*.npy")
    )

    print(f"{folder}: {len(files)} samples")

    for file in files:

        mfcc = np.load(file)

        X.append(mfcc)
        y.append(label)


X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=np.int32)

print("\nTotal samples:", len(X))
print("Feature shape:", X.shape)


# ============================================================
# Add CNN channel dimension
# ============================================================

X = X[..., np.newaxis]

print("CNN input shape:", X.shape)


# ============================================================
# Train/Test split
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# ============================================================
# Validation split
# ============================================================

X_train, X_val, y_train, y_val = train_test_split(
    X_train,
    y_train,
    test_size=0.20,
    random_state=42,
    stratify=y_train
)

print("\nTraining:", X_train.shape)
print("Validation:", X_val.shape)
print("Testing:", X_test.shape)


# ============================================================
# Normalize MFCCs
# ============================================================

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


# ============================================================
# Tiny CNN
# ============================================================

model = tf.keras.Sequential([

    tf.keras.layers.Input(
        shape=X_train.shape[1:]
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
        len(CLASSES),
        activation="softmax"
    )
])


# ============================================================
# Compile
# ============================================================

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


model.summary()


# ============================================================
# Training callbacks
# ============================================================

callbacks = [

    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=15,
        restore_best_weights=True
    ),

    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=5,
        min_lr=1e-6
    )
]


# ============================================================
# Train
# ============================================================

history = model.fit(

    X_train,
    y_train,

    validation_data=(
        X_val,
        y_val
    ),

    epochs=100,

    batch_size=8,

    callbacks=callbacks,

    verbose=1
)


# ============================================================
# Evaluate
# ============================================================

test_loss, test_accuracy = model.evaluate(
    X_test,
    y_test,
    verbose=0
)

print("\n================================")
print("TEST RESULTS")
print("================================")

print(
    f"Test accuracy: "
    f"{test_accuracy * 100:.2f}%"
)


# ============================================================
# Predictions
# ============================================================

predictions = model.predict(
    X_test,
    verbose=0
)

y_pred = np.argmax(
    predictions,
    axis=1
)


# ============================================================
# Classification report
# ============================================================

print("\nClassification Report:\n")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=CLASSES,
        zero_division=0
    )
)


# ============================================================
# Confusion matrix
# ============================================================

cm = confusion_matrix(
    y_test,
    y_pred
)

print("\nConfusion Matrix:")
print(cm)


plt.figure(figsize=(8, 6))

plt.imshow(
    cm,
    interpolation="nearest"
)

plt.xticks(
    range(len(CLASSES)),
    CLASSES,
    rotation=45
)

plt.yticks(
    range(len(CLASSES)),
    CLASSES
)

plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.title(
    "TinyML 5-Class Confusion Matrix"
)

for i in range(len(CLASSES)):
    for j in range(len(CLASSES)):

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
# Training curves
# ============================================================

plt.figure(figsize=(8, 5))

plt.plot(
    history.history["accuracy"],
    label="Training"
)

plt.plot(
    history.history["val_accuracy"],
    label="Validation"
)

plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training vs Validation Accuracy")
plt.legend()
plt.tight_layout()
plt.show()


# ============================================================
# Save model
# ============================================================

model_path = os.path.join(
    MODEL_DIR,
    "voice_5class_baseline.keras"
)

model.save(model_path)

print(
    f"\nModel saved to:\n{model_path}"
)