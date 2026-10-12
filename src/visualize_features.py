import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

FEATURE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "features"
)

CLASSES = [
    "light_on",
    "light_off",
    "fan_on",
    "fan_off",
    "unknown",
    "silence"
]

X = []
y = []

for label_index, label in enumerate(CLASSES):

    files = glob.glob(
        os.path.join(FEATURE_DIR, label, "*.npy")
    )

    for file in files:

        mfcc = np.load(file)

        X.append(mfcc.flatten())
        y.append(label_index)

X = np.array(X)
y = np.array(y)

print("Feature matrix:", X.shape)

pca = PCA(n_components=2)
X_2d = pca.fit_transform(X)

plt.figure(figsize=(9, 6))

for label_index, label in enumerate(CLASSES):

    indices = y == label_index

    plt.scatter(
        X_2d[indices, 0],
        X_2d[indices, 1],
        label=label
    )

plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("MFCC Feature Distribution")
plt.legend()
plt.tight_layout()
plt.show()

print(
    "Explained variance:",
    pca.explained_variance_ratio_
)

print(
    "Total:",
    np.sum(pca.explained_variance_ratio_)
)