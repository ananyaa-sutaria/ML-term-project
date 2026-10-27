"""Preprocess the Sign Language MNIST dataset.

Usage:
    python preprocess_sign_mnist.py <data_dir> [out_dir]

<data_dir> must contain sign_mnist_train.csv and sign_mnist_test.csv
(https://www.kaggle.com/datasets/datamunge/sign-language-mnist).
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

SEED = 42
# Labels are 0-24 but J (9) and Z (25) need motion, so 9 and 25 never appear.
LETTERS = [c for c in "ABCDEFGHIKLMNOPQRSTUVWXY"]  # 24 static letters, in label order


def load(path):
    df = pd.read_csv(path)
    return df["label"].to_numpy(), df.drop(columns="label").to_numpy(dtype=np.uint8)


def validate(name, y, X):
    print(f"\n== {name} ==")
    print(f"images: {X.shape[0]}, pixels per image: {X.shape[1]}")
    assert X.shape[1] == 28 * 28, "expected 784 pixels per image"
    print(f"pixel range: {X.min()}-{X.max()}")
    print(f"duplicate images: {pd.DataFrame(X).duplicated().sum()}")
    print(f"unexpected labels: {sorted(set(y) - set(range(25)) - {9})}")


def to_letters(y):
    # Map raw labels (skipping 9) to contiguous 0-23 class ids.
    return np.where(y > 9, y - 1, y)


def main(data_dir, out_dir):
    data_dir, out_dir = Path(data_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    y_tr_raw, X_tr = load(data_dir / "sign_mnist_train.csv")
    y_te_raw, X_te = load(data_dir / "sign_mnist_test.csv")
    validate("train", y_tr_raw, X_tr)
    validate("test", y_te_raw, X_te)

    y_tr, y_te = to_letters(y_tr_raw), to_letters(y_te_raw)

    # Class distribution
    counts = pd.Series(y_tr).value_counts().sort_index()
    counts.index = [LETTERS[i] for i in counts.index]
    print("\nTrain class counts:\n", counts.to_string())
    ax = counts.plot.bar(figsize=(9, 3), title="Train class distribution")
    ax.figure.tight_layout()
    ax.figure.savefig(out_dir / "class_distribution.png", dpi=120)
    plt.close(ax.figure)

    # Scale to [0, 1] and reshape to (N, 28, 28, 1)
    X_tr = (X_tr / 255.0).astype(np.float32).reshape(-1, 28, 28, 1)
    X_te = (X_te / 255.0).astype(np.float32).reshape(-1, 28, 28, 1)

    # Hold out a stratified validation set from the training data
    X_train, X_val, y_train, y_val = train_test_split(
        X_tr, y_tr, test_size=0.15, stratify=y_tr, random_state=SEED
    )
    print(f"\nsplit sizes - train: {len(X_train)}, val: {len(X_val)}, test: {len(X_te)}")

    np.savez_compressed(
        out_dir / "sign_mnist_processed.npz",
        X_train=X_train, y_train=y_train,
        X_val=X_val, y_val=y_val,
        X_test=X_te, y_test=y_te,
        letters=np.array(LETTERS),
    )

    # Sample grid, one image per class
    fig, axes = plt.subplots(3, 8, figsize=(12, 5))
    for i, ax in enumerate(axes.ravel()):
        ax.imshow(X_train[np.argmax(y_train == i)].squeeze(), cmap="gray")
        ax.set_title(LETTERS[i])
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_dir / "samples.png", dpi=120)
    print(f"saved outputs to {out_dir}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "processed")
