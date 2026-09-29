"""
Shared data handling: loading MNIST, drawing the laboratory subset,
normalizing, saving and loading the prepared splits, and corrupting
images with Gaussian or salt and pepper noise (Section 13).
"""
import os

import numpy as np

import config

SPLIT_NAMES = ["x_train", "y_train", "x_val", "y_val", "x_test", "y_test"]

# fixed offsets so the noisy train, validation and test sets are always
# generated with the same random numbers, in every script that needs them
NOISE_SEED_OFFSETS = {"train": 101, "val": 202, "test": 303}


def load_raw_mnist():
    from tensorflow import keras

    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    return x_train, y_train, x_test, y_test


def select_subset(x, y, n, seed):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(x))[:n]
    return x[idx], y[idx]


def normalize(x):
    """uint8 images in [0, 255] -> float32 images in [0, 1] with a channel axis."""
    return (x.astype("float32") / 255.0)[..., np.newaxis]


def flatten(x):
    return np.asarray(x).reshape(len(x), -1)


def unflatten(x):
    return np.asarray(x).reshape(len(x), *config.IMG_SHAPE)


def save_splits(splits):
    os.makedirs(config.DATA_DIR, exist_ok=True)
    for name in SPLIT_NAMES:
        np.save(os.path.join(config.DATA_DIR, f"{name}.npy"), splits[name])


def load_splits():
    splits = {}
    for name in SPLIT_NAMES:
        path = os.path.join(config.DATA_DIR, f"{name}.npy")
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} not found. Run 2_preprocessing.py first.")
        splits[name] = np.load(path)
    return splits


def add_gaussian_noise(x, sigma, seed):
    rng = np.random.default_rng(seed)
    noisy = x + rng.normal(0.0, sigma, size=x.shape)
    return np.clip(noisy, 0.0, 1.0).astype("float32")


def add_salt_pepper_noise(x, p, seed):
    """Replaces a proportion p of pixels: half set to 0 (pepper), half set to 1 (salt)."""
    rng = np.random.default_rng(seed)
    noisy = np.array(x, dtype="float32", copy=True)
    u = rng.random(size=x.shape)
    noisy[u < p / 2] = 0.0
    noisy[(u >= p / 2) & (u < p)] = 1.0
    return noisy


def add_noise(x, kind, level, split):
    seed = config.SEED + NOISE_SEED_OFFSETS[split] + int(round(level * 1000))
    if kind == "gaussian":
        return add_gaussian_noise(x, level, seed)
    if kind == "salt_pepper":
        return add_salt_pepper_noise(x, level, seed)
    raise ValueError(f"unknown noise type: {kind}")


def noise_label(kind, level):
    if kind == "gaussian":
        return f"Gaussian, sigma = {level}"
    return f"Salt and pepper, p = {level}"


def noise_tag(kind, level):
    return f"{kind}_{str(level).replace('.', 'p')}"
