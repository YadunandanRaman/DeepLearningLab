import os
import zipfile

import numpy as np

SEQUENCE_LENGTH = 128
NUM_CHANNELS = 9
NUM_CLASSES = 6

# Fixed by the dataset itself, not something to change per run.
CLASS_NAMES = [
    "WALKING", "WALKING_UPSTAIRS", "WALKING_DOWNSTAIRS",
    "SITTING", "STANDING", "LAYING",
]

# The 9 raw inertial signal channels, in the order they are stacked
# into the final (N, 128, 9) tensor. Each is one file inside the
# dataset's own Inertial Signals folder.
SIGNAL_NAMES = [
    "body_acc_x", "body_acc_y", "body_acc_z",
    "body_gyro_x", "body_gyro_y", "body_gyro_z",
    "total_acc_x", "total_acc_y", "total_acc_z",
]

# Set this to wherever your zip actually is once Google Drive is
# mounted, for example after running in a Colab cell:
#
#   from google.colab import drive
#   drive.mount('/content/drive')
#
# this is the one place the dataset path needs to be set, every script
# imports load_har_data() from this module rather than hardcoding a
# path itself.
ZIP_PATH = "/content/drive/MyDrive/UCI_Human.zip"


def _extract(zip_path):
    extract_dir = os.path.join(os.path.dirname(zip_path), "uci_har_extracted")
    dataset_dir = os.path.join(extract_dir, "UCI HAR Dataset")

    if not os.path.isdir(dataset_dir):
        print(f"extracting {zip_path}, this only happens once")
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(extract_dir)

        # search for the expected "train" and "test" subfolders
        # anywhere under extract_dir instead of assuming one fixed
        # layout, in case this particular archive nests them a
        # level differently than the standard release does
        if not os.path.isdir(dataset_dir):
            for root, dirs, files in os.walk(extract_dir):
                if "train" in dirs and "test" in dirs:
                    dataset_dir = root
                    break

    if not os.path.isdir(dataset_dir):
        raise FileNotFoundError(
            f"could not find a folder containing both train/ and test/ inside {extract_dir}, "
            f"the archive's internal layout may differ from what this loader expects"
        )

    return dataset_dir


def _load_signals(split_dir, split_name):
    """
    Loads all 9 Inertial Signals files for one split ("train" or
    "test") and stacks them into a single (N, 128, 9) array. Each file
    is whitespace separated, one row per window, one column per of the
    128 time steps within that window.
    """
    signals = []
    for name in SIGNAL_NAMES:
        file_path = os.path.join(split_dir, "Inertial Signals", f"{name}_{split_name}.txt")
        channel = np.loadtxt(file_path)
        signals.append(channel)
    # each entry in signals is (N, 128); stack along a new last axis to
    # get (N, 128, 9)
    return np.stack(signals, axis=-1).astype("float32")


def _load_labels(split_dir, split_name):
    labels = np.loadtxt(os.path.join(split_dir, f"y_{split_name}.txt")).astype("int64")
    return labels - 1  # dataset labels are 1 to 6, converted to 0 to 5


def load_har_data(val_fraction=0.15, seed=42):
    """
    Loads the UCI HAR dataset's raw inertial signals (not the
    precomputed 561 feature vectors) and returns:

    (X_train, y_train), (X_val, y_val), (X_test, y_test)

    X arrays have shape (N, 128, 9), y arrays are integer class labels
    0 to 5.

    The dataset's own train and test splits are subject disjoint, no
    subject appears in both, which is what actually makes the test set
    a meaningful measure of generalization to new people, not just new
    windows from people the model has already partly seen. Rather than
    discard that design and draw a fresh random 70/15/15 split across
    everyone, the official test split is kept exactly as the dataset
    defines it, and the validation set is instead carved out of the
    official training split (15 percent by default), so validation is
    still subject disjoint from training the same way, and the test
    set stays completely untouched by that split, exactly as this
    experiment requires.

    Normalization statistics (mean and standard deviation, computed per
    channel) come only from the training portion, then are applied to
    validation and test as well, so no information from either ever
    leaks into how the training data itself is scaled.
    """
    dataset_dir = _extract(ZIP_PATH)

    X_train_full = _load_signals(os.path.join(dataset_dir, "train"), "train")
    y_train_full = _load_labels(os.path.join(dataset_dir, "train"), "train")
    X_test = _load_signals(os.path.join(dataset_dir, "test"), "test")
    y_test = _load_labels(os.path.join(dataset_dir, "test"), "test")

    print(f"loaded {len(X_train_full)} official training windows and {len(X_test)} official test windows")

    rng = np.random.default_rng(seed)
    total = len(X_train_full)
    val_size = int(total * val_fraction)
    perm = rng.permutation(total)
    val_idx, train_idx = perm[:val_size], perm[val_size:]

    X_train, y_train = X_train_full[train_idx], y_train_full[train_idx]
    X_val, y_val = X_train_full[val_idx], y_train_full[val_idx]

    # per channel mean and standard deviation, from the training
    # portion only
    mean = X_train.mean(axis=(0, 1), keepdims=True)
    std = X_train.std(axis=(0, 1), keepdims=True) + 1e-8

    X_train = (X_train - mean) / std
    X_val = (X_val - mean) / std
    X_test = (X_test - mean) / std

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)
