import os
import pickle
import numpy as np

# If you have already extracted cifar-10-python.tar.gz yourself (for
# example from Google Drive, via
#   !tar -xvf /content/drive/MyDrive/cifar-10-python.tar -C /content/ cifar-10-batches-py/
# ), set this to wherever the cifar-10-batches-py folder ended up.
# This is the one place the dataset path needs to be set, every script
# that loads CIFAR-10 imports load_data() from this module instead of
# hardcoding a path itself.
DATASET_DIR = "/content/cifar-10-batches-py"


def _unpickle(file_path):
    with open(file_path, "rb") as f:
        # encoding="bytes" is required for Python 3 to read this dataset correctly
        return pickle.load(f, encoding="bytes")


def _load_batch(file_path):
    batch = _unpickle(file_path)
    images = batch[b"data"]
    labels = batch[b"labels"]

    # each row is 3072 values: 1024 red, then 1024 green, then 1024
    # blue, each itself a flattened 32 x 32 channel, in that order
    images = images.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    labels = np.array(labels, dtype="uint8").reshape(-1, 1)
    return images, labels


def load_data(dataset_dir=DATASET_DIR):
    """
    Drop in replacement for tensorflow.keras.datasets.cifar10.load_data().

    If dataset_dir exists and has the five data_batch_* files plus
    test_batch, they are read directly, matching the batches you get
    from extracting cifar-10-python.tar.gz yourself. If the folder is
    not found, this falls back to the standard Keras downloader instead,
    so the same scripts work whether or not you have a local copy.

    Returns the same (X_train, y_train), (X_test, y_test) structure as
    the Keras loader: X as uint8 images of shape (N, 32, 32, 3), y as a
    column vector of shape (N, 1) with class indices 0 to 9.
    """
    train_batch_names = [f"data_batch_{i}" for i in range(1, 6)]
    has_local_copy = os.path.isdir(dataset_dir) and all(
        os.path.isfile(os.path.join(dataset_dir, name)) for name in train_batch_names + ["test_batch"]
    )

    if not has_local_copy:
        print(f"no local CIFAR-10 batches found at {dataset_dir}, downloading via Keras instead")
        from tensorflow.keras.datasets import cifar10
        return cifar10.load_data()

    print(f"loading CIFAR-10 from local batches at {dataset_dir}")

    train_images, train_labels = [], []
    for name in train_batch_names:
        images, labels = _load_batch(os.path.join(dataset_dir, name))
        train_images.append(images)
        train_labels.append(labels)

    X_train = np.concatenate(train_images, axis=0)
    y_train = np.concatenate(train_labels, axis=0)
    X_test, y_test = _load_batch(os.path.join(dataset_dir, "test_batch"))

    return (X_train, y_train), (X_test, y_test)
