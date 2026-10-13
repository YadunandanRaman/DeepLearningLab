"""
Dataset handling (Section 3): tf_flowers, 224 x 224, ImageNet preprocessing, identical augmentation for
every model, and one fixed stratified 70 / 15 / 15 split that is reused by every experiment.

The dataset is downloaded automatically (keras get_file, the same 3670 photographs as tf_flowers) the
first time, resized to 224 x 224 and cached as a uint8 array, so nothing is read from disk twice.
"""
import json
import os

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications.resnet50 import preprocess_input

import config

DATA_URL = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
IMAGES_NPY = "images_224.npy"
LABELS_NPY = "labels.npy"
SPLIT_JSON = "split_indices.json"


def _cache(name):
    return os.path.join(config.DATA_DIR, name)


def load_all_images():
    """Return (images uint8 [N,224,224,3], labels int [N]) for the full dataset, using the cache if present."""
    config.make_dirs()
    if config.SYNTHETIC:                               # random class-tinted images for the dry run
        rng = np.random.RandomState(config.SEED)
        labels = np.arange(250) % config.NUM_CLASSES
        base = rng.randint(0, 256, (250, config.IMG_SIZE, config.IMG_SIZE, 3))
        tint = np.array([[200, 60, 60], [60, 200, 60], [60, 60, 200], [200, 200, 60], [200, 60, 200]])[labels]
        images = np.clip(0.5 * base + 0.5 * tint[:, None, None, :], 0, 255).astype(np.uint8)
        return images, labels.astype(np.int64)
    xi, yi = os.path.join(config.CACHE_DIR, IMAGES_NPY), os.path.join(config.CACHE_DIR, LABELS_NPY)
    if os.path.exists(xi) and os.path.exists(yi):                      # local cache, rebuilt automatically if missing
        return np.load(xi), np.load(yi)

    archive = keras.utils.get_file("flower_photos.tgz", DATA_URL, untar=True, cache_subdir="datasets")
    root = archive if os.path.isdir(archive) else os.path.join(os.path.dirname(archive), "flower_photos")
    if not os.path.isdir(os.path.join(root, "daisy")):
        root = os.path.join(root, "flower_photos")
    images, labels = [], []
    for idx, cls in enumerate(config.CLASS_NAMES):
        folder = os.path.join(root, cls)
        files = sorted(f for f in os.listdir(folder) if f.lower().endswith((".jpg", ".jpeg", ".png")))
        print(f"  {cls:11s}: {len(files)} images")
        for f in files:
            img = tf.io.decode_image(tf.io.read_file(os.path.join(folder, f)), channels=3, expand_animations=False)
            img = tf.image.resize(img, (config.IMG_SIZE, config.IMG_SIZE), method="bilinear")
            images.append(tf.cast(tf.round(img), tf.uint8).numpy())
            labels.append(idx)
    images, labels = np.stack(images), np.array(labels, dtype=np.int64)
    np.save(xi, images)
    np.save(yi, labels)
    return images, labels


def make_split(labels):
    """Stratified 70 / 15 / 15 split with a fixed seed; the indices are stored so every script reuses them."""
    from sklearn.model_selection import train_test_split

    path = _cache(SPLIT_JSON)
    if os.path.exists(path):
        with open(path) as f:
            s = json.load(f)
        return {k: np.array(v) for k, v in s.items()}
    idx = np.arange(len(labels))
    tr, rest = train_test_split(idx, test_size=config.SPLIT[1] + config.SPLIT[2], stratify=labels,
                                random_state=config.SEED)
    va, te = train_test_split(rest, test_size=config.SPLIT[2] / (config.SPLIT[1] + config.SPLIT[2]),
                              stratify=labels[rest], random_state=config.SEED)
    if config.QUICK:                                  # tiny stratified subsets for the smoke test
        rng = np.random.RandomState(config.SEED)
        tr, va, te = (rng.permutation(a)[:n] for a, n in zip((tr, va, te), config.QUICK_SUBSET))
    split = {"train": np.sort(tr), "val": np.sort(va), "test": np.sort(te)}
    with open(path, "w") as f:
        json.dump({k: v.tolist() for k, v in split.items()}, f)
    return split


_CACHE = {}


def load_split():
    """Dict with x_train/y_train/x_val/y_val/x_test/y_test (images uint8, labels int) plus the indices."""
    if "split" in _CACHE:
        return _CACHE["split"]
    images, labels = load_all_images()
    split = make_split(labels)
    out = {"indices": split}
    for k in ("train", "val", "test"):
        out[f"x_{k}"], out[f"y_{k}"] = images[split[k]], labels[split[k]]
    _CACHE["split"] = out
    return out


def preprocess_array(x_uint8):
    """ImageNet preprocessing (RGB -> BGR, mean subtraction). Always works on a float copy."""
    return preprocess_input(np.array(x_uint8, dtype=np.float32, copy=True))


def augmenter():
    """The augmentation used for EVERY model (Section 3.1). Applied to training images only, on the GPU
    (the layers are part of the training graph, so the CPU is not a bottleneck)."""
    s = config.SEED
    f32 = {"dtype": "float32"}
    return keras.Sequential([
        keras.layers.RandomFlip(config.AUG_FLIP, seed=s, **f32),
        keras.layers.RandomRotation(config.AUG_ROTATION, seed=s, **f32),
        keras.layers.RandomZoom(config.AUG_ZOOM, seed=s, **f32),
        keras.layers.RandomTranslation(config.AUG_TRANSLATION, config.AUG_TRANSLATION, seed=s, **f32),
        keras.layers.RandomContrast(config.AUG_CONTRAST, seed=s, **f32),
    ], name="augmentation")


def make_dataset(x, y, train, batch_size=None):
    """tf.data pipeline of uint8 image batches and one hot labels. Augmentation and ImageNet preprocessing are applied
    on the GPU by the training model (train_utils.make_training_model), so the data is only shuffled and batched here."""
    batch_size = batch_size or config.BATCH_SIZE
    onehot = keras.utils.to_categorical(y, config.NUM_CLASSES).astype("float32")
    ds = tf.data.Dataset.from_tensor_slices((x, onehot))
    if train:
        ds = ds.shuffle(len(x), seed=config.SEED, reshuffle_each_iteration=True)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def class_counts():
    d = load_split()
    rows = []
    for i, c in enumerate(config.CLASS_NAMES):
        rows.append([c] + [int((d[f"y_{k}"] == i).sum()) for k in ("train", "val", "test")])
    return rows
