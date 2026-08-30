import tensorflow as tf
import tensorflow_datasets as tfds

IMAGE_SIZE = (224, 224)
NUM_CLASSES = 37

# tensorflow_datasets' oxford_iiit_pet ships with these 37 breed names,
# in this exact order, as its label ClassLabel feature. Verified
# directly against tfds.builder("oxford_iiit_pet").info before writing
# this file, rather than assumed.
CLASS_NAMES = [
    "Abyssinian", "american_bulldog", "american_pit_bull_terrier", "basset_hound",
    "beagle", "Bengal", "Birman", "Bombay", "boxer", "British_Shorthair",
    "chihuahua", "Egyptian_Mau", "english_cocker_spaniel", "english_setter",
    "german_shorthaired", "great_pyrenees", "havanese", "japanese_chin",
    "keeshond", "leonberger", "Maine_Coon", "miniature_pinscher", "newfoundland",
    "Persian", "pomeranian", "pug", "Ragdoll", "Russian_Blue", "saint_bernard",
    "samoyed", "scottish_terrier", "shiba_inu", "Siamese", "Sphynx",
    "staffordshire_bull_terrier", "wheaten_terrier", "yorkshire_terrier",
]

# The official split has no separate validation set, only train (3,680
# images) and test (3,669 images). Used as a fallback if
# dataset.cardinality() cannot be resolved before the validation split
# is taken below, which can happen depending on Colab's tfds version.
OFFICIAL_TRAIN_SIZE = 3680


def _resize(image, label):
    image = tf.image.resize(image, IMAGE_SIZE)
    image = tf.cast(image, tf.float32)
    return image, label


def load_datasets(val_fraction=0.2, seed=42):
    """
    Loads the Oxford-IIIT Pet dataset (37 cat and dog breeds) and
    resizes every image to 224 x 224 x 3, as the lab manual specifies.

    Returns three unbatched tf.data.Dataset objects of (image, label)
    pairs, images as float32 in the original 0 to 255 range:

    - train_ds: 80 percent of the official training split (about 2,944
      images), used to fit model weights.
    - val_ds: the remaining 20 percent of the official training split
      (about 736 images), used for model selection across every study
      in this experiment, which initialization, which regularizer,
      which hyperparameters.
    - test_ds: the official test split (3,669 images), completely
      untouched by every script except the final evaluation in Section
      12, exactly as the manual requires.

    Images are left unnormalized: MobileNetV2's own required
    preprocessing (rescaling to -1 to 1) is applied inside the model
    itself in model_builder.py, and expects 0 to 255 input, not an
    already rescaled image.
    """
    train_val_ds, test_ds = tfds.load(
        "oxford_iiit_pet",
        split=["train", "test"],
        as_supervised=True,
        shuffle_files=True,
    )

    train_val_ds = train_val_ds.map(_resize, num_parallel_calls=tf.data.AUTOTUNE)
    test_ds = test_ds.map(_resize, num_parallel_calls=tf.data.AUTOTUNE)

    total = int(train_val_ds.cardinality().numpy())
    if total <= 0:
        total = OFFICIAL_TRAIN_SIZE
    val_size = int(total * val_fraction)

    train_val_ds = train_val_ds.shuffle(total, seed=seed, reshuffle_each_iteration=False)
    val_ds = train_val_ds.take(val_size)
    train_ds = train_val_ds.skip(val_size)

    return train_ds, val_ds, test_ds


def as_numpy(dataset, limit=None):
    """
    Materializes a tf.data.Dataset of (image, label) pairs into numpy
    arrays. Used only by the cross validation script in Section 11,
    which needs the training and validation pools combined into plain
    arrays to hand to sklearn's KFold, and by the final evaluation
    script for the same reason. Every other script trains directly off
    the tf.data.Dataset objects load_datasets returns, without ever
    pulling the whole thing into memory at once.
    """
    images, labels = [], []
    for image, label in dataset.take(limit) if limit else dataset:
        images.append(image.numpy())
        labels.append(label.numpy())
    import numpy as np
    return np.stack(images).astype("uint8"), np.array(labels)
