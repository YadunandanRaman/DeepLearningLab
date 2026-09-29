"""
Step 2: prepare the data (Sections 3 and 4 of the lab manual).

  * draws the laboratory subset: 10000 training and 2000 test images
  * holds out 10 percent of the training images for validation
  * scales pixels from [0, 255] to [0, 1] and adds the channel axis
  * prints the tensor shapes the fully connected (784) and
    convolutional (28 x 28 x 1) models receive
  * saves the prepared arrays to outputs/saved_data/

Labels are saved only for visualization (latent space colouring and
error analysis). No model ever uses them as a target: the target of
every autoencoder is the image itself.
"""
import numpy as np

import config
from data_utils import flatten, load_raw_mnist, normalize, save_splits, select_subset
from train_utils import save_json


def main():
    config.make_dirs()
    config.set_seed()

    x_train_raw, y_train_raw, x_test_raw, y_test_raw = load_raw_mnist()

    x_pool, y_pool = select_subset(x_train_raw, y_train_raw, config.N_TRAIN, config.SEED)
    x_test, y_test = select_subset(x_test_raw, y_test_raw, config.N_TEST, config.SEED + 1)

    n_val = int(config.N_TRAIN * config.VAL_FRACTION)
    x_val, y_val = x_pool[:n_val], y_pool[:n_val]
    x_train, y_train = x_pool[n_val:], y_pool[n_val:]

    print("before preprocessing:")
    print(f"  x_train {x_train.shape} {x_train.dtype}, range [{x_train.min()}, {x_train.max()}]")
    print(f"  x_val   {x_val.shape}")
    print(f"  x_test  {x_test.shape}")

    splits = {
        "x_train": normalize(x_train), "y_train": y_train.astype("int64"),
        "x_val": normalize(x_val), "y_val": y_val.astype("int64"),
        "x_test": normalize(x_test), "y_test": y_test.astype("int64"),
    }

    print("\nafter preprocessing (convolutional input, 28 x 28 x 1):")
    for name in ("x_train", "x_val", "x_test"):
        arr = splits[name]
        print(f"  {name:<7} {arr.shape} {arr.dtype}, range [{arr.min():.1f}, {arr.max():.1f}]")

    print("\nflattened view (fully connected input, 28 x 28 = 784):")
    for name in ("x_train", "x_val", "x_test"):
        print(f"  {name:<7} {flatten(splits[name]).shape}")

    batch = splits["x_train"][:32]
    print(f"\none training batch of 32 images: {batch.shape}")
    print(f"the same batch flattened:        {flatten(batch).shape}")

    save_splits(splits)

    summary = {
        "n_train": int(len(x_train)),
        "n_val": int(len(x_val)),
        "n_test": int(len(x_test)),
        "image_shape": list(config.IMG_SHAPE),
        "flat_dim": config.FLAT_DIM,
        "train_class_counts": np.bincount(y_train, minlength=10).tolist(),
        "val_class_counts": np.bincount(y_val, minlength=10).tolist(),
        "test_class_counts": np.bincount(y_test, minlength=10).tolist(),
        "seed": config.SEED,
    }
    save_json(summary, config.result_path("02_data_split.json"))
    print("\nsaved prepared arrays to", config.DATA_DIR)


if __name__ == "__main__":
    main()
