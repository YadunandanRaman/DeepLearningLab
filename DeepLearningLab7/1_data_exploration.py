"""
Step 1: load MNIST and explore it (Section 3 of the lab manual).

Prints the raw dataset dimensions and pixel statistics and saves two
plots: one sample image of every digit, and the class distribution of
the full training set against the laboratory subset every later script
uses.
"""
import numpy as np
import matplotlib.pyplot as plt

import config
from data_utils import load_raw_mnist, select_subset
from plot_style import setup_fonts, save_fig
from visualize import show_image_grid


def main():
    config.make_dirs()
    config.set_seed()
    setup_fonts()

    x_train, y_train, x_test, y_test = load_raw_mnist()

    print("raw MNIST")
    print(f"  training images : {x_train.shape}, dtype {x_train.dtype}")
    print(f"  training labels : {y_train.shape}")
    print(f"  test images     : {x_test.shape}, dtype {x_test.dtype}")
    print(f"  test labels     : {y_test.shape}")
    print(f"  pixel range     : [{x_train.min()}, {x_train.max()}]")
    print(f"  mean pixel      : {x_train.mean():.2f}")
    print(f"  classes         : {np.unique(y_train).tolist()}")

    first_idx = [int(np.where(y_train == d)[0][0]) for d in range(10)]
    show_image_grid(
        x_train[first_idx] / 255.0, 1, 10,
        config.plot_path("01_sample_images"),
        titles=[f"Digit {d}" for d in range(10)],
    )

    # the same subset 2_preprocessing.py draws, to show it stays balanced
    _, y_sub = select_subset(x_train, y_train, config.N_TRAIN, config.SEED)
    full_pct = np.bincount(y_train, minlength=10) / len(y_train) * 100
    sub_pct = np.bincount(y_sub, minlength=10) / len(y_sub) * 100

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    pos = np.arange(10)
    width = 0.38
    ax.bar(pos - width / 2, full_pct, width, color="steelblue", label=f"Full training set ({len(y_train)} images)")
    ax.bar(pos + width / 2, sub_pct, width, color="darkorange", label=f"Laboratory subset ({config.N_TRAIN} images)")
    ax.set_xticks(pos)
    ax.set_xlabel("Digit")
    ax.set_ylabel("Share of images (%)")
    ax.grid(True, axis="y", linestyle=":", linewidth=0.6)
    ax.legend(loc="lower right")
    fig.tight_layout()
    print("saved", save_fig(fig, config.plot_path("01_class_distribution")))
    plt.close(fig)

    print("\nclass counts (full training set / laboratory subset):")
    for d in range(10):
        print(f"  digit {d}: {int((y_train == d).sum()):5d} / {int((y_sub == d).sum()):4d}")


if __name__ == "__main__":
    main()
