"""
Shared figure helpers so every script draws image grids, curves and
latent scatter plots the same way. Every figure goes through save_fig(),
so all of them are EPS at 600 DPI.

EPS does not support transparency, so no plot here uses alpha.
"""
import numpy as np
import matplotlib.pyplot as plt

from plot_style import save_fig


def _img(x):
    return np.asarray(x, dtype=np.float32).reshape(28, 28)


def _clean_axis(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def _finish(fig, path):
    fig.tight_layout()
    out = save_fig(fig, path)
    plt.close(fig)
    print("saved", out)
    return out


def show_image_rows(rows, row_labels, path, col_titles=None, row_cmaps=None, row_vmax=None):
    """
    rows[r][c] is one 28 x 28 image. Each row is a category (for example
    Original, Reconstruction, Error) and each column is one test sample.
    """
    n_rows, n_cols = len(rows), len(rows[0])
    row_cmaps = row_cmaps or ["gray"] * n_rows
    row_vmax = row_vmax or [1.0] * n_rows
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(1.1 * n_cols + 1.0, 1.15 * n_rows + 0.3), squeeze=False)
    for r in range(n_rows):
        vmax = max(float(row_vmax[r]), 1e-8)
        for c in range(n_cols):
            ax = axes[r][c]
            ax.imshow(_img(rows[r][c]), cmap=row_cmaps[r], vmin=0.0, vmax=vmax)
            _clean_axis(ax)
            if c == 0:
                ax.set_ylabel(row_labels[r], fontsize=9)
            if r == 0 and col_titles is not None:
                ax.set_title(col_titles[c], fontsize=8)
    return _finish(fig, path)


def show_image_grid(images, n_rows, n_cols, path, titles=None):
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(1.1 * n_cols, 1.15 * n_rows), squeeze=False)
    for i, ax in enumerate(axes.ravel()):
        if i < len(images):
            ax.imshow(_img(images[i]), cmap="gray", vmin=0.0, vmax=1.0)
            if titles is not None:
                ax.set_title(titles[i], fontsize=8)
        _clean_axis(ax)
    return _finish(fig, path)


def plot_curves(x, series, xlabel, ylabel, path, marker="o", xticks=None, figsize=(6.5, 4.2)):
    """series is {legend label: y values}; every curve shares one y quantity."""
    fig, ax = plt.subplots(figsize=figsize)
    for label, values in series.items():
        ax.plot(x, values, marker=marker, markersize=4, linewidth=1.4, label=label)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if xticks is not None:
        ax.set_xticks(xticks)
    ax.grid(True, linestyle=":", linewidth=0.6)
    ax.legend()
    return _finish(fig, path)


def plot_loss_curves(history, train_key, val_key, ylabel, path):
    epochs = np.arange(1, len(history[train_key]) + 1)
    return plot_curves(
        epochs,
        {"Training": history[train_key], "Validation": history[val_key]},
        "Epoch", ylabel, path, marker="o",
    )


def plot_latent_scatter(z, labels, path, xlabel="$z_1$", ylabel="$z_2$", paths=None):
    """
    2D latent codes coloured by digit label. Labels are only used for
    colouring, they are never given to the model.
    paths is an optional list of {"points": (k, 2) array, "label": str}
    drawn on top (for example an interpolation path).
    """
    fig, ax = plt.subplots(figsize=(7.0, 5.8))
    cmap = plt.get_cmap("tab10")
    for d in range(10):
        m = labels == d
        ax.scatter(z[m, 0], z[m, 1], s=3, color=cmap(d), label=str(d), rasterized=True)
    if paths:
        styles = [("black", "o"), ("dimgray", "s")]
        for i, p in enumerate(paths):
            color, marker = styles[i % len(styles)]
            pts = np.asarray(p["points"])
            ax.plot(pts[:, 0], pts[:, 1], color=color, marker=marker, markersize=4,
                    linewidth=1.3, label=p["label"])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, linestyle=":", linewidth=0.6)
    ax.legend(title="Digit", markerscale=4, bbox_to_anchor=(1.02, 1.0), loc="upper left", fontsize=8)
    return _finish(fig, path)
