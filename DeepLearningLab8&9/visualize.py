"""Shared plotting helpers (images, heatmaps, overlays, feature maps). Figures are saved as EPS by plot_style.save_fig."""
import numpy as np
import tensorflow as tf
from matplotlib import cm

import config
import plot_style

plot_style.setup_fonts()
import matplotlib.pyplot as plt  # noqa: E402  (after font setup)

CLASS_TITLES = [c.capitalize() for c in config.CLASS_NAMES]
MODEL_COLORS = {}


def color_for(name):
    if name not in MODEL_COLORS:
        MODEL_COLORS[name] = f"C{len(MODEL_COLORS) % 10}"
    return MODEL_COLORS[name]


def normalize01(a):
    a = np.asarray(a, dtype=np.float32)
    return (a - a.min()) / (a.max() - a.min() + 1e-8)


def resize_map(m, size=config.IMG_SIZE):
    """Bilinear resize of a 2D map to size x size, scaled to [0, 1]."""
    r = tf.image.resize(np.asarray(m, dtype=np.float32)[None, ..., None], (size, size), method="bilinear")
    return normalize01(r[0, ..., 0].numpy())


def overlay(img_uint8, heat01, alpha=config.OVERLAY_ALPHA):
    """Blend a jet coloured heatmap onto the image in numpy (EPS has no transparency)."""
    rgb = cm.jet(np.clip(heat01, 0, 1))[..., :3] * 255.0
    return ((1 - alpha) * img_uint8.astype(np.float32) + alpha * rgb).astype(np.uint8)


def feature_image(fmap):
    """Mean over channels of an H x W x C feature map, scaled to [0, 1]."""
    return normalize01(np.asarray(fmap, dtype=np.float32).mean(-1))


def show(ax, img, title=None, cmap=None, fontsize=9):
    if cmap:
        ax.imshow(img, cmap=cmap, vmin=0, vmax=1)
    else:
        ax.imshow(img)
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=fontsize)


def pick_examples(y_true, preds, ref_models, n_correct=config.N_QUALITATIVE_CORRECT,
                  n_wrong=config.N_QUALITATIVE_WRONG, seed=config.SEED):
    """Correct examples: every model in ref_models is right (one per class where possible).
    Wrong examples: images misclassified by the most models, preferring different true classes."""
    rng = np.random.RandomState(seed)
    ok = np.stack([preds[m] == y_true for m in ref_models])                  # (models, N)
    all_ok = np.where(ok.all(0))[0]
    n_wrong_models = (~ok).sum(0)
    correct = []
    for c in rng.permutation(config.NUM_CLASSES):
        cand = [i for i in all_ok if y_true[i] == c]
        if cand and len(correct) < n_correct:
            correct.append(int(rng.choice(cand)))
    for i in rng.permutation(all_ok):
        if len(correct) >= n_correct:
            break
        if int(i) not in correct:
            correct.append(int(i))
    order = np.argsort(-n_wrong_models, kind="stable")
    wrong, used = [], set()
    for i in order:
        if n_wrong_models[i] == 0 or len(wrong) >= n_wrong:
            break
        if y_true[i] not in used or len(used) >= config.NUM_CLASSES:
            wrong.append(int(i)); used.add(y_true[i])
    for i in order:                                                         # fill up if the classes repeated
        if len(wrong) >= n_wrong or n_wrong_models[i] == 0:
            break
        if int(i) not in wrong:
            wrong.append(int(i))
    return correct, wrong
