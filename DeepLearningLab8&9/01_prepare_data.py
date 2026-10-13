"""
Step 1 (Sections 3 and 31): dataset download, 224 x 224 cache, fixed stratified 70/15/15 split, class counts,
and the figure with sample input images from all five classes.

Outputs
  cache/images_224.npy, labels.npy (local disk), outputs/saved_data/split_indices.json (reused by every script)
  outputs/results/01_split_counts.csv / .tex
  outputs/plots/01_sample_images.eps                                    (required visualization 1)
"""
import numpy as np

import config
import data_utils
import plot_style
import report_utils
import visualize as viz

config.make_dirs()
d = data_utils.load_split()
print({k: v.shape for k, v in d.items() if k.startswith("x_")})

header = ["Class", "Train", "Validation", "Test", "Total"]
rows = []
for name, tr, va, te in data_utils.class_counts():
    rows.append([name, tr, va, te, tr + va + te])
rows.append(["Total"] + [sum(r[i] for r in rows) for i in range(1, 5)])
report_utils.write_csv(config.result_path("01_split_counts.csv"), header, rows)
report_utils.write_latex_table(config.result_path("01_split_counts.tex"), header, rows,
                               "Number of images per class in each split (stratified 70/15/15, seed 42).",
                               "tab:split_counts")
for r in rows:
    print(r)

# sample images from all five classes (train split, fixed seed)
n_per_class = 4
rng = np.random.RandomState(config.SEED)
fig, axes = viz.plt.subplots(config.NUM_CLASSES, n_per_class, figsize=(2.2 * n_per_class, 2.3 * config.NUM_CLASSES))
for c in range(config.NUM_CLASSES):
    ids = np.where(d["y_train"] == c)[0]
    ids = rng.choice(ids, min(n_per_class, len(ids)), replace=False)
    for j in range(n_per_class):
        ax = axes[c, j]
        if j < len(ids):
            ax.imshow(d["x_train"][ids[j]])
        ax.axis("off")
        if j == 0:
            ax.set_title(config.CLASS_NAMES[c].capitalize(), loc="left", fontsize=11)
fig.tight_layout()
print("saved", plot_style.save_fig(fig, config.plot_path("01_sample_images")))
