import numpy as np
import matplotlib.pyplot as plt

from data_loader import load_datasets, CLASS_NAMES, NUM_CLASSES
from plot_style import setup_fonts, save_fig

setup_fonts()

train_ds, val_ds, test_ds = load_datasets()

train_count = int(train_ds.cardinality().numpy())
val_count = int(val_ds.cardinality().numpy())
test_count = int(test_ds.cardinality().numpy())

print("Training images:", train_count)
print("Validation images:", val_count)
print("Testing images (untouched until Section 12):", test_count)
print("Number of classes:", NUM_CLASSES)

out_dir = "outputs/plots"

# ten sample images from the training set
fig, axes = plt.subplots(2, 5, figsize=(14, 6))
sample_batch = train_ds.shuffle(200, seed=42).take(10)
for ax, (image, label) in zip(axes.ravel(), sample_batch):
    ax.imshow(image.numpy().astype("uint8"))
    ax.set_title(CLASS_NAMES[int(label)], fontsize=9)
    ax.axis("off")
fig.tight_layout()
save_fig(fig, f"{out_dir}/sample_images")
plt.close(fig)

# class distribution across the combined training and validation pool
counts = np.zeros(NUM_CLASSES, dtype=int)
for _, label in train_ds.concatenate(val_ds):
    counts[int(label)] += 1

fig, ax = plt.subplots(figsize=(14, 6))
ax.bar(CLASS_NAMES, counts, color="steelblue", label="Training and Validation Images")
ax.set_xlabel("Breed")
ax.set_ylabel("Number of Images")
ax.legend()
plt.setp(ax.get_xticklabels(), rotation=90, ha="center", fontsize=7)
fig.tight_layout()
save_fig(fig, f"{out_dir}/class_distribution")
plt.close(fig)

print("\nimages per class: min =", counts.min(), "max =", counts.max(), "mean =", round(counts.mean(), 1))
print("saved sample_images and class_distribution plots to", out_dir)
