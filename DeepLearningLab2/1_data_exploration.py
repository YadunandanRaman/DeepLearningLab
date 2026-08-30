import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.datasets import fashion_mnist

from plot_style import setup_fonts, save_fig

setup_fonts()

CLASS_NAMES = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
               "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

print("Training images shape:", X_train.shape)
print("Training labels shape:", y_train.shape)
print("Testing images shape:", X_test.shape)
print("Testing labels shape:", y_test.shape)
print("Number of classes:", len(np.unique(y_train)))
print("Pixel value range:", X_train.min(), "to", X_train.max())

out_dir = "outputs/plots"

# ten sample images
fig, axes = plt.subplots(2, 5, figsize=(12, 5))
rng = np.random.default_rng(42)
sample_idx = rng.choice(len(X_train), size=10, replace=False)
for ax, idx in zip(axes.ravel(), sample_idx):
    ax.imshow(X_train[idx], cmap="gray")
    ax.set_title(CLASS_NAMES[y_train[idx]], fontsize=10)
    ax.axis("off")
fig.tight_layout()
save_fig(fig, f"{out_dir}/sample_images")
plt.close(fig)

# class distribution
counts = np.bincount(y_train)
fig, ax = plt.subplots(figsize=(9, 5))
ax.bar(CLASS_NAMES, counts, color="steelblue", label="Training Samples")
ax.set_xlabel("Class")
ax.set_ylabel("Number of Images")
ax.legend()
plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
fig.tight_layout()
save_fig(fig, f"{out_dir}/class_distribution")
plt.close(fig)

print("\nclass counts:")
for name, count in zip(CLASS_NAMES, counts):
    print(f"  {name}: {count}")

print("\nsaved sample_images and class_distribution plots to", out_dir)
