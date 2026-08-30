import numpy as np
import matplotlib.pyplot as plt

from data_loader import load_data
from plot_style import setup_fonts, save_fig

setup_fonts()

CLASS_NAMES = ["airplane", "automobile", "bird", "cat", "deer",
               "dog", "frog", "horse", "ship", "truck"]

(X_train, y_train), (X_test, y_test) = load_data()

y_train = y_train.flatten()
y_test = y_test.flatten()

print("Training images shape:", X_train.shape)
print("Training labels shape:", y_train.shape)
print("Testing images shape:", X_test.shape)
print("Testing labels shape:", y_test.shape)
print("Number of classes:", len(np.unique(y_train)))
print("Pixel value range before normalization:", X_train.min(), "to", X_train.max())

# normalized purely to satisfy this task's own reporting requirement.
# The transfer learning models built from Task 2 onward take raw,
# unnormalized pixels as input instead, since each pretrained backbone
# (MobileNetV2, VGG16, ResNet50) applies its own specific preprocessing
# internally, and that preprocessing expects 0 to 255 input, not a
# generic 0 to 1 rescale. See model_builder.py for details.
X_train_normalized = X_train.astype("float32") / 255.0
print("Pixel value range after normalization:", X_train_normalized.min(), "to", X_train_normalized.max())

out_dir = "outputs/plots"

fig, axes = plt.subplots(2, 5, figsize=(12, 5.5))
rng = np.random.default_rng(42)
sample_idx = rng.choice(len(X_train), size=10, replace=False)
for ax, idx in zip(axes.ravel(), sample_idx):
    ax.imshow(X_train[idx])
    ax.set_title(CLASS_NAMES[y_train[idx]], fontsize=10)
    ax.axis("off")
fig.tight_layout()
save_fig(fig, f"{out_dir}/sample_images")
plt.close(fig)

print("\nsaved sample_images plot to", out_dir)
