import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.models import load_model, Model

from data_loader import load_data
from plot_style import setup_fonts, save_fig

setup_fonts()

CLASS_NAMES = ["airplane", "automobile", "bird", "cat", "deer",
               "dog", "frog", "horse", "ship", "truck"]

MAPS_SHOWN = 8  # at least 8, as Task 4 asks for

(X_train, y_train), (X_test, y_test) = load_data()
y_test_labels = y_test.flatten()
X_test = X_test.astype("float32") / 255.0

model = load_model("outputs/saved_model/cnn_model.keras")

sample_idx = 0
sample_image = X_test[sample_idx:sample_idx + 1]
sample_label = CLASS_NAMES[y_test_labels[sample_idx]]

feature_map_model = Model(inputs=model.input, outputs=model.get_layer("conv1").output)
feature_maps = feature_map_model.predict(sample_image, verbose=0)[0]
num_filters = feature_maps.shape[-1]

fig, axes = plt.subplots(2, 4, figsize=(14, 7))
for i, ax in enumerate(axes.ravel()):
    ax.imshow(feature_maps[:, :, i], cmap="viridis")
    ax.set_title(f"Filter {i + 1}", fontsize=10)
    ax.axis("off")
fig.suptitle(f"Feature Maps: conv1 ({num_filters} filters total, {MAPS_SHOWN} shown), input: {sample_label}",
             fontsize=12, y=1.02)
fig.tight_layout()
save_fig(fig, "outputs/plots/feature_maps_conv1")
plt.close(fig)

print(f"conv1 output shape: {feature_maps.shape}")
print(f"showed {MAPS_SHOWN} of {num_filters} filters for sample image ({sample_label})")
print("saved feature_maps_conv1 plot to outputs/plots")
