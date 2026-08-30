import numpy as np
import matplotlib.pyplot as plt

from perceptron import Perceptron
from plot_style import setup_fonts, save_fig

setup_fonts()

X_train = np.load("outputs/saved_model/X_train.npy")
y_train = np.load("outputs/saved_model/y_train.npy")

model = Perceptron(n_features=X_train.shape[1], lr=0.01, epochs=50, seed=42)
model.fit(X_train, y_train)

# save weights/bias so evaluate.py can use them without retraining
np.save("outputs/saved_model/weights.npy", model.w)
np.save("outputs/saved_model/bias.npy", np.array([model.b]))

out_dir = "outputs/plots"

# training error vs epoch
fig, ax = plt.subplots(figsize=(7, 5))
epochs = range(1, len(model.errors_per_epoch) + 1)
ax.plot(epochs, model.errors_per_epoch, marker="o", color="red", label="Training Error")
ax.set_xlabel("Epoch")
ax.set_ylabel("Misclassified Samples")
ax.legend()
fig.tight_layout()
save_fig(fig, f"{out_dir}/training_error")
plt.close(fig)

# weight evolution
weights_arr = np.array(model.weights_per_epoch)
feature_names = ["variance", "skewness", "curtosis", "entropy"]
fig, ax = plt.subplots(figsize=(7, 5))
for i, name in enumerate(feature_names):
    ax.plot(epochs, weights_arr[:, i], marker=".", label=f"w_{name}")
ax.set_xlabel("Epoch")
ax.set_ylabel("Weight Value")
ax.legend()
fig.tight_layout()
save_fig(fig, f"{out_dir}/weight_evolution")
plt.close(fig)

# bias evolution
fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(epochs, model.bias_per_epoch, marker="o", color="green", label="Bias")
ax.set_xlabel("Epoch")
ax.set_ylabel("Bias Value")
ax.legend()
fig.tight_layout()
save_fig(fig, f"{out_dir}/bias_evolution")
plt.close(fig)

print("final weights:", model.w)
print("final bias:", model.b)
print("saved model + plots (EPS, 600 DPI, Times New Roman)")
