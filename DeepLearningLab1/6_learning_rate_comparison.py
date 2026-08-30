import numpy as np
import matplotlib.pyplot as plt

from perceptron import Perceptron
from plot_style import setup_fonts, save_fig

setup_fonts()

X_train = np.load("outputs/saved_model/X_train.npy")
y_train = np.load("outputs/saved_model/y_train.npy")
X_test = np.load("outputs/saved_model/X_test.npy")
y_test = np.load("outputs/saved_model/y_test.npy")

learning_rates = [0.001, 0.01, 0.1]
colors = ["green", "orange", "red"]

fig, ax = plt.subplots(figsize=(7.5, 5.5))

for lr, color in zip(learning_rates, colors):
    model = Perceptron(n_features=X_train.shape[1], lr=lr, epochs=50, seed=42)
    model.fit(X_train, y_train, verbose=False)

    y_pred = model.predict(X_test)
    acc = np.mean(y_pred == y_test)

    print(f"lr={lr}: epochs_run={len(model.errors_per_epoch)}, "
          f"final_train_errors={model.errors_per_epoch[-1]}, test_acc={acc:.4f}")

    epochs = range(1, len(model.errors_per_epoch) + 1)
    ax.plot(epochs, model.errors_per_epoch, marker="o", color=color, label=f"eta = {lr}")

ax.set_xlabel("Epoch")
ax.set_ylabel("Misclassified Samples")
ax.legend()
fig.tight_layout()
save_fig(fig, "outputs/plots/learning_rate_comparison")
plt.close(fig)

print("saved plot (EPS, 600 DPI, Times New Roman)")
