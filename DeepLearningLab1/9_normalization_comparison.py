import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from perceptron import Perceptron
from plot_style import setup_fonts, save_fig

setup_fonts()

df = pd.read_csv("data/banknote_authentication.csv")
feature_cols = ["variance", "skewness", "curtosis", "entropy"]
X = df[feature_cols].values.astype(float)
y = df["class"].values.astype(int)

# same 80/20 split as 3_preprocessing.py, so this is a fair comparison
rng = np.random.default_rng(42)
idx = rng.permutation(len(X))
n_test = int(len(X) * 0.2)
test_idx, train_idx = idx[:n_test], idx[n_test:]

X_train_raw, X_test_raw = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]

# normalized version (same as 3_preprocessing.py)
mean = X_train_raw.mean(axis=0)
std = X_train_raw.std(axis=0)
X_train_norm = (X_train_raw - mean) / std

# train on raw, unscaled features
model_raw = Perceptron(n_features=4, lr=0.01, epochs=50, seed=42)
model_raw.fit(X_train_raw, y_train, verbose=False)

# train on normalized features
model_norm = Perceptron(n_features=4, lr=0.01, epochs=50, seed=42)
model_norm.fit(X_train_norm, y_train, verbose=False)

print("without normalization: final training errors =", model_raw.errors_per_epoch[-1])
print("with normalization:    final training errors =", model_norm.errors_per_epoch[-1])

fig, ax = plt.subplots(figsize=(7.5, 5.5))
ax.plot(range(1, len(model_raw.errors_per_epoch) + 1), model_raw.errors_per_epoch,
        marker="o", color="red", label="Without normalization")
ax.plot(range(1, len(model_norm.errors_per_epoch) + 1), model_norm.errors_per_epoch,
        marker="o", color="green", label="With normalization")
ax.set_xlabel("Epoch")
ax.set_ylabel("Misclassified Samples")
ax.legend()
fig.tight_layout()
save_fig(fig, "outputs/plots/normalization_comparison")
plt.close(fig)

print("saved plot (EPS, 600 DPI, Times New Roman)")
