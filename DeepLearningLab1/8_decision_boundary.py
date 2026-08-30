import numpy as np
import matplotlib.pyplot as plt

from perceptron import Perceptron
from plot_style import setup_fonts, save_fig

setup_fonts()

# train on just two features (variance, skewness) so we can draw a 2D boundary
X_train = np.load("outputs/saved_model/X_train.npy")
y_train = np.load("outputs/saved_model/y_train.npy")

# columns: 0=variance, 1=skewness, 2=curtosis, 3=entropy
X2 = X_train[:, [0, 1]]

model = Perceptron(n_features=2, lr=0.01, epochs=50, seed=42)
model.fit(X2, y_train, verbose=False)

w = model.w
b = model.b

fig, ax = plt.subplots(figsize=(6.5, 5.5))
for cls, color, label in [(0, "green", "Authentic"), (1, "red", "Forged")]:
    mask = y_train == cls
    ax.scatter(X2[mask, 0], X2[mask, 1], c=color, label=label, alpha=0.6, edgecolor="k", s=25)

# boundary: w0*x + w1*y + b = 0  ->  y = -(w0*x + b) / w1
x_vals = np.linspace(X2[:, 0].min() - 1, X2[:, 0].max() + 1, 200)
if abs(w[1]) > 1e-8:
    y_vals = -(w[0] * x_vals + b) / w[1]
    ax.plot(x_vals, y_vals, "k--", linewidth=2, label="Decision Boundary")

ax.set_xlabel("Variance")
ax.set_ylabel("Skewness")
ax.legend()
fig.tight_layout()
save_fig(fig, "outputs/plots/decision_boundary")
plt.close(fig)

print("saved decision boundary plot (EPS, 600 DPI, Times New Roman)")
