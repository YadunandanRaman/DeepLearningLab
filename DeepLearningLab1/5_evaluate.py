import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from perceptron import Perceptron
from plot_style import setup_fonts, save_fig

setup_fonts()

X_test = np.load("outputs/saved_model/X_test.npy")
y_test = np.load("outputs/saved_model/y_test.npy")
w = np.load("outputs/saved_model/weights.npy")
b = np.load("outputs/saved_model/bias.npy")[0]

model = Perceptron(n_features=X_test.shape[1])
model.w = w
model.b = b

y_pred = model.predict(X_test)

tn = np.sum((y_test == 0) & (y_pred == 0))
fp = np.sum((y_test == 0) & (y_pred == 1))
fn = np.sum((y_test == 1) & (y_pred == 0))
tp = np.sum((y_test == 1) & (y_pred == 1))

accuracy = (tp + tn) / (tp + tn + fp + fn)
precision = tp / (tp + fp) if (tp + fp) else 0
recall = tp / (tp + fn) if (tp + fn) else 0
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0

print(f"accuracy:  {accuracy:.4f}")
print(f"precision: {precision:.4f}")
print(f"recall:    {recall:.4f}")
print(f"f1 score:  {f1:.4f}")

cm = np.array([[tn, fp], [fn, tp]])
print("\nconfusion matrix:")
print(cm)

fig, ax = plt.subplots(figsize=(5.5, 4.5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Greens",
    xticklabels=["Authentic", "Forged"],
    yticklabels=["Authentic", "Forged"],
    cbar_kws={"label": "Count"},
    ax=ax,
)
ax.set_xlabel("Predicted Label")
ax.set_ylabel("Actual Label")
fig.tight_layout()
save_fig(fig, "outputs/plots/confusion_matrix")
plt.close(fig)

print("saved confusion matrix plot (EPS, 600 DPI, Times New Roman)")
