import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow.keras.models import load_model
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

from plot_style import setup_fonts, save_fig

setup_fonts()

CLASS_NAMES = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
               "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

X_test = np.load("outputs/saved_model/X_test.npy")
y_test_labels = np.load("outputs/saved_model/y_test_labels.npy")

model = load_model("outputs/saved_model/baseline_model.keras")

y_pred_probs = model.predict(X_test, verbose=0)
y_pred = np.argmax(y_pred_probs, axis=1)

# macro averaging is used since Fashion-MNIST has 10 balanced classes,
# so every class contributes equally to precision/recall/F1
accuracy = accuracy_score(y_test_labels, y_pred)
precision = precision_score(y_test_labels, y_pred, average="macro")
recall = recall_score(y_test_labels, y_pred, average="macro")
f1 = f1_score(y_test_labels, y_pred, average="macro")

print(f"accuracy:  {accuracy:.4f}")
print(f"precision: {precision:.4f}")
print(f"recall:    {recall:.4f}")
print(f"f1 score:  {f1:.4f}")

print("\nclassification report:")
report_text = classification_report(y_test_labels, y_pred, target_names=CLASS_NAMES)
print(report_text)

with open("outputs/results/baseline_classification_report.txt", "w") as f:
    f.write(report_text)

pd.DataFrame([{
    "accuracy": accuracy,
    "precision": precision,
    "recall": recall,
    "f1_score": f1,
}]).to_csv("outputs/results/baseline_metrics.csv", index=False)

cm = confusion_matrix(y_test_labels, y_pred)

fig, ax = plt.subplots(figsize=(8, 7))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=CLASS_NAMES,
    yticklabels=CLASS_NAMES,
    cbar_kws={"label": "Count"},
    ax=ax,
)
ax.set_xlabel("Predicted Label")
ax.set_ylabel("Actual Label")
plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
fig.tight_layout()
save_fig(fig, "outputs/plots/confusion_matrix")
plt.close(fig)

print("\nsaved confusion matrix plot, classification report, and metrics csv")
