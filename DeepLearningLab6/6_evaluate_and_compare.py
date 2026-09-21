import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import load_model
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix,
)

from data_loader import load_har_data, CLASS_NAMES, NUM_CLASSES
from plot_style import setup_fonts, save_fig

setup_fonts()

MODEL_TYPES = ["rnn", "lstm", "gru"]
LABELS = {"rnn": "RNN", "lstm": "LSTM", "gru": "GRU"}

(X_train, y_train), (X_val, y_val), (X_test, y_test) = load_har_data()

out_dir = "outputs/plots"
metrics_rows = []

for model_type in MODEL_TYPES:
    label = LABELS[model_type]
    print(f"\n=== evaluating {label} ===")
    model = load_model(f"outputs/saved_model/{model_type}_model.keras")
    summary = pd.read_csv(f"outputs/results/{model_type}_summary.csv").iloc[0]

    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, labels=list(range(NUM_CLASSES)), average="macro", zero_division=0)
    recall = recall_score(y_test, y_pred, labels=list(range(NUM_CLASSES)), average="macro", zero_division=0)
    f1 = f1_score(y_test, y_pred, labels=list(range(NUM_CLASSES)), average="macro", zero_division=0)

    print(f"{label}: accuracy={accuracy:.4f}, macro precision={precision:.4f}, "
          f"macro recall={recall:.4f}, macro F1={f1:.4f}")

    metrics_rows.append({
        "model": label,
        "accuracy": accuracy,
        "macro_precision": precision,
        "macro_recall": recall,
        "macro_f1": f1,
        "parameters": int(summary["parameters"]),
        "training_time_seconds": float(summary["training_time_seconds"]),
    })

    # Plot 4: one confusion matrix per model
    cm = confusion_matrix(y_test, y_pred, labels=list(range(NUM_CLASSES)))
    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
        cbar_kws={"label": "Count"}, ax=ax,
    )
    ax.set_xlabel("Predicted Activity")
    ax.set_ylabel("Actual Activity")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    fig.tight_layout()
    save_fig(fig, f"{out_dir}/{model_type}_confusion_matrix")
    plt.close(fig)

comparison_df = pd.DataFrame(metrics_rows)
comparison_df.to_csv("outputs/results/model_comparison.csv", index=False)
print("\nmodel comparison:")
print(comparison_df.to_string(index=False))

# Plot 5: bar chart comparing test accuracy and macro F1 across the 3 models
fig, ax = plt.subplots(figsize=(8, 5.5))
x = np.arange(len(MODEL_TYPES))
width = 0.35
ax.bar(x - width / 2, comparison_df["accuracy"], width, label="Accuracy", color="steelblue")
ax.bar(x + width / 2, comparison_df["macro_f1"], width, label="Macro F1", color="crimson")
ax.set_xticks(x)
ax.set_xticklabels(comparison_df["model"])
ax.set_xlabel("Model")
ax.set_ylabel("Score")
ax.legend()
fig.tight_layout()
save_fig(fig, f"{out_dir}/model_performance_comparison")
plt.close(fig)

print(f"\nsaved confusion matrices for all 3 models and model_performance_comparison plot to {out_dir}")
