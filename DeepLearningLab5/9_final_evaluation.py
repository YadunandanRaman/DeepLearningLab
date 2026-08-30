import os
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
)

from data_loader import load_datasets, as_numpy, CLASS_NAMES, NUM_CLASSES
from model_builder import build_model, compile_model
from plot_style import setup_fonts, save_fig

setup_fonts()

# This must match whichever configuration actually won in
# 8_cross_validation.py, by name and by parameters, update both if that
# script's results point to a different configuration once it has
# actually been run.
BEST_CONFIG_NAME = "C2: He + Dropout"
BEST_CONFIG = dict(initializer="he", regularization="dropout", dropout_rate=0.25,
                    optimizer_name="adam", learning_rate=0.001)
EPOCHS = 15
BATCH_SIZE = 32

train_ds, val_ds, test_ds = load_datasets()

print("materializing the complete training pool (training plus validation) into memory")
X_train_full, y_train_full = as_numpy(train_ds.concatenate(val_ds))
X_train_full = X_train_full.astype("float32")
print("training pool shape:", X_train_full.shape)

print("materializing the untouched test set, used here for the first and only time")
X_test, y_test = as_numpy(test_ds)
X_test = X_test.astype("float32")
print("test set shape:", X_test.shape)

model, base_model = build_model(
    initializer=BEST_CONFIG["initializer"],
    regularization=BEST_CONFIG["regularization"],
    dropout_rate=BEST_CONFIG["dropout_rate"],
    freeze_base=True,
)
compile_model(model, optimizer_name=BEST_CONFIG["optimizer_name"], learning_rate=BEST_CONFIG["learning_rate"])
total_params = model.count_params()

start = time.time()
model.fit(
    X_train_full, y_train_full,
    validation_split=0.1,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    verbose=2,
)
training_time = time.time() - start

model.save("outputs/saved_model/final_model.keras")

y_pred_probs = model.predict(X_test, verbose=0)
y_pred = np.argmax(y_pred_probs, axis=1)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, labels=list(range(NUM_CLASSES)), average="macro", zero_division=0)
recall = recall_score(y_test, y_pred, labels=list(range(NUM_CLASSES)), average="macro", zero_division=0)
f1 = f1_score(y_test, y_pred, labels=list(range(NUM_CLASSES)), average="macro", zero_division=0)

print(f"\nfinal test accuracy:  {accuracy:.4f}")
print(f"final test precision: {precision:.4f}")
print(f"final test recall:    {recall:.4f}")
print(f"final test f1 score:  {f1:.4f}")
print(f"training time: {training_time:.2f}s")
print(f"total parameters: {total_params:,}")

report_text = classification_report(
    y_test, y_pred, labels=list(range(NUM_CLASSES)), target_names=CLASS_NAMES, zero_division=0
)
print("\nclassification report:")
print(report_text)
with open("outputs/results/final_classification_report.txt", "w") as f:
    f.write(report_text)

# Mean CV accuracy and standard deviation are pulled directly from
# 8_cross_validation.py's saved results for the same configuration
# name, rather than retyped by hand, if that script has not been run
# yet, these two fields are left blank instead of guessed.
mean_cv_accuracy, cv_sd = None, None
cv_results_path = "outputs/results/cross_validation.csv"
if os.path.isfile(cv_results_path):
    cv_df = pd.read_csv(cv_results_path)
    matching_row = cv_df[cv_df["configuration"] == BEST_CONFIG_NAME]
    if len(matching_row) == 1:
        mean_cv_accuracy = float(matching_row["mean"].iloc[0])
        cv_sd = float(matching_row["sd"].iloc[0])

pd.DataFrame([{
    "configuration": BEST_CONFIG_NAME,
    "mean_cv_accuracy": mean_cv_accuracy,
    "cv_standard_deviation": cv_sd,
    "test_accuracy": accuracy,
    "precision": precision,
    "recall": recall,
    "f1_score": f1,
    "training_time_seconds": training_time,
    "total_parameters": total_params,
}]).to_csv("outputs/results/final_metrics.csv", index=False)

out_dir = "outputs/plots"

cm = confusion_matrix(y_test, y_pred, labels=list(range(NUM_CLASSES)))
fig, ax = plt.subplots(figsize=(16, 14))
sns.heatmap(
    cm, annot=False, cmap="Blues",
    xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
    cbar_kws={"label": "Count"}, ax=ax,
)
ax.set_xlabel("Predicted Label")
ax.set_ylabel("Actual Label")
plt.setp(ax.get_xticklabels(), rotation=90, ha="center", fontsize=6)
plt.setp(ax.get_yticklabels(), fontsize=6)
fig.tight_layout()
save_fig(fig, f"{out_dir}/confusion_matrix")
plt.close(fig)

# misclassified images, optional Plot 15
misclassified_idx = np.where(y_pred != y_test)[0]
print(f"\n{len(misclassified_idx)} of {len(y_test)} test images misclassified")

rng = np.random.default_rng(42)
shown_idx = rng.choice(misclassified_idx, size=min(10, len(misclassified_idx)), replace=False)

fig, axes = plt.subplots(2, 5, figsize=(15, 7))
for ax, idx in zip(axes.ravel(), shown_idx):
    ax.imshow(X_test[idx].astype("uint8"))
    true_name = CLASS_NAMES[y_test[idx]]
    pred_name = CLASS_NAMES[y_pred[idx]]
    ax.set_title(f"true: {true_name}\npredicted: {pred_name}", fontsize=8)
    ax.axis("off")
fig.tight_layout()
save_fig(fig, f"{out_dir}/misclassified_images")
plt.close(fig)

print("\nsaved confusion_matrix and misclassified_images plots, classification report, and metrics csv")
