import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from scikeras.wrappers import KerasClassifier
from tensorflow.keras.models import load_model

from model_builder import build_search_model
from plot_style import setup_fonts, save_fig

setup_fonts()

X_train = np.load("outputs/saved_model/X_train.npy")
y_train_labels = np.load("outputs/saved_model/y_train_labels.npy")
X_test = np.load("outputs/saved_model/X_test.npy")
y_test_labels = np.load("outputs/saved_model/y_test_labels.npy")

# -----------------------------------------------------------------------
# Search space (Section 7 of the lab manual)
# -----------------------------------------------------------------------
param_distributions = {
    "hidden_layers": [1, 2, 3],
    "hidden_neurons": [32, 64, 128, 256],
    "learning_rate": [0.1, 0.01, 0.001],
    "batch_size": [16, 32, 64, 128],
    "epochs": [10, 20, 30],
    "optimizer_name": ["sgd", "adam", "rmsprop"],
    "activation": ["relu", "tanh", "sigmoid"],
    "dropout_rate": [0.0, 0.2, 0.5],
}

# This search space has up to 3 x 4 x 3 x 4 x 3 x 3 x 3 x 3 = 3,888 possible
# combinations. Trying all of them with 5-fold cross validation on the full
# 60,000 image training set is not realistic on typical classroom hardware,
# which is exactly why the lab manual recommends RandomizedSearchCV over
# GridSearchCV. RandomizedSearchCV only samples N_ITER combinations below,
# and the search itself runs on a random subsample of the training data.
# The final chosen configuration is retrained on the FULL training set
# afterward, so this shortcut only affects how the search is scored, not
# the model that is actually delivered.
N_ITER = 10
CV_FOLDS = 5
SEARCH_SAMPLE_SIZE = 6000

rng = np.random.default_rng(42)
search_idx = rng.choice(len(X_train), size=SEARCH_SAMPLE_SIZE, replace=False)
X_search = X_train[search_idx]
y_search = y_train_labels[search_idx]

base_clf = KerasClassifier(
    model=build_search_model,
    hidden_layers=2,
    hidden_neurons=128,
    activation="relu",
    dropout_rate=0.0,
    optimizer_name="adam",
    learning_rate=0.001,
    epochs=10,
    batch_size=32,
    verbose=0,
)

search = RandomizedSearchCV(
    estimator=base_clf,
    param_distributions=param_distributions,
    n_iter=N_ITER,
    cv=CV_FOLDS,
    random_state=42,
    n_jobs=1,
    verbose=2,
)

print(f"running randomized search: {N_ITER} candidates x {CV_FOLDS} folds = {N_ITER * CV_FOLDS} model fits")
print(f"search subsample size: {SEARCH_SAMPLE_SIZE} of {len(X_train)} training images\n")

search.fit(X_search, y_search)

print("\nbest hyperparameters found:")
for key, value in search.best_params_.items():
    print(f"  {key}: {value}")
print(f"best cross validation accuracy: {search.best_score_:.4f}")

# -----------------------------------------------------------------------
# Inspect how much each hyperparameter affected the search, useful for
# answering "which hyperparameter had the greatest impact" in the report.
# -----------------------------------------------------------------------
results_df = pd.DataFrame(search.cv_results_)

print("\nmean cv accuracy by hyperparameter value (within the sampled candidates):")
for param in param_distributions.keys():
    col = f"param_{param}"
    if col in results_df.columns:
        grouped = results_df.groupby(col)["mean_test_score"].mean().sort_values(ascending=False)
        print(f"\n{param}:")
        print(grouped.to_string())

results_df.to_csv("outputs/results/hyperparameter_search_cv_results.csv", index=False)

# -----------------------------------------------------------------------
# Retrain the optimized model on the FULL training set
# -----------------------------------------------------------------------
best_params = search.best_params_

optimized_model = build_search_model(
    hidden_layers=best_params["hidden_layers"],
    hidden_neurons=best_params["hidden_neurons"],
    activation=best_params["activation"],
    dropout_rate=best_params["dropout_rate"],
    optimizer_name=best_params["optimizer_name"],
    learning_rate=best_params["learning_rate"],
)

start = time.time()
optimized_model.fit(
    X_train, y_train_labels,
    epochs=best_params["epochs"],
    batch_size=best_params["batch_size"],
    verbose=2,
)
optimized_training_time = time.time() - start

optimized_model.save("outputs/saved_model/optimized_model.keras")

# -----------------------------------------------------------------------
# Evaluate the optimized model on the test set
# -----------------------------------------------------------------------
y_pred_probs = optimized_model.predict(X_test, verbose=0)
y_pred_optimized = np.argmax(y_pred_probs, axis=1)

optimized_accuracy = accuracy_score(y_test_labels, y_pred_optimized)
optimized_precision = precision_score(y_test_labels, y_pred_optimized, average="macro")
optimized_recall = recall_score(y_test_labels, y_pred_optimized, average="macro")
optimized_f1 = f1_score(y_test_labels, y_pred_optimized, average="macro")

print(f"\noptimized model test accuracy:  {optimized_accuracy:.4f}")
print(f"optimized model test precision: {optimized_precision:.4f}")
print(f"optimized model test recall:    {optimized_recall:.4f}")
print(f"optimized model test f1 score:  {optimized_f1:.4f}")
print(f"optimized model training time:  {optimized_training_time:.2f} seconds")

best_params_out = dict(best_params)
best_params_out["cv_accuracy"] = search.best_score_
best_params_out["test_accuracy"] = optimized_accuracy
best_params_out["training_time_seconds"] = optimized_training_time
pd.DataFrame([best_params_out]).to_csv("outputs/results/best_hyperparameters.csv", index=False)

# -----------------------------------------------------------------------
# Compare with the baseline model trained in 4_train_model.py
# -----------------------------------------------------------------------
baseline_model = load_model("outputs/saved_model/baseline_model.keras")
y_pred_baseline_probs = baseline_model.predict(X_test, verbose=0)
y_pred_baseline = np.argmax(y_pred_baseline_probs, axis=1)

baseline_accuracy = accuracy_score(y_test_labels, y_pred_baseline)
baseline_precision = precision_score(y_test_labels, y_pred_baseline, average="macro")
baseline_recall = recall_score(y_test_labels, y_pred_baseline, average="macro")
baseline_f1 = f1_score(y_test_labels, y_pred_baseline, average="macro")

try:
    baseline_time_df = pd.read_csv("outputs/results/baseline_training_time.csv")
    baseline_training_time = float(baseline_time_df["training_time_seconds"].iloc[0])
except FileNotFoundError:
    baseline_training_time = float("nan")
    print("\nbaseline_training_time.csv not found, run 4_train_model.py first for a complete comparison")

comparison = pd.DataFrame({
    "Metric": ["Accuracy", "Precision", "Recall", "F1-score", "Training Time (s)"],
    "Baseline": [baseline_accuracy, baseline_precision, baseline_recall, baseline_f1, baseline_training_time],
    "Optimized": [optimized_accuracy, optimized_precision, optimized_recall, optimized_f1, optimized_training_time],
})
comparison.to_csv("outputs/results/performance_comparison.csv", index=False)
print("\nperformance comparison:")
print(comparison.to_string(index=False))

# -----------------------------------------------------------------------
# Plot 8: Hyperparameter Search Results
# -----------------------------------------------------------------------
sorted_results = results_df.sort_values("mean_test_score").reset_index(drop=True)
colors = ["steelblue"] * len(sorted_results)
best_row = sorted_results["mean_test_score"].idxmax()
colors[best_row] = "crimson"

fig, ax = plt.subplots(figsize=(9, 5.5))
ax.bar(range(len(sorted_results)), sorted_results["mean_test_score"], color=colors)
ax.set_xlabel("Candidate (sorted by score)")
ax.set_ylabel("Mean Cross Validation Accuracy")
handles = [
    mpatches.Patch(color="steelblue", label="Other Candidates"),
    mpatches.Patch(color="crimson", label="Best Candidate"),
]
ax.legend(handles=handles)
fig.tight_layout()
save_fig(fig, "outputs/plots/hyperparameter_search_results")
plt.close(fig)

# -----------------------------------------------------------------------
# Plot 9: Best Model Accuracy Comparison
# -----------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6, 5.5))
ax.bar(["Baseline", "Optimized"], [baseline_accuracy, optimized_accuracy], color=["steelblue", "crimson"])
ax.set_xlabel("Model")
ax.set_ylabel("Test Accuracy")
handles = [
    mpatches.Patch(color="steelblue", label="Baseline"),
    mpatches.Patch(color="crimson", label="Optimized"),
]
ax.legend(handles=handles)
fig.tight_layout()
save_fig(fig, "outputs/plots/best_model_comparison")
plt.close(fig)

print("\nsaved hyperparameter search plots and result tables to outputs/plots and outputs/results")
