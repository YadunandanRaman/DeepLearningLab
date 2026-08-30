import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold

from data_loader import load_datasets, as_numpy
from model_builder import build_model, compile_model
from plot_style import setup_fonts, save_fig

setup_fonts()

# The four configurations below are reasonable, representative
# defaults, one plain baseline, one combining a generally strong
# initializer with dropout, one testing L2 instead, and one testing a
# smaller learning rate, they are not claimed to be the four best
# configurations for this dataset. Section 11 asks specifically for
# "3 to 4 promising configurations", selected using the results of
# Sections 5 through 10, so replace C1 through C4 below with whichever
# configurations actually performed best once those scripts have been
# run, before trusting this script's conclusions.
CONFIGS = {
    "C1: Baseline": dict(initializer="xavier", regularization=None, dropout_rate=0.0,
                          optimizer_name="adam", learning_rate=0.001),
    "C2: He + Dropout": dict(initializer="he", regularization="dropout", dropout_rate=0.25,
                              optimizer_name="adam", learning_rate=0.001),
    "C3: L2": dict(initializer="xavier", regularization="l2", dropout_rate=0.0,
                    optimizer_name="adam", learning_rate=0.001),
    "C4: Lower LR + Dropout": dict(initializer="xavier", regularization="dropout", dropout_rate=0.25,
                                    optimizer_name="adam", learning_rate=0.0001),
}

K_FOLDS = 5
EPOCHS = 5
BATCH_SIZE = 32

# The K fold split runs over the combined training and validation pool
# only. The independent test set is not touched here, or anywhere in
# this script, exactly as Section 11 requires: it is reserved entirely
# for the final evaluation in Section 12, after a configuration has
# already been chosen using these cross validation results.
train_ds, val_ds, _ = load_datasets()
print("materializing the training and validation pool into memory for K fold splitting")
X, y = as_numpy(train_ds.concatenate(val_ds))
print("pool shape:", X.shape, "labels shape:", y.shape)

kfold = KFold(n_splits=K_FOLDS, shuffle=True, random_state=42)

all_results = {}
for config_name, params in CONFIGS.items():
    print(f"\n=== {config_name} ===")
    fold_accuracies = []

    for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(X), start=1):
        print(f"fold {fold_idx} of {K_FOLDS}")
        X_train_fold, y_train_fold = X[train_idx], y[train_idx]
        X_val_fold, y_val_fold = X[val_idx], y[val_idx]

        model, base_model = build_model(
            initializer=params["initializer"],
            regularization=params["regularization"],
            dropout_rate=params["dropout_rate"],
            freeze_base=True,
        )
        compile_model(model, optimizer_name=params["optimizer_name"], learning_rate=params["learning_rate"])

        model.fit(
            X_train_fold.astype("float32"), y_train_fold,
            validation_data=(X_val_fold.astype("float32"), y_val_fold),
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            verbose=2,
        )
        _, fold_accuracy = model.evaluate(X_val_fold.astype("float32"), y_val_fold, verbose=0)
        fold_accuracies.append(fold_accuracy)
        print(f"fold {fold_idx} accuracy: {fold_accuracy:.4f}")

    mean_accuracy = float(np.mean(fold_accuracies))
    sd_accuracy = float(np.std(fold_accuracies))
    all_results[config_name] = {
        "fold_accuracies": fold_accuracies,
        "mean": mean_accuracy,
        "sd": sd_accuracy,
    }
    print(f"{config_name}: {mean_accuracy:.4f} +/- {sd_accuracy:.4f}")

rows = []
for config_name, result in all_results.items():
    row = {"configuration": config_name}
    for i, acc in enumerate(result["fold_accuracies"], start=1):
        row[f"F{i}"] = acc
    row["mean"] = result["mean"]
    row["sd"] = result["sd"]
    rows.append(row)
results_df = pd.DataFrame(rows)
results_df.to_csv("outputs/results/cross_validation.csv", index=False)
print("\ncross validation summary:")
print(results_df.to_string(index=False))

out_dir = "outputs/plots"

# Plot 13: 5-Fold Cross Validation Accuracy, with standard deviation error bars
fig, ax = plt.subplots(figsize=(9, 6))
names = list(all_results.keys())
means = [all_results[n]["mean"] for n in names]
sds = [all_results[n]["sd"] for n in names]
ax.bar(names, means, yerr=sds, capsize=6, color="steelblue", label="Mean CV Accuracy")
ax.set_xlabel("Configuration")
ax.set_ylabel("Mean Validation Accuracy")
ax.legend()
plt.setp(ax.get_xticklabels(), rotation=15, ha="right")
fig.tight_layout()
save_fig(fig, f"{out_dir}/cross_validation_accuracy")
plt.close(fig)

print("\nsaved cross_validation_accuracy plot to", out_dir)
