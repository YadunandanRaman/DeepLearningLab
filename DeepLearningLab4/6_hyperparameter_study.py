import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from tensorflow.keras.utils import to_categorical

from data_loader import load_data
from model_builder import build_transfer_model, compile_model, unfreeze_last_block
from plot_style import setup_fonts, save_fig

setup_fonts()

# Training eight transfer learning models on the full 50,000 image
# training set, some of them with part of the pretrained base unfrozen,
# would take far longer than this comparison needs. Every configuration
# below is instead trained on the same random subsample of the training
# set, changing only the hyperparameter being studied away from the
# Task 3 baseline (learning rate 0.001, batch size 32, 10 epochs, Adam,
# 128 dense units, frozen base). This is enough to see the direction and
# rough size of each hyperparameter's effect, not to fully optimize any
# one of them.
ARCHITECTURE = "mobilenetv2"
SUBSAMPLE_SIZE = 5000
DEFAULT_EPOCHS = 10

(X_train, y_train), (X_test, y_test) = load_data()
y_train = y_train.flatten()
X_train = X_train.astype("float32")
y_train_onehot = to_categorical(y_train, num_classes=10)

rng = np.random.default_rng(42)
subset_idx = rng.choice(len(X_train), size=SUBSAMPLE_SIZE, replace=False)
X_sub = X_train[subset_idx]
y_sub = y_train_onehot[subset_idx]

# each entry changes exactly one hyperparameter away from the baseline
configs = {
    "Baseline": dict(learning_rate=0.001, batch_size=32, epochs=DEFAULT_EPOCHS,
                      optimizer="adam", dense_units=128, frozen="all"),
    "Learning Rate 0.0001": dict(learning_rate=0.0001, batch_size=32, epochs=DEFAULT_EPOCHS,
                                  optimizer="adam", dense_units=128, frozen="all"),
    "Batch Size 16": dict(learning_rate=0.001, batch_size=16, epochs=DEFAULT_EPOCHS,
                           optimizer="adam", dense_units=128, frozen="all"),
    "Batch Size 64": dict(learning_rate=0.001, batch_size=64, epochs=DEFAULT_EPOCHS,
                           optimizer="adam", dense_units=128, frozen="all"),
    "20 Epochs": dict(learning_rate=0.001, batch_size=32, epochs=20,
                       optimizer="adam", dense_units=128, frozen="all"),
    "SGD Optimizer": dict(learning_rate=0.001, batch_size=32, epochs=DEFAULT_EPOCHS,
                           optimizer="sgd", dense_units=128, frozen="all"),
    "Dense Units 256": dict(learning_rate=0.001, batch_size=32, epochs=DEFAULT_EPOCHS,
                             optimizer="adam", dense_units=256, frozen="all"),
    "Frozen Partial": dict(learning_rate=0.001, batch_size=32, epochs=DEFAULT_EPOCHS,
                            optimizer="adam", dense_units=128, frozen="partial"),
}

results = []
for name, params in configs.items():
    print(f"\ntraining configuration: {name}")
    model, base_model = build_transfer_model(
        architecture=ARCHITECTURE, dense_units=params["dense_units"], freeze_base=True
    )
    if params["frozen"] == "partial":
        unfreeze_last_block(base_model, architecture=ARCHITECTURE)

    compile_model(model, optimizer_name=params["optimizer"], learning_rate=params["learning_rate"])

    start = time.time()
    history = model.fit(
        X_sub, y_sub,
        validation_split=0.1,
        epochs=params["epochs"],
        batch_size=params["batch_size"],
        verbose=2,
    )
    elapsed = time.time() - start

    final_val_acc = history.history["val_accuracy"][-1]
    print(f"{name}: final validation accuracy={final_val_acc:.4f}, training time={elapsed:.2f}s")

    results.append({
        "configuration": name,
        "final_val_accuracy": final_val_acc,
        "training_time_seconds": elapsed,
    })

results_df = pd.DataFrame(results)
results_df.to_csv("outputs/results/hyperparameter_study.csv", index=False)
print("\nhyperparameter study results:")
print(results_df.to_string(index=False))

fig, ax = plt.subplots(figsize=(11, 6))
colors = ["steelblue"] * len(results_df)
best_idx = results_df["final_val_accuracy"].idxmax()
colors[best_idx] = "crimson"

ax.bar(range(len(results_df)), results_df["final_val_accuracy"], color=colors)
ax.set_xticks(range(len(results_df)))
ax.set_xticklabels(results_df["configuration"], rotation=30, ha="right")
ax.set_xlabel("Configuration")
ax.set_ylabel("Final Validation Accuracy")
handles = [
    mpatches.Patch(color="steelblue", label="Other Configurations"),
    mpatches.Patch(color="crimson", label="Best Configuration"),
]
ax.legend(handles=handles)
fig.tight_layout()
save_fig(fig, "outputs/plots/hyperparameter_study")
plt.close(fig)

print("\nsaved hyperparameter study plot and results table")
