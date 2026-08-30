import pandas as pd
import matplotlib.pyplot as plt

from data_loader import load_datasets
from model_builder import build_model, compile_model
from plot_style import setup_fonts, save_fig

setup_fonts()

# Following the manual's experimental rule directly: change one
# hyperparameter at a time while keeping everything else fixed at this
# baseline. Learning rate, batch size, and dropout rate are each swept
# independently below, never combined, so each plot isolates exactly
# one variable's effect on validation accuracy.
BASELINE_LR = 0.001
BASELINE_BATCH_SIZE = 32
BASELINE_DROPOUT = 0.0
EPOCHS = 8

train_ds, val_ds, _ = load_datasets()


def train_one(learning_rate, batch_size, dropout_rate):
    train_batched = train_ds.batch(batch_size).prefetch(1)
    val_batched = val_ds.batch(batch_size).prefetch(1)

    regularization = "dropout" if dropout_rate > 0 else None
    model, base_model = build_model(
        initializer="xavier", regularization=regularization,
        dropout_rate=dropout_rate, freeze_base=True,
    )
    compile_model(model, optimizer_name="adam", learning_rate=learning_rate)

    history = model.fit(train_batched, validation_data=val_batched, epochs=EPOCHS, verbose=2)
    return max(history.history["val_accuracy"])


results = []

# Plot 8: Learning Rate vs Validation Accuracy
print("\n--- learning rate sweep ---")
lr_results = {}
for lr in [0.001, 0.0001]:
    print(f"\nlearning rate = {lr}")
    best_val_acc = train_one(lr, BASELINE_BATCH_SIZE, BASELINE_DROPOUT)
    lr_results[lr] = best_val_acc
    print(f"learning rate {lr}: best validation accuracy = {best_val_acc:.4f}")
    results.append({"hyperparameter": "learning_rate", "value": lr, "best_val_accuracy": best_val_acc})

# Plot 9: Batch Size vs Validation Accuracy
print("\n--- batch size sweep ---")
batch_results = {}
for batch_size in [16, 32, 64]:
    print(f"\nbatch size = {batch_size}")
    best_val_acc = train_one(BASELINE_LR, batch_size, BASELINE_DROPOUT)
    batch_results[batch_size] = best_val_acc
    print(f"batch size {batch_size}: best validation accuracy = {best_val_acc:.4f}")
    results.append({"hyperparameter": "batch_size", "value": batch_size, "best_val_accuracy": best_val_acc})

# Plot 10: Dropout Rate vs Validation Accuracy
print("\n--- dropout rate sweep ---")
dropout_results = {}
for dropout_rate in [0.0, 0.25, 0.5]:
    print(f"\ndropout rate = {dropout_rate}")
    best_val_acc = train_one(BASELINE_LR, BASELINE_BATCH_SIZE, dropout_rate)
    dropout_results[dropout_rate] = best_val_acc
    print(f"dropout rate {dropout_rate}: best validation accuracy = {best_val_acc:.4f}")
    results.append({"hyperparameter": "dropout_rate", "value": dropout_rate, "best_val_accuracy": best_val_acc})

pd.DataFrame(results).to_csv("outputs/results/hyperparameter_tuning.csv", index=False)

out_dir = "outputs/plots"

fig, ax = plt.subplots(figsize=(6, 5.5))
ax.bar([str(k) for k in lr_results], list(lr_results.values()), color="steelblue", label="Validation Accuracy")
ax.set_xlabel("Learning Rate")
ax.set_ylabel("Validation Accuracy")
ax.legend()
fig.tight_layout()
save_fig(fig, f"{out_dir}/hp_learning_rate")
plt.close(fig)

fig, ax = plt.subplots(figsize=(6, 5.5))
ax.bar([str(k) for k in batch_results], list(batch_results.values()), color="steelblue", label="Validation Accuracy")
ax.set_xlabel("Batch Size")
ax.set_ylabel("Validation Accuracy")
ax.legend()
fig.tight_layout()
save_fig(fig, f"{out_dir}/hp_batch_size")
plt.close(fig)

fig, ax = plt.subplots(figsize=(6, 5.5))
ax.bar([str(k) for k in dropout_results], list(dropout_results.values()), color="steelblue", label="Validation Accuracy")
ax.set_xlabel("Dropout Rate")
ax.set_ylabel("Validation Accuracy")
ax.legend()
fig.tight_layout()
save_fig(fig, f"{out_dir}/hp_dropout_rate")
plt.close(fig)

print("\nsaved hp_learning_rate, hp_batch_size, and hp_dropout_rate plots to", out_dir)
