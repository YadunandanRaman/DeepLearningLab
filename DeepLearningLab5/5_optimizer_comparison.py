import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from data_loader import load_datasets
from model_builder import build_model, compile_model
from plot_style import setup_fonts, save_fig

setup_fonts()

# Every configuration here is identical except the optimizer, so any
# difference in convergence speed, stability, or final accuracy can be
# attributed to the optimizer alone. "Momentum" is SGD with a momentum
# term added (momentum=0.9), the standard way to get SGD with Momentum
# in Keras without a separate optimizer class.
OPTIMIZERS = ["sgd", "momentum", "rmsprop", "adam"]
LABELS = {"sgd": "SGD", "momentum": "Momentum", "rmsprop": "RMSProp", "adam": "Adam"}
EPOCHS = 10
BATCH_SIZE = 32
LEARNING_RATE = 0.001
CONVERGENCE_THRESHOLD = 0.02  # epoch to converge: loss within this of its own final value

train_ds, val_ds, _ = load_datasets()
train_ds_batched = train_ds.batch(BATCH_SIZE).prefetch(1)
val_ds_batched = val_ds.batch(BATCH_SIZE).prefetch(1)

histories = {}
summary_rows = []
for optimizer_name in OPTIMIZERS:
    label = LABELS[optimizer_name]
    print(f"\ntraining with {label}")
    model, base_model = build_model(initializer="xavier", regularization=None, freeze_base=True)
    compile_model(model, optimizer_name=optimizer_name, learning_rate=LEARNING_RATE)

    start = time.time()
    history = model.fit(
        train_ds_batched,
        validation_data=val_ds_batched,
        epochs=EPOCHS,
        verbose=2,
    )
    elapsed = time.time() - start
    histories[optimizer_name] = history.history

    final_loss = history.history["loss"][-1]
    best_val_acc = max(history.history["val_accuracy"])
    # epoch to converge: first epoch whose training loss is already
    # within CONVERGENCE_THRESHOLD of the final epoch's loss
    losses = np.array(history.history["loss"])
    converged_mask = np.abs(losses - final_loss) <= CONVERGENCE_THRESHOLD
    epoch_to_converge = int(np.argmax(converged_mask)) + 1 if converged_mask.any() else EPOCHS

    print(f"{label}: final loss={final_loss:.4f}, best val accuracy={best_val_acc:.4f}, "
          f"epoch to converge={epoch_to_converge}, time={elapsed:.2f}s")

    summary_rows.append({
        "optimizer": label,
        "final_loss": final_loss,
        "best_val_accuracy": best_val_acc,
        "epoch_to_converge": epoch_to_converge,
        "training_time_seconds": elapsed,
    })

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv("outputs/results/optimizer_comparison.csv", index=False)
print("\noptimizer comparison summary:")
print(summary_df.to_string(index=False))

out_dir = "outputs/plots"
epochs_range = range(1, EPOCHS + 1)
colors = {"sgd": "gray", "momentum": "orange", "rmsprop": "steelblue", "adam": "crimson"}

# Plot 6: Training Loss vs Epoch, one curve per optimizer
fig, ax = plt.subplots(figsize=(8, 5.5))
for optimizer_name in OPTIMIZERS:
    ax.plot(epochs_range, histories[optimizer_name]["loss"], marker="o",
            color=colors[optimizer_name], label=LABELS[optimizer_name])
ax.set_xlabel("Epoch")
ax.set_ylabel("Training Loss")
ax.legend()
fig.tight_layout()
save_fig(fig, f"{out_dir}/optimizer_training_loss")
plt.close(fig)

# Plot 7: Validation Accuracy vs Epoch, one curve per optimizer
fig, ax = plt.subplots(figsize=(8, 5.5))
for optimizer_name in OPTIMIZERS:
    ax.plot(epochs_range, histories[optimizer_name]["val_accuracy"], marker="o",
            color=colors[optimizer_name], label=LABELS[optimizer_name])
ax.set_xlabel("Epoch")
ax.set_ylabel("Validation Accuracy")
ax.legend()
fig.tight_layout()
save_fig(fig, f"{out_dir}/optimizer_validation_accuracy")
plt.close(fig)

print("\nsaved optimizer_training_loss and optimizer_validation_accuracy plots to", out_dir)
