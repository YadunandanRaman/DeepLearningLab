import pandas as pd
import matplotlib.pyplot as plt

from data_loader import load_datasets
from model_builder import build_model, compile_model
from plot_style import setup_fonts, save_fig

setup_fonts()

# Every configuration in this script is identical except the new
# classifier head's weight initialization, so any difference in the
# resulting curves can be attributed to initialization alone. The
# pretrained MobileNetV2 base stays frozen throughout, exactly as in
# every other study in this experiment unless a section is explicitly
# about fine tuning (Section 10).
INITIALIZERS = ["zero", "random", "xavier", "he"]
EPOCHS = 8
BATCH_SIZE = 32
LEARNING_RATE = 0.001

train_ds, val_ds, _ = load_datasets()
train_ds_batched = train_ds.batch(BATCH_SIZE).prefetch(1)
val_ds_batched = val_ds.batch(BATCH_SIZE).prefetch(1)

histories = {}
for initializer in INITIALIZERS:
    print(f"\ntraining with {initializer} initialization")
    model, base_model = build_model(initializer=initializer, regularization=None, freeze_base=True)
    compile_model(model, optimizer_name="adam", learning_rate=LEARNING_RATE)

    history = model.fit(
        train_ds_batched,
        validation_data=val_ds_batched,
        epochs=EPOCHS,
        verbose=2,
    )
    histories[initializer] = history.history
    final_val_acc = history.history["val_accuracy"][-1]
    print(f"{initializer}: final training loss={history.history['loss'][-1]:.4f}, "
          f"final validation accuracy={final_val_acc:.4f}")

# save the raw per epoch numbers for every initializer, not just the
# final epoch, so the report can quote exact figures rather than read
# them off a plot
rows = []
for initializer, hist in histories.items():
    for epoch in range(EPOCHS):
        rows.append({
            "initializer": initializer,
            "epoch": epoch + 1,
            "training_loss": hist["loss"][epoch],
            "validation_accuracy": hist["val_accuracy"][epoch],
        })
pd.DataFrame(rows).to_csv("outputs/results/weight_initialization_history.csv", index=False)

out_dir = "outputs/plots"
epochs_range = range(1, EPOCHS + 1)
colors = {"zero": "gray", "random": "orange", "xavier": "steelblue", "he": "crimson"}

# Plot 1: Training Loss vs Epoch, one curve per initializer
fig, ax = plt.subplots(figsize=(8, 5.5))
for initializer in INITIALIZERS:
    ax.plot(epochs_range, histories[initializer]["loss"], marker="o",
            color=colors[initializer], label=initializer.capitalize())
ax.set_xlabel("Epoch")
ax.set_ylabel("Training Loss")
ax.legend()
fig.tight_layout()
save_fig(fig, f"{out_dir}/init_training_loss")
plt.close(fig)

# Plot 2: Validation Accuracy vs Epoch, one curve per initializer
fig, ax = plt.subplots(figsize=(8, 5.5))
for initializer in INITIALIZERS:
    ax.plot(epochs_range, histories[initializer]["val_accuracy"], marker="o",
            color=colors[initializer], label=initializer.capitalize())
ax.set_xlabel("Epoch")
ax.set_ylabel("Validation Accuracy")
ax.legend()
fig.tight_layout()
save_fig(fig, f"{out_dir}/init_validation_accuracy")
plt.close(fig)

print("\nsaved init_training_loss and init_validation_accuracy plots to", out_dir)
