import pandas as pd
import matplotlib.pyplot as plt

from data_loader import load_datasets
from model_builder import build_model, compile_model
from plot_style import setup_fonts, save_fig

setup_fonts()

# Every configuration here is identical except the new classifier
# head's regularization, none, L2, Dropout, or Batch Normalization, so
# any difference in the generalization gap (the distance between
# training and validation curves) can be attributed to regularization
# alone. The base stays frozen throughout, as in every study except
# Section 10's fine tuning.
REGULARIZATIONS = [None, "l2", "dropout", "batchnorm"]
LABELS = {None: "No Regularization", "l2": "L2", "dropout": "Dropout", "batchnorm": "Batch Norm"}
EPOCHS = 10
BATCH_SIZE = 32
LEARNING_RATE = 0.001

train_ds, val_ds, _ = load_datasets()
train_ds_batched = train_ds.batch(BATCH_SIZE).prefetch(1)
val_ds_batched = val_ds.batch(BATCH_SIZE).prefetch(1)

histories = {}
for regularization in REGULARIZATIONS:
    label = LABELS[regularization]
    print(f"\ntraining with {label}")
    model, base_model = build_model(initializer="xavier", regularization=regularization, freeze_base=True)
    compile_model(model, optimizer_name="adam", learning_rate=LEARNING_RATE)

    history = model.fit(
        train_ds_batched,
        validation_data=val_ds_batched,
        epochs=EPOCHS,
        verbose=2,
    )
    histories[regularization] = history.history

    final_gap = history.history["accuracy"][-1] - history.history["val_accuracy"][-1]
    print(f"{label}: final training accuracy={history.history['accuracy'][-1]:.4f}, "
          f"final validation accuracy={history.history['val_accuracy'][-1]:.4f}, "
          f"generalization gap={final_gap:.4f}")

rows = []
for regularization, hist in histories.items():
    for epoch in range(EPOCHS):
        rows.append({
            "regularization": LABELS[regularization],
            "epoch": epoch + 1,
            "training_accuracy": hist["accuracy"][epoch],
            "validation_accuracy": hist["val_accuracy"][epoch],
            "training_loss": hist["loss"][epoch],
            "validation_loss": hist["val_loss"][epoch],
        })
pd.DataFrame(rows).to_csv("outputs/results/regularization_history.csv", index=False)

out_dir = "outputs/plots"
epochs_range = range(1, EPOCHS + 1)

# Plot 3: Training and Validation Accuracy vs Epoch, one subplot per
# regularization variant so each variant's own generalization gap is
# directly visible
fig, axes = plt.subplots(2, 2, figsize=(11, 8))
for ax, regularization in zip(axes.ravel(), REGULARIZATIONS):
    hist = histories[regularization]
    ax.plot(epochs_range, hist["accuracy"], marker="o", color="green", label="Training Accuracy")
    ax.plot(epochs_range, hist["val_accuracy"], marker="o", color="orange", label="Validation Accuracy")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_title(LABELS[regularization], fontsize=11)
    ax.legend(fontsize=8)
fig.tight_layout()
save_fig(fig, f"{out_dir}/regularization_accuracy")
plt.close(fig)

# Plot 4: Training and Validation Loss vs Epoch, same layout
fig, axes = plt.subplots(2, 2, figsize=(11, 8))
for ax, regularization in zip(axes.ravel(), REGULARIZATIONS):
    hist = histories[regularization]
    ax.plot(epochs_range, hist["loss"], marker="o", color="red", label="Training Loss")
    ax.plot(epochs_range, hist["val_loss"], marker="o", color="purple", label="Validation Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(LABELS[regularization], fontsize=11)
    ax.legend(fontsize=8)
fig.tight_layout()
save_fig(fig, f"{out_dir}/regularization_loss")
plt.close(fig)

print("\nsaved regularization_accuracy and regularization_loss plots to", out_dir)
