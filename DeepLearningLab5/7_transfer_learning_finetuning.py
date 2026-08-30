import time

import pandas as pd
import matplotlib.pyplot as plt

from data_loader import load_datasets
from model_builder import build_model, compile_model, get_base_model, unfreeze_last_block
from plot_style import setup_fonts, save_fig

setup_fonts()

# Case A: feature extraction, the pretrained base stays frozen and only
# the new classifier head trains, exactly like every other study in
# this experiment. Case B: fine tuning, continues from Case A's already
# trained model, unfreezes the base's last block, and continues
# training with a smaller learning rate, one of the two the manual
# suggests (1e-4), since retraining pretrained weights with as large a
# learning rate as the one used to train a new head from scratch risks
# damaging those weights before they can adapt gently.
FEATURE_EXTRACTION_EPOCHS = 8
FINE_TUNE_EPOCHS = 6
BATCH_SIZE = 32
FEATURE_EXTRACTION_LR = 0.001
FINE_TUNE_LR = 0.0001

train_ds, val_ds, _ = load_datasets()
train_batched = train_ds.batch(BATCH_SIZE).prefetch(1)
val_batched = val_ds.batch(BATCH_SIZE).prefetch(1)

print("Case A: feature extraction, frozen base")
model, base_model = build_model(initializer="xavier", regularization=None, freeze_base=True)
compile_model(model, optimizer_name="adam", learning_rate=FEATURE_EXTRACTION_LR)

start = time.time()
history_a = model.fit(train_batched, validation_data=val_batched, epochs=FEATURE_EXTRACTION_EPOCHS, verbose=2)
time_a = time.time() - start
print(f"Case A finished: final validation accuracy={history_a.history['val_accuracy'][-1]:.4f}, "
      f"time={time_a:.2f}s")

model.save("outputs/saved_model/feature_extraction_model.keras")

print("\nCase B: fine tuning, unfreezing the last block")
base_model = get_base_model(model)
unfreeze_last_block(base_model)
compile_model(model, optimizer_name="adam", learning_rate=FINE_TUNE_LR)

start = time.time()
history_b = model.fit(train_batched, validation_data=val_batched, epochs=FINE_TUNE_EPOCHS, verbose=2)
time_b = time.time() - start
print(f"Case B finished: final validation accuracy={history_b.history['val_accuracy'][-1]:.4f}, "
      f"time={time_b:.2f}s")

model.save("outputs/saved_model/fine_tuned_model.keras")

# combine both phases onto one continuous epoch timeline, the standard
# way this kind of plot is shown, feature extraction first, fine tuning
# picking up exactly where it left off
combined_val_acc = history_a.history["val_accuracy"] + history_b.history["val_accuracy"]
combined_train_loss = history_a.history["loss"] + history_b.history["loss"]
combined_val_loss = history_a.history["val_loss"] + history_b.history["val_loss"]
switch_epoch = FEATURE_EXTRACTION_EPOCHS
total_epochs = FEATURE_EXTRACTION_EPOCHS + FINE_TUNE_EPOCHS
epochs_range = range(1, total_epochs + 1)

rows = []
for epoch, (val_acc, train_loss, val_loss) in enumerate(
    zip(combined_val_acc, combined_train_loss, combined_val_loss), start=1
):
    rows.append({
        "epoch": epoch,
        "phase": "feature_extraction" if epoch <= switch_epoch else "fine_tuning",
        "validation_accuracy": val_acc,
        "training_loss": train_loss,
        "validation_loss": val_loss,
    })
pd.DataFrame(rows).to_csv("outputs/results/transfer_learning_finetuning.csv", index=False)

pd.DataFrame({
    "phase": ["feature_extraction", "fine_tuning"],
    "training_time_seconds": [time_a, time_b],
}).to_csv("outputs/results/transfer_learning_finetuning_time.csv", index=False)

out_dir = "outputs/plots"

# Plot 11: Feature Extraction vs Fine Tuning, Validation Accuracy vs Epoch
fig, ax = plt.subplots(figsize=(9, 5.5))
ax.plot(range(1, switch_epoch + 1), combined_val_acc[:switch_epoch], marker="o",
        color="steelblue", label="Feature Extraction")
ax.plot(range(switch_epoch + 1, total_epochs + 1), combined_val_acc[switch_epoch:], marker="o",
        color="crimson", label="Fine Tuning")
ax.axvline(switch_epoch + 0.5, color="gray", linestyle="--", linewidth=1)
ax.set_xlabel("Epoch")
ax.set_ylabel("Validation Accuracy")
ax.legend()
fig.tight_layout()
save_fig(fig, f"{out_dir}/transfer_vs_finetune_accuracy")
plt.close(fig)

# Plot 12: Training and Validation Loss, before and after fine tuning
fig, ax = plt.subplots(figsize=(9, 5.5))
ax.plot(epochs_range, combined_train_loss, marker="o", color="red", label="Training Loss")
ax.plot(epochs_range, combined_val_loss, marker="o", color="purple", label="Validation Loss")
ax.axvline(switch_epoch + 0.5, color="gray", linestyle="--", linewidth=1)
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss")
ax.legend()
fig.tight_layout()
save_fig(fig, f"{out_dir}/transfer_vs_finetune_loss")
plt.close(fig)

print("\nsaved transfer_vs_finetune_accuracy and transfer_vs_finetune_loss plots to", out_dir)
print("the dashed vertical line in both plots marks where fine tuning begins")
