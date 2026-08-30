import pandas as pd
import matplotlib.pyplot as plt

from data_loader import load_datasets
from model_builder import build_model, compile_model
from plot_style import setup_fonts, save_fig

setup_fonts()

# Section 7's numerical Batch Normalization example (mean 5, variance
# 5, normalized values approximately [-1.342, -0.447, 0.447, 1.342] for
# x = [2, 4, 6, 8]) is pure arithmetic and needs no code, it is worked
# out directly in the report. This script produces Plot 5, the
# empirical follow up: a direct two way comparison, identical models
# with and without Batch Normalization in the classifier head, holding
# everything else fixed, which is exactly the "no regularization"
# versus "batchnorm" pair already trained in 3_regularization.py. Both
# are retrained here on their own for a focused, standalone Plot 5
# rather than making the reader extract this one comparison from that
# script's larger four way grid.
EPOCHS = 10
BATCH_SIZE = 32
LEARNING_RATE = 0.001

train_ds, val_ds, _ = load_datasets()
train_ds_batched = train_ds.batch(BATCH_SIZE).prefetch(1)
val_ds_batched = val_ds.batch(BATCH_SIZE).prefetch(1)

histories = {}
for use_bn, label in [(False, "Without BN"), (True, "With BN")]:
    print(f"\ntraining {label}")
    regularization = "batchnorm" if use_bn else None
    model, base_model = build_model(initializer="xavier", regularization=regularization, freeze_base=True)
    compile_model(model, optimizer_name="adam", learning_rate=LEARNING_RATE)

    history = model.fit(
        train_ds_batched,
        validation_data=val_ds_batched,
        epochs=EPOCHS,
        verbose=2,
    )
    histories[label] = history.history
    print(f"{label}: final validation accuracy={history.history['val_accuracy'][-1]:.4f}")

rows = []
for label, hist in histories.items():
    for epoch in range(EPOCHS):
        rows.append({
            "configuration": label,
            "epoch": epoch + 1,
            "validation_accuracy": hist["val_accuracy"][epoch],
        })
pd.DataFrame(rows).to_csv("outputs/results/batch_norm_comparison.csv", index=False)

out_dir = "outputs/plots"
epochs_range = range(1, EPOCHS + 1)

# Plot 5: With BN vs Without BN, Validation Accuracy vs Epoch
fig, ax = plt.subplots(figsize=(8, 5.5))
ax.plot(epochs_range, histories["Without BN"]["val_accuracy"], marker="o", color="steelblue", label="Without BN")
ax.plot(epochs_range, histories["With BN"]["val_accuracy"], marker="o", color="crimson", label="With BN")
ax.set_xlabel("Epoch")
ax.set_ylabel("Validation Accuracy")
ax.legend()
fig.tight_layout()
save_fig(fig, f"{out_dir}/batch_norm_comparison")
plt.close(fig)

print("\nsaved batch_norm_comparison plot to", out_dir)
