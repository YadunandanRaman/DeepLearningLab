import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from model_builder import build_baseline_model
from plot_style import setup_fonts, save_fig

setup_fonts()

X_train = np.load("outputs/saved_model/X_train.npy")
y_train = np.load("outputs/saved_model/y_train.npy")

model = build_baseline_model()
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

start = time.time()
history = model.fit(
    X_train, y_train,
    validation_split=0.1,
    epochs=20,
    batch_size=32,
    verbose=2,
)
training_time = time.time() - start

model.save("outputs/saved_model/baseline_model.keras")

pd.DataFrame({"training_time_seconds": [training_time]}).to_csv(
    "outputs/results/baseline_training_time.csv", index=False
)

print(f"\nbaseline training time: {training_time:.2f} seconds")
print("final training accuracy:", history.history["accuracy"][-1])
print("final validation accuracy:", history.history["val_accuracy"][-1])

out_dir = "outputs/plots"
epochs_range = range(1, len(history.history["accuracy"]) + 1)

# training accuracy vs epoch
fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(epochs_range, history.history["accuracy"], marker="o", color="green", label="Training Accuracy")
ax.set_xlabel("Epoch")
ax.set_ylabel("Accuracy")
ax.legend()
fig.tight_layout()
save_fig(fig, f"{out_dir}/training_accuracy")
plt.close(fig)

# validation accuracy vs epoch
fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(epochs_range, history.history["val_accuracy"], marker="o", color="orange", label="Validation Accuracy")
ax.set_xlabel("Epoch")
ax.set_ylabel("Accuracy")
ax.legend()
fig.tight_layout()
save_fig(fig, f"{out_dir}/validation_accuracy")
plt.close(fig)

# training loss vs epoch
fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(epochs_range, history.history["loss"], marker="o", color="red", label="Training Loss")
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss")
ax.legend()
fig.tight_layout()
save_fig(fig, f"{out_dir}/training_loss")
plt.close(fig)

# validation loss vs epoch
fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(epochs_range, history.history["val_loss"], marker="o", color="purple", label="Validation Loss")
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss")
ax.legend()
fig.tight_layout()
save_fig(fig, f"{out_dir}/validation_loss")
plt.close(fig)

print("saved trained model and 4 training curve plots to", out_dir)
