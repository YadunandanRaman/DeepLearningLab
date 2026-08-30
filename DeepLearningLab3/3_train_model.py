import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical

from data_loader import load_data
from model_builder import build_cnn
from plot_style import setup_fonts, save_fig

setup_fonts()

# Task 6 hyperparameters, exactly as specified in the lab manual
OPTIMIZER = "adam"
EPOCHS = 20
BATCH_SIZE = 32

(X_train, y_train), (X_test, y_test) = load_data()
y_train = y_train.flatten()
X_train = X_train.astype("float32") / 255.0
y_train_onehot = to_categorical(y_train, num_classes=10)

model = build_cnn(pooling_type="max")
model.summary()
print(f"\ntotal parameters: {model.count_params():,}")

model.compile(optimizer=OPTIMIZER, loss="categorical_crossentropy", metrics=["accuracy"])

start = time.time()
history = model.fit(
    X_train, y_train_onehot,
    validation_split=0.1,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    verbose=2,
)
training_time = time.time() - start

model.save("outputs/saved_model/cnn_model.keras")

pd.DataFrame({"training_time_seconds": [training_time]}).to_csv(
    "outputs/results/training_time.csv", index=False
)

print(f"\ntraining time: {training_time:.2f} seconds")
print("final training accuracy:", history.history["accuracy"][-1])
print("final validation accuracy:", history.history["val_accuracy"][-1])

out_dir = "outputs/plots"
epochs_range = range(1, len(history.history["accuracy"]) + 1)

fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(epochs_range, history.history["accuracy"], marker="o", color="green", label="Training Accuracy")
ax.set_xlabel("Epoch")
ax.set_ylabel("Accuracy")
ax.legend()
fig.tight_layout()
save_fig(fig, f"{out_dir}/training_accuracy")
plt.close(fig)

fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(epochs_range, history.history["val_accuracy"], marker="o", color="orange", label="Validation Accuracy")
ax.set_xlabel("Epoch")
ax.set_ylabel("Accuracy")
ax.legend()
fig.tight_layout()
save_fig(fig, f"{out_dir}/validation_accuracy")
plt.close(fig)

fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(epochs_range, history.history["loss"], marker="o", color="red", label="Training Loss")
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss")
ax.legend()
fig.tight_layout()
save_fig(fig, f"{out_dir}/training_loss")
plt.close(fig)

fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(epochs_range, history.history["val_loss"], marker="o", color="purple", label="Validation Loss")
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss")
ax.legend()
fig.tight_layout()
save_fig(fig, f"{out_dir}/validation_loss")
plt.close(fig)

print("\nsaved trained model and 4 training curve plots to", out_dir)
