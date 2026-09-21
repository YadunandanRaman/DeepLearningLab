import time

import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt

from data_loader import load_har_data
from model_builder import build_sequence_classifier
from plot_style import setup_fonts, save_fig

setup_fonts()

# Suggested training configuration from Section 9, used identically
# for RNN, LSTM, and GRU (Section 12: "replace only the recurrent
# layer"), so any difference in the results is attributable to the
# recurrent layer type alone.
RECURRENT_TYPE = "gru"
UNITS = 32
DROPOUT_RATE = 0.2
LEARNING_RATE = 0.001
BATCH_SIZE = 32
EPOCHS = 30

(X_train, y_train), (X_val, y_val), (X_test, y_test) = load_har_data()

model = build_sequence_classifier(recurrent_type=RECURRENT_TYPE, units=UNITS, dropout_rate=DROPOUT_RATE)
model.summary()
total_params = model.count_params()
print(f"\n{RECURRENT_TYPE} total trainable parameters: {total_params:,}")

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

start = time.time()
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    verbose=2,
)
training_time = time.time() - start
print(f"\ntraining time: {training_time:.2f} seconds")

model.save(f"outputs/saved_model/{RECURRENT_TYPE}_model.keras")

pd.DataFrame({
    "epoch": range(1, EPOCHS + 1),
    "training_loss": history.history["loss"],
    "validation_loss": history.history["val_loss"],
    "training_accuracy": history.history["accuracy"],
    "validation_accuracy": history.history["val_accuracy"],
}).to_csv(f"outputs/results/{RECURRENT_TYPE}_history.csv", index=False)

pd.DataFrame({
    "model": [RECURRENT_TYPE],
    "parameters": [total_params],
    "training_time_seconds": [training_time],
}).to_csv(f"outputs/results/{RECURRENT_TYPE}_summary.csv", index=False)

out_dir = "outputs/plots"
epochs_range = range(1, EPOCHS + 1)

# Plot 2: Training and Validation Loss
fig, ax = plt.subplots(figsize=(8, 5.5))
ax.plot(epochs_range, history.history["loss"], color="red", label="Training Loss")
ax.plot(epochs_range, history.history["val_loss"], color="purple", label="Validation Loss")
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss")
ax.legend()
fig.tight_layout()
save_fig(fig, f"{out_dir}/{RECURRENT_TYPE}_loss")
plt.close(fig)

# Plot 3: Training and Validation Accuracy
fig, ax = plt.subplots(figsize=(8, 5.5))
ax.plot(epochs_range, history.history["accuracy"], color="green", label="Training Accuracy")
ax.plot(epochs_range, history.history["val_accuracy"], color="orange", label="Validation Accuracy")
ax.set_xlabel("Epoch")
ax.set_ylabel("Accuracy")
ax.legend()
fig.tight_layout()
save_fig(fig, f"{out_dir}/{RECURRENT_TYPE}_accuracy")
plt.close(fig)

print(f"\nsaved {RECURRENT_TYPE}_loss and {RECURRENT_TYPE}_accuracy plots to {out_dir}")
