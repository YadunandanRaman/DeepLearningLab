import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical

from data_loader import load_data
from model_builder import compile_model, unfreeze_last_block, get_base_model
from plot_style import setup_fonts, save_fig

setup_fonts()

ARCHITECTURE = "mobilenetv2"
FINE_TUNE_LEARNING_RATE = 0.0001  # smaller than Task 3's 0.001
FINE_TUNE_EPOCHS = 8
BATCH_SIZE = 32

(X_train, y_train), (X_test, y_test) = load_data()
y_train = y_train.flatten()
y_test = y_test.flatten()
y_train_onehot = to_categorical(y_train, num_classes=10)
y_test_onehot = to_categorical(y_test, num_classes=10)
X_train = X_train.astype("float32")
X_test = X_test.astype("float32")

model = load_model("outputs/saved_model/transfer_model.keras")

print("evaluating the Task 3 model, frozen base, before fine tuning")
before_loss, before_accuracy = model.evaluate(X_test, y_test_onehot, verbose=0)
print(f"test accuracy before fine tuning: {before_accuracy:.4f}")

# Unfreezing the last block only, rather than the whole base, keeps
# most of the pretrained ImageNet features fixed and only adapts the
# highest level, most task specific features to CIFAR-10. The learning
# rate is also dropped by a factor of 10 from Task 3's 0.001. Fine
# tuning with a learning rate as large as the one used to train the
# new head from scratch risks large gradient updates that damage the
# pretrained weights before they have a chance to adapt gently.
base_model = get_base_model(model)
unfreeze_last_block(base_model, architecture=ARCHITECTURE)
compile_model(model, optimizer_name="adam", learning_rate=FINE_TUNE_LEARNING_RATE)

trainable_params = sum(w.shape.num_elements() for w in model.trainable_weights)
print(f"trainable parameters during fine tuning: {trainable_params:,}")

start = time.time()
history = model.fit(
    X_train, y_train_onehot,
    validation_split=0.1,
    epochs=FINE_TUNE_EPOCHS,
    batch_size=BATCH_SIZE,
    verbose=2,
)
fine_tune_time = time.time() - start

print("\nevaluating after fine tuning")
after_loss, after_accuracy = model.evaluate(X_test, y_test_onehot, verbose=0)
print(f"test accuracy after fine tuning: {after_accuracy:.4f}")

model.save("outputs/saved_model/fine_tuned_model.keras")

comparison = pd.DataFrame({
    "Stage": ["Before Fine Tuning (Task 3)", "After Fine Tuning (Task 4)"],
    "Test Accuracy": [before_accuracy, after_accuracy],
    "Test Loss": [before_loss, after_loss],
})
comparison.to_csv("outputs/results/fine_tuning_comparison.csv", index=False)
pd.DataFrame({"fine_tune_time_seconds": [fine_tune_time]}).to_csv(
    "outputs/results/task4_training_time.csv", index=False
)

print("\nfine tuning comparison:")
print(comparison.to_string(index=False))
print(f"\nfine tuning time: {fine_tune_time:.2f} seconds")

out_dir = "outputs/plots"
epochs_range = range(1, len(history.history["accuracy"]) + 1)

fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(epochs_range, history.history["accuracy"], marker="o", color="green", label="Fine Tuning Training Accuracy")
ax.set_xlabel("Epoch")
ax.set_ylabel("Accuracy")
ax.legend()
fig.tight_layout()
save_fig(fig, f"{out_dir}/fine_tuning_training_accuracy")
plt.close(fig)

fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(epochs_range, history.history["val_accuracy"], marker="o", color="orange", label="Fine Tuning Validation Accuracy")
ax.set_xlabel("Epoch")
ax.set_ylabel("Accuracy")
ax.legend()
fig.tight_layout()
save_fig(fig, f"{out_dir}/fine_tuning_validation_accuracy")
plt.close(fig)

fig, ax = plt.subplots(figsize=(6, 5.5))
ax.bar(comparison["Stage"], comparison["Test Accuracy"], color=["steelblue", "crimson"])
ax.set_xlabel("Stage")
ax.set_ylabel("Test Accuracy")
handles = [
    mpatches.Patch(color="steelblue", label="Before Fine Tuning"),
    mpatches.Patch(color="crimson", label="After Fine Tuning"),
]
ax.legend(handles=handles)
plt.setp(ax.get_xticklabels(), rotation=15, ha="right")
fig.tight_layout()
save_fig(fig, f"{out_dir}/fine_tuning_comparison")
plt.close(fig)

print("\nsaved fine tuned model and fine tuning plots to", out_dir)
