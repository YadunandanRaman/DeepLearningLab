import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score

from data_loader import load_har_data, NUM_CLASSES
from model_builder import build_sequence_classifier
from plot_style import setup_fonts, save_fig

setup_fonts()

# Truncating to the first T time steps of each 128 length window,
# consistently across every sequence length tested, so the comparison
# isolates how much temporal context the model gets to see rather than
# introducing a different windowing strategy at each length.
SEQUENCE_LENGTHS = [32, 64, 128]
MODEL_TYPES = ["rnn", "lstm", "gru"]
UNITS = 32
DROPOUT_RATE = 0.2
LEARNING_RATE = 0.001
BATCH_SIZE = 32
EPOCHS = 20  # reduced from Section 9's 30, this is a comparison across 9 runs, not a final model

(X_train, y_train), (X_val, y_val), (X_test, y_test) = load_har_data()

results = {}
for seq_len in SEQUENCE_LENGTHS:
    X_train_t = X_train[:, :seq_len, :]
    X_val_t = X_val[:, :seq_len, :]
    X_test_t = X_test[:, :seq_len, :]

    for model_type in MODEL_TYPES:
        print(f"\ntraining {model_type} with sequence length {seq_len}")
        model = build_sequence_classifier(
            recurrent_type=model_type, input_shape=(seq_len, X_train.shape[-1]),
            num_classes=NUM_CLASSES, units=UNITS, dropout_rate=DROPOUT_RATE,
        )
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        model.fit(
            X_train_t, y_train, validation_data=(X_val_t, y_val),
            epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=0,
        )

        y_pred = np.argmax(model.predict(X_test_t, verbose=0), axis=1)
        f1 = f1_score(y_test, y_pred, labels=list(range(NUM_CLASSES)), average="macro", zero_division=0)
        print(f"{model_type} at T={seq_len}: macro F1={f1:.4f}")
        results[(seq_len, model_type)] = f1

rows = []
for seq_len in SEQUENCE_LENGTHS:
    row = {"sequence_length": seq_len}
    for model_type in MODEL_TYPES:
        row[f"{model_type}_f1"] = results[(seq_len, model_type)]
    rows.append(row)

results_df = pd.DataFrame(rows)
results_df.to_csv("outputs/results/sequence_length_study.csv", index=False)
print("\nsequence length study results:")
print(results_df.to_string(index=False))

# Plot 6: Sequence Length vs Test F1 score, one line per model
fig, ax = plt.subplots(figsize=(8, 5.5))
colors = {"rnn": "gray", "lstm": "steelblue", "gru": "crimson"}
for model_type in MODEL_TYPES:
    ax.plot(results_df["sequence_length"], results_df[f"{model_type}_f1"],
            marker="o", color=colors[model_type], label=model_type.upper())
ax.set_xlabel("Sequence Length (T)")
ax.set_ylabel("Test Macro F1 Score")
ax.set_xticks(SEQUENCE_LENGTHS)
ax.legend()
fig.tight_layout()
save_fig(fig, "outputs/plots/sequence_length_vs_f1")
plt.close(fig)

print("\nsaved sequence_length_vs_f1 plot to outputs/plots")
