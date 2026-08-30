import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Input, MaxPooling2D, AveragePooling2D

from data_loader import load_data
from model_builder import build_cnn
from plot_style import setup_fonts, save_fig

setup_fonts()

# ------------------------------------------------------------------
# Output size: max pooling and average pooling with the same pool size
# and stride always produce identical output shapes, since spatial
# reduction depends only on the pooling window and stride, not on
# whether the window takes a maximum or an average. This is confirmed
# directly with real layers below, rather than only stated.
# ------------------------------------------------------------------
sample_input = Input(shape=(32, 32, 16))
max_shape = MaxPooling2D(pool_size=2)(sample_input).shape
avg_shape = AveragePooling2D(pool_size=2)(sample_input).shape
print(f"max pooling output shape: {max_shape}")
print(f"average pooling output shape: {avg_shape}")
print(f"output shapes match: {max_shape == avg_shape}")

# ------------------------------------------------------------------
# Accuracy: two full CNNs, identical to the Task 6 architecture except
# for the pooling layer, trained under matched conditions for a
# reduced epoch budget, to compare accuracy directly rather than
# assuming which pooling strategy performs better.
# ------------------------------------------------------------------
EPOCHS = 10
BATCH_SIZE = 32

(X_train, y_train), (X_test, y_test) = load_data()
y_train = y_train.flatten()
y_test_labels = y_test.flatten()
X_train = X_train.astype("float32") / 255.0
X_test = X_test.astype("float32") / 255.0
y_train_onehot = to_categorical(y_train, num_classes=10)

results = {}
for pooling_type in ["max", "average"]:
    print(f"\ntraining with {pooling_type} pooling")
    model = build_cnn(pooling_type=pooling_type)
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

    start = time.time()
    model.fit(
        X_train, y_train_onehot,
        validation_split=0.1,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=2,
    )
    elapsed = time.time() - start

    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
    test_accuracy = accuracy_score(y_test_labels, y_pred)
    results[pooling_type] = {"test_accuracy": test_accuracy, "training_time_seconds": elapsed}
    print(f"{pooling_type} pooling: test accuracy={test_accuracy:.4f}, training time={elapsed:.2f}s")

comparison_df = pd.DataFrame(results).T
comparison_df.to_csv("outputs/results/pooling_comparison.csv")
print("\npooling comparison:")
print(comparison_df.to_string())

fig, ax = plt.subplots(figsize=(6, 5.5))
ax.bar(["Max Pooling", "Average Pooling"],
       [results["max"]["test_accuracy"], results["average"]["test_accuracy"]],
       color=["steelblue", "crimson"], label="Test Accuracy")
ax.set_xlabel("Pooling Strategy")
ax.set_ylabel("Test Accuracy")
ax.legend()
fig.tight_layout()
save_fig(fig, "outputs/plots/pooling_comparison")
plt.close(fig)

print("\nsaved pooling comparison plot and results table")
