import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from tensorflow.keras.utils import to_categorical

from data_loader import load_data
from model_builder import build_cnn
from plot_style import setup_fonts, save_fig

setup_fonts()

# ------------------------------------------------------------------
# Additional Exercise 5: increase the number of convolution filters
# from 16 to 64 and analyze the effect on accuracy and computation
# time. Both conv layers use the same filter count in each variant,
# isolating filter count as a single, clean variable rather than
# changing both layers by different, confounded amounts the way the
# 32 then 64 baseline architecture does.
#
# Two full CNNs, identical to the Task 6 architecture except for this
# one setting, are trained under matched conditions, the same 10 epoch,
# batch size 32 recipe used for the Task 5 pooling comparison, on the
# complete training set.
# ------------------------------------------------------------------
EPOCHS = 10
BATCH_SIZE = 32
FILTER_COUNTS = [16, 64]

(X_train, y_train), (X_test, y_test) = load_data()
y_train = y_train.flatten()
y_test_labels = y_test.flatten()
X_train = X_train.astype("float32") / 255.0
X_test = X_test.astype("float32") / 255.0
y_train_onehot = to_categorical(y_train, num_classes=10)

results = {}
for num_filters in FILTER_COUNTS:
    print(f"\ntraining with {num_filters} filters in both convolutional layers")
    model = build_cnn(pooling_type="max", num_filters=num_filters)
    total_params = model.count_params()
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

    results[num_filters] = {
        "parameters": total_params,
        "test_accuracy": test_accuracy,
        "training_time_seconds": elapsed,
    }
    print(f"{num_filters} filters: params={total_params:,}, "
          f"test accuracy={test_accuracy:.4f}, training time={elapsed:.2f}s")

comparison_df = pd.DataFrame(results).T
comparison_df.index.name = "num_filters"
comparison_df.to_csv("outputs/results/filter_count_comparison.csv")
print("\nfilter count comparison:")
print(comparison_df.to_string())

labels = [f"{n} Filters" for n in FILTER_COUNTS]
accuracies = [results[n]["test_accuracy"] for n in FILTER_COUNTS]

fig, ax = plt.subplots(figsize=(6, 5.5))
ax.bar(labels, accuracies, color=["steelblue", "crimson"], label="Test Accuracy")
ax.set_xlabel("Filter Count")
ax.set_ylabel("Test Accuracy")
ax.legend()
fig.tight_layout()
save_fig(fig, "outputs/plots/filter_count_comparison")
plt.close(fig)

print("\nsaved filter count comparison plot and results table")
