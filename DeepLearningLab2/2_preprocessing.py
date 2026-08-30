import numpy as np

from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.utils import to_categorical

(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

print("Before preprocessing:")
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)
print("pixel value range:", X_train.min(), "to", X_train.max())

# flatten 28x28 images into 784 length vectors (Section 5, steps 1 and 3)
X_train_flat = X_train.reshape(X_train.shape[0], 28 * 28)
X_test_flat = X_test.reshape(X_test.shape[0], 28 * 28)

# normalize pixels to [0, 1] (Section 5, step 4)
X_train_flat = X_train_flat.astype("float32") / 255.0
X_test_flat = X_test_flat.astype("float32") / 255.0

# one-hot encode labels (Section 5, step 5), used for the baseline model's
# categorical_crossentropy loss in Task 4
y_train_onehot = to_categorical(y_train, num_classes=10)
y_test_onehot = to_categorical(y_test, num_classes=10)

print("\nAfter preprocessing:")
print("X_train shape:", X_train_flat.shape)
print("X_test shape:", X_test_flat.shape)
print("y_train shape (one-hot):", y_train_onehot.shape)
print("y_test shape (one-hot):", y_test_onehot.shape)
print("pixel value range:", X_train_flat.min(), "to", X_train_flat.max())

np.save("outputs/saved_model/X_train.npy", X_train_flat)
np.save("outputs/saved_model/X_test.npy", X_test_flat)
np.save("outputs/saved_model/y_train.npy", y_train_onehot)
np.save("outputs/saved_model/y_test.npy", y_test_onehot)

# integer labels are also saved since sklearn's metrics functions and the
# hyperparameter search in 6_hyperparameter_search.py both expect a 1D
# target vector rather than one-hot vectors
np.save("outputs/saved_model/y_train_labels.npy", y_train)
np.save("outputs/saved_model/y_test_labels.npy", y_test)

print("\nsaved preprocessed arrays to outputs/saved_model/")
