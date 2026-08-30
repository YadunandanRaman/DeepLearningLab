import numpy as np
import pandas as pd

df = pd.read_csv("data/banknote_authentication.csv")

feature_cols = ["variance", "skewness", "curtosis", "entropy"]
X = df[feature_cols].values.astype(float)
y = df["class"].values.astype(int)

# 80/20 split
rng = np.random.default_rng(42)
idx = rng.permutation(len(X))
n_test = int(len(X) * 0.2)
test_idx, train_idx = idx[:n_test], idx[n_test:]

X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]

# normalize using training stats
mean = X_train.mean(axis=0)
std = X_train.std(axis=0)
X_train = (X_train - mean) / std
X_test = (X_test - mean) / std

print("train size:", X_train.shape[0])
print("test size:", X_test.shape[0])
print("feature means (train, after scaling):", X_train.mean(axis=0).round(4))
print("feature stds (train, after scaling):", X_train.std(axis=0).round(4))

np.save("outputs/saved_model/X_train.npy", X_train)
np.save("outputs/saved_model/X_test.npy", X_test)
np.save("outputs/saved_model/y_train.npy", y_train)
np.save("outputs/saved_model/y_test.npy", y_test)

print("saved train/test arrays to outputs/saved_model/")
