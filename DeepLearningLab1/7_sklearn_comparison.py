import numpy as np
from sklearn.linear_model import Perceptron as SklearnPerceptron

from perceptron import Perceptron

X_train = np.load("outputs/saved_model/X_train.npy")
y_train = np.load("outputs/saved_model/y_train.npy")
X_test = np.load("outputs/saved_model/X_test.npy")
y_test = np.load("outputs/saved_model/y_test.npy")

# our implementation
ours = Perceptron(n_features=X_train.shape[1], lr=0.01, epochs=50, seed=42)
ours.fit(X_train, y_train, verbose=False)
our_acc = np.mean(ours.predict(X_test) == y_test)

# sklearn's implementation
sk = SklearnPerceptron(eta0=0.01, max_iter=50, tol=1e-3, random_state=42)
sk.fit(X_train, y_train)
sk_acc = np.mean(sk.predict(X_test) == y_test)

print(f"our perceptron accuracy:     {our_acc:.4f}")
print(f"sklearn perceptron accuracy: {sk_acc:.4f}")
