import numpy as np

# Single layer perceptron with step activation.
# z = w.x + b, y_hat = 1 if z >= 0 else 0
# w = w + lr * (y - y_hat) * x
# b = b + lr * (y - y_hat)

class Perceptron:
    def __init__(self, n_features, lr=0.01, epochs=50, seed=42):
        rng = np.random.default_rng(seed)
        self.w = rng.normal(0, 0.01, n_features)
        self.b = 0.0
        self.lr = lr
        self.epochs = epochs

        self.errors_per_epoch = []
        self.weights_per_epoch = []
        self.bias_per_epoch = []

    def step(self, z):
        return np.where(z >= 0, 1, 0)

    def net_input(self, X):
        return np.dot(X, self.w) + self.b

    def predict(self, X):
        return self.step(self.net_input(X))

    def fit(self, X, y, verbose=True):
        for epoch in range(1, self.epochs + 1):
            errors = 0
            for xi, yi in zip(X, y):
                y_hat = self.step(np.dot(xi, self.w) + self.b)
                update = self.lr * (yi - y_hat)
                if update != 0:
                    errors += 1
                    self.w += update * xi
                    self.b += update

            self.errors_per_epoch.append(errors)
            self.weights_per_epoch.append(self.w.copy())
            self.bias_per_epoch.append(self.b)

            if verbose:
                print(f"epoch {epoch}: misclassified={errors}, w={self.w}, b={self.b:.4f}")

            if errors == 0:
                print(f"converged at epoch {epoch}")
                break

        return self
