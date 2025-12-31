####################################
# Logistic Regression
####################################
import numpy as np
import pickle

class CustomLogisticRegression:
    def __init__(self, lr=0.01, epochs=1000, reg_strength=0.0):
        self.lr = lr
        self.epochs = epochs
        self.reg_strength = reg_strength
        self.weights = None
        self.bias = 0.0

    def sigmoid(self, z):
        z = np.clip(z, -500, 500)  # prevent overflow
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        X = np.array(X, dtype=np.float64)
        y = np.array(y, dtype=np.float64)

        # Normalize X
        self.X_min = X.min(axis=0)
        self.X_max = X.max(axis=0)
        X = (X - self.X_min) / (self.X_max - self.X_min + 1e-8)

        m, n = X.shape
        self.weights = np.zeros(n, dtype=np.float64)
        self.bias = 0.0

        for epoch in range(self.epochs):
            linear = np.dot(X, self.weights) + self.bias
            predictions = self.sigmoid(linear)
            error = predictions - y
            dw = (np.dot(X.T, error) + self.reg_strength * self.weights) / m
            db = np.sum(error) / m

            self.weights -= self.lr * dw
            self.bias -= self.lr * db

            if epoch % 100 == 0 or epoch == self.epochs - 1:
                loss = -np.mean(y * np.log(predictions + 1e-9) + (1 - y) * np.log(1 - predictions + 1e-9))

    def predict_proba(self, X):
        X = np.array(X, dtype=np.float64)
        X = (X - self.X_min) / (self.X_max - self.X_min + 1e-8)
        return self.sigmoid(np.dot(X, self.weights) + self.bias)

    def predict(self, X):
        return (self.predict_proba(X) >= 0.5).astype(int)

    def save(self, filename):
        with open(filename, "wb") as f:
            pickle.dump(self.classifiers, f)

    def load_pretrained(self, filename):
        with open(filename, "rb") as f:
            self.classifiers = pickle.load(f)
            self.classes = np.array(list(self.classifiers.keys()))


class CustomMulticlassLogisticRegression:
    def __init__(self, lr=0.01, epochs=1000, reg_strength=0.0):
        self.lr = lr
        self.epochs = epochs
        self.reg_strength = reg_strength
        self.classifiers = {}

    def fit(self, X, y):
        self.classes = np.unique(y)
        for cls in self.classes:
            y_binary = (y == cls).astype(int)
            model = CustomLogisticRegression(
                lr=self.lr, epochs=self.epochs, reg_strength=self.reg_strength
            )
            model.fit(X, y_binary)
            self.classifiers[cls] = model

    def predict_proba(self, X):
        return {cls: model.predict_proba(X) for cls, model in self.classifiers.items()}

    def predict(self, X):
        probs = self.predict_proba(X)
        probs_matrix = np.column_stack([probs[cls] for cls in self.classes])
        return self.classes[np.argmax(probs_matrix, axis=1)]

    def save(self, filename):
        with open(filename, "wb") as f:
            pickle.dump(self.classifiers, f)

    def load_pretrained(self, filename):
        with open(filename, "rb") as f:
            self.classifiers = pickle.load(f)
            self.classes = np.array(list(self.classifiers.keys()))

