import numpy as np
from activations import softmax
from losses import cross_entropy_loss
from utils import one_hot_encode


class MulticlassLogisticRegression:
    def __init__(self, learning_rate=0.01, max_iter=1000, tol=1e-4, lambda_reg=0.1, random_state=None):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tol = tol
        self.lambda_reg = lambda_reg
        self.random_state = random_state
        self.weights = None
        self.classes_ = None

    def fit(self, X, y):

        self.classes_ = np.unique(y)
        num_classes = len(self.classes_)
        y_int = np.searchsorted(self.classes_, y)
        y_oh = one_hot_encode(y_int, num_classes)

        X_b = np.c_[X, np.ones(X.shape[0])]

        if self.random_state is not None:
            np.random.seed(self.random_state)
        self.weights = np.random.randn(X_b.shape[1], num_classes) * 0.01

        prev_loss = float('inf')
        for i in range(self.max_iter):

            logits = X_b @ self.weights
            probs = softmax(logits)

            loss = cross_entropy_loss(y_oh, probs)
            reg_loss = 0.5 * self.lambda_reg * \
                np.sum(self.weights[:-1]**2)
            total_loss = loss + reg_loss

            if abs(prev_loss - total_loss) < self.tol:
                break
            prev_loss = total_loss

            grad = X_b.T @ (probs - y_oh) / X_b.shape[0]

            grad[:-1] += self.lambda_reg * self.weights[:-1]

            self.weights -= self.learning_rate * grad

    def predict_proba(self, X):
        X_b = np.c_[X, np.ones(X.shape[0])]
        logits = X_b @ self.weights
        return softmax(logits)

    def predict(self, X):
        probs = self.predict_proba(X)
        return self.classes_[np.argmax(probs, axis=1)]

    def score(self, X, y):
        y_pred = self.predict(X)
        return np.mean(y == y_pred)
