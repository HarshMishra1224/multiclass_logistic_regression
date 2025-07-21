import numpy as np


def cross_entropy_loss(y_true, y_pred, epsilon=1e-12):
    """Compute cross-entropy loss between true labels and predicted probabilities."""
    y_pred = np.clip(y_pred, epsilon, 1. - epsilon)  # Avoid log(0)
    return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))
