import numpy as np


def one_hot_encode(y, num_classes):
    """Convert class labels to one-hot encoded vectors."""
    return np.eye(num_classes)[y]


def accuracy_score(y_true, y_pred):
    """Compute classification accuracy."""
    return np.mean(y_true == y_pred)
