import numpy as np


def softmax(z):
    """Compute softmax values for each row of z."""
    shift_z = z - np.max(z, axis=1, keepdims=True)  # Avoid overflow
    exp_z = np.exp(shift_z)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)
