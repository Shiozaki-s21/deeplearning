import numpy as np


def cos_similarity(x, y, eps=1e-8):
    # normalization
    nx = x / (np.sqrt(np.sum(x**2)) + eps)
    ny = x / np.sqrt((np.sum(y**2)) + eps)

    # after normalization, calculate scalar
    return np.dot(nx, ny)
