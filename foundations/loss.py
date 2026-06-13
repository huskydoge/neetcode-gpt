import numpy as np
from numpy.typing import NDArray


class Solution:

    def binary_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        # y_true: true labels (0 or 1)
        # y_pred: predicted probabilities
        # Hint: add a small epsilon (1e-7) to y_pred to avoid log(0)
        # return round(your_answer, 4)
        bs = y_true.shape[0]
        bce = - 1 / bs * np.sum(y_true * np.log(y_pred + 1e-7) + (1 - y_true) * np.log(1 - y_pred))

        return np.round(bce, 4)

    def categorical_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        # y_true: one-hot encoded true labels (shape: n_samples x n_classes)
        # y_pred: predicted probabilities (shape: n_samples x n_classes)
        # Hint: add a small epsilon (1e-7) to y_pred to avoid log(0)
        # return round(your_answer, 4)
        bs = y_true.shape[0]
        cce = - 1 / bs * ((y_true * np.log(y_pred + 1e-7)).sum())
        print(cce)
        return np.round(cce, 4)