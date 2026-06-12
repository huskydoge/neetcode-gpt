import numpy as np
from numpy.typing import NDArray
from typing import Tuple


class Solution:
    def train(self, X: NDArray[np.float64], y: NDArray[np.float64], epochs: int, lr: float) -> Tuple[NDArray[np.float64], float]:
        # X: (n_samples, n_features)
        # y: (n_samples,) targets
        # epochs: number of training iterations
        # lr: learning rate
        #
        # Model: y_hat = X @ w + b
        # Loss: MSE = (1/n) * sum((y_hat - y)^2)
        # Initialize w = zeros, b = 0
        # return (np.round(w, 5), round(b, 5))

        bs = X.shape[0]
        n_features = X.shape[1]
        w = np.zeros((n_features, ))
        b = 0

        for ep in range(epochs):
            y_hat = X @ w + b

            loss = ((y_hat - y)**2).mean() 

            grad_w = X.T @ (2 / bs * (y_hat - y)) # (bs, )
            grad_b = 2 / bs * (y_hat - y).sum()

            w -= grad_w * lr
            b -= grad_b * lr

        return (np.round(w, 5), round(b, 5))


        
