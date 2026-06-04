import numpy as np
from typing import Tuple, List


class Solution:
    def batch_norm(
        self,
        x: List[List[float]],
        gamma: List[float],
        beta: List[float],
        running_mean: List[float],
        running_var: List[float],
        momentum: float,
        eps: float,
        training: bool
    ) -> Tuple[List[List[float]], List[float], List[float]]:

        x = np.asarray(x, dtype=float)
        gamma = np.asarray(gamma, dtype=float)
        beta = np.asarray(beta, dtype=float)
        running_mean = np.asarray(running_mean, dtype=float)
        running_var = np.asarray(running_var, dtype=float)

        if training:
            mean = x.mean(axis=0)
            var = x.var(axis=0)

            # update running stats
            running_mean = (1 - momentum) * running_mean + momentum * mean
            running_var = (1 - momentum) * running_var + momentum * var
        else:
            mean = running_mean
            var = running_var

        # in-place: x = x - mean
        x -= mean

        # in-place: x = x / sqrt(var + eps)
        x /= np.sqrt(var + eps)

        # in-place: x = x * gamma + beta
        x *= gamma
        x += beta

        return (
            np.round(x, 4).tolist(),
            np.round(running_mean, 4).tolist(),
            np.round(running_var, 4).tolist()
        )