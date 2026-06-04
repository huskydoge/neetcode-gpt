import numpy as np
from typing import List


class Solution:
    def rms_norm(self, x: List[float], gamma: List[float], eps: float) -> List[float]:
        # Implement RMS Normalization (similar to LayerNorm but without mean centering or beta)
        # Normalize x, then scale by gamma
        # Return result rounded to 4 decimal places as a list
        np_gamma = np.array(gamma, dtype="float")
        np_x = np.array(x,dtype="float")
        rms = np.sqrt((np.square(x).mean(axis=-1, keepdims=True) + eps))
        res = np_gamma * np_x / rms

        res = np.round(res, 4).tolist()

        return res
