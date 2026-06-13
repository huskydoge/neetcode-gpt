import numpy as np
from numpy.typing import NDArray


class Solution:
    def get_positional_encoding(self, seq_len: int, d_model: int) -> NDArray[np.float64]:
        # PE(pos, 2i)   = sin(pos / 10000^(2i / d_model))
        # PE(pos, 2i+1) = cos(pos / 10000^(2i / d_model))
        #
        # Hint: Use np.arange() to create position and dimension index vectors,
        # then compute all values at once with broadcasting (no loops needed).
        # Assign sine to even columns (PE[:, 0::2]) and cosine to odd columns (PE[:, 1::2]).
        # Round to 5 decimal places.

        pos_indices = np.arange(seq_len)
        d_indices = np.arange(d_model // 2)

        sin_values = np.sin(pos_indices[:, None] / 10000**(2 * d_indices / d_model))
        cos_values = np.cos(pos_indices[:, None] / 10000**(2 * d_indices / d_model))

        PE = np.zeros((seq_len, d_model))

        PE[:, 0::2] = sin_values
        PE[:, 1::2] = cos_values

        return np.round(PE, 5)
