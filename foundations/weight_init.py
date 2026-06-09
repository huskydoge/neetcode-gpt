import torch
import torch.nn as nn
import math
from typing import List


class Solution:

    def xavier_init(self, fan_in: int, fan_out: int) -> List[List[float]]:
        # Return a (fan_out x fan_in) weight matrix using Xavier/Glorot normal initialization
        # Use torch.manual_seed(0) for reproducibility
        # Round to 4 decimal places and return as nested list
        torch.manual_seed(0)
        std = math.sqrt(2 / (fan_in + fan_out))
        weights = torch.empty((fan_out, fan_in)).normal_(0, std)
        return weights.round(decimals=4).tolist()

    def kaiming_init(self, fan_in: int, fan_out: int) -> List[List[float]]:
        # Return a (fan_out x fan_in) weight matrix using Kaiming/He normal initialization (for ReLU)
        # Use torch.manual_seed(0) for reproducibility
        # Round to 4 decimal places and return as nested list
        torch.manual_seed(0)
        std = math.sqrt(2 / (fan_in))
        weights = torch.empty((fan_out, fan_in)).normal_(0, std)
        return weights.round(decimals=4).tolist()

    def check_activations(self, num_layers: int, input_dim: int, hidden_dim: int, init_type: str) -> List[float]:
        torch.manual_seed(0)

        weights = []

        for i in range(num_layers):
            fan_in = input_dim if i == 0 else hidden_dim
            fan_out = hidden_dim

            if init_type == "xavier":
                std = math.sqrt(2 / (fan_in + fan_out))
            elif init_type == "kaiming":
                std = math.sqrt(2 / fan_in)
            elif init_type == "random":
                std = 1

            W = torch.randn(fan_out, fan_in) * std
            weights.append(W)

        x = torch.randn(input_dim)

        stds = []
        for W in weights:
            x = torch.relu(W @ x)
            stds.append(round(x.std().item(), 2))

        return stds