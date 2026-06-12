import torch
import torch.nn as nn
from torchtyping import TensorType

class Solution(nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        # Architecture: Linear(784, 512) -> ReLU -> Dropout(0.2) -> Linear(512, 10) -> Sigmoid
        self.ln_1 = nn.Linear(784, 512)
        self.act_1 = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.ln_2 = nn.Linear(512, 10)
        self.act_2 = nn.Sigmoid()

    def forward(self, images: TensorType[float]) -> TensorType[float]:
        torch.manual_seed(0)
        # images shape: (batch_size, 784)
        # Return the model's prediction to 4 decimal places

        h = self.ln_1(images)
        h = self.act_1(h)
        h = self.dropout(h)
        h = self.ln_2(h)
        logits = torch.round(self.act_2(h), decimals = 4)



        return logits
