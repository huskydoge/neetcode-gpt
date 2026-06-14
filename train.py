import torch
import torch.nn as nn
import torch.nn.functional as F

# The GPT model is provided for you. It returns raw logits (not probabilities).
# You only need to implement the training loop below.

class Solution:
    def train(self, model: nn.Module, data: torch.Tensor, epochs: int, context_length: int, batch_size: int, lr: float) -> float:
        # Train the GPT model using AdamW and cross_entropy loss.
        # For each epoch: seed with torch.manual_seed(epoch),
        # sample batches from data, run forward/backward, update weights.
        # Return the final loss rounded to 4 decimals.
        optim = torch.optim.AdamW(model.parameters(), lr = lr)
        for ep in range(epochs):
            torch.manual_seed(ep)

            indices = torch.randint(0, len(data) - context_length, (batch_size, ))
            offsets = torch.arange(context_length)
            ids = indices[:, None] + offsets[None, :]

            X = data[ids]
            Y = data[ids + 1]

            logits = model(X)
            model_dim = logits.shape[-1]

            loss = F.cross_entropy(logits.reshape(-1,model_dim), Y.reshape(-1))

            optim.zero_grad()
            loss.backward()

            optim.step()
        
        return round(loss.item(), 4)


