import torch
import torch.nn as nn
from torchtyping import TensorType

class GroupedQueryAttention(nn.Module):
    def __init__(self, model_dim: int, num_heads: int, num_kv_heads: int):
        super().__init__()
        torch.manual_seed(0)
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = model_dim // num_heads

        self.q_proj = nn.Linear(model_dim, num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(model_dim, num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(model_dim, num_kv_heads * self.head_dim, bias=False)
        self.output_proj = nn.Linear(num_heads * self.head_dim, model_dim, bias=False)

    def forward(self, x: TensorType[float]) -> TensorType[float]:
        B, T, D = x.shape

        # 1. Project x into Q, K, V using the projection layers
        # 2. Reshape into heads: Q has num_heads, K and V have num_kv_heads
        # 3. Expand K, V by repeating each KV head (num_heads // num_kv_heads) times
        # 4. Compute scaled dot-product attention with causal mask
        # 5. Concatenate heads and apply output projection
        # 6. Return rounded output (decimals=4)
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        Q = Q.reshape(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.reshape(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2).repeat_interleave(self.num_heads // self.num_kv_heads, dim = 1)
        V = V.reshape(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2).repeat_interleave(self.num_heads // self.num_kv_heads, dim = 1) # (B, h, T, D)
        causal_mask = torch.tril(torch.ones((T, T)))

        scores = torch.softmax((Q @ K.transpose(-1, -2) / self.head_dim ** 0.5).masked_fill(causal_mask == 0, float("-inf")), dim = -1) # (B, h, T, T)

        o = scores @ V # (B, h, T, D)
        o = o.transpose(1, 2).reshape(B, T, -1)

        return torch.round(self.output_proj(o), decimals = 4)






