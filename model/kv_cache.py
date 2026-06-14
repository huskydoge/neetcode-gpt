import torch
import torch.nn as nn
from typing import Tuple, Optional

class KVCache:
    def __init__(self):
        self.cache_k: Optional[torch.Tensor] = None  # (batch, seq_len, model_dim)
        self.cache_v: Optional[torch.Tensor] = None

    def update(self, new_k: torch.Tensor, new_v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Append new_k and new_v to the cache along the sequence dimension (dim=1).
        # On the first call, initialize the cache with the given tensors.
        # Return the full (cached) K and V tensors.
        if self.cache_k is None:
            self.cache_k = new_k
        else:
            self.cache_k = torch.cat((self.cache_k, new_k), dim = 1)
        if self.cache_v is None:
            self.cache_v = new_v
        else:
            self.cache_v = torch.cat((self.cache_v, new_v), dim = 1)

        return self.cache_k, self.cache_v
        

    def clear(self):
        self.cache_k = None
        self.cache_v = None

class CachedAttention(nn.Module):
    def __init__(self, model_dim: int):
        super().__init__()
        torch.manual_seed(0)
        self.q_proj = nn.Linear(model_dim, model_dim, bias=False)
        self.k_proj = nn.Linear(model_dim, model_dim, bias=False)
        self.v_proj = nn.Linear(model_dim, model_dim, bias=False)

    def forward(self, x: torch.Tensor, kv_cache: Optional[KVCache] = None) -> Tuple[torch.Tensor, KVCache]:
        # 1. Project x into Q, K, V using the linear layers
        # 2. If kv_cache is None, create a new KVCache
        # 3. Update the cache with the new K and V
        # 4. Compute scaled dot-product attention using Q and the full cached K, V
        # 5. Return (rounded output, kv_cache)
        print(x.shape)
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)
        model_dim = x.shape[-1]

        if kv_cache is None:
            kv_cache = KVCache()
        
        kv_cache.update(K, V)
        print(Q.shape)
        print(kv_cache.cache_k.shape)
        scores = torch.softmax(Q @ kv_cache.cache_k.transpose(-1, -2) / (model_dim) ** 0.5, dim = -1)
        print(scores.shape)
        print(kv_cache.cache_v.shape)
        output = scores @ kv_cache.cache_v

        return (output, kv_cache)

