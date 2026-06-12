import torch
import torch.nn as nn
from torchtyping import TensorType
from typing import List

class Solution:
    def get_dataset(self, positive: List[str], negative: List[str]) -> TensorType[float]:
        # 1. Build vocabulary: collect all unique words, sort them, assign integer IDs starting at 1
        # 2. Encode each sentence by replacing words with their IDs
        # 3. Combine positive + negative into one list of tensors
        # 4. Pad shorter sequences with 0s using nn.utils.rnn.pad_sequence(tensors, batch_first=True)
        vocab = set()
        for sentence in positive + negative:
            vocab.update(sentence.split())

        vocab = sorted(vocab)

        token_dict = {}

        for i, token in enumerate(vocab):
            token_dict[token] = i + 1

        res = []

        for sentence in positive + negative:
            res.append(torch.tensor([token_dict[token] for token in sentence.split()]))

        padded = nn.utils.rnn.pad_sequence(res, batch_first=True)

        return padded

