from typing import List, Dict

class Solution:
    def tokenize_numbers(self, numbers: List[int], vocab: Dict[str, int]) -> List[List[str]]:
        # Tokenize each number using greedy left-to-right longest match.
        # Return a list of token lists showing how each number gets split.

        res = []

        for number in numbers:
            i = 0
            s = str(number)
            split = []
            cur = 0
            while i < len(s):
                for j in range(i+1, len(s) + 1):
                    if s[i:j] in vocab:
                        cur = j
                    else:
                        continue
                if cur > i:
                    split.append(s[i:cur])
                i = cur
            res.append(split)
        
        return res



    def count_tokens(self, text: str, vocab: Dict[str, int]) -> int:
        # Count how many tokens the text uses with greedy tokenization.
        # Use greedy left-to-right longest match.
        i = 0
        split = []
        s = text
        while i < len(s):
            cur = 0
            for j in range(i+1, len(s) + 1):
                if s[i:j] in vocab:
                    cur = j
                else:
                    continue
            if cur > i:
                split.append(s[i:cur])
            i = cur

        return len(split)
        

    def fertility_score(self, text: str, vocab: Dict[str, int]) -> float:
        # Compute tokens-per-word ratio (fertility).
        # Higher = more expensive and less efficient.
        # Round to 4 decimal places.
        tokens = self.count_tokens(text, vocab)
        words = len(text.split(" "))

        return round(tokens / words, 4)
