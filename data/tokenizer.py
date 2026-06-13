from typing import List


class Solution:
    def get_merges(self, corpus: str, num_merges: int) -> List[List[str]]:
        # 1. Split corpus into a list of individual characters
        # 2. For each merge step:
        #    a. Count frequency of all adjacent token pairs
        #    b. Find the most frequent pair (break ties lexicographically)
        #    c. Merge all non-overlapping occurrences left to right
        #    d. Record the merge as [token_a, token_b]
        # 3. Return the list of merges performed

        chars = list(corpus)
        print(chars)

        merge_performed = []

        for merge in range(num_merges):
            token_pairs = {}
            print(f"merge: {merge} === \n")
            for i in range(len(chars) - 1):
                adj_pair = (chars[i], chars[i + 1])
                if adj_pair in token_pairs:
                    token_pairs[adj_pair] += 1
                else:
                    token_pairs[adj_pair] = 1

            # find 
            print(token_pairs)

            best_pair = min(
                token_pairs.keys(),
                key=lambda pair: (-token_pairs[pair], pair)
            )

            print(best_pair)
            merge_performed.append(best_pair)

            # start merge
            res = []
            i = 0
            while i < (len(chars) - 1):
                adj_pair = (chars[i], chars[i + 1])

                if adj_pair == best_pair:
                    res.append((best_pair[0] + best_pair[1]))
                    i += 2
                else:
                    res.append(adj_pair[0])
                    i += 1

            if i == len(chars) - 1:
                res.append(chars[i])
            print(res)
            chars = res

        return merge_performed


