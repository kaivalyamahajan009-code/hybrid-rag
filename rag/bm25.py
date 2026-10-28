import math
from collections import Counter

from rag.utils import tokenize


class BM25:
    """Okapi BM25 over tokenized chunk texts."""

    def __init__(self, texts: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        docs = [tokenize(t) for t in texts]
        self.n = len(docs)
        self.doc_len = [len(d) for d in docs]
        self.avgdl = sum(self.doc_len) / self.n if self.n else 0.0
        self.tf = [Counter(d) for d in docs]
        df: dict[str, int] = {}
        for tokens in docs:
            for term in set(tokens):
                df[term] = df.get(term, 0) + 1
        self.idf = {
            term: math.log((self.n - freq + 0.5) / (freq + 0.5) + 1.0)
            for term, freq in df.items()
        }

    def search(self, query: str, k: int, allowed: set[int] | None = None) -> list[tuple[int, float]]:
        q = tokenize(query)
        hits = []
        for i, tf in enumerate(self.tf):
            if allowed is not None and i not in allowed:
                continue
            score = 0.0
            for term in q:
                if term not in tf:
                    continue
                f = tf[term]
                denom = f + self.k1 * (1 - self.b + self.b * self.doc_len[i] / self.avgdl)
                score += self.idf[term] * (f * (self.k1 + 1)) / denom
            if score > 0:
                hits.append((i, score))
        hits.sort(key=lambda x: x[1], reverse=True)
        return hits[:k]
