import faiss
import numpy as np


class VectorStore:
    """FAISS inner-product index; row i is chunk i."""

    def __init__(self, dim: int):
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)

    def add(self, vectors: np.ndarray) -> None:
        self.index.add(np.ascontiguousarray(vectors, dtype=np.float32))

    def search(self, query_vec, k: int, allowed: set[int] | None = None) -> list[tuple[int, float]]:
        query = np.ascontiguousarray(query_vec, dtype=np.float32).reshape(1, -1)
        n = self.index.ntotal
        search_k = n if allowed is not None else min(k, n)
        scores, idxs = self.index.search(query, search_k)
        hits = []
        for score, idx in zip(scores[0], idxs[0]):
            idx = int(idx)
            if idx < 0:
                continue
            if allowed is not None and idx not in allowed:
                continue
            hits.append((idx, float(score)))
            if len(hits) >= k:
                break
        return hits

    def save(self, path: str) -> None:
        faiss.write_index(self.index, str(path))

    @staticmethod
    def load(path: str) -> "VectorStore":
        index = faiss.read_index(str(path))
        store = VectorStore(index.d)
        store.index = index
        return store
