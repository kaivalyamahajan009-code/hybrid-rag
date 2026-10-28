import math

from rag.schema import SearchResult


class Reranker:
    """Cross-encoder reranker; stores a probability in rerank_score."""

    def __init__(self, model_name: str):
        from sentence_transformers import CrossEncoder

        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, results: list[SearchResult], top_k: int) -> list[SearchResult]:
        pairs = [(query, f"{r.chunk.section}: {r.chunk.text}") for r in results]
        logits = self.model.predict(pairs)
        for result, logit in zip(results, logits):
            result.rerank_score = 1 / (1 + math.exp(-float(logit)))
        results.sort(key=lambda r: r.rerank_score, reverse=True)
        return results[:top_k]
