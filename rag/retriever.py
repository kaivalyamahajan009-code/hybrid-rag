"""Hybrid retrieval: entity filter, vector search, BM25, fusion, rerank."""

from rag.schema import SearchResult
from rag.utils import normalize


class HybridRetriever:
    """FAISS + BM25, with an NER metadata filter and optional reranking."""

    def __init__(self, chunks, embedder, vector_store, bm25, reranker, ner, config):
        self.chunks = chunks
        self.embedder = embedder
        self.vector_store = vector_store
        self.bm25 = bm25
        self.reranker = reranker
        self.ner = ner
        self.config = config

    def search(self, query: str, mode: str = "hybrid", top_k: int | None = None, rerank: bool | None = None):
        top_k = self.config.top_k if top_k is None else top_k
        allowed = self._allowed(query)
        vector_hits = []
        bm25_hits = []
        if mode in ("hybrid", "vector"):
            query_vec = self.embedder.encode([query])[0]
            vector_hits = self.vector_store.search(query_vec, self.config.top_k_vector, allowed)
        if mode in ("hybrid", "bm25"):
            bm25_hits = self.bm25.search(query, self.config.top_k_bm25, allowed)
        fused = _fuse(self.chunks, vector_hits, bm25_hits, self.config.rrf_k)
        do_rerank = self.config.use_reranker if rerank is None else rerank
        if do_rerank and self.reranker is not None and fused:
            window = fused[: self.config.rerank_candidates]
            return self.reranker.rerank(query, window, top_k)
        return fused[:top_k]

    def _allowed(self, query):
        if not self.config.use_metadata_filter:
            return None
        wanted = _by_label(self.ner.extract(query))
        if not wanted:
            return None
        matched = {i for i, chunk in enumerate(self.chunks) if _matches(chunk, wanted)}
        # a filter that keeps almost nothing is usually a bad NER hit, so search everything
        if len(matched) < self.config.min_filtered_chunks:
            return None
        return matched


def _by_label(entities):
    grouped = {}
    for text, label in entities:
        value = normalize(text)
        if not value:
            continue
        bucket = grouped.setdefault(label, [])
        if value not in bucket:
            bucket.append(value)
    return grouped


def _matches(chunk, wanted):
    # every label must hit; two values under one label are alternatives
    for label, values in wanted.items():
        stored = [normalize(value) for value in chunk.entities.get(label, [])]
        if not any(_contains(have, need) for need in values for have in stored):
            return False
    return True


def _contains(stored, query_value):
    # whole words only, so "acme" matches "acme corp" but "york" doesn't match "yorkshire"
    return f" {query_value} " in f" {stored} "


def _fuse(chunks, vector_hits, bm25_hits, rrf_k):
    scores = {}
    vector_rank = {}
    bm25_rank = {}
    _add_ranks(scores, vector_rank, vector_hits, rrf_k)
    _add_ranks(scores, bm25_rank, bm25_hits, rrf_k)
    order = sorted(scores, key=lambda i: (-scores[i], i))
    return [
        SearchResult(
            chunk=chunks[i],
            score=scores[i],
            vector_rank=vector_rank.get(i),
            bm25_rank=bm25_rank.get(i),
        )
        for i in order
    ]


def _add_ranks(scores, ranks, hits, rrf_k):
    for rank, (idx, _score) in enumerate(hits, start=1):
        ranks[idx] = rank
        # 1/(k + rank) from each list, so a chunk found by both searches floats up
        scores[idx] = scores.get(idx, 0.0) + 1.0 / (rrf_k + rank)
