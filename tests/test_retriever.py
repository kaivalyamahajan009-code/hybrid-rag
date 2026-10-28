from config import Config
from rag.retriever import HybridRetriever
from rag.schema import Chunk


class FakeEmbedder:
    def encode(self, texts):
        return [[0.0] for _ in texts]


class FakeIndex:
    """Returns chunk indexes in a fixed order, honouring the metadata filter."""

    def __init__(self, order):
        self.order = list(order)

    def search(self, query, k, allowed=None):
        hits = []
        for idx in self.order:
            if allowed is not None and idx not in allowed:
                continue
            hits.append((idx, 1.0))
            if len(hits) == k:
                break
        return hits


class FakeNER:
    def __init__(self, entities=None):
        self.entities = entities or []

    def extract(self, text):
        return list(self.entities)


class FakeReranker:
    def rerank(self, query, results, top_k):
        ordered = list(reversed(results))
        for i, result in enumerate(ordered):
            result.rerank_score = 1.0 - 0.1 * i
        return ordered[:top_k]


def chunk(i, org=None):
    entities = {"ORGANIZATION": [org]} if org else {}
    return Chunk(f"c{i}", "report.pdf", "Overview", 1, 1, f"text {i} {org or ''}", entities)


def build(chunks, vector_order, bm25_order, ner=None, reranker=None, **cfg):
    return HybridRetriever(
        chunks,
        FakeEmbedder(),
        FakeIndex(vector_order),
        FakeIndex(bm25_order),
        reranker,
        ner or FakeNER(),
        Config(**cfg),
    )


def test_rrf_promotes_chunk_found_by_both():
    # vector: c0, c1. bm25: c1, c2. Only c1 is in both lists.
    chunks = [chunk(i) for i in range(3)]
    ret = build(chunks, [0, 1], [1, 2])
    results = ret.search("revenue", rerank=False)
    assert [r.chunk.id for r in results] == ["c1", "c0", "c2"]
    assert results[0].vector_rank == 2
    assert results[0].bm25_rank == 1


def test_vector_and_bm25_modes():
    chunks = [chunk(i) for i in range(3)]
    ret = build(chunks, [0, 1], [1, 2])
    vector = ret.search("revenue", mode="vector", rerank=False)
    bm25 = ret.search("revenue", mode="bm25", rerank=False)
    assert [r.chunk.id for r in vector] == ["c0", "c1"]
    assert [r.chunk.id for r in bm25] == ["c1", "c2"]
    assert all(r.bm25_rank is None for r in vector)
    assert all(r.vector_rank is None for r in bm25)


def test_metadata_filter_keeps_query_entity():
    chunks = [
        chunk(0, "meridian"),
        chunk(1, "northwind"),
        chunk(2, "northwind logistics"),
        chunk(3, "northwind"),
    ]
    ner = FakeNER([("Northwind", "ORGANIZATION")])
    ret = build(chunks, [0, 1, 2, 3], [0, 1, 2, 3], ner=ner, min_filtered_chunks=3)
    results = ret.search("What about Northwind?", rerank=False)
    assert {r.chunk.id for r in results} == {"c1", "c2", "c3"}


def test_filter_skipped_when_too_few_match():
    chunks = [chunk(0, "meridian"), chunk(1, "northwind"), chunk(2, "acme")]
    ner = FakeNER([("Northwind", "ORGANIZATION")])
    ret = build(chunks, [0, 1, 2], [0, 1, 2], ner=ner, min_filtered_chunks=3)
    results = ret.search("What about Northwind?", rerank=False)
    assert [r.chunk.id for r in results] == ["c0", "c1", "c2"]


def test_reranker_reorders():
    chunks = [chunk(i) for i in range(3)]
    ret = build(chunks, [0, 1, 2], [0, 1, 2], reranker=FakeReranker())
    results = ret.search("revenue")
    assert [r.chunk.id for r in results] == ["c2", "c1", "c0"]
    assert results[0].rerank_score == 1.0
