from rag.bm25 import BM25


def test_rare_term_ranks_first():
    texts = [
        "revenue grew last year across every product line",
        "the xylophone shipment arrived late last year",
        "product line revenue was steady last year",
    ]
    hits = BM25(texts).search("xylophone last year", k=3)
    assert hits[0][0] == 1
    assert hits[0][1] > hits[1][1]


def test_allowed_restricts_results():
    texts = [
        "alpha token appears here",
        "alpha token appears again",
        "alpha token also here",
    ]
    hits = BM25(texts).search("alpha token", k=5, allowed={0, 2})
    assert {i for i, _ in hits} == {0, 2}
    assert all(score > 0 for _, score in hits)


def test_zero_score_docs_not_returned():
    texts = [
        "cats sit on mats",
        "unrelated vegetables and fruit",
        "more cats on more mats",
    ]
    hits = BM25(texts).search("cats mats", k=10)
    assert [i for i, _ in hits] == [0, 2]
    assert all(score > 0 for _, score in hits)
