import json

from rag.evaluate import compare, evaluate, load_questions
from rag.schema import Answer, Chunk, SearchResult


def _chunk(cid, doc, lo, hi):
    return Chunk(cid, doc, "Intro", lo, hi, "text")


C1 = _chunk("c1", "report.pdf", 1, 2)
C2 = _chunk("c2", "report.pdf", 3, 4)
C3 = _chunk("c3", "report.pdf", 5, 5)
C4 = _chunk("c4", "other.pdf", 1, 1)

Q1 = {"id": "q1", "question": "What is revenue?", "relevant": [{"file": "report.pdf", "page": 2}], "answerable": True}
Q2 = {"id": "q2", "question": "Where is HQ?", "relevant": [{"file": "report.pdf", "page": 5}], "answerable": True}
Q3 = {"id": "q3", "question": "Zorblax?", "relevant": [], "answerable": False}
Q4 = {"id": "q4", "question": "Balloons?", "relevant": [], "answerable": False}
Q5 = {"id": "q5", "question": "Missing page?", "relevant": [{"file": "report.pdf", "page": 10}], "answerable": True}


class FakePipeline:
    def __init__(self, retrieved, answers):
        self.retrieved = retrieved
        self.generator = self
        self._answers = answers
        self.calls = []

    def search(self, question, mode="hybrid", top_k=5, rerank=False):
        self.calls.append((mode, rerank))
        return self.retrieved[question][:top_k]

    def answer(self, question, results):
        return self._answers[question]


def _pipeline():
    sr = lambda c: SearchResult(c, 1.0)
    retrieved = {
        Q1["question"]: [sr(C2), sr(C1), sr(C3)],
        Q2["question"]: [sr(C3), sr(C4)],
        Q3["question"]: [sr(C2)],
        Q4["question"]: [sr(C1)],
        Q5["question"]: [sr(C2), sr(C4)],
    }
    answers = {
        Q1["question"]: Answer(Q1["question"], "Revenue grew. [1][2]", [sr(C2), sr(C1)], cited=[1, 2], grounded=True),
        Q2["question"]: Answer(Q2["question"], "HQ is Chicago. [1]", [sr(C3)], cited=[1], grounded=True),
        Q3["question"]: Answer(Q3["question"], "Not enough evidence.", [sr(C2)], abstained=True),
        Q4["question"]: Answer(Q4["question"], "Twelve balloons.", [sr(C1)], cited=[1], grounded=False),
        Q5["question"]: Answer(Q5["question"], "Something invented.", [sr(C2)], grounded=False),
    }
    return FakePipeline(retrieved, answers)


def test_metrics_by_hand():
    # Q1 hit@2, cites 1/2; Q2 hit@1, cite ok; Q5 miss + ungrounded
    # Q3 abstains; Q4 answers an unanswerable question
    m = evaluate(_pipeline(), [Q1, Q2, Q3, Q4, Q5], mode="hybrid", rerank=True)
    assert m["recall@1"] == 1 / 3
    assert m["recall@3"] == 2 / 3
    assert m["recall@5"] == 2 / 3
    assert m["mrr"] == 0.5
    assert m["citation_accuracy"] == 0.75
    assert m["hallucination_rate"] == 0.4
    assert m["abstention_accuracy"] == 0.5


def test_compare_runs_three_modes():
    pipe = _pipeline()
    rows = compare(pipe, [Q1, Q2, Q3, Q4, Q5])
    assert [name for name, _ in rows] == ["vector-only", "bm25-only", "hybrid + rerank"]
    assert pipe.calls.count(("vector", False)) == 5
    assert pipe.calls.count(("bm25", False)) == 5
    assert pipe.calls.count(("hybrid", True)) == 5


def test_load_questions(tmp_path):
    path = tmp_path / "q.jsonl"
    path.write_text(json.dumps(Q1) + "\n\n" + json.dumps(Q3) + "\n")
    loaded = load_questions(path)
    assert len(loaded) == 2
    assert loaded[0]["id"] == "q1"
    assert loaded[0]["relevant"] == [{"file": "report.pdf", "page": 2}]
    assert loaded[1]["answerable"] is False
