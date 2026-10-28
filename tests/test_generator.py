import rag.generator as generator
from config import Config
from rag.generator import AnswerGenerator
from rag.schema import Chunk, SearchResult

ABSTAIN = "I couldn't find enough evidence in the documents to answer this."


def hit(text, rerank=None, n=0):
    chunk = Chunk(f"c{n}", "report.pdf", "Results", 2, 2, text)
    return SearchResult(chunk, 1.0, rerank_score=rerank)


def stub_ollama(monkeypatch, content, captured=None):
    def post(url, json):
        if captured is not None:
            captured.append(json)

        class Resp:
            def json(self):
                return {"message": {"content": content}}

        return Resp()

    monkeypatch.setattr(generator.requests, "post", post)


def test_abstain_on_low_rerank_score():
    gen = AnswerGenerator(Config())
    answer = gen.answer(
        "What was Northwind revenue in 2024?",
        [hit("Northwind revenue was 12 million in 2024.", rerank=0.05)],
    )
    assert answer.abstained
    assert answer.text == ABSTAIN


def test_abstain_on_low_query_coverage():
    gen = AnswerGenerator(Config())
    answer = gen.answer(
        "What was Northwind revenue in 2024?",
        [hit("The garden club meets on Tuesdays.")],
    )
    assert answer.abstained
    assert answer.text == ABSTAIN
    assert answer.cited == []


def test_extractive_cites_the_right_source():
    gen = AnswerGenerator(Config())
    results = [
        hit("Northwind revenue was 12 million in 2024.", rerank=0.9, n=0),
        hit("The office kitchen was repainted last spring.", rerank=0.4, n=1),
    ]
    answer = gen.answer("What was Northwind revenue in 2024?", results)
    assert not answer.abstained
    assert answer.grounded
    assert answer.cited == [1]
    assert "[1]" in answer.text
    assert "[2]" not in answer.text
    assert "12" in answer.text


def test_distractor_company_not_appended():
    # rank 2 has the higher rerank score, so the two sentences score close together
    gen = AnswerGenerator(Config())
    results = [
        hit("Northwind revenue reached 12 million during 2024.", rerank=0.5, n=0),
        hit("Meridian Logistics revenue reached 12 million during 2024.", rerank=1.0, n=1),
    ]
    answer = gen.answer("What was Northwind revenue during 2024?", results)
    assert "Northwind" in answer.text
    assert "Meridian" not in answer.text
    assert answer.cited == [1]


def test_pdf_citation_marker_is_not_ours():
    # three sources, so a leftover "[3]" would be a real citation of the last chunk
    gen = AnswerGenerator(Config())
    results = [
        hit("Northwind revenue was 12 million in 2024 [3].", rerank=0.9, n=0),
        hit("The office kitchen was repainted last spring.", rerank=0.4, n=1),
        hit("Meridian Logistics opened a warehouse in Ohio.", rerank=0.4, n=2),
    ]
    answer = gen.answer("What was Northwind revenue in 2024?", results)
    assert answer.cited == [1]
    assert "[1]" in answer.text
    assert "[3]" not in answer.text
    assert answer.grounded


def test_grounding_flags_bad_number_and_citation(monkeypatch):
    stub_ollama(
        monkeypatch,
        "Revenue was 99 million dollars [1]. "
        "Northwind revenue was 12 million dollars overall [9].",
    )
    gen = AnswerGenerator(Config(llm_model="tiny"))
    results = [hit("Northwind revenue was 12 million dollars overall.", rerank=0.9)]
    answer = gen.answer("What was Northwind revenue?", results)
    assert not answer.abstained
    assert not answer.grounded
    assert any("99" in sentence for sentence in answer.unsupported)
    assert any("[9]" in sentence for sentence in answer.unsupported)
    assert "[9]" not in answer.text
    assert answer.cited == [1]


def test_not_found_abstains(monkeypatch):
    captured = []
    stub_ollama(monkeypatch, "NOT_FOUND", captured)
    gen = AnswerGenerator(Config(llm_model="tiny"))
    answer = gen.answer(
        "What was Northwind revenue?",
        [hit("Northwind revenue was 12 million.", rerank=0.9)],
    )
    assert answer.abstained and answer.text == ABSTAIN
    payload = captured[0]
    assert payload["model"] == "tiny" and payload["stream"] is False and payload["messages"]
