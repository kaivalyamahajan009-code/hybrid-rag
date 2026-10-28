import numpy as np
import pymupdf

from config import Config
from rag.pipeline import RAGPipeline
from rag.utils import tokenize


class BowEmbedder:
    def __init__(self, model_name):
        self.dim = 64

    def encode(self, texts):
        vecs = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            for token in tokenize(text):
                vecs[i, hash(token) % self.dim] += 1.0
            norm = np.linalg.norm(vecs[i])
            if norm:
                vecs[i] /= norm
        return vecs


class Gazetteer:
    def extract(self, text):
        names = ("Acme Corp", "Northwind Analytics")
        return [(name, "ORGANIZATION") for name in names if name.lower() in text.lower()]


def write_pdf(path):
    doc = pymupdf.open()
    pages = [
        ("1 Overview", "Acme Corp stores spare pumps in the warehouse near the dock."),
        ("2 Results", "The river project in Oslo reported answer code 42 for Northwind Analytics."),
    ]
    for title, body in pages:
        page = doc.new_page()
        page.insert_text((72, 110), title, fontsize=18, fontname="hebo")
        page.insert_text((72, 150), body, fontsize=11)
    doc.save(path)
    doc.close()


def test_ingest_ask_save_and_filter(tmp_path, monkeypatch):
    monkeypatch.setattr("rag.pipeline.Embedder", BowEmbedder)
    pdf_dir = tmp_path / "pdfs"
    pdf_dir.mkdir()
    write_pdf(pdf_dir / "notes.pdf")
    config = Config(index_dir=str(tmp_path / "index"), use_reranker=False, min_filtered_chunks=1)
    pipe = RAGPipeline(config, ner=Gazetteer())
    pipe.ingest([pdf_dir])
    question = "What answer code did the river project in Oslo report?"
    answer = pipe.ask(question)
    chunk = answer.sources[answer.cited[0] - 1].chunk
    assert chunk.page_start == 2 and "42" in answer.text
    ranks = lambda p: [(r.chunk.id, r.score, r.vector_rank, r.bm25_rank) for r in p.search(question)]
    assert ranks(RAGPipeline.load(config, ner=Gazetteer())) == ranks(pipe)
    only = pipe.search("Where does Acme Corp store spare pumps?")
    assert [hit.chunk.page_start for hit in only] == [1]
    assert only[0].chunk.entities["ORGANIZATION"] == ["acme corp"]
