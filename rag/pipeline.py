import json
from pathlib import Path

from rag.bm25 import BM25
from rag.chunker import chunk_sections
from rag.embedder import Embedder
from rag.generator import AnswerGenerator
from rag.ner import LABELS, load_ner
from rag.pdf_loader import load_pdf
from rag.reranker import Reranker
from rag.retriever import HybridRetriever
from rag.schema import Chunk
from rag.utils import normalize
from rag.vector_store import VectorStore


class RAGPipeline:
    """Load models, index PDFs, and answer questions."""

    def __init__(self, config, ner=None):
        self.config = config
        self.embedder = Embedder(config.embed_model)
        self.reranker = Reranker(config.rerank_model) if config.use_reranker else None
        self.ner = ner or load_ner(config.ner)
        self.generator = AnswerGenerator(config)
        self.chunks = []

    def ingest(self, paths):
        self.chunks = []
        for path in pdf_paths(paths):
            chunks = chunk_sections(
                load_pdf(path), path.name, self.config.chunk_size, self.config.chunk_overlap,
            )
            for chunk in chunks:
                chunk.entities = entities_of(self.ner, chunk.text)
            self.chunks.extend(chunks)
            print(f"{path.name}: {len(chunks)} chunks")
        self._index(embed=True)
        self.save()

    def _index(self, embed):
        # section title is indexed too, so a heading can match the query
        texts = [f"{c.section}: {c.text}" for c in self.chunks]
        if embed:
            self.vector_store = VectorStore(self.embedder.dim)
            self.vector_store.add(self.embedder.encode(texts))
        self.retriever = HybridRetriever(
            self.chunks, self.embedder, self.vector_store, BM25(texts),
            self.reranker, self.ner, self.config,
        )

    def search(self, question, mode="hybrid", top_k=None, rerank=None):
        return self.retriever.search(question, mode=mode, top_k=top_k, rerank=rerank)

    def ask(self, question, mode="hybrid", rerank=None):
        return self.generator.answer(question, self.search(question, mode=mode, rerank=rerank))

    def save(self):
        folder = Path(self.config.index_dir)
        folder.mkdir(parents=True, exist_ok=True)
        with open(folder / "chunks.json", "w") as f:
            json.dump([chunk.to_dict() for chunk in self.chunks], f)
        self.vector_store.save(folder / "faiss.index")

    @classmethod
    def load(cls, config, ner=None):
        pipe = cls(config, ner=ner)
        folder = Path(config.index_dir)
        with open(folder / "chunks.json") as f:
            pipe.chunks = [Chunk.from_dict(row) for row in json.load(f)]
        pipe.vector_store = VectorStore.load(folder / "faiss.index")
        pipe._index(embed=False)  # faiss was loaded; BM25 is rebuilt from the chunks
        return pipe


def pdf_paths(paths):
    files = []
    for raw in paths:
        path = Path(raw)
        files.extend(sorted(path.glob("*.pdf")) if path.is_dir() else [path])
    return files


def entities_of(ner, text):
    grouped = {}
    for value, label in ner.extract(text):
        norm = normalize(value)
        if label in LABELS and norm:
            grouped.setdefault(label, set()).add(norm)
    return {label: sorted(values) for label, values in grouped.items()}
