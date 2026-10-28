from dataclasses import dataclass


@dataclass
class Config:
    index_dir: str = "index"

    # models
    embed_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # chunking (in words)
    chunk_size: int = 220
    chunk_overlap: int = 40

    # retrieval
    top_k_vector: int = 30
    top_k_bm25: int = 30
    rrf_k: int = 60
    rerank_candidates: int = 20
    top_k: int = 5
    use_reranker: bool = True
    use_metadata_filter: bool = True
    min_filtered_chunks: int = 3  # fall back to no filter if fewer chunks match the entities

    # grounding
    min_rerank_score: float = 0.2  # abstain if the best chunk scores below this
    min_query_coverage: float = 0.5  # same idea when the reranker is off
    min_sentence_support: float = 0.5  # token overlap an answer sentence needs with its source

    # generation: None = extractive answers (no LLM), or an Ollama model name like "llama3.1"
    llm_model: str | None = None
    ollama_url: str = "http://localhost:11434/api/chat"

    # NER: "module:function" that returns an object with extract(text)
    ner: str | None = None
