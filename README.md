# Hybrid RAG over PDFs

I built this as a placement-interview project. You put PDFs in a folder, the script indexes them, and you ask questions. The answer quotes a sentence from the documents and cites the file and page. If the pages do not contain the answer, it says it could not find enough evidence.

What it does:

- Answers questions from PDFs, with section-wise chunks and a citation that names the page.
- Uses PyMuPDF to read the text, the page numbers, and the headings.
- Embeds chunks with a sentence-transformer and searches them in FAISS.
- Has a hook for a BiLSTM/BiGRU NER model (people, organizations, dates, locations, money). The model is a separate project; this repo only stores what it returns.
- Retrieves with FAISS and BM25 together, and can drop chunks that do not match the NER tags on the question.
- Reranks that list with a cross-encoder, then checks the answer against the source text.
- Reports Recall@K, MRR, citation accuracy, and hallucination rate, next to a vector-only baseline.

## Pipeline

```
PDF -> load_pdf (headings, page numbers, repeated headers/footers removed)
    -> chunk_sections (one section, ~220 words, a little overlap)
    -> NER entities stored on the chunk
    -> FAISS on "Section: text", and BM25 on the same string
question -> metadata filter -> vector + BM25 -> RRF -> cross-encoder -> top 5
    -> quote the best sentence, or ask Ollama, and keep it only when a source supports it
```

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
.\.venv\Scripts\python -m pytest -q -p no:cacheprovider
```

The first `ingest` downloads `all-MiniLM-L6-v2` and `ms-marco-MiniLM-L-6-v2` if they are not cached. The tests fake the embedder, so they do not download a model.

## Usage

Eight sample reports are in `data/pdfs`. The questions are in `data/eval_questions.jsonl`.

```powershell
.\.venv\Scripts\python main.py ingest data/pdfs
.\.venv\Scripts\python main.py ask "What was the consolidated revenue of Northwind Analytics for the fiscal year ended 31 December 2024?" --show-sources
.\.venv\Scripts\python main.py ask "What is the melting point of ytterbium on the planet Zorblax?"
.\.venv\Scripts\python main.py evaluate data/eval_questions.jsonl
```

`ask` takes `--mode hybrid|vector|bm25` and `--no-rerank`. `--show-sources` prints every retrieved chunk with its vector rank, BM25 rank, rerank score, and entities. A normal ask only prints cited lines, like `[1] northwind_annual_report_2024.pdf, p. 2 (2 Financial Results)`. `--llm llama3.1` calls a local Ollama model.

## How each part works

- **Parsing.** PyMuPDF gives me every line with its font size, boldness and page. The most common font size is the body text; a short line that's noticeably bigger (or bold on its own) is a heading. Lines repeated in the top/bottom margin on most pages are running headers/footers, so I drop them along with page numbers.
- **Chunking.** Chunks never cross a section, so each one has a clean section title and page range for citations. I pack whole sentences up to ~220 words and repeat the last ~40 words in the next chunk so a fact on the boundary isn't lost.
- **Embeddings + FAISS.** `all-MiniLM-L6-v2` embeds `"section: text"`. The vectors are normalized, so FAISS inner product is cosine similarity.
- **BM25.** I wrote a small Okapi BM25 over the same text. It catches exact things like product codes and contract numbers that embeddings tend to blur.
- **Metadata filtering.** NER runs on every chunk at ingest and on the question at query time. If the question mentions e.g. an organization, only chunks tagged with it are searched. If fewer than 3 chunks match I skip the filter, since that usually means a bad NER hit.
- **Fusion.** Reciprocal rank fusion: each chunk gets `1 / (60 + rank)` from the vector list and from the BM25 list. Being near the top of both beats being first in only one.
- **Reranking.** A cross-encoder (`ms-marco-MiniLM-L-6-v2`) reads the question and each of the top 20 chunks together and rescores them. It's slower but much more accurate, and its score (0-1) is what I use to decide whether there's enough evidence.
- **Grounding.** If the best rerank score is below 0.2 the system says it couldn't find enough evidence instead of guessing. Otherwise it quotes the best-matching sentence(s) with a `[n]` citation (or asks Ollama with `--llm`). Every answer sentence is then checked against its cited source: enough word overlap, and every number in it must appear in the source. Anything that fails is flagged.
- **Evaluation.** Each question has gold `{file, page}` pairs, or `answerable: false`. Recall@K/MRR check whether a gold page shows up in the results, citation accuracy checks the pages the answer actually cites, and hallucination rate counts answers to unanswerable questions plus answers that fail the grounding check.

## Results

I ran `python main.py evaluate data/eval_questions.jsonl` on the eight PDFs: 34 questions with a gold page, and 6 that should be refused. No NER model is loaded, so the metadata filter is idle. Hybrid here is vectors, BM25, and the cross-encoder.

```
metric               vector-only  bm25-only  hybrid + rerank
-------------------  -----------  ---------  ---------------
recall@1             0.8824       0.8235     1.0000
recall@3             1.0000       0.9706     1.0000
recall@5             1.0000       1.0000     1.0000
mrr                  0.9314       0.8995     1.0000
citation_accuracy    0.9545       0.8485     1.0000
hallucination_rate   0.0000       0.0000     0.0000
abstention_accuracy  1.0000       1.0000     1.0000
```

Vector-only Recall@1 is 0.88 and MRR is 0.93. BM25 citation accuracy is 0.85 because several reports repeat the same sentences with a different company or year. Hybrid plus the reranker is 1.00 on Recall@1, MRR, and citation accuracy, and all six unanswerable questions abstain, so the hallucination rate is 0. The revenue question quotes "$48,600,000" from `northwind_annual_report_2024.pdf`, page 2, section "2 Financial Results". The ytterbium question abstains.

## Plugging in the NER model

`extract(text)` should return `(text, label)` pairs. `bio_to_entities` builds those from BIO tags. Save this as `my_ner.py` in the repo root:

```python
from rag.ner import bio_to_entities

class MyNER:
    def predict(self, tokens):
        return ["O"] * len(tokens)  # replace with the BiLSTM / BiGRU

    def extract(self, text):
        tokens = text.split()
        return bio_to_entities(tokens, self.predict(tokens))

def load_model():
    return MyNER()
```

Tags are stored during ingest, and loading an old index does not retag the chunks. Run ingest again after the model is ready, and pass the same `--ner` when you ask.

```powershell
.\.venv\Scripts\python main.py ingest data/pdfs --ner my_ner:load_model
.\.venv\Scripts\python main.py ask "Where is Northwind Analytics headquartered?" --ner my_ner:load_model
```

## Project structure

```
main.py            ingest, ask, evaluate
config.py          models, chunk size, thresholds
rag/pdf_loader.py  headings and pages     rag/chunker.py    section-wise chunks
rag/ner.py         BIO hook               rag/embedder.py   sentence-transformer
rag/vector_store.py  FAISS                rag/bm25.py       keyword index
rag/retriever.py   filter, fusion         rag/reranker.py   cross-encoder
rag/generator.py   answer and grounding   rag/pipeline.py   ingest, ask, save, load
rag/evaluate.py    metrics                rag/schema.py     Chunk, Answer
data/pdfs/         eight sample reports
data/eval_questions.jsonl
tests/             pytest, embedder faked
```

## What I'd improve next

- Plug in the real BiLSTM NER and see if the metadata filter separates the near-duplicate reports.
- Write a short answer when the fact is split across two chunks. The default writer only quotes a sentence.
- Handle tables and two-column pages. PyMuPDF's line order gets messy there.
- Rerank fewer than 20 chunks. That step is what makes `ask` slow.
- Hold out a few questions before moving the 0.2 rerank cutoff, so this table is not the set I tuned on.
