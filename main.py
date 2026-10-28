"""Command line for ingest, ask, and evaluate."""

import argparse

from config import Config
from rag.evaluate import compare, load_questions, print_table
from rag.pipeline import RAGPipeline


def main():
    shared = argparse.ArgumentParser(add_help=False)
    shared.add_argument("--ner", help="module:function that loads an NER model")
    shared.add_argument("--llm", help="Ollama model name, instead of copying sentences")

    parser = argparse.ArgumentParser(description="Hybrid RAG over PDFs")
    sub = parser.add_subparsers(dest="cmd", required=True)

    ingest = sub.add_parser("ingest", parents=[shared], help="read PDFs and build the index")
    ingest.add_argument("paths", nargs="+", help="PDF files or folders")

    ask = sub.add_parser("ask", parents=[shared], help="answer one question")
    ask.add_argument("question")
    ask.add_argument("--mode", default="hybrid", choices=["hybrid", "vector", "bm25"])
    ask.add_argument("--no-rerank", action="store_true")
    ask.add_argument("--show-sources", action="store_true")

    evaluate = sub.add_parser("evaluate", parents=[shared], help="score a JSONL question file")
    evaluate.add_argument("questions")

    args = parser.parse_args()
    config = Config(ner=args.ner, llm_model=args.llm)

    if args.cmd == "ingest":
        RAGPipeline(config).ingest(args.paths)
        return

    pipe = RAGPipeline.load(config)
    if args.cmd == "ask":
        rerank = False if args.no_rerank else None
        show_answer(pipe.ask(args.question, mode=args.mode, rerank=rerank), args.show_sources)
        return

    print_table(compare(pipe, load_questions(args.questions)))


def show_answer(answer, show_sources):
    print(answer.text)
    if show_sources:
        rows = list(enumerate(answer.sources, start=1))
    else:
        rows = [(n, answer.sources[n - 1]) for n in answer.cited]
    if rows:
        print("\nSources:")
    for n, hit in rows:
        chunk = hit.chunk
        print(f"[{n}] {chunk.doc}, {chunk.pages()} ({chunk.section})")
        if show_sources:
            rerank = "-" if hit.rerank_score is None else f"{hit.rerank_score:.3f}"
            print(f"    vector rank {hit.vector_rank}, bm25 rank {hit.bm25_rank}, rerank {rerank}")
            print(f"    entities {chunk.entities}")
    if answer.unsupported:
        print("Warning: unsupported sentences:")
        for sentence in answer.unsupported:
            print(f"  {sentence}")


if __name__ == "__main__":
    main()
