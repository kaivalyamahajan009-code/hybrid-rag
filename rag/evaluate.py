import json
import os


def load_questions(path):
    """JSONL with id, question, relevant [{file, page}], answerable."""
    questions = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            questions.append(json.loads(line))
    return questions


def is_relevant(chunk, q):
    """True if the chunk covers a gold file + page from the question."""
    chunk_file = os.path.basename(chunk.doc).lower()
    for ref in q["relevant"]:
        if os.path.basename(ref["file"]).lower() != chunk_file:
            continue
        if chunk.page_start <= ref["page"] <= chunk.page_end:
            return True
    return False


def _hit_at_k(results, q, k):
    return any(is_relevant(r.chunk, q) for r in results[:k])


def _reciprocal_rank(results, q):
    for i, r in enumerate(results, start=1):
        if is_relevant(r.chunk, q):
            return 1.0 / i
    return 0.0


def _citation_score(answer, q):
    hits = 0
    for n in answer.cited:
        if is_relevant(answer.sources[n - 1].chunk, q):
            hits += 1
    return hits / len(answer.cited)


def _is_hallucination(answer, q):
    if answer.abstained:
        return False
    if not q["answerable"]:
        return True
    return not answer.grounded


def evaluate(pipeline, questions, mode, rerank, ks=(1, 3, 5)):
    """Score one retrieval setup. Recall/MRR use answerable questions only."""
    max_k = max(ks)
    recall_hits = {k: 0 for k in ks}
    n_answerable = 0
    mrr_sum = 0.0
    cite_sum = 0.0
    n_cited = 0
    n_hallu = 0
    n_unanswerable = 0
    n_abstain_ok = 0

    for q in questions:
        results = pipeline.search(q["question"], mode=mode, top_k=max_k, rerank=rerank)
        answer = pipeline.generator.answer(q["question"], results[:5])

        if q["answerable"]:
            n_answerable += 1
            for k in ks:
                if _hit_at_k(results, q, k):
                    recall_hits[k] += 1
            mrr_sum += _reciprocal_rank(results, q)
            if not answer.abstained and answer.cited:
                cite_sum += _citation_score(answer, q)
                n_cited += 1
        else:
            n_unanswerable += 1
            if answer.abstained:
                n_abstain_ok += 1

        if _is_hallucination(answer, q):
            n_hallu += 1

    metrics = {}
    for k in ks:
        metrics[f"recall@{k}"] = recall_hits[k] / n_answerable if n_answerable else 0.0
    metrics["mrr"] = mrr_sum / n_answerable if n_answerable else 0.0
    metrics["citation_accuracy"] = cite_sum / n_cited if n_cited else 0.0
    metrics["hallucination_rate"] = n_hallu / len(questions) if questions else 0.0
    metrics["abstention_accuracy"] = n_abstain_ok / n_unanswerable if n_unanswerable else 0.0
    return metrics


def compare(pipeline, questions):
    """vector-only, bm25-only, and hybrid + rerank on the same questions."""
    setups = [
        ("vector-only", "vector", False),
        ("bm25-only", "bm25", False),
        ("hybrid + rerank", "hybrid", True),
    ]
    return [(name, evaluate(pipeline, questions, mode, rerank)) for name, mode, rerank in setups]


def print_table(rows):
    """Aligned ASCII table; rows are (name, metrics) from compare()."""
    names = [name for name, _ in rows]
    keys = list(rows[0][1].keys())
    table = [["metric"] + names]
    for key in keys:
        table.append([key] + [f"{metrics[key]:.4f}" for _, metrics in rows])

    widths = [max(len(row[c]) for row in table) for c in range(len(table[0]))]
    for i, row in enumerate(table):
        print("  ".join(cell.ljust(widths[j]) for j, cell in enumerate(row)).rstrip())
        if i == 0:
            print("  ".join("-" * w for w in widths))
