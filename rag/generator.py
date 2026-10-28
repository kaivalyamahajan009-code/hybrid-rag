import math
import re

import requests

from rag.schema import Answer
from rag.utils import split_sentences, tokenize

_ABSTAIN = "I couldn't find enough evidence in the documents to answer this."
_CITE = re.compile(r"\[(\d{1,3})\]")
_NUMBER = re.compile(r"\d+(?:\.\d+)?")
_WORD = re.compile(r"[A-Za-z0-9][\w.'&-]*")


class AnswerGenerator:
    """Extractive sentences, or an Ollama model when config.llm_model is set."""

    def __init__(self, config):
        self.config = config

    def answer(self, question: str, results: list) -> Answer:
        if not _enough(question, results, self.config):
            return _abstain(question, results)
        if self.config.llm_model:
            draft = _ollama(question, results, self.config)
        else:
            draft = _extractive(question, results)
        if not draft or "NOT_FOUND" in draft:
            return _abstain(question, results)
        text, cited, unsupported = _ground(draft, results, self.config.min_sentence_support)
        if not text:
            return _abstain(question, results)
        return Answer(
            question=question,
            text=text,
            sources=results,
            cited=cited,
            grounded=bool(cited) and not unsupported,
            unsupported=unsupported,
        )


def _abstain(question, results):
    return Answer(question=question, text=_ABSTAIN, sources=results, abstained=True)


def _enough(question, results, config):
    if not results:
        return False
    if results[0].rerank_score is not None:
        return results[0].rerank_score >= config.min_rerank_score
    query = tokenize(question)
    if not query:
        return False
    best = 0.0
    for result in results[:3]:
        present = set(tokenize(result.chunk.text))
        best = max(best, sum(token in present for token in query) / len(query))
    return best >= config.min_query_coverage


def _ollama(question, results, config):
    blocks = [f"[{n}] ({hit.chunk.doc}, {hit.chunk.pages()})\n{_strip(hit.chunk.text)}"
              for n, hit in enumerate(results, start=1)]
    messages = [
        {"role": "system", "content": (
            "Answer only from the numbered sources and cite them with [n]. "
            "If the answer is not there, reply NOT_FOUND."
        )},
        {"role": "user", "content": "\n\n".join(blocks) + "\n\nQuestion: " + question},
    ]
    body = {"model": config.llm_model, "messages": messages, "stream": False}
    return requests.post(config.ollama_url, json=body).json()["message"]["content"]


def _extractive(question, results):
    weights = []
    for rank, result in enumerate(results, start=1):
        # rerank scores are ~1.0 for most on-topic chunks, so the rank is used as well
        confidence = 1.0 if result.rerank_score is None else result.rerank_score
        weights.append(confidence / rank)
    rows, seen = [], set()
    for src, result in enumerate(results):
        for pos, sentence in enumerate(split_sentences(result.chunk.text)):
            # drop "[3]" already printed in the PDF before we add our own citation
            text = _strip(sentence)
            if not text or text.casefold() in seen:
                continue
            seen.add(text.casefold())
            rows.append((src, pos, text))
    if not rows:
        return ""
    token_sets = [set(tokenize(text)) for _, _, text in rows]
    df = {}
    for tokens in token_sets:
        for token in tokens:
            df[token] = df.get(token, 0) + 1
    n = len(rows)
    idf = {token: math.log((n + 1) / (count + 1)) + 1.0 for token, count in df.items()}
    query = set(tokenize(question))
    scored = []
    for (src, pos, text), tokens in zip(rows, token_sets):
        overlap = sum(idf[token] for token in query & tokens)
        if overlap:
            scored.append((overlap * weights[src], src, pos, text))
    if not scored:
        return ""
    scored.sort(key=lambda item: (-item[0], item[1], item[2]))
    best_score, best_src, _, best_text = scored[0]
    # best sentence, plus at most one more that scores at least 60% of it
    floor = best_score * 0.6
    # a sentence from another chunk must mention the same names/numbers, otherwise a
    # question about Northwind's revenue could also get Meridian's revenue appended
    needed = _anchors(question) & set(tokenize(best_text))
    chosen = []
    for score, src, pos, text in scored:
        if len(chosen) >= 2 or score < floor:
            break
        if src == best_src or needed <= set(tokenize(text)):
            chosen.append((src, pos, text))
    chosen.sort()
    return " ".join(_cite(text, [src + 1]) for src, _, text in chosen)


def _anchors(query):
    words = _WORD.findall(query)
    picked = [w for i, w in enumerate(words) if any(c.isdigit() for c in w) or (i > 0 and w[0].isupper())]
    return set(tokenize(" ".join(picked)))


def _strip(text):
    text = _CITE.sub(" ", text)
    text = re.sub(r"\s+([.,;:!?])", r"\1", text)
    return re.sub(r"\s+", " ", text).strip()


def _cite(sentence, markers):
    text = sentence.strip()
    if not markers:
        return text
    tags = " ".join(f"[{n}]" for n in markers)
    if text and text[-1] in ".!?":
        return f"{text[:-1]} {tags}{text[-1]}"
    return f"{text} {tags}".strip()


def _ground(text, results, min_support):
    cited, seen, unsupported, parts = [], set(), [], []
    n_sources = len(results)
    for sentence in split_sentences(text):
        markers = [int(m) for m in _CITE.findall(sentence)]
        valid = []
        for marker in markers:
            if not 1 <= marker <= n_sources or marker in valid:
                continue
            valid.append(marker)
            if marker not in seen:
                seen.add(marker)
                cited.append(marker)
        body = _strip(sentence)
        if _bad_sentence(body, markers, valid, results, min_support):
            unsupported.append(sentence.strip())
        parts.append(_cite(body, valid))
    return " ".join(part for part in parts if part), cited, unsupported


def _figures(text):
    return set(_NUMBER.findall(re.sub(r"(?<=\d),(?=\d{3})", "", text)))


def _bad_sentence(body, markers, valid, results, min_support):
    tokens = tokenize(body)
    if len(tokens) < 3:
        return False
    # uncited sentences can be backed by any source; a citation like [9] that points nowhere backs nothing
    pool = list(results) if not markers else [results[m - 1] for m in valid]
    if not pool:
        return True
    wanted = set(tokens)
    best, numbers = 0.0, set()
    for result in pool:
        source = _strip(result.chunk.text)
        have = set(tokenize(source))
        best = max(best, len(wanted & have) / len(wanted))
        numbers.update(_figures(source))
    return best < min_support or not _figures(body) <= numbers
