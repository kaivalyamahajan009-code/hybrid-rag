from pathlib import Path

from rag.schema import Chunk
from rag.utils import split_sentences


def chunk_sections(sections, doc_name, size=220, overlap=40) -> list[Chunk]:
    """Split each section into ~`size`-word chunks. A chunk never crosses a section."""
    name, stem = Path(doc_name).name, Path(doc_name).stem
    chunks = []
    for section in sections:
        sentences = []  # (sentence, page)
        for text, page in section.paragraphs:
            for sentence in split_sentences(text):
                words = sentence.split()
                # tables and lists can give one huge "sentence", so cut those into pieces
                for i in range(0, len(words), size):
                    sentences.append((" ".join(words[i:i + size]), page))

        for group in pack(sentences, size, overlap):
            pages = [page for _, page in group]
            chunks.append(Chunk(
                id=f"{stem}_{len(chunks)}",
                doc=name,
                section=section.title,
                page_start=min(pages),
                page_end=max(pages),
                text=" ".join(s for s, _ in group),
            ))
    return chunks


def pack(sentences, size, overlap):
    groups, current, new = [], [], []
    for sentence in sentences:
        if current and word_count(current + [sentence]) > size:
            groups.append(current)
            # start the next chunk with the last few sentences of this one
            current = last_sentences(current, overlap)
            while current and word_count(current + [sentence]) > size:
                current = current[1:]
            new = []
        current.append(sentence)
        new.append(sentence)

    if groups and word_count(new) < 25:
        groups[-1] += new  # tiny leftover: add it to the previous chunk instead
    elif current:
        groups.append(current)
    return groups


def last_sentences(group, max_words):
    tail = []
    for sentence in reversed(group):
        if word_count(tail + [sentence]) > max_words:
            break
        tail.insert(0, sentence)
    return tail


def word_count(group) -> int:
    return sum(len(s.split()) for s, _ in group)
