"""Hook for the NER model.

The BiLSTM/BiGRU NER is a separate project. Anything with an
`extract(text) -> list[(entity_text, label)]` method can be plugged in, either by
passing it to RAGPipeline or with `--ner my_module:load_model` on the command line.
Until then NoNER is used and metadata filtering simply does nothing.
"""

import importlib

LABELS = ["PERSON", "ORGANIZATION", "DATE", "LOCATION", "MONEY"]


class NoNER:
    def extract(self, text: str) -> list[tuple[str, str]]:
        return []


def load_ner(spec: str | None):
    if not spec:
        return NoNER()
    module_name, func_name = spec.split(":")
    return getattr(importlib.import_module(module_name), func_name)()


def bio_to_entities(tokens: list[str], tags: list[str]) -> list[tuple[str, str]]:
    """Turn BIO tags from the NER model into (text, label) pairs.

    ["Acme", "Corp", "paid", "$5"], ["B-ORGANIZATION", "I-ORGANIZATION", "O", "B-MONEY"]
    -> [("Acme Corp", "ORGANIZATION"), ("$5", "MONEY")]
    """
    entities = []
    current, label = [], None
    for token, tag in zip(tokens, tags):
        if tag.startswith("I-") and label == tag[2:]:
            current.append(token)
            continue
        if current:
            entities.append((" ".join(current), label))
        current, label = ([token], tag[2:]) if tag != "O" else ([], None)
    if current:
        entities.append((" ".join(current), label))
    return entities
