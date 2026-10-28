import re

STOPWORDS = set(
    """a an and are as at be been but by can did do does for from had has have he her his
    how i if in into is it its me my of on or our she so than that the their them then there
    these they this those to was we were what when where which who whom why will with would
    you your""".split()
)


def tokenize(text: str) -> list[str]:
    text = re.sub(r"(?<=\d),(?=\d{3})", "", text.lower())  # 1,200 -> 1200
    return [t for t in re.findall(r"[a-z0-9]+(?:[.'-][a-z0-9]+)*", text) if t not in STOPWORDS]


def split_sentences(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"(])", text)
    # keep a trailing citation like "... grew 5%. [1]" with its sentence
    sentences = []
    for p in parts:
        m = re.match(r"^((?:\[\d+\]\s*)+)(.*)$", p)
        if m and sentences:
            sentences[-1] += " " + m.group(1).strip()
            p = m.group(2)
        if p.strip():
            sentences.append(p.strip())
    return sentences


def normalize(text: str) -> str:
    """Canonical form used to compare entity values ("Acme Corp." -> "acme corp")."""
    text = re.sub(r"(?<=\d),(?=\d{3})", "", text.lower())
    text = re.sub(r"[^\w\s$.%-]", " ", text)
    text = re.sub(r"\.(?!\d)", " ", text)
    return re.sub(r"\s+", " ", text).strip()
