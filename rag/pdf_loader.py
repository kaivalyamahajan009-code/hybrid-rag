import re
from collections import Counter, defaultdict
from dataclasses import dataclass

import pymupdf

from rag.schema import Section

PAGE_NUMBER = re.compile(r"^(page\s+)?\d+(\s+of\s+\d+)?$", re.IGNORECASE)


@dataclass
class Line:
    text: str
    page: int
    size: float
    bold: bool
    block: int
    in_margin: bool
    heading: bool = False


def load_pdf(path) -> list[Section]:
    """Read a PDF into sections, using font size / bold text to spot the headings."""
    lines = []
    with pymupdf.open(path) as doc:
        for page in doc:
            lines += read_lines(page)
        n_pages = len(doc)
    lines = remove_headers_footers(lines, n_pages)
    mark_headings(lines)
    return group_into_sections(lines)


def read_lines(page) -> list[Line]:
    lines = []
    margin = page.rect.height * 0.08
    for b, block in enumerate(page.get_text("dict", sort=True)["blocks"]):
        if block["type"] != 0:  # image block
            continue
        for line in block["lines"]:
            spans = [s for s in line["spans"] if s["text"].strip()]
            if not spans:
                continue
            main = max(spans, key=lambda s: len(s["text"]))  # font of the longest span
            y = (line["bbox"][1] + line["bbox"][3]) / 2
            lines.append(Line(
                text="".join(s["text"] for s in spans).strip(),
                page=page.number + 1,
                size=round(main["size"], 1),
                bold=bool(main["flags"] & 16) or "bold" in main["font"].lower(),
                block=b,
                in_margin=y < margin or y > page.rect.height - margin,
            ))
    return lines


def remove_headers_footers(lines, n_pages):
    # a margin line that repeats on at least half the pages is a running header/footer
    def key(line):
        return re.sub(r"\d+", "#", line.text.lower())

    pages_seen = defaultdict(set)
    for line in lines:
        if line.in_margin:
            pages_seen[key(line)].add(line.page)
    repeated = {k for k, pages in pages_seen.items() if n_pages >= 3 and len(pages) >= n_pages / 2}

    return [
        line for line in lines
        if not (line.in_margin and (key(line) in repeated or PAGE_NUMBER.match(line.text)))
    ]


def mark_headings(lines):
    # the most common font size (counted by characters) is the body text
    sizes = Counter()
    for line in lines:
        sizes[line.size] += len(line.text)
    if not sizes:
        return
    body = sizes.most_common(1)[0][0]
    lines_per_block = Counter((line.page, line.block) for line in lines)

    for line in lines:
        short = len(line.text.split()) <= 12 and not line.text.endswith(".")
        has_letters = any(c.isalpha() for c in line.text)
        alone = lines_per_block[(line.page, line.block)] == 1
        line.heading = short and has_letters and (line.size >= body * 1.15 or (line.bold and alone))


def group_into_sections(lines) -> list[Section]:
    sections = [Section("Introduction")]  # for text before the first heading
    i = 0
    while i < len(lines):
        # consecutive lines from the same block form one paragraph (or one multi-line heading)
        j = i + 1
        while (j < len(lines) and lines[j].page == lines[i].page
               and lines[j].block == lines[i].block and lines[j].heading == lines[i].heading):
            j += 1
        text = join_lines(lines[i:j])
        if lines[i].heading:
            sections.append(Section(text))
        else:
            sections[-1].paragraphs.append((text, lines[i].page))
        i = j
    return [s for s in sections if s.paragraphs]


def join_lines(lines) -> str:
    text = "\n".join(line.text for line in lines)
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)  # re-join hyphenated words
    return re.sub(r"\s+", " ", text).strip()
