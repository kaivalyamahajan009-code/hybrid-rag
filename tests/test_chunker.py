import pymupdf

from rag.chunker import chunk_sections
from rag.pdf_loader import load_pdf
from rag.schema import Section
from rag.utils import split_sentences


def _sentence(n, label):
    return " ".join([label] + ["word"] * (n - 1)) + "."


def _block(start, count, n=10):
    return " ".join(_sentence(n, f"S{i}") for i in range(start, start + count))


def test_pdf_sections_and_pages(tmp_path):
    path = tmp_path / "notes.pdf"
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((250, 824), "1", fontsize=10)
    page.insert_text((72, 110), "1 Overview", fontsize=18, fontname="hebo")
    page.insert_text((72, 150), "Alpha starts the overview on page one.", fontsize=11)
    page = doc.new_page()
    page.insert_text((250, 824), "2", fontsize=10)
    page.insert_text((72, 110), "Beta continues the overview on page two.", fontsize=11)
    page.insert_text((72, 200), "2 Methods", fontsize=18, fontname="hebo")
    page.insert_text((72, 240), "Gamma explains the method used here.", fontsize=11)
    doc.save(path)
    doc.close()

    sections = load_pdf(path)
    assert [section.title for section in sections] == ["1 Overview", "2 Methods"]
    assert [page for _, page in sections[0].paragraphs] == [1, 2]
    overview = " ".join(text for text, _ in sections[0].paragraphs)
    assert "Alpha starts" in overview and "Beta continues" in overview
    assert "Gamma" not in overview
    assert all(text not in {"1", "2"} for section in sections for text, _ in section.paragraphs)

    chunks = chunk_sections(sections, "notes.pdf")
    assert chunks[0].id == "notes_0" and chunks[0].doc == "notes.pdf"
    assert chunks[0].section == "1 Overview"
    assert chunks[0].page_start == 1 and chunks[0].page_end == 2
    methods = chunks[-1]
    assert methods.section == "2 Methods"
    assert methods.page_start == methods.page_end == 2
    assert "Gamma" in methods.text and "Alpha" not in methods.text


def test_chunks_stay_inside_section_size_and_pages():
    body = Section("Body", [(_block(0, 7), 1), (_block(7, 7), 4)])
    other = Section("Other", [(_block(20, 3), 9)])
    chunks = chunk_sections([body, other], "report.pdf", size=50, overlap=20)

    body_chunks = [chunk for chunk in chunks if chunk.section == "Body"]
    other_chunks = [chunk for chunk in chunks if chunk.section == "Other"]
    assert other_chunks[0].text.startswith("S20")
    assert other_chunks[0].page_start == other_chunks[0].page_end == 9
    assert all("S20" not in chunk.text for chunk in body_chunks)
    assert all(len(chunk.text.split()) <= 50 for chunk in chunks)
    assert [chunk.id for chunk in chunks] == [f"report_{i}" for i in range(len(chunks))]

    first, second = body_chunks[0], body_chunks[1]
    shared = " ".join(split_sentences(first.text)[-2:])
    assert second.text.startswith(shared)
    assert 0 < len(shared.split()) <= 20
    assert first.page_start == first.page_end == 1
    assert "S7 word" in second.text
    assert second.page_start == 1 and second.page_end == 4


def test_tiny_tail_merges_and_long_sentence_splits():
    merged = chunk_sections([Section("Only", [(_block(0, 6), 3)])], "a.pdf", size=50, overlap=20)
    assert len(merged) == 1
    assert "S0" in merged[0].text and "S5" in merged[0].text
    assert merged[0].page_start == merged[0].page_end == 3

    long = " ".join(["Start"] + ["word"] * 89) + "."
    parts = chunk_sections([Section("S", [(long, 2)])], "b.pdf", size=30, overlap=0)
    assert [len(chunk.text.split()) for chunk in parts] == [30, 30, 30]
    assert parts[0].page_start == parts[-1].page_end == 2
