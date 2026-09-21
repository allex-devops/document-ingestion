import fitz
import pytest

from docingest.layout import layout_text, read_layout


def make_pdf(path, items, width=595, height=842):
    """items: (x0, y0, x1, y1, text, fontsize, fontname)"""
    doc = fitz.open()
    page = doc.new_page(width=width, height=height)
    for x0, y0, x1, y1, text, size, font in items:
        page.insert_textbox(fitz.Rect(x0, y0, x1, y1), text, fontsize=size, fontname=font)
    doc.save(path)
    doc.close()


BODY = (
    "Retrieval systems split long documents into passages and score each passage against the query, "
    "then hand the best few to a language model that writes the final answer for the reader. "
)


def two_column(path):
    # paragraphs need several lines each: two short lines side by side get merged into one wide block
    # by the PDF reader, which is not what a real two-column paper looks like
    items = [(40, 40, 555, 80, "A Paper About Whales", 20, "hebo")]
    for i in (1, 2, 3):
        y = 60 + i * 200
        items.append((40, y, 280, y + 150, f"LEFT{i} " + BODY * 2, 10, "helv"))
        items.append((320, y, 555, y + 150, f"RIGHT{i} " + BODY * 2, 10, "helv"))
    make_pdf(path, items)


def test_two_column_page_is_read_down_each_column(tmp_path):
    two_column(tmp_path / "p.pdf")
    texts = [b.text for b in read_layout(tmp_path / "p.pdf")]
    assert texts[0] == "A Paper About Whales"
    assert [t.split()[0] for t in texts[1:]] == ["LEFT1", "LEFT2", "LEFT3", "RIGHT1", "RIGHT2", "RIGHT3"]


def test_reading_it_naively_top_to_bottom_would_have_interleaved(tmp_path):
    # guards the test above: if the columns weren't really side by side it would prove nothing
    two_column(tmp_path / "p.pdf")
    with fitz.open(tmp_path / "p.pdf") as doc:
        raw = sorted(doc[0].get_text("blocks"), key=lambda b: (round(b[1]), b[0]))
    assert [b[4].split()[0] for b in raw][:5] == ["A", "LEFT1", "RIGHT1", "LEFT2", "RIGHT2"]


def test_single_column_page_stays_top_to_bottom(tmp_path):
    items = [(50, 60 + i * 100, 540, 150 + i * 100, f"Paragraph {i} spans the whole width of the page.", 11, "helv") for i in range(4)]
    make_pdf(tmp_path / "p.pdf", items)
    assert [b.text.split()[1] for b in read_layout(tmp_path / "p.pdf")] == ["0", "1", "2", "3"]


def test_headings_are_picked_out_by_size_and_boldness(tmp_path):
    items = [
        (50, 40, 540, 80, "Introduction", 18, "helv"),
        (50, 100, 540, 180, "Body text that goes on for a while and is set in the normal size.", 10, "helv"),
        (50, 200, 540, 220, "Bold Label", 10, "hebo"),
        (50, 240, 540, 320, "More body text in the normal size that belongs to the section above.", 10, "helv"),
    ]
    make_pdf(tmp_path / "p.pdf", items)
    got = {b.text.split("\n")[0][:12]: b.heading for b in read_layout(tmp_path / "p.pdf")}
    assert got == {"Introduction": True, "Body text th": False, "Bold Label": True, "More body te": False}


def test_blocks_are_numbered_in_reading_order_across_pages(tmp_path):
    doc = fitz.open()
    for text in ("first page text here", "second page text here"):
        doc.new_page().insert_textbox(fitz.Rect(50, 50, 500, 100), text)
    doc.save(tmp_path / "p.pdf")
    doc.close()
    blocks = read_layout(tmp_path / "p.pdf")
    assert [(b.id, b.page) for b in blocks] == [(1, 1), (2, 2)]
    # same size and weight as everything else, so neither is a heading
    assert layout_text(blocks) == "[B1 p1] first page text here\n\n[B2 p2] second page text here"


def test_empty_pdf_gives_no_blocks(tmp_path):
    doc = fitz.open()
    doc.new_page()
    doc.save(tmp_path / "blank.pdf")
    doc.close()
    assert read_layout(tmp_path / "blank.pdf") == []
