"""Calls a real chat and embedding model, so it uses a few API tokens.

Needs LLM_BASE_URL, LLM_API_KEY, LLM_MODEL and EMBED_MODEL (see .env.example). The OCR test also needs a
model that accepts images: set VISION_MODEL if your chat model doesn't.

    uv run --env-file .env pytest -m live -s
"""
import os

import fitz
import pytest
from conftest import scanned_pdf
from ragchat.rag import ConfigError, llm_from_env

from docingest.fields import INVOICE, extract_fields
from docingest.ocr import ocr_tesseract, ocr_vlm

pytestmark = pytest.mark.live


@pytest.fixture(scope="module")
def llm():
    try:
        return llm_from_env()
    except ConfigError as e:
        pytest.skip(str(e))


def make_invoice(path):
    doc = fitz.open()
    page = doc.new_page()
    for x, y, text, size, font in [
        (40, 40, "ACME Supplies Ltd", 18, "hebo"),
        (40, 100, "Invoice No: INV-2024-0042", 11, "helv"),
        (40, 130, "Date: 15 March 2024", 11, "helv"),
        (40, 200, "Widgets x 10 ....... 1,000.00", 11, "helv"),
        (40, 230, "Delivery ....... 250.00", 11, "helv"),
        (40, 300, "Total due: $1,250.00", 11, "helv"),
    ]:
        page.insert_textbox(fitz.Rect(x, y, 500, y + 40), text, fontsize=size, fontname=font)
    doc.save(path)
    doc.close()


def test_both_ocr_engines_read_a_scan(tmp_path, llm):
    scanned_pdf(tmp_path / "scan.pdf", "INVOICE TOTAL 450 DUE FRIDAY")
    with fitz.open(tmp_path / "scan.pdf") as doc:
        png = doc[0].get_pixmap(dpi=200).tobytes("png")
    tess = ocr_tesseract(png)
    vlm = ocr_vlm(png, llm, os.environ.get("VISION_MODEL"))
    print(f"\ntesseract: {tess.strip()!r}\nvision model: {vlm.strip()!r}")
    for text in (tess, vlm):
        assert "INVOICE" in text.upper() and "450" in text


def test_real_model_extracts_invoice_fields_with_evidence(tmp_path, llm):
    make_invoice(tmp_path / "inv.pdf")
    got = {r.name: r for r in extract_fields(tmp_path / "inv.pdf", INVOICE, llm)}
    for r in got.values():
        print(f"{r.name:16} {r.status:12} p.{r.page} {r.value!r}")
    assert {r.status for r in got.values()} == {"found"}
    assert got["invoice_number"].value == "INV-2024-0042"
    assert "1,250.00" in got["total"].value


def test_scan_to_answer_with_citation(tmp_path, llm):
    """A scanned page and a real paper in one folder: sync, then ask about the scan."""
    import json
    import shutil

    from ragchat.rag import answer, embedder
    from ragchat.store import Store
    from shared.corpus import pdf_path
    from shared.paths import DATA_DIR

    from docingest.pipeline import Ingester

    questions = DATA_DIR / "evalset" / "questions.jsonl"
    if not questions.exists():
        pytest.skip("question set not generated yet")
    paper = json.loads(questions.read_text().splitlines()[0])["paper_id"]

    docs = tmp_path / "docs"
    docs.mkdir()
    shutil.copy(pdf_path(paper), docs / "paper.pdf")
    scanned_pdf(docs / "memo.pdf", "The maintenance window is every Tuesday at 3 pm.", fontsize=22)

    embed = embedder(llm)
    store = Store(tmp_path / "idx", embed_model=llm.embed_model)
    report = Ingester(store, embed, tmp_path / "state.json").sync(docs)
    got = answer("When is the maintenance window?", store, llm, embed, min_score=0.6)
    print(f"\nsync: {report.added}, {report.ocr_pages} page(s) by OCR, {report.chunks} chunks")
    print(f"answer: {got.text!r}\nsources: {[(s.source, s.page, round(s.score, 2)) for s in got.sources]}")
    assert report.ocr_pages == 1 and not report.failed
    assert got.grounded and "tuesday" in got.text.lower()
    assert any(s.source == "memo.pdf" and s.page == 1 for s in got.sources)
