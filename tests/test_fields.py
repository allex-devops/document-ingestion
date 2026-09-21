import json

import fitz
import pytest

from docingest.fields import INVOICE, FieldSpec, extract_fields, normalise, parse_reply
from docingest.layout import read_layout


@pytest.fixture
def invoice(tmp_path):
    doc = fitz.open()
    page = doc.new_page()
    lines = [
        (40, 40, "ACME Supplies Ltd", 18, "hebo"),
        (40, 100, "Invoice No: INV-2024-0042", 11, "helv"),
        (40, 130, "Date: 15 March 2024", 11, "helv"),
        (40, 200, "Widgets x 10 ....... 1,000.00", 11, "helv"),
        (40, 230, "Delivery ....... 250.00", 11, "helv"),
        (40, 300, "Total due: $1,250.00", 11, "helv"),
    ]
    for x, y, text, size, font in lines:
        page.insert_textbox(fitz.Rect(x, y, 500, y + 40), text, fontsize=size, fontname=font)
    doc.save(tmp_path / "inv.pdf")
    doc.close()
    return tmp_path / "inv.pdf"


def block_id(pdf, needle):
    return next(b.id for b in read_layout(pdf) if needle in b.text)


class ScriptedLLM:
    def __init__(self, fields):
        self.fields, self.calls = fields, []

    def chat(self, messages, **kw):
        self.calls.append((messages, kw))
        return {"content": json.dumps({"fields": self.fields})}


def entry(value, block):
    return {"value": value, "block": block}


def by_name(results):
    return {r.name: r for r in results}


def test_found_fields_come_back_with_page_and_position(invoice):
    llm = ScriptedLLM({
        "vendor": entry("ACME Supplies Ltd", block_id(invoice, "ACME")),
        "invoice_number": entry("INV-2024-0042", block_id(invoice, "INV-2024")),
        "invoice_date": entry("15 March 2024", block_id(invoice, "15 March")),
        "total": entry("$1,250.00", block_id(invoice, "Total due")),
    })
    got = by_name(extract_fields(invoice, INVOICE, llm))
    assert {r.status for r in got.values()} == {"found"}
    total = got["total"]
    assert total.page == 1 and total.block_id == block_id(invoice, "Total due")
    x0, y0, x1, y1 = total.bbox
    assert 30 < x0 < 60 and 290 < y0 < 320  # where the line was drawn, not just any box


def test_a_value_that_is_not_in_the_document_is_flagged_not_trusted(invoice):
    llm = ScriptedLLM({"invoice_number": entry("INV-9999-0001", block_id(invoice, "INV-2024"))})
    r = by_name(extract_fields(invoice, [FieldSpec("invoice_number", "the id")], llm))["invoice_number"]
    assert (r.status, r.value, r.page, r.bbox) == ("unsupported", "INV-9999-0001", None, None)


def test_a_wrong_block_citation_is_corrected_when_the_value_really_exists(invoice):
    llm = ScriptedLLM({"total": entry("$1,250.00", block_id(invoice, "ACME"))})
    r = extract_fields(invoice, [FieldSpec("total", "the total")], llm)[0]
    assert r.status == "found" and r.block_id == block_id(invoice, "Total due")


def test_a_missing_block_number_still_finds_the_value(invoice):
    llm = ScriptedLLM({"total": entry("$1,250.00", None)})
    assert extract_fields(invoice, [FieldSpec("total", "the total")], llm)[0].status == "found"


def test_number_formatting_differences_do_not_cause_false_alarms(invoice):
    llm = ScriptedLLM({"total": entry("1250.00", None)})
    assert extract_fields(invoice, [FieldSpec("total", "the total")], llm)[0].status == "found"


def test_null_and_absent_fields_are_missing_not_unsupported(invoice):
    llm = ScriptedLLM({"vendor": entry(None, None)})  # "purchase_order" isn't in the reply at all
    specs = [FieldSpec("vendor", "v"), FieldSpec("purchase_order", "po")]
    got = by_name(extract_fields(invoice, specs, llm))
    assert got["vendor"].status == got["purchase_order"].status == "missing"


def test_the_prompt_carries_labelled_blocks_and_asks_for_json(invoice):
    llm = ScriptedLLM({})
    extract_fields(invoice, INVOICE, llm)
    messages, kw = llm.calls[0]
    text = messages[0]["content"]
    assert "[B1 p1" in text and "INV-2024-0042" in text and "- total:" in text
    assert kw["temperature"] == 0 and kw["response_format"] == {"type": "json_object"}


def test_a_reply_that_is_not_json_is_an_error_not_a_silent_empty_result(invoice):
    class Chatty:
        def chat(self, messages, **kw):
            return {"content": "Sure! The total is $1,250."}

    with pytest.raises(ValueError, match="did not return JSON"):
        extract_fields(invoice, INVOICE, Chatty())


def test_parse_reply_copes_with_a_code_fence_and_rejects_broken_json():
    assert parse_reply('```json\n{"fields": {"a": {"value": "x"}}}\n```') == {"a": {"value": "x"}}
    with pytest.raises(ValueError, match="broken JSON"):
        parse_reply('{"fields": {oops}}')  # has braces, so it gets as far as the JSON parser


def test_normalise_ignores_case_spacing_commas_and_currency_signs():
    assert normalise("$1,250.00") == normalise("1250.00") == "1250.00"
    assert normalise("Acme  Supplies\nLtd") == normalise("acme supplies ltd")
