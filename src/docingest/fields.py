import json
import re
from dataclasses import dataclass
from pathlib import Path

from .layout import Block, layout_text, read_layout


@dataclass
class FieldSpec:
    name: str
    description: str


@dataclass
class Extracted:
    name: str
    value: str | None
    status: str  # "found", "missing" (model said it isn't there) or "unsupported" (value isn't in the document)
    page: int | None = None
    bbox: tuple[float, float, float, float] | None = None
    block_id: int | None = None


INVOICE = [
    FieldSpec("vendor", "The company issuing the invoice"),
    FieldSpec("invoice_number", "The invoice number or id"),
    FieldSpec("invoice_date", "The invoice date, as written"),
    FieldSpec("total", "The total amount due, as written"),
]

PROMPT = """Extract the requested fields from the document below.

The document is split into blocks labelled like [B7 p1]. For each field give the exact text as it appears
in the document and the number of the block it came from. If a field is not in the document, use null.
Never guess or reformat values.

Reply with JSON only: {{"fields": {{"<name>": {{"value": "<text or null>", "block": <block number or null>}}}}}}

Fields:
{fields}

Document:
{document}"""


def normalise(text: str) -> str:
    # so "$1,250.00" matches "1250.00" and line breaks or spacing don't matter
    return re.sub(r"[\s,$€£]", "", text.lower())


def parse_reply(reply: str) -> dict:
    match = re.search(r"\{.*\}", reply, re.S)
    if not match:
        raise ValueError("the model did not return JSON")
    try:
        return json.loads(match.group(0)).get("fields", {})
    except json.JSONDecodeError as e:
        raise ValueError(f"the model returned broken JSON: {e}")


def locate(value: str, cited: int | None, blocks: list[Block]) -> Block | None:
    """The block that really contains `value`, trying the one the model cited first."""
    want = normalise(value)
    if not want:
        return None
    by_id = {b.id: b for b in blocks}
    ordered = ([by_id[cited]] if cited in by_id else []) + [b for b in blocks if b.id != cited]
    return next((b for b in ordered if want in normalise(b.text)), None)


def extract_fields(pdf: Path, specs: list[FieldSpec], llm, model: str | None = None) -> list[Extracted]:
    blocks = read_layout(pdf)
    prompt = PROMPT.format(
        fields="\n".join(f"- {s.name}: {s.description}" for s in specs), document=layout_text(blocks)
    )
    kw = {"model": model} if model else {}
    reply = llm.chat([{"role": "user", "content": prompt}], temperature=0, response_format={"type": "json_object"}, **kw)
    got = parse_reply(reply.get("content") or "")

    out = []
    for spec in specs:
        entry = got.get(spec.name) or {}
        value = entry.get("value")
        if value in (None, ""):
            out.append(Extracted(spec.name, None, "missing"))
            continue
        cited = entry.get("block") if isinstance(entry.get("block"), int) else None
        block = locate(str(value), cited, blocks)
        if block is None:
            # the model made it up or reformatted it; better to flag than to hand it on as fact
            out.append(Extracted(spec.name, str(value), "unsupported"))
        else:
            out.append(Extracted(spec.name, str(value), "found", block.page, block.bbox, block.id))
    return out
