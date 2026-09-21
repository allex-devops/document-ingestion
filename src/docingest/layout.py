from dataclasses import dataclass, replace
from pathlib import Path
from statistics import median

import fitz  # pymupdf

FULL_WIDTH = 0.6  # blocks wider than this share of the page span the columns (titles, wide figures)


@dataclass
class Block:
    id: int
    page: int
    bbox: tuple[float, float, float, float]
    text: str
    size: float
    bold: bool
    heading: bool = False


def page_blocks(page: fitz.Page, page_no: int) -> list[Block]:
    out = []
    for raw in page.get_text("dict")["blocks"]:
        if raw.get("type") != 0:  # 1 is an image
            continue
        lines, sizes, bold = [], [], False
        for line in raw["lines"]:
            spans = [s for s in line["spans"] if s["text"].strip()]
            if spans:
                lines.append(" ".join(s["text"].strip() for s in spans))
                sizes += [s["size"] for s in spans]
                bold = bold or any(s["flags"] & 16 or "bold" in s["font"].lower() for s in spans)
        if lines:
            out.append(Block(0, page_no, tuple(raw["bbox"]), "\n".join(lines), max(sizes), bold))
    return out


def reading_order(blocks: list[Block], page_width: float) -> list[Block]:
    """Top to bottom, except that a two-column page is read down the left column, then the right."""
    by_position = sorted(blocks, key=lambda b: (b.bbox[1], b.bbox[0]))
    narrow = [b for b in blocks if b.bbox[2] - b.bbox[0] < FULL_WIDTH * page_width]
    left = sorted((b for b in narrow if (b.bbox[0] + b.bbox[2]) / 2 < page_width / 2), key=lambda b: b.bbox[1])
    right = sorted((b for b in narrow if (b.bbox[0] + b.bbox[2]) / 2 >= page_width / 2), key=lambda b: b.bbox[1])

    # a real two-column page has at least two blocks per side with a gap between them; anything else is
    # just short blocks on a normal page, so plain top-to-bottom is right
    two_columns = len(left) >= 2 and len(right) >= 2 and max(b.bbox[2] for b in left) <= min(b.bbox[0] for b in right)
    if not two_columns:
        return by_position

    top = min(b.bbox[1] for b in narrow)
    wide = [b for b in by_position if b not in narrow]
    # wide blocks above the columns go first, ones between or below them go last. That suits footnotes and
    # captions, but a wide figure in the middle of the page ends up in the wrong place
    return [b for b in wide if b.bbox[1] < top] + left + right + [b for b in wide if b.bbox[1] >= top]


def read_layout(path: Path) -> list[Block]:
    """Every text block in the document in reading order, numbered, with headings marked."""
    ordered: list[Block] = []
    with fitz.open(path) as doc:
        for i, page in enumerate(doc):
            ordered += reading_order(page_blocks(page, i + 1), page.rect.width)
    if not ordered:
        return []
    body = median(b.size for b in ordered)
    return [
        replace(
            b,
            id=n,
            heading=b.size >= 1.15 * body or (b.bold and len(b.text) < 80 and "\n" not in b.text),
        )
        for n, b in enumerate(ordered, 1)
    ]


def layout_text(blocks: list[Block]) -> str:
    """The document as text a model can read, with each block labelled so it can point back at one."""
    return "\n\n".join(f"[B{b.id} p{b.page}{' heading' if b.heading else ''}] {b.text}" for b in blocks)
