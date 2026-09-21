import re
import unicodedata
from collections import Counter


def strip_repeated_lines(pages: list[str], min_share: float = 0.5) -> list[str]:
    """Remove running headers and footers: short lines that show up on at least half the pages."""
    if len(pages) < 3:
        return pages  # with two pages "repeated" just means "both", which is too weak to trust
    seen = Counter()
    for text in pages:
        seen.update({line.strip() for line in text.splitlines() if 0 < len(line.strip()) < 80})
    repeated = {line for line, n in seen.items() if n / len(pages) >= min_share}
    out = []
    for text in pages:
        kept = [
            line
            for line in text.splitlines()
            if line.strip() not in repeated and not re.fullmatch(r"\s*(page\s+)?\d{1,4}(\s+of\s+\d+)?\s*", line, re.I)
        ]
        out.append("\n".join(kept))
    return out


def clean_page(text: str) -> str:
    # NFKC turns ligatures like "ﬁ" into "fi" and non-breaking spaces into plain ones
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    # rejoin words split across lines ("retriev-\nal"), but only when the next bit is lowercase.
    # "state-of-\nthe-art" survives that way; the price is a few splits we miss
    text = re.sub(r"([a-z])-\n([a-z])", r"\1\2", text)
    # hard-wrapped lines become one paragraph, but a line ending in punctuation stays a break
    text = re.sub(r"(?<![.!?:;\n])\n(?!\n)", " ", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()
