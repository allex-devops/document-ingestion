import re
import shutil
import zlib

import fitz
import numpy as np
import pytest
from ragchat.store import Store

DIM = 256


def fake_embed(texts, kind="document"):
    """Hashed bag of words: texts that share words land close together."""
    out = np.zeros((len(texts), DIM), dtype=np.float32)
    for row, text in enumerate(texts):
        for word in re.findall(r"[a-z]+", text.lower()):
            out[row, zlib.crc32(word.encode()) % DIM] += 1
    norms = np.linalg.norm(out, axis=1, keepdims=True)
    return out / np.where(norms == 0, 1, norms)


class CountingEmbed:
    def __init__(self):
        self.calls = 0

    def __call__(self, texts, kind="document"):
        self.calls += 1
        return fake_embed(texts, kind)


def text_pdf(path, pages, fontsize=11):
    doc = fitz.open()
    for text in pages:
        doc.new_page().insert_textbox(fitz.Rect(50, 50, 550, 780), text, fontsize=fontsize)
    doc.save(path)
    doc.close()


def scanned_pdf(path, text, fontsize=28):
    """A PDF whose only content is a picture of text, like a scan: no text layer to extract."""
    src = fitz.open()
    src.new_page().insert_textbox(fitz.Rect(50, 50, 550, 780), text, fontsize=fontsize)
    png = src[0].get_pixmap(dpi=200).tobytes("png")
    out = fitz.open()
    out.new_page().insert_image(fitz.Rect(0, 0, 595, 842), stream=png)
    out.save(path)
    out.close()


needs_tesseract = pytest.mark.skipif(shutil.which("tesseract") is None, reason="tesseract not installed")


@pytest.fixture
def store(tmp_path):
    return Store(tmp_path / "index")
