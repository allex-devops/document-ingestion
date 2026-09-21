import hashlib
import json
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from ragchat.ingest import Chunk
from ragchat.rag import Embedder
from ragchat.store import Store
from shared.docs import chunk_text

from .clean import clean_page, strip_repeated_lines
from .extract import SUPPORTED, extract
from .ocr import ocr_tesseract


@dataclass
class SyncReport:
    added: list[str] = field(default_factory=list)
    updated: list[str] = field(default_factory=list)
    unchanged: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)
    failed: dict[str, str] = field(default_factory=dict)
    chunks: int = 0
    ocr_pages: int = 0


def file_sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


class Ingester:
    """Keeps a folder and a vector index in step, redoing only what changed."""

    def __init__(
        self,
        store: Store,
        embed: Embedder,
        state_file: Path,
        ocr: Callable[[bytes], str] = ocr_tesseract,
        size: int = 800,
        overlap: int = 120,
        ocr_name: str = "tesseract",
    ):
        self.store, self.embed, self.state_file = store, embed, Path(state_file)
        self.ocr, self.size, self.overlap = ocr, size, overlap
        # saved with the state, so changing any of these redoes every file
        self.settings = {"size": size, "overlap": overlap, "ocr": ocr_name}

    def _load(self) -> dict:
        try:
            state = json.loads(self.state_file.read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            return {"settings": self.settings, "files": {}}
        if state.get("settings") != self.settings:
            return {"settings": self.settings, "files": {}}
        return state

    def _save(self, state: dict) -> None:
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=self.state_file.parent)
        with os.fdopen(fd, "w") as f:
            json.dump(state, f, indent=1)
        os.replace(tmp, self.state_file)  # swap in a finished file so a crash can't leave half of one

    def chunks_for(self, path: Path, name: str) -> tuple[list[Chunk], int]:
        doc = extract(path, self.ocr)
        texts = [clean_page(t) for t in strip_repeated_lines([p.text for p in doc.pages])]
        chunks = []
        for page, text in zip(doc.pages, texts):
            for i, piece in enumerate(chunk_text(text, self.size, self.overlap)):
                chunks.append(Chunk(id=f"{name}:{page.number}:{i}", text=piece, source=name, page=page.number))
        return chunks, doc.ocr_pages

    def sync(self, folder: Path) -> SyncReport:
        folder = Path(folder)
        state = self._load()
        files = state["files"]
        report = SyncReport()

        on_disk = {p.relative_to(folder).as_posix(): p for p in sorted(folder.rglob("*")) if p.suffix.lower() in SUPPORTED}
        for name, path in on_disk.items():
            sha = file_sha(path)
            if files.get(name, {}).get("sha") == sha:
                report.unchanged.append(name)
                continue
            try:
                chunks, ocr_pages = self.chunks_for(path, name)
                # delete first, or a shorter new version leaves its old tail chunks behind
                self.store.delete_source(name)
                if chunks:
                    self.store.add(chunks, self.embed([c.text for c in chunks], "document"))
            except Exception as e:  # one bad file shouldn't stop the rest of the folder
                report.failed[name] = str(e)[:200]
                continue
            (report.updated if name in files else report.added).append(name)
            files[name] = {"sha": sha, "chunks": len(chunks), "ocr_pages": ocr_pages}
            report.chunks += len(chunks)
            report.ocr_pages += ocr_pages
            self._save(state)  # after each file, so an interrupted run resumes instead of restarting

        for name in [n for n in files if n not in on_disk]:
            self.store.delete_source(name)
            del files[name]
            report.removed.append(name)
        self._save(state)
        return report
