import argparse
import os
from pathlib import Path

from ragchat.rag import ConfigError, answer, embedder, llm_from_env
from ragchat.store import Store

from .fields import INVOICE, extract_fields
from .ocr import ocr_tesseract, ocr_vlm
from .pipeline import Ingester, SyncReport


def print_report(r: SyncReport) -> None:
    print(f"added {len(r.added)}, updated {len(r.updated)}, unchanged {len(r.unchanged)}, removed {len(r.removed)}")
    print(f"{r.chunks} new chunks, {r.ocr_pages} pages read by OCR")
    for name, err in r.failed.items():
        print(f"  FAILED {name}: {err}")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="docingest")
    ap.add_argument("--data", default="data")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sync = sub.add_parser("sync", help="index a folder, redoing only files that changed")
    sync.add_argument("folder", type=Path)
    sync.add_argument("--ocr", choices=["tesseract", "vlm"], default="tesseract")
    ask = sub.add_parser("ask", help="ask a question, with page citations")
    ask.add_argument("question")
    fields = sub.add_parser("fields", help="pull invoice fields out of a PDF, with where each one was found")
    fields.add_argument("pdf", type=Path)
    args = ap.parse_args(argv)

    try:
        llm = llm_from_env()
    except ConfigError as e:
        ap.error(str(e))
    embed = embedder(llm)
    store = Store(Path(args.data) / "index")

    if args.cmd == "sync":
        vision_model = os.environ.get("VISION_MODEL")
        ocr = ocr_tesseract if args.ocr == "tesseract" else (lambda png: ocr_vlm(png, llm, vision_model))
        ingester = Ingester(store, embed, Path(args.data) / "state.json", ocr=ocr, ocr_name=args.ocr)
        print_report(ingester.sync(args.folder))
    elif args.cmd == "fields":
        for f in extract_fields(args.pdf, INVOICE, llm):
            where = f"p.{f.page}" if f.page else "-"
            print(f"{f.name:16} {f.status:12} {where:5} {f.value}")
    else:
        a = answer(args.question, store, llm, embed)
        print(a.text)
        for i, s in enumerate(a.sources, 1):
            print(f"  [{i}] {s.source} p.{s.page} (score {s.score:.2f})")


if __name__ == "__main__":
    main()
