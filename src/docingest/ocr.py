import base64
import subprocess


class OCRError(RuntimeError):
    pass


def ocr_tesseract(png: bytes, lang: str = "eng") -> str:
    """Fast and offline, but only as good as the scan."""
    try:
        proc = subprocess.run(
            ["tesseract", "stdin", "stdout", "-l", lang], input=png, capture_output=True, timeout=120, check=False
        )
    except FileNotFoundError:
        raise OCRError("tesseract is not installed (brew install tesseract)")
    if proc.returncode != 0:
        raise OCRError(proc.stderr.decode(errors="replace")[:200])
    return proc.stdout.decode("utf-8", errors="replace")


def ocr_vlm(png: bytes, llm, model: str | None = None) -> str:
    """Slower, but copes with layouts and messy scans that trip up tesseract. Without a model it uses the client's own, which has to accept images."""
    data_uri = "data:image/png;base64," + base64.b64encode(png).decode()
    kw = {"model": model} if model else {}
    reply = llm.chat(
        [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Transcribe all the text in this image exactly as written. Output only the text."},
                    {"type": "image_url", "image_url": {"url": data_uri}},
                ],
            }
        ],
        **kw,
    )
    return reply.get("content") or ""
