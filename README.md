# Document Ingestion

Turns PDFs, Word files and scans into clean, searchable, citable chunks. It keeps a folder and its index in step, and it can pull structured fields (like an invoice number or total) out of a PDF along with the page and position each value came from.

## What it does

- **Extracts** text from PDF, DOCX, TXT and Markdown files. A PDF page with no text layer is treated as a scan and sent to OCR, either tesseract or a vision model.
- **Cleans** it: fixes ligatures and stray control characters, rejoins hyphenated line breaks, and strips running headers, footers and page numbers.
- **Chunks and indexes** each page in a local Chroma index, keeping the file name and page number so answers can cite them.
- **Stays in step with the folder.** Run `sync` again after adding, editing or deleting files and only the changes are redone. An interrupted run picks up where it stopped, and one bad file doesn't stop the rest.
- **Answers questions** from the index with page citations.
- **Extracts fields** from complex documents. It reads multi-column pages in the right order, marks headings, and checks every value the model returns against the source text.

## Requirements

- Python 3.12 and [`uv`](https://docs.astral.sh/uv/)
- [`tesseract`](https://github.com/tesseract-ocr/tesseract) for OCR (`brew install tesseract`)
- An API key for OpenAI or Google Gemini (any OpenAI-compatible endpoint works), used for chat and embeddings. `--ocr vlm` also needs a model that can read images.

Your document text is sent to the provider you choose.

## Setup

```bash
uv sync
cp .env.example .env    # uncomment one provider block and paste your key
```

## Run

```bash
# index a folder; run it again after adding, editing or deleting files and only the changes are redone
uv run --env-file .env python -m docingest sync path/to/folder
uv run --env-file .env python -m docingest sync path/to/folder --ocr vlm     # vision model instead of tesseract
uv run --env-file .env python -m docingest ask "What were the main findings?"

# pull fields from an invoice PDF: value, whether it was found in the text, and the page
uv run --env-file .env python -m docingest fields path/to/invoice.pdf
```

`sync` prints what was added, updated, unchanged and removed, and how many pages needed OCR. Changing the chunk size or the OCR engine reprocesses everything, since the old chunks would no longer match. The index and sync state live in `./data`; use `--data` to put them somewhere else.

Each extracted field comes back as one of:

| Status | Meaning |
|---|---|
| `found` | the value is in the document; page and position are attached |
| `missing` | the model says the document doesn't have it |
| `unsupported` | the model gave a value that isn't in the document, so don't trust it |

`fields` looks for an invoice's vendor, number, date and total. To extract something else, define your own list of `FieldSpec`s in `fields.py`.

## Configuration

Set these in `.env`. The first four are required and have no defaults: nothing runs until you've chosen a provider and given it your own key.

| Variable | Purpose |
|---|---|
| `LLM_BASE_URL` | the provider's OpenAI-compatible endpoint |
| `LLM_API_KEY` | your provider key |
| `LLM_MODEL` | chat model name |
| `EMBED_MODEL` | embedding model name |
| `VISION_MODEL` | optional; the model `--ocr vlm` uses, if `LLM_MODEL` can't read images |

If you change `EMBED_MODEL`, delete the data folder and sync again, because vectors from different models can't be compared.

## Tests

```bash
uv run pytest                                  # offline, no key needed; includes a real tesseract run on a generated scan
uv run --env-file .env pytest -m live -s       # real models; uses API tokens
```

## Known limits

Word files count as one page, so their citations always say p.1. Reading order handles the usual two-column page but not a wide figure in the middle of the columns. It also depends on the PDF reader splitting the text into blocks, which it does for real multi-line paragraphs but not for single short lines side by side.
