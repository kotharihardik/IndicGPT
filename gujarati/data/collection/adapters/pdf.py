"""Collection method `pdf`: extract the text layer of local PDF files.

The files are downloaded separately by `fetch_ia_pdfs.py`; this adapter only
reads them.

The distinction matters for the provenance labelling, because the assignment
counts OCR as manual collection: this adapter performs no OCR. PyMuPDF
extracts the text layer already embedded in the file. Recognition of an
image-only scan would require the Gujarati Tesseract model, which is not
installed (`brew install tesseract-lang`, or `apt install tesseract-ocr-guj`).
Documents are therefore labelled `pdf_text_layer` rather than `manual_ocr`.

This method contributed the least of the four, approximately 3.7M characters.
Most archive.org books are image-only scans carrying no text layer, and
several of those that do have one use a broken font encoding that produces
mojibake and is rejected by the script filter. Those limitations motivated the
`ia_fulltext` adapter.
"""

from __future__ import annotations

from .base import ROOT, clean_text, make_doc, reject_reason, split_long_text
from .progress import Progress, log


async def collect_pdf(_session, src: dict, seen: set, prog: Progress):
    """Yield Documents from the text layer of every PDF in `local_dir`.

    The first argument is the shared aiohttp session, unused here because the
    files are already local. It stays in the signature so the registry in
    __init__.py can call every adapter the same way.
    """
    import pymupdf

    pdf_dir = ROOT / src["local_dir"]
    if not pdf_dir.exists():
        log(f"  {src['name']}: {pdf_dir} missing - run fetch_ia_pdfs.py first")
        return

    files = sorted(pdf_dir.rglob("*.pdf"))
    log(f"  {src['name']}: {len(files)} PDFs on disk")
    limit = int(src.get("split_chars", 300_000))

    for path in files:
        if path.name in seen:
            continue
        seen.add(path.name)

        try:
            with pymupdf.open(path) as doc:
                pages = [doc[i].get_text("text") for i in range(doc.page_count)]
        except Exception as exc:
            # Encrypted, truncated or not really a PDF. The reason is tallied
            # so a systematic failure shows up in the run log.
            prog.reject(f"pdf-{type(exc).__name__}")
            continue

        text = clean_text("\n\n".join(pages))
        if not text:
            prog.reject("pdf-no-text-layer")     # image-only scan
            continue

        parts = split_long_text(text, limit)
        for n, part in enumerate(parts, 1):
            reason = reject_reason(part)
            if reason:
                # Nearly always 'not-gujarati', from a broken embedded font
                # encoding producing mojibake. Rejecting it is correct.
                prog.reject(reason)
                continue
            yield make_doc(src, part, src.get("url", ""), path.stem,
                           {"file": path.name, "part": n, "n_parts": len(parts),
                            "pages": len(pages), "ocr": False,
                            "extraction": "pymupdf_text_layer"},
                           method="pdf_text_layer")
