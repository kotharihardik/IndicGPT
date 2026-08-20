"""Collection method `pdf`: extract the text layer of local PDF files.

The single source using this method is court_pdf: 199 PDFs of Supreme Court of
Nepal judgments, procedural codes and statutes. It is retained despite its small
yield because it is the only formal legal register in the corpus; every other
source is news, business, health or literature, and statutory language differs
substantially from all of them.

The distinction matters for provenance labelling, because the assignment counts
OCR as manual collection: this adapter performs no OCR. PyMuPDF extracts the
text layer already embedded in the file, so documents are labelled
`pdf_text_layer` rather than `manual_ocr`.

The realised yield was far below the sampled estimate. A 6-PDF sample indicated
that roughly half carried a usable Devanagari text layer, which supported a 20M
character budget; the full run produced 961K characters, approximately 12%
usable. Most of the remainder are image-only scans with no text layer, and some
judgments are written in English. This pattern of over-optimistic small-sample
estimates recurred throughout the project, and is why the budgets in the source
registry are treated as ceilings rather than predictions.
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
        log(f"  {src['name']}: {pdf_dir} missing - download the PDFs first")
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
                # Usually 'not-devanagari' (an English-language judgment) or
                # 'not-nepali' (a Sanskrit-heavy legal citation).
                prog.reject(reason)
                continue
            yield make_doc(src, part, src.get("url", ""), path.stem,
                           {"file": path.name, "part": n, "n_parts": len(parts),
                            "pages": len(pages), "ocr": False,
                            "extraction": "pymupdf_text_layer"},
                           method="pdf_text_layer")
