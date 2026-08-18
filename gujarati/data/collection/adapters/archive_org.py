"""Collection method `ia_fulltext`: retrieve archive.org's own OCR output.

This adapter replaces the `pdf` route for archive.org material, which was
blocked by two independent problems. Scanned PDFs are served at 10-300 KB/s,
giving a logged ETA of 341 hours for the 12 GB required, and most of those
scans are image-only, which would require the Gujarati Tesseract model that is
not installed on this machine.

Archive.org performs OCR server-side and publishes the result as a plain
DjVuTXT file. Retrieving that file avoids both problems: the transfer is a few
hundred KB rather than tens of MB, and no local recognition is required. On a
20-item sample, 15 items exposed a DjVuTXT file (the remainder are
lending-restricted and return 403), and 13 of those 15 passed the Gujarati
script filter, averaging 377K characters each.

This became the principal non-news source at 152.75M characters, approximately
41x the yield of the PDF method.

Provenance note: documents from this adapter are labelled `manual_web`, not
`manual_ocr`. The recognition was performed by archive.org; this pipeline
retrieves and cleans the result.
"""

from __future__ import annotations

import asyncio
from urllib.parse import quote

import aiohttp

from .base import clean_text, make_doc, reject_reason, split_long_text
from .progress import Progress, log

#: Cursor-paginated search over archive.org's metadata.
SCRAPE_API = "https://archive.org/services/search/v1/scrape"


async def collect_ia_fulltext(session, src: dict, seen: set, prog: Progress):
    """Yield Documents built from archive.org's stored OCR text.

    Two steps per book, because the text file name is not derivable from the
    item identifier: it keeps the original title, spaces and all. So look up
    the item metadata, find the file whose format is DjVuTXT, then download it.
    """
    query = src.get("ia_query", "language:(gujarati) AND mediatype:(texts)")
    limit = int(src.get("split_chars", 300_000))

    # Each book is one slow, mostly-idle download (measured: 44s average for a
    # few hundred KB). One at a time put the ETA at 39 hours. Fetching 8 at
    # once brought it under two, because almost all of that time is waiting on
    # network latency rather than local processing.
    batch_size = int(src.get("batch", 8))
    cursor, n_items = None, 0

    async def fetch_one(item: dict):
        """Return (identifier, title, raw_text) for one item, or None."""
        ident = item.get("identifier")
        if not ident:
            return None

        try:
            async with session.get(f"https://archive.org/metadata/{ident}",
                                   timeout=aiohttp.ClientTimeout(total=60)) as r:
                meta = await r.json() if r.status == 200 else {}
        except Exception:
            prog.reject("ia-metadata-failed")
            return None

        # Skip history/ copies, which are earlier revisions of the same file.
        names = [f["name"] for f in meta.get("files", [])
                 if f.get("format") == "DjVuTXT"
                 and not f["name"].startswith("history/")]
        if not names:
            prog.reject("ia-no-ocr-text")
            return None

        url = f"https://archive.org/download/{ident}/{quote(names[0])}"
        try:
            async with session.get(
                    url, timeout=aiohttp.ClientTimeout(total=300)) as r:
                if r.status != 200:
                    # 403 means a lending-restricted item. Nothing to fix.
                    prog.reject(f"ia-http-{r.status}")
                    return None
                raw = await r.text(errors="replace")
        except Exception as exc:
            prog.reject(f"ia-{type(exc).__name__}")
            return None
        return ident, item.get("title", "") or ident, raw

    while True:
        params = {"q": query, "fields": "identifier,title", "count": "100"}
        if cursor:
            params["cursor"] = cursor
        try:
            async with session.get(SCRAPE_API, params=params,
                                   timeout=aiohttp.ClientTimeout(total=90)) as r:
                if r.status != 200:
                    log(f"  {src['name']}: scrape API returned {r.status} - stopping")
                    return
                blob = await r.json()
        except Exception as exc:
            log(f"  {src['name']}: scrape API failed ({type(exc).__name__}) - stopping")
            return

        items = blob.get("items") or []
        if not items:
            return
        n_items += len(items)

        todo = [i for i in items if i.get("identifier")
                and i["identifier"] not in seen]
        for i in todo:
            seen.add(i["identifier"])

        for start in range(0, len(todo), batch_size):
            chunk = todo[start:start + batch_size]
            got = await asyncio.gather(*(fetch_one(i) for i in chunk))

            for res in got:
                if res is None:
                    continue
                ident, title, raw = res
                text = clean_text(raw)
                if not text:
                    prog.reject("ia-empty")
                    continue

                parts = split_long_text(text, limit)
                for n, part in enumerate(parts, 1):
                    reason = reject_reason(part)
                    if reason:
                        prog.reject(reason)
                        continue
                    yield make_doc(
                        src, part, f"https://archive.org/details/{ident}", title,
                        {"identifier": ident, "part": n, "n_parts": len(parts),
                         "ocr_by": "archive.org (server-side, not us)",
                         "extraction": "archive_org_djvu_txt"},
                        method="ia_fulltext")

        cursor = blob.get("cursor")
        if not cursor:
            log(f"  {src['name']}: scanned {n_items:,} archive.org items")
            return
