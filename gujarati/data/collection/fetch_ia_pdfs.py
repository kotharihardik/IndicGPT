"""Download Gujarati book PDFs from archive.org into data/raw/pdfs/books/.

This is a downloader, not a collector: it only puts files on disk. The `pdf`
adapter in adapters/pdf.py is what turns them into corpus documents, and the
`ia_books_pdf` source in manual_sources.yaml is what points at this directory.

    python data/collection/fetch_ia_pdfs.py --dry-run
    python data/collection/fetch_ia_pdfs.py --max-items 700 --max-gb 20

It writes manifest.csv alongside the PDFs, recording identifier, title,
creator, date, licenceurl and rights for every item. That file is the
provenance table for this source. Rerunning reads the manifest first and skips
whatever it already has, so an interrupted download resumes.

Worth knowing before using this: it was largely superseded by the
`ia_fulltext` method (adapters/archive_org.py), which pulls archive.org's own
OCR text instead of the PDFs. Downloads here run at 10-300 KB/s and most of
the scans have no text layer to read, so this route produced 3.7M characters
against ia_fulltext's 152.8M from the same provider. It is kept because its
output is in the corpus and the run should stay reproducible.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path
from urllib.parse import quote

import requests

SCRAPE_API = "https://archive.org/services/search/v1/scrape"
METADATA_API = "https://archive.org/metadata/{identifier}"
DOWNLOAD_URL = "https://archive.org/download/{identifier}/{filename}"

FIELDS = ["identifier", "title", "creator", "date", "year",
          "licenseurl", "rights", "imagecount", "item_size", "collection"]

TEXT_FIRST = ["Text PDF", "Additional Text PDF", "Image Container PDF"]
IMAGE_FIRST = ["Image Container PDF", "Additional Text PDF", "Text PDF"]

#: The archive.org collection this source draws from.
BOOKS_QUERY = ("collection:booksbylanguage_gujarati AND mediatype:texts "
               "AND format:PDF")
DEFAULT_OUT = "data/raw/pdfs/books"


def human_bytes(n: float) -> str:
    """Format a byte count for logging."""
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}PB"


def fmt_dur(seconds: float) -> str:
    """Human-readable duration."""
    if seconds <= 0 or seconds == float("inf"):
        return "?"
    s = int(seconds)
    if s < 60:
        return f"{s}s"
    if s < 3600:
        return f"{s // 60}m{s % 60:02d}s"
    return f"{s // 3600}h{(s % 3600) // 60:02d}m"


def scrape_search(session: requests.Session, query: str, limit: int,
                  delay: float) -> list:
    """Page through the archive.org scraping API and return item metadata."""
    items, cursor = [], None
    while len(items) < limit:
        params = {"q": query, "fields": ",".join(FIELDS),
                  "count": min(1000, limit - len(items))}
        if cursor:
            params["cursor"] = cursor
        try:
            r = session.get(SCRAPE_API, params=params, timeout=60)
            r.raise_for_status()
            payload = r.json()
        except (requests.RequestException, ValueError) as exc:
            print(f"  ! search failed: {exc}")
            break
        batch = payload.get("items", [])
        if not batch:
            break
        items.extend(batch)
        print(f"  search: {len(items):,} / {payload.get('total', 0):,} matching items",
              flush=True)
        cursor = payload.get("cursor")
        if not cursor:
            break
        time.sleep(delay)
    return items[:limit]


def pick_pdf(session: requests.Session, identifier: str, prefer: str,
             max_mb: float, delay: float) -> tuple:
    """Return (filename, size_bytes, meta) for one item, or (None, 0, meta).

    Skips items archive.org marks as access-restricted (lending library):
    those are not freely downloadable and must not be worked around.
    """
    try:
        r = session.get(METADATA_API.format(identifier=identifier), timeout=60)
        r.raise_for_status()
        blob = r.json()
    except (requests.RequestException, ValueError) as exc:
        return None, 0, {"skip_reason": f"metadata failed: {exc}"}
    finally:
        time.sleep(delay)

    meta = blob.get("metadata", {}) or {}
    for flag in ("access-restricted-item", "access-restricted"):
        if str(meta.get(flag, "")).lower() == "true":
            return None, 0, {**meta, "skip_reason": flag}

    order = IMAGE_FIRST if prefer == "image" else TEXT_FIRST
    by_format: dict = {}
    for f in blob.get("files", []):
        fmt = f.get("format", "")
        if fmt in order and f.get("name", "").lower().endswith(".pdf"):
            by_format.setdefault(fmt, f)

    for fmt in order:
        f = by_format.get(fmt)
        if not f:
            continue
        size = int(f.get("size", 0) or 0)
        if max_mb and size > max_mb * 1024 * 1024:
            continue
        return f["name"], size, {**meta, "chosen_format": fmt}

    return None, 0, {**meta, "skip_reason": "no suitable pdf derivative"}


def download(session: requests.Session, identifier: str, filename: str,
             dest: Path, delay: float, expect: int = 0) -> int:
    """Stream one PDF to disk with inline progress. Returns bytes written."""
    url = DOWNLOAD_URL.format(identifier=identifier, filename=quote(filename))
    tmp = dest.with_suffix(dest.suffix + ".part")
    written = 0
    t0 = time.time()
    try:
        with session.get(url, timeout=180, stream=True) as r:
            r.raise_for_status()
            total = int(r.headers.get("Content-Length", expect) or 0)
            last = t0
            with open(tmp, "wb") as fh:
                for chunk in r.iter_content(chunk_size=1 << 16):
                    fh.write(chunk)
                    written += len(chunk)
                    now = time.time()
                    if total > 1 << 22 and now - last > 2.0:   # >4MB: live bar
                        last = now
                        pct = 100 * written / total if total else 0
                        speed = written / max(0.001, now - t0)
                        print(f"      {identifier[:38]:<40} {pct:5.1f}%  "
                              f"{human_bytes(written)}/{human_bytes(total)}  "
                              f"{human_bytes(speed)}/s", end="\r", flush=True)
        if total > 1 << 22:
            print(" " * 100, end="\r")
        tmp.rename(dest)
    except (requests.RequestException, OSError) as exc:
        print(f"  ! download {identifier}: {exc}")
        tmp.unlink(missing_ok=True)
        return 0
    finally:
        time.sleep(delay)
    return written


MANIFEST_COLUMNS = ["identifier", "title", "creator", "date", "publisher",
                    "licenseurl", "rights", "chosen_format", "pdf_file",
                    "bytes", "imagecount", "status"]


def load_manifest(path: Path) -> dict:
    """Load an existing manifest so a rerun skips items already processed."""
    if not path.exists():
        return {}
    with open(path, newline="", encoding="utf-8") as fh:
        return {row["identifier"]: row for row in csv.DictReader(fh)}


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Download Gujarati book PDFs from archive.org.")
    ap.add_argument("--out", default=DEFAULT_OUT, help="output directory")
    ap.add_argument("--query", default=BOOKS_QUERY,
                    help="override the archive.org search query")
    ap.add_argument("--max-items", type=int, default=700)
    ap.add_argument("--max-gb", type=float, default=20.0)
    ap.add_argument("--max-file-mb", type=float, default=400.0)
    ap.add_argument("--prefer", choices=["text", "image"], default="text",
                    help="which PDF derivative to take. 'text' prefers one "
                         "with an embedded text layer, which is the only kind "
                         "the pdf adapter can read")
    ap.add_argument("--open-only", action="store_true",
                    help="keep only items with an explicit licenseurl")
    ap.add_argument("--delay", type=float, default=1.0)
    ap.add_argument("--contact", default="hardikkothari300@gmail.com")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    query = args.query
    out_dir = Path(args.out)

    if "example.com" in args.contact:
        print("Pass a real --contact address.")
        sys.exit(1)

    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.csv"
    seen = load_manifest(manifest_path)

    print(f"query    : {query}")
    print(f"out      : {out_dir}")
    print(f"manifest : {len(seen):,} items already processed\n")

    session = requests.Session()
    session.headers.update({
        "User-Agent": f"LMA-Monsoon2026-StudentCrawler/0.1 "
                      f"(academic corpus collection; contact: {args.contact})"
    })

    items = scrape_search(session, query, args.max_items, args.delay)
    print(f"got {len(items):,} candidate items\n")
    if not items:
        print("No items matched. Check the query on archive.org's advanced search.")
        return

    if args.dry_run:
        est = sum(int(i.get("item_size", 0) or 0) for i in items)
        licensed = sum(1 for i in items if i.get("licenseurl"))
        pages = sum(int(i.get("imagecount", 0) or 0) for i in items)
        print("DRY RUN - nothing downloaded")
        print(f"  items with explicit licenseurl : {licensed:,} / {len(items):,}")
        print(f"  total item size (all files)    : {human_bytes(est)}")
        print(f"  rough PDF-only estimate        : {human_bytes(est * 0.5)}")
        print(f"  total scanned pages            : {pages:,}")
        print(f"  rough text yield @1200 ch/page : {pages * 1200 / 1e6:.0f}M chars "
              f"(~{pages * 1200 / 3.5 / 1e6:.0f}M tokens)")
        print("\n  first 10:")
        for i in items[:10]:
            print(f"    {i['identifier']:<45} {str(i.get('title'))[:48]}")
        return

    budget = args.max_gb * 1024 ** 3
    total = kept = skipped = 0
    new_rows = []
    t0 = time.time()

    try:
        for n, item in enumerate(items, 1):
            ident = item["identifier"]
            if ident in seen:
                continue
            if args.open_only and not item.get("licenseurl"):
                skipped += 1
                new_rows.append({"identifier": ident, "title": item.get("title", ""),
                                 "status": "skipped:no-licenseurl"})
                continue

            fname, size, meta = pick_pdf(session, ident, args.prefer,
                                         args.max_file_mb, args.delay)
            if not fname:
                skipped += 1
                reason = str(meta.get("skip_reason", "?"))
                new_rows.append({"identifier": ident, "title": item.get("title", ""),
                                 "status": "skipped:" + reason})
                print(f"  [{n}/{len(items)}] SKIP {ident[:40]:<42} {reason}")
                continue

            dest = out_dir / f"{ident}.pdf"
            written = dest.stat().st_size if dest.exists() else \
                download(session, ident, fname, dest, args.delay, size)
            if not written:
                skipped += 1
                new_rows.append({"identifier": ident, "status": "failed:download"})
                continue

            total += written
            kept += 1
            new_rows.append({
                "identifier": ident, "title": item.get("title", ""),
                "creator": item.get("creator", ""),
                "date": item.get("date", "") or item.get("year", ""),
                "publisher": meta.get("publisher", ""),
                "licenseurl": item.get("licenseurl", "") or meta.get("licenseurl", ""),
                "rights": item.get("rights", "") or meta.get("rights", ""),
                "chosen_format": meta.get("chosen_format", ""),
                "pdf_file": dest.name, "bytes": written,
                "imagecount": item.get("imagecount", ""), "status": "ok",
            })

            el = time.time() - t0
            speed = total / el if el else 0
            eta = (budget - total) / speed if speed else 0
            print(f"  [{n}/{len(items)}] {ident[:38]:<40} "
                  f"{human_bytes(written):>9}  |  kept {kept:,}  |  "
                  f"{human_bytes(total)}/{args.max_gb}GB "
                  f"({100 * total / budget:.1f}%)  |  "
                  f"{human_bytes(speed)}/s  |  ETA {fmt_dur(eta)}", flush=True)

            if total >= budget:
                print(f"\nhit --max-gb budget ({args.max_gb} GB), stopping")
                break
    except KeyboardInterrupt:
        print("\ninterrupted - manifest written, rerun to resume")
    finally:
        write_header = not manifest_path.exists()
        with open(manifest_path, "a", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=MANIFEST_COLUMNS, extrasaction="ignore")
            if write_header:
                w.writeheader()
            w.writerows(new_rows)

    print(f"\ndownloaded {kept:,} PDFs / {human_bytes(total)} "
          f"({skipped:,} skipped) in {fmt_dur(time.time() - t0)}")
    print(f"provenance table: {manifest_path}")
    print("\nnext:  python3 data/collection/collect_manual.py --only ia_books_pdf")


if __name__ == "__main__":
    main()
