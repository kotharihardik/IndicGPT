#!/usr/bin/env python3
"""Collect the Nepali MANUAL corpus (Phase 1, Model L).

Phase 1 requires that at least 20% of the final training tokens be manually
collected. The corpus target is ~500M tokens, so the floor is ~100M manual
tokens; the other ~80% comes from a public dataset via ``collect_public.py``.

Three collection methods, chosen per source by ``method:`` in
``data/sources/manual_sources.yaml``:

``wp_rest``
    The site's own public WordPress REST API. One request to
    ``/wp-json/wp/v2/posts?per_page=100`` returns **100 complete article
    bodies** as JSON, with no navigation, adverts or scripts. This is where
    almost all of the corpus comes from.

``sitemap``
    For sites with no REST API: read ``sitemap.xml``, fetch each article page,
    extract the body with trafilatura. One request per article, so ~70x more
    expensive per character. Used only for register diversity.

``pdf``
    PyMuPDF text-layer extraction from Gujarati books already downloaded to
    ``data/raw/pdfs/`` by ``fetch_ia_pdfs.py``. Small contribution.

Why the REST API decides the design
-----------------------------------
Measured on these sites:

======================================  ==================
method                                  kept text per second
======================================  ==================
wp_rest                                 ~570 K chars/s
sitemap (one request per article)        ~7 K chars/s
the original serial crawler              ~0.9 K chars/s
======================================  ==================

The first version of this collector was a serial crawler with a 2-second sleep
per page that also fetched every page twice (once for text, once for its
links). It managed 10.3M characters in 3h16m and then stalled because homepage
link-discovery cannot reach a news archive. The REST API is faster for a
structural reason, not because we request more aggressively: it amortises one
request across 100 documents and transfers no markup. Per-host request rates
here are *lower* than a browser loading the same page.

Usage
-----
::

    python data/collection/collect_manual.py --check    # robots + sizes, writes nothing
    python data/collection/collect_manual.py            # collect to target
    python data/collection/collect_manual.py --status   # progress vs the 20% floor
    python data/collection/collect_manual.py --only abtakmedia --limit-chars 1000000

Safe to interrupt with Ctrl-C and rerun: resume state is the collected data
itself. At start-up each source's existing shards are scanned for the article
URLs already stored, and those are skipped. There is no separate state file to
get out of sync with the corpus.
"""

from __future__ import annotations

import argparse
import asyncio
import gzip
import html as htmllib
import json
import os
import re
import sys
import time
import unicodedata
import urllib.robotparser as robotparser
from pathlib import Path
from urllib.parse import unquote, urlparse

import aiohttp
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from corpus_io import (Document, JsonlWriter, MANUAL_PDF, MANUAL_WEB,  # noqa: E402
                       human, lang_root, load_config)

ROOT = lang_root(Path(__file__))
CFG = load_config(ROOT)
LANG = CFG["language"]
CRAWL = CFG.get("crawl", {})
UA = CRAWL["user_agent"]

#: Estimated characters per token, used only for progress reporting: the
#: tokenizer does not exist yet during collection. Deliberately pessimistic
#: (high) so we over-collect rather than find a shortfall at the deadline.
#: ``data/splits/corpus_stats.py --tokenizer`` reports the exact counts later.
CHARS_PER_TOKEN = CFG.get("tokens", {}).get("chars_per_token", 4.0)

#: The manual-token floor, plus headroom for what cleaning and dedup remove.
_T = CFG["targets"]
MANUAL_FLOOR = float(_T["corpus_tokens"]) * float(_T["manual_fraction"])
TARGET_TOKENS = MANUAL_FLOOR * float(_T["collection_headroom"])

#: Fields to request from the WP API. Asking for only these makes the response
#: several times smaller than the default payload.
WP_FIELDS = "id,date,link,title,content"

#: Never worth fetching from a sitemap.
SKIP_EXT = (".jpg", ".jpeg", ".png", ".gif", ".webp", ".svg", ".ico", ".mp4",
            ".mp3", ".zip", ".css", ".js", ".json", ".xml", ".epub", ".djvu")

DEVANAGARI = re.compile(r"[\u0900-\u097F]")

#: Nepali has NO exclusive characters - it shares Devanagari with Hindi,
#: Marathi, Sanskrit and Bhojpuri. Gujarati could be identified by its
#: Unicode block and Sindhi by 17 exclusive letters; neither method exists
#: here. Language identification is therefore LEXICAL: these are
#: high-frequency Nepali function words that Hindi does not use, and the
#: Hindi equivalents that Nepali does not use.
#: Measured on real pages: Nepali news scores 11-14 NEPALI_MARKERS per
#: 1000 characters and 0.0 HINDI_MARKERS. Hindi pages invert this exactly.
NEPALI_MARKERS = (" छ ", " छन् ", " हो ", " हुन् ", " भएको ", " गरेको ",
                  " लाई ", " र ", " थियो ", " गर्न ", " भने ", " पनि ",
                  " तर ", " नेपाल")
HINDI_MARKERS = (" है ", " हैं ", " और ", " किया ", " नहीं ", " कहा ",
                 " गया ", " लिए ", " रहे ", " करने ", " था ", " थी ")
LATIN = re.compile(r"[A-Za-z]")
LETTER = re.compile(r"[^\W\d_]", re.UNICODE)
TAG = re.compile(r"<[^>]+>")
SCRIPT_STYLE = re.compile(r"(?is)<(script|style|noscript)\b.*?</\1>")

#: Trailing "Also read:" link lists that WordPress themes append inside the
#: article body on these sites.
BOILERPLATE_TAIL = re.compile(
    r"(?:यो\s*पनि\s*पढ्नुहोस्|यी\s*पनि\s*पढ्नुहोस्|थप\s*पढ्नुहोस्|"
    r"सम्बन्धित\s*समाचार|Also\s*Read)\s*[:：]?.*$", re.S | re.I)

#: Common Nepali function words. Text containing none of these is a menu or
#: a list of names, not prose.
STOPWORDS = ("छ", "हो", "र", "पनि", "तर", "लाई", "भएको", "गरेको", "हुन्",
             "छन्", "भने", "गर्न", "देखि", "सम्म", "बाट", "मा")


# --------------------------------------------------------------------------- #
# Text cleaning and the quality filter
# --------------------------------------------------------------------------- #


def clean_text(raw: str, is_html: bool = False) -> str:
    """Normalize text and, for HTML input, strip markup.

    NFC normalization happens here as well as in ``preprocessing/preprocess.py``
    because the tokenizer must see one spelling per word, and because
    downstream duplicate detection compares normalized text.
    """
    if not raw:
        return ""
    if is_html:
        raw = SCRIPT_STYLE.sub(" ", raw)
        raw = re.sub(r"(?i)</(p|div|br|li|h[1-6]|tr)\s*>", "\n", raw)
        raw = htmllib.unescape(TAG.sub(" ", raw))
    text = unicodedata.normalize("NFC", raw)
    text = text.replace("​", "").replace("﻿", "")
    text = BOILERPLATE_TAIL.sub("", text)
    text = re.sub(r"[ \t ]+", " ", text)
    text = "\n".join(line.strip() for line in text.split("\n"))
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def reject_reason(text: str) -> str:
    """Return '' to accept the document, or a short reason to reject it.

    This is a cheap early filter, not a replacement for
    ``preprocessing/preprocess.py``, which applies the strict thresholds later.
    It is deliberately looser than that stage: discarding a document here is
    irreversible, whereas discarding it in preprocessing is a config change
    and a rerun of a local script. Collect slightly wide, filter strictly later.
    """
    if len(text) < 200:
        return "too-short"
    if text.count(" ") < 20:
        return "too-few-words"

    letters = len(LETTER.findall(text))
    if not letters:
        return "no-letters"
    if len(DEVANAGARI.findall(text)) / letters < 0.60:
        return "not-devanagari"

    # Devanagari alone proves nothing - Hindi looks identical at this level.
    padded = " " + text + " "
    per_1k = max(len(text) / 1000, 1)
    nepali = sum(padded.count(w) for w in NEPALI_MARKERS) / per_1k
    hindi = sum(padded.count(w) for w in HINDI_MARKERS) / per_1k
    if nepali < 2.0:
        return "not-nepali"          # Devanagari, but not Nepali vocabulary
    if hindi > nepali:
        return "hindi-not-nepali"    # Hindi page inside a Nepali crawl

    if len(LATIN.findall(text)) / letters > 0.35:
        return "latin-heavy"
    if sum(1 for w in STOPWORDS if w in text) < 2:
        return "no-stopwords"
    return ""


# --------------------------------------------------------------------------- #
# HTTP
# --------------------------------------------------------------------------- #

_robots: dict = {}


async def robots_ok(session: aiohttp.ClientSession, url: str) -> bool:
    """Check robots.txt for our user agent, fetching it once per host.

    Checked at every run rather than recorded once in the source YAML, because
    it changes: ``sandesh.com`` returned DISALLOW on an early probe and ALLOW
    when re-probed. An unreachable robots.txt is treated as permission.
    """
    host = urlparse(url).netloc
    if host not in _robots:
        parser = robotparser.RobotFileParser()
        try:
            async with session.get(f"{urlparse(url).scheme}://{host}/robots.txt",
                                   timeout=aiohttp.ClientTimeout(total=20)) as r:
                parser.parse((await r.text(errors="ignore")).splitlines()
                             if r.status == 200 else [""])
        except Exception:
            parser.parse([""])
        _robots[host] = parser
    try:
        return _robots[host].can_fetch(UA, url)
    except Exception:
        return True


async def fetch(session: aiohttp.ClientSession, url: str, as_json: bool = False,
                retries: int = 2, timeout: float = 45.0):
    """GET a URL, retrying transient failures. Returns parsed JSON, text, or None.

    Retries only 429/5xx and network errors; a 404 is final. Pass ``retries=0``
    for URLs that might simply not exist - some hosts hang instead of returning
    404, and retrying such a URL at a 45s timeout wastes minutes to learn
    nothing.
    """
    for attempt in range(retries + 1):
        try:
            async with session.get(
                    url, timeout=aiohttp.ClientTimeout(total=timeout)) as r:
                if r.status == 200:
                    body = await r.text(errors="replace")
                    if not as_json:
                        return body
                    try:
                        return json.loads(body)
                    except ValueError:
                        return None
                if r.status not in (408, 429, 500, 502, 503, 504):
                    return None
        except Exception:
            pass
        if attempt < retries:
            await asyncio.sleep(2 ** attempt)
    return None


# --------------------------------------------------------------------------- #
# Progress reporting
# --------------------------------------------------------------------------- #


def fmt_dur(seconds: float) -> str:
    """Human-readable duration, e.g. '2h14m'."""
    if seconds != seconds or seconds in (float("inf"), float("-inf")) or seconds < 0:
        return "?"
    s = int(seconds)
    if s < 60:
        return f"{s}s"
    if s < 3600:
        return f"{s // 60}m{s % 60:02d}s"
    return f"{s // 3600}h{(s % 3600) // 60:02d}m"


class Progress:
    """Live progress for one source, plus the run total.

    Exists because a silent collector and a broken collector look identical
    from the outside; the run log is also the evidence for the throughput and
    cleaning-funnel numbers reported in Phase 1.
    """

    total_chars = 0        # across all sources in this run, including resumed
    log_file = None

    def __init__(self, name: str, already: int = 0, budget: int = 0,
                 every: float = 10.0):
        self.name = name
        self.already = already
        self.budget = budget
        self.every = every
        self.start = time.time()
        self.last = 0.0
        self.docs = 0
        self.chars = 0
        self.requests = 0
        self.rejected: dict = {}

    def keep(self, n_chars: int) -> None:
        self.docs += 1
        self.chars += n_chars
        Progress.total_chars += n_chars

    def reject(self, reason: str) -> None:
        self.rejected[reason] = self.rejected.get(reason, 0) + 1

    @property
    def source_chars(self) -> int:
        """This source's total, including what earlier runs collected."""
        return self.already + self.chars

    @property
    def elapsed(self) -> float:
        return time.time() - self.start

    def line(self, tail: str = "") -> str:
        rate = self.chars / self.elapsed if self.elapsed else 0
        done = Progress.total_chars
        eta = (TARGET_TOKENS * CHARS_PER_TOKEN - done) / rate if rate else 0
        parts = [f"[{self.name}]", f"{self.docs:,} docs",
                 f"{human(self.source_chars)} ch",
                 f"~{human(self.source_chars / CHARS_PER_TOKEN)} tok",
                 f"{human(rate)} ch/s", f"{self.requests:,} req"]
        if self.budget:
            parts.append(f"{100 * self.source_chars / self.budget:.0f}% of budget")
        parts.append(f"ETA {fmt_dur(eta)}")
        parts.append(fmt_dur(self.elapsed))
        if tail:
            parts.append(tail)
        return "  |  ".join(parts)

    def tick(self, force: bool = False) -> None:
        now = time.time()
        if force or now - self.last >= self.every:
            self.last = now
            log("  " + self.line())

    def done(self, why: str = "") -> None:
        log("  " + self.line("DONE" + (f" ({why})" if why else "")))
        if self.rejected:
            top = sorted(self.rejected.items(), key=lambda kv: -kv[1])[:5]
            log("       rejected: " + ", ".join(f"{r} x{n:,}" for r, n in top))


def log(message: str) -> None:
    """Print to the terminal and append to ``logs/collect_manual.log``."""
    print(message, flush=True)
    if Progress.log_file:
        Progress.log_file.write(f"{time.strftime('%H:%M:%S')} {message}\n")
        Progress.log_file.flush()


# --------------------------------------------------------------------------- #
# What is already collected  (this is also the resume mechanism)
# --------------------------------------------------------------------------- #


def existing(source_name: str) -> tuple:
    """Return (n_docs, n_chars, set_of_urls) already stored for a source.

    Reads the shards themselves rather than a separate bookkeeping file, so
    resume state can never disagree with the corpus. Tolerates a shard whose
    tail was truncated by an interrupted run.
    """
    raw_dir = CFG["paths"]["raw_manual"]
    docs = chars = 0
    urls = set()
    if not raw_dir.exists():
        return 0, 0, urls
    for path in sorted(raw_dir.glob(f"{source_name}.part-*.jsonl.gz")):
        try:
            with gzip.open(path, "rt", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except ValueError:
                        continue
                    docs += 1
                    chars += len(rec.get("text", ""))
                    if rec.get("source_url"):
                        urls.add(rec["source_url"])
        except (OSError, EOFError, gzip.BadGzipFile):
            continue        # truncated tail of an interrupted run
    return docs, chars, urls


def make_doc(src: dict, text: str, url: str, title: str = "",
             extra: dict = None, method: str = "") -> Document:
    """Build a Document with the provenance fields the 20% audit needs.

    Provenance records what actually happened, not what the source is
    nominally for. PDFs are the case that matters: with no Gujarati Tesseract
    model installed, ``collect_pdf`` reads embedded text layers and performs no
    OCR at all. Those documents are therefore ``manual_pdf`` /
    ``pdf_text_layer``; only genuinely recognised scans would be ``manual_ocr``
    / ``pdf_ocr``. Both count toward the >=20% manual requirement.
    """
    is_pdf = src["method"] == "pdf"
    return Document(
        text=text, language=LANG, source_name=src["name"],
        source_type="manual",
        collection_method=method or ("pdf_text_layer" if is_pdf else "scrape"),
        manual_or_public=MANUAL_PDF if is_pdf else MANUAL_WEB,
        source_url=url, title=title or None, license=src.get("license", ""),
        extra={"method": src["method"], **(extra or {})})


# --------------------------------------------------------------------------- #
# Method 1: WordPress REST API   (the source of almost all the corpus)
# --------------------------------------------------------------------------- #


async def collect_wp(session, src: dict, seen: set, prog: Progress):
    """Yield documents from a WordPress site, newest posts first.

    Pages are fetched in small concurrent batches. Plain ``?page=N`` pagination
    is enough here: the largest budget needs about 600 pages, and these hosts
    serve ``page=1600`` (some cap lower, which is why an empty or failed page
    ends the source rather than raising).
    """
    api = f"https://{urlparse(src['url']).netloc}/wp-json/wp/v2/posts"
    if not await robots_ok(session, api):
        log(f"  {src['name']}: robots.txt DISALLOWS {api} - skipping")
        return

    per_page = int(src.get("per_page", 100))
    batch = int(src.get("batch", 4))

    # We always read newest-first and take every post, so what is already
    # stored is a prefix of the archive. On resume, skip straight past it
    # instead of re-requesting hundreds of pages only to discard them. The
    # margin covers articles published since the last run; anything already
    # stored is still caught by the URL check below.
    page = max(1, len(seen) // per_page - 5)
    if page > 1:
        log(f"  {src['name']}: {len(seen):,} posts already stored, "
            f"resuming from page {page}")

    while True:
        urls = [f"{api}?per_page={per_page}&page={p}&orderby=date&order=desc"
                f"&_fields={WP_FIELDS}" for p in range(page, page + batch)]
        results = await asyncio.gather(*(fetch(session, u, as_json=True)
                                        for u in urls))
        prog.requests += len(urls)

        got_full_page = False
        for payload in results:
            if not isinstance(payload, list) or not payload:
                continue
            if len(payload) >= per_page:
                got_full_page = True
            for post in payload:
                link = post.get("link") or ""
                if link in seen:
                    # Resuming re-walks the pages already collected, because
                    # resume state is the stored data rather than a saved page
                    # number. Tick anyway so a long skip is visibly progress
                    # and not a hang.
                    prog.reject("already-collected")
                    prog.tick()
                    continue
                seen.add(link)
                text = clean_text((post.get("content") or {}).get("rendered", ""),
                                  is_html=True)
                reason = reject_reason(text)
                if reason:
                    prog.reject(reason)
                    continue
                title = clean_text((post.get("title") or {}).get("rendered", ""),
                                   is_html=True)
                yield make_doc(src, text, link, title,
                               {"wp_id": post.get("id"),
                                "published": post.get("date")})

        # Any short page means we reached the end of the archive (or the host's
        # page ceiling). Either way there is nothing further to ask for.
        if not got_full_page:
            return
        page += batch


# --------------------------------------------------------------------------- #
# Method 2: sitemap -> article pages   (register diversity only)
# --------------------------------------------------------------------------- #


def wanted_url(src: dict, url: str) -> bool:
    """True if a sitemap URL is on-host, not an asset, and not denied."""
    if not url.startswith("http"):
        return False
    hosts = [h.removeprefix("www.") for h in
             (src.get("hosts") or [urlparse(src["url"]).netloc])]
    host = urlparse(url).netloc.removeprefix("www.")
    if not any(host == h or host.endswith("." + h) for h in hosts):
        return False
    if url.lower().split("?")[0].endswith(SKIP_EXT):
        return False
    # Test the decoded form too, so patterns with Gujarati characters match the
    # percent-encoded URLs a crawler actually sees.
    decoded = unquote(url)
    return not any(re.search(p, url) or re.search(p, decoded)
                   for p in src.get("deny", []))


async def sitemap_urls(session, src: dict, prog: Progress) -> list:
    """Walk the sitemap tree (indexes included) and return article URLs."""
    from bs4 import BeautifulSoup

    declared = list(src.get("sitemaps") or [])
    if not declared:
        robots = await fetch(session, f"https://{urlparse(src['url']).netloc}"
                                      f"/robots.txt", retries=1, timeout=20)
        declared = re.findall(r"(?im)^\s*sitemap:\s*(\S+)", robots or "")
    # A guessed path gets one short attempt: some hosts hang on a missing
    # sitemap instead of returning 404.
    queue = [(u, False) for u in declared] + [
        (f"https://{urlparse(src['url']).netloc}/sitemap.xml", True)]

    max_pages = int(src.get("max_pages", 20000))
    seen_maps, out = set(), []
    while queue and len(out) < max_pages and len(seen_maps) < 400:
        url, guessed = queue.pop(0)
        if url in seen_maps:
            continue
        seen_maps.add(url)
        body = await fetch(session, url, retries=0 if guessed else 1,
                           timeout=10 if guessed else 30)
        prog.requests += 1
        if not body:
            continue
        try:
            soup = BeautifulSoup(body, "xml")
        except Exception:
            continue
        locs = [loc.get_text(strip=True) for loc in soup.find_all("loc")]
        if soup.find("sitemapindex"):
            queue += [(u, False) for u in locs if u not in seen_maps]
        else:
            out += [u for u in locs if wanted_url(src, u)]

    out = list(dict.fromkeys(out))
    if len(out) < 100:
        # No usable sitemap. Fall back to following links from the seed page.
        #
        # This is the classic crawl the sitemap path normally replaces, and it
        # is slower and less complete - but a large part of the Nepali web
        # outside news publishes no sitemap at all. Health, education,
        # agriculture and legal sites were all reachable and rich in Nepali
        # text yet invisible to sitemap discovery, so without this fallback
        # those registers are simply absent from the corpus.
        #
        # Bounded breadth-first, same host only, so it cannot wander off.
        out += await crawl_links(session, src, prog, max_pages)
        out = list(dict.fromkeys(out))
    return out[:max_pages]


async def crawl_links(session, src: dict, prog: Progress, max_pages: int) -> list:
    """Breadth-first link discovery for sites with no sitemap.

    Returns article URLs only; the caller fetches and extracts them. Pages are
    re-fetched by the caller rather than parsed here, which costs one extra
    request per discovered page but keeps this function simple and lets the
    normal quality gate apply uniformly.
    """
    from bs4 import BeautifulSoup
    from urllib.parse import urljoin, urldefrag

    seed = src["url"]
    max_depth = int(src.get("max_depth", 3))
    frontier = [(seed, 0)]
    seen = {seed}
    found: list = []

    while frontier and len(found) < max_pages and len(seen) < max_pages * 3:
        batch, frontier = frontier[:8], frontier[8:]
        pages = await asyncio.gather(
            *(fetch(session, u, retries=0, timeout=20) for u, _ in batch))
        prog.requests += len(batch)
        for (url, depth), body in zip(batch, pages):
            if not body:
                continue
            if wanted_url(src, url) and url != seed:
                found.append(url)
            if depth >= max_depth:
                continue
            try:
                soup = BeautifulSoup(body, "lxml")
            except Exception:
                continue
            for a in soup.find_all("a", href=True):
                nxt = urldefrag(urljoin(url, a["href"]))[0]
                if nxt not in seen and wanted_url(src, nxt):
                    seen.add(nxt)
                    frontier.append((nxt, depth + 1))
        if len(found) % 200 < 8:
            log(f"    {src['name']}: link-crawl found {len(found):,} URLs "
                f"(frontier {len(frontier):,})")
    return found


async def collect_sitemap(session, src: dict, seen: set, prog: Progress):
    """Yield documents by fetching each article page listed in the sitemap."""
    import trafilatura

    if not await robots_ok(session, src["url"]):
        log(f"  {src['name']}: robots.txt DISALLOWS {src['url']} - skipping")
        return

    urls = [u for u in await sitemap_urls(session, src, prog) if u not in seen]
    log(f"  {src['name']}: {len(urls):,} new article URLs from sitemap")
    if not urls:
        return

    concurrency = int(src.get("concurrency", 8))
    queue: asyncio.Queue = asyncio.Queue(maxsize=concurrency * 4)
    sem = asyncio.Semaphore(concurrency)
    DONE = object()

    async def one(url: str):
        async with sem:
            html = await fetch(session, url)
            prog.requests += 1
            if not html:
                await queue.put(("", url))
                return
            raw = trafilatura.extract(html, url=url, include_comments=False,
                                      include_tables=False, favor_precision=True)
            await queue.put((clean_text(raw or ""), url))

    async def producer():
        await asyncio.gather(*(one(u) for u in urls))
        await queue.put(DONE)

    task = asyncio.create_task(producer())
    try:
        while True:
            item = await queue.get()
            if item is DONE:
                break
            text, url = item
            seen.add(url)
            if not text:
                prog.reject("no-text-extracted")
                continue
            reason = reject_reason(text)
            if reason:
                prog.reject(reason)
                continue
            yield make_doc(src, text, url)
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


# --------------------------------------------------------------------------- #
# Method 3: local PDFs
# --------------------------------------------------------------------------- #


def collect_pdf(src: dict, seen: set, prog: Progress):
    """Yield documents from the text layer of PDFs already on disk.

    Text-layer extraction only. Real OCR of image-only scans needs the Gujarati
    Tesseract model, which is not installed by default:
    ``brew install tesseract-lang`` or ``apt install tesseract-ocr-guj``.
    Books also arrive far larger than the length filter allows, so they are
    split at paragraph boundaries rather than discarded.
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
            prog.reject(f"pdf-{type(exc).__name__}")
            continue

        text = clean_text("\n\n".join(pages))
        if not text:
            prog.reject("pdf-no-text-layer")
            continue

        # Split long books into parts at blank lines so no sentence is cut.
        parts, buf, size = [], [], 0
        for para in text.split("\n\n"):
            if size + len(para) > limit and buf:
                parts.append("\n\n".join(buf))
                buf, size = [], 0
            buf.append(para)
            size += len(para) + 2
        if buf:
            parts.append("\n\n".join(buf))

        for n, part in enumerate(parts, 1):
            reason = reject_reason(part)
            if reason:
                # Usually 'not-gujarati': a broken embedded font encoding
                # produced mojibake. Rejecting is correct.
                prog.reject(reason)
                continue
            yield make_doc(src, part, src.get("url", ""), path.stem,
                           {"file": path.name, "part": n, "n_parts": len(parts),
                            "pages": len(pages), "ocr": False,
                            "extraction": "pymupdf_text_layer"},
                           method="pdf_text_layer")


# --------------------------------------------------------------------------- #
# Driving one source
# --------------------------------------------------------------------------- #


async def run_source(session, src: dict, limit_chars: int,
                     deadline: float) -> None:
    """Collect one source into its own shard, up to its character budget."""
    name, method = src["name"], src["method"]
    budget = int(src.get("max_chars", 0))
    prev_docs, prev_chars, seen = existing(name)

    log(f"\n--- {name}  ({method})  budget "
        f"{human(budget) if budget else 'none'} chars ---")
    if prev_chars:
        log(f"  resuming: {prev_docs:,} docs / {human(prev_chars)} chars "
            f"already collected, {len(seen):,} URLs known")
    if budget and prev_chars >= budget:
        log("  budget already met - skipping")
        return

    prog = Progress(name, already=prev_chars, budget=budget)
    out_dir = CFG["paths"]["raw_manual"]
    out_dir.mkdir(parents=True, exist_ok=True)
    shard = out_dir / f"{name}.part-{int(time.time())}-{os.getpid()}.jsonl.gz"
    writer = JsonlWriter(shard)
    why = ""

    try:
        if method == "wp_rest":
            stream = collect_wp(session, src, seen, prog)
        elif method == "sitemap":
            stream = collect_sitemap(session, src, seen, prog)
        elif method == "pdf":
            async def as_async():
                for doc in collect_pdf(src, seen, prog):
                    yield doc
            stream = as_async()
        else:
            log(f"  ! unknown method {method!r}")
            return

        async for doc in stream:
            writer.write(doc)
            prog.keep(doc.n_chars)
            prog.tick()
            if budget and prog.source_chars >= budget:
                why = "source budget reached"
                break
            if limit_chars and Progress.total_chars >= limit_chars:
                why = "--limit-chars reached"
                break
            if deadline and time.time() > deadline:
                why = "--max-hours reached"
                break
    except asyncio.CancelledError:
        why = "interrupted"
        raise
    finally:
        writer.close()
        prog.done(why)
        log(f"  shard: {shard.name} ({writer.n:,} docs / {human(writer.chars)} chars)")


# --------------------------------------------------------------------------- #
# Reports
# --------------------------------------------------------------------------- #


def load_sources(only: list, include_disabled: bool) -> list:
    """Load ``manual_sources.yaml``, merging the shared ``defaults:`` block."""
    path = ROOT / "data" / "sources" / "manual_sources.yaml"
    blob = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    defaults = blob.get("defaults") or {}
    out = [{**defaults, **s} for s in (blob.get("sources") or [])]
    if not include_disabled:
        out = [s for s in out if s.get("enabled", True)]
    if only:
        out = [s for s in out if s["name"] in only]
    return out


def report_status(sources: list) -> None:
    """Print cumulative progress toward the 20% manual-token floor."""
    print("\nNEPALI MANUAL COLLECTION STATUS")
    print(f"  token estimate at {CHARS_PER_TOKEN} chars/token "
          f"(exact counts: data/splits/corpus_stats.py --tokenizer)\n")
    print(f"  {'source':<24}{'method':<10}{'docs':>9}{'chars':>11}"
          f"{'~tokens':>11}{'budget':>10}{'':>7}")
    print("  " + "-" * 82)

    total_docs = total_chars = 0
    by_method: dict = {}
    for src in sources:
        docs, chars, _ = existing(src["name"])
        total_docs += docs
        total_chars += chars
        by_method[src["method"]] = by_method.get(src["method"], 0) + chars
        budget = int(src.get("max_chars", 0))
        pct = f"{100 * chars / budget:.0f}%" if budget else "-"
        print(f"  {src['name']:<24}{src['method']:<10}{docs:>9,}"
              f"{human(chars):>11}{human(chars / CHARS_PER_TOKEN):>11}"
              f"{(human(budget) if budget else '-'):>10}{pct:>7}")

    print("  " + "-" * 82)
    tokens = total_chars / CHARS_PER_TOKEN
    print(f"  {'TOTAL':<24}{'':<10}{total_docs:>9,}{human(total_chars):>11}"
          f"{human(tokens):>11}")
    print("\n  by method:")
    for method, chars in sorted(by_method.items(), key=lambda kv: -kv[1]):
        share = 100 * chars / total_chars if total_chars else 0
        print(f"    {method:<10}{human(chars):>10} chars  "
              f"{human(chars / CHARS_PER_TOKEN):>9} tok  ({share:4.1f}%)")

    floor = MANUAL_FLOOR
    for label, goal in (("hard floor (20% of 500M)", floor),
                        ("collection target", TARGET_TOKENS)):
        frac = min(1.0, tokens / goal)
        bar = "#" * int(frac * 34) + "." * (34 - int(frac * 34))
        print(f"\n  {label}: {human(goal)} tokens")
        print(f"  [{bar}] {100 * tokens / goal:5.1f}%")
    if tokens < TARGET_TOKENS:
        need = TARGET_TOKENS * CHARS_PER_TOKEN - total_chars
        print(f"\n  still need ~{human(need)} chars "
              f"(~{human(need / CHARS_PER_TOKEN)} tokens)")
    else:
        print("\n  target met.")


async def report_check(sources: list) -> None:
    """Check robots.txt and archive size for every source. Writes nothing."""
    print("\nPRE-FLIGHT CHECK (nothing is written)\n")
    async with aiohttp.ClientSession(headers={"User-Agent": UA}) as session:
        for src in sources:
            name, method = src["name"], src["method"]
            if method == "pdf":
                pdf_dir = ROOT / src["local_dir"]
                n = len(list(pdf_dir.rglob("*.pdf"))) if pdf_dir.exists() else 0
                print(f"  {name:<24}{method:<10}{n} PDFs on disk")
                continue

            allowed = await robots_ok(session, src["url"])
            note = ""
            if method == "wp_rest" and allowed:
                api = (f"https://{urlparse(src['url']).netloc}"
                       f"/wp-json/wp/v2/posts?per_page=1&_fields=id")
                data = await fetch(session, api, as_json=True, retries=1)
                note = "REST API ok" if isinstance(data, list) and data \
                    else "REST API UNREACHABLE"
            print(f"  {name:<24}{method:<10}"
                  f"{'ALLOW' if allowed else 'DISALLOW':<10}{note}")

    budget = sum(int(s.get("max_chars", 0)) for s in sources)
    print(f"\n  sum of budgets: {human(budget)} chars "
          f"(~{human(budget / CHARS_PER_TOKEN)} tokens)")
    print(f"  collection target: {human(TARGET_TOKENS)} tokens")


# --------------------------------------------------------------------------- #


async def collect_all(sources: list, args) -> None:
    """Collect every selected source in registry order."""
    deadline = time.time() + args.max_hours * 3600 if args.max_hours else 0.0
    # Seed the run total with what is already on disk so the ETA and the
    # percentage-of-target reflect real progress, not just this run.
    Progress.total_chars = sum(existing(s["name"])[1] for s in sources)

    log("=" * 88)
    log(f"NEPALI MANUAL COLLECTION   {time.strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"  sources: {', '.join(s['name'] for s in sources)}")
    log(f"  target : {human(TARGET_TOKENS)} manual tokens "
        f"(~{human(TARGET_TOKENS * CHARS_PER_TOKEN)} chars)")
    log(f"  already: {human(Progress.total_chars)} chars collected")
    log("=" * 88)

    started = time.time()
    connector = aiohttp.TCPConnector(limit=40, ttl_dns_cache=600)
    async with aiohttp.ClientSession(
            connector=connector,
            headers={"User-Agent": UA, "Accept-Encoding": "gzip, deflate"}) as s:
        for src in sources:
            if deadline and time.time() > deadline:
                log("\n  time box reached - stopping")
                break
            if args.limit_chars and Progress.total_chars >= args.limit_chars:
                log("\n  --limit-chars reached - stopping")
                break
            if Progress.total_chars >= TARGET_TOKENS * CHARS_PER_TOKEN:
                log("\n  collection target reached - stopping")
                break
            await run_source(s, src, args.limit_chars, deadline)

    tokens = Progress.total_chars / CHARS_PER_TOKEN
    log("\n" + "=" * 88)
    log(f"TOTAL {human(Progress.total_chars)} chars  ~{human(tokens)} tokens "
        f"({100 * tokens / TARGET_TOKENS:.1f}% of target) "
        f"in {fmt_dur(time.time() - started)}")
    log("=" * 88)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Collect the Gujarati manual corpus (Phase 1).")
    ap.add_argument("--only", nargs="*", default=[], metavar="NAME",
                    help="restrict to these source names")
    ap.add_argument("--include-disabled", action="store_true")
    ap.add_argument("--limit-chars", type=int, default=0,
                    help="stop after this many characters in total")
    ap.add_argument("--max-hours", type=float, default=0.0,
                    help="stop after this many hours (resumable)")
    ap.add_argument("--check", action="store_true",
                    help="check robots.txt and reachability, write nothing")
    ap.add_argument("--status", action="store_true",
                    help="report progress toward the 20% floor and exit")
    args = ap.parse_args()

    sources = load_sources(args.only, args.include_disabled)
    if not sources:
        print("No sources selected. Check --only spelling and 'enabled:' in "
              "data/sources/manual_sources.yaml.")
        sys.exit(1)

    if args.status:
        report_status(sources)
        return
    if args.check:
        asyncio.run(report_check(sources))
        return

    # Logs are filed by topic so phases 2 and 3 can add their own without the
    # directory becoming an unsearchable pile of run logs.
    log_dir = CFG["paths"]["logs"] / "collection"
    log_dir.mkdir(parents=True, exist_ok=True)
    Progress.log_file = open(log_dir / "collect_manual.log", "a", encoding="utf-8")
    try:
        asyncio.run(collect_all(sources, args))
    except KeyboardInterrupt:
        log("\ninterrupted - shards are flushed, rerun to resume")
    finally:
        Progress.log_file.close()
    print("\nprogress: python3 data/collection/collect_manual.py --status")


if __name__ == "__main__":
    main()
