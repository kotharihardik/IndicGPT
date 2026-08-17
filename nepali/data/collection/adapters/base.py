"""Shared components required by every collection adapter.

Configuration, text cleaning, the quality gate, the Document record and the
resume scan are defined here so that the adapter modules contain only the
logic specific to their source type.

The quality gate is the component that differs most from the Gujarati
pipeline, because Nepali cannot be identified by script alone. See
`reject_reason` below.

Cleaning is performed in two stages, of which this file is the first and the
more permissive. `reject_reason` discards a document irrecoverably, whereas the
stricter thresholds in `data/preprocessing/preprocess.py` only exclude it from
the cleaned corpus and can be re-applied at any time. Collection is therefore
deliberately permissive and preprocessing strict.
"""

from __future__ import annotations

import gzip
import html as htmllib
import json
import re
import sys
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from corpus_io import (Document, MANUAL_PDF, MANUAL_WEB,  # noqa: E402
                       lang_root, load_config)

ROOT = lang_root(Path(__file__))
CFG = load_config(ROOT)
LANG = CFG["language"]
UA = CFG.get("crawl", {})["user_agent"]

#: Characters per token, used only for the progress display. The tokenizer
#: does not exist during collection, so this is an estimate. It is set high
#: deliberately: over-collection is inexpensive, whereas discovering a
#: shortfall after collection is not. Exact counts are produced later by
#: `data/splits/corpus_stats.py --tokenizer`.
CHARS_PER_TOKEN = CFG.get("tokens", {}).get("chars_per_token", 4.0)

#: Read from configs/pipeline.yaml rather than hardcoded, so the requirement
#: and the collector cannot drift apart. 20% of 500M tokens is the 100M manual
#: floor; collection_headroom (1.25 here) is the extra margin for what
#: preprocessing and dedup remove afterwards.
_T = CFG["targets"]
MANUAL_FLOOR = float(_T["corpus_tokens"]) * float(_T["manual_fraction"])
TARGET_TOKENS = MANUAL_FLOOR * float(_T["collection_headroom"])

#: File types that are never worth fetching from a sitemap.
SKIP_EXT = (".jpg", ".jpeg", ".png", ".gif", ".webp", ".svg", ".ico", ".mp4",
            ".mp3", ".zip", ".css", ".js", ".json", ".xml", ".epub", ".djvu")

DEVANAGARI = re.compile(r"[ऀ-ॿ]")
LATIN = re.compile(r"[A-Za-z]")
LETTER = re.compile(r"[^\W\d_]", re.UNICODE)
TAG = re.compile(r"<[^>]+>")
SCRIPT_STYLE = re.compile(r"(?is)<(script|style|noscript)\b.*?</\1>")

#: Nepali has no exclusive characters at all. It shares Devanagari with Hindi,
#: Marathi, Sanskrit and Bhojpuri, so a script check cannot identify it. For
#: Gujarati the Unicode block alone is proof, and Sindhi has 17 exclusive
#: Arabic letters; Nepali has nothing equivalent, so identification here has to
#: be LEXICAL. These are high-frequency Nepali function words Hindi does not
#: use, plus the Hindi equivalents Nepali does not use.
#: Measured on real pages: Nepali news scores 11-14 NEPALI_MARKERS per 1000
#: characters and 0.0 HINDI_MARKERS. Hindi pages invert this exactly, which is
#: what makes the ratio test in `reject_reason` reliable.
NEPALI_MARKERS = (" छ ", " छन् ", " हो ", " हुन् ", " भएको ", " गरेको ",
                  " लाई ", " र ", " थियो ", " गर्न ", " भने ", " पनि ",
                  " तर ", " नेपाल")
HINDI_MARKERS = (" है ", " हैं ", " और ", " किया ", " नहीं ", " कहा ",
                 " गया ", " लिए ", " रहे ", " करने ", " था ", " थी ")

#: "Also read:" link lists. WordPress themes put these *inside* the article
#: body, so trafilatura keeps them (correctly, as far as it can tell) and only
#: a Nepali-aware rule can strip them.
BOILERPLATE_TAIL = re.compile(
    r"(?:यो\s*पनि\s*पढ्नुहोस्|यी\s*पनि\s*पढ्नुहोस्|थप\s*पढ्नुहोस्|"
    r"सम्बन्धित\s*समाचार|Also\s*Read)\s*[:：]?.*$", re.S | re.I)

#: Common Nepali function words. A page with none of these is a menu or a list
#: of names, not prose.
STOPWORDS = ("छ", "हो", "र", "पनि", "तर", "लाई", "भएको", "गरेको", "हुन्",
             "छन्", "भने", "गर्न", "देखि", "सम्म", "बाट", "मा")


def clean_text(raw: str, is_html: bool = False) -> str:
    """Strip markup (if any) and normalise whitespace and Unicode.

    Pass `is_html=True` for raw HTML, which is what the WordPress API
    returns. Sitemap pages go through trafilatura first, so they arrive here
    as plain text already.

    Order matters in the HTML branch: <script> has to go *with its contents*
    before the generic tag strip, or the JavaScript stays and only the angle
    brackets go. Block-closing tags become newlines first, otherwise the whole
    article collapses onto one line.

    NFC also runs in preprocess.py. That is on purpose, not an oversight: the
    URL de-duplication and the shard files both compare normalised text, so it
    cannot wait for a later stage.
    """
    if not raw:
        return ""
    if is_html:
        raw = SCRIPT_STYLE.sub(" ", raw)
        raw = re.sub(r"(?i)</(p|div|br|li|h[1-6]|tr)\s*>", "\n", raw)
        raw = htmllib.unescape(TAG.sub(" ", raw))
    text = unicodedata.normalize("NFC", raw)
    text = text.replace("​", "").replace("﻿", "")   # zero-width, BOM
    text = BOILERPLATE_TAIL.sub("", text)
    text = re.sub(r"[ \t ]+", " ", text)
    text = "\n".join(line.strip() for line in text.split("\n"))
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def reject_reason(text: str) -> str:
    """Return '' to keep the document, or a short reason string to drop it.

    The two Devanagari checks are the important ones and they do different
    jobs. The script ratio only proves the page is Devanagari, which a Hindi
    page satisfies just as well. The marker counts then decide *which*
    Devanagari language it is: a page needs enough Nepali function words per
    1000 characters, and it must not contain more Hindi markers than Nepali
    ones.

    Both thresholds are deliberately loose here. `preprocess.py` re-checks with
    the strict values (3.0 markers per 1000 chars, and a 1.5x Nepali:Hindi
    ratio). Dropping a page here means re-crawling to get it back; dropping it
    in preprocessing is just a config edit.

    The returned string is counted in the progress display. That is how I
    found the Hindi leakage on two of the sites, which showed up as thousands
    of 'hindi-not-nepali' rejections rather than as silently bad data.
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

    # Devanagari alone proves nothing, because Hindi looks identical here.
    # Pad with spaces so a marker at the very start or end still matches.
    padded = " " + text + " "
    per_1k = max(len(text) / 1000, 1)
    nepali = sum(padded.count(w) for w in NEPALI_MARKERS) / per_1k
    hindi = sum(padded.count(w) for w in HINDI_MARKERS) / per_1k
    if nepali < 2.0:
        return "not-nepali"           # Devanagari, but not Nepali vocabulary
    if hindi > nepali:
        return "hindi-not-nepali"     # a Hindi page inside a Nepali crawl

    if len(LATIN.findall(text)) / letters > 0.35:
        return "latin-heavy"
    if sum(1 for w in STOPWORDS if w in text) < 2:
        return "no-stopwords"
    return ""


def existing(source_name: str) -> tuple:
    """Return (docs, chars, urls) already stored for one source.

    This is the whole resume mechanism. It reads the shard files themselves
    instead of a separate progress file, so the resume state cannot drift out
    of sync with what is actually on disk. A shard whose tail was cut off by
    Ctrl-C is skipped rather than crashing the run.
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
            continue          # truncated tail from an interrupted run
    return docs, chars, urls


def split_long_text(text: str, limit: int) -> list:
    """Cut a long document into parts of at most roughly `limit` characters.

    Court judgments arrive as one very long blob, well past the `max_chars`
    quality filter, so they would be thrown away whole. Splitting on blank
    lines keeps paragraphs and sentences intact; splitting on a raw character
    count would cut words in half and corrupt the tokenizer training text.
    """
    parts, buf, size = [], [], 0
    for para in text.split("\n\n"):
        if size + len(para) > limit and buf:
            parts.append("\n\n".join(buf))
            buf, size = [], 0
        buf.append(para)
        size += len(para) + 2      # +2 accounts for the rejoined blank line
    if buf:
        parts.append("\n\n".join(buf))
    return parts


def make_doc(src: dict, text: str, url: str, title: str = "",
             extra: dict = None, method: str = "") -> Document:
    """Build one Document with the provenance fields the 20% audit needs.

    The label records what actually happened, not what the source is nominally
    for. The `pdf` method is the case that matters: it reads an embedded text
    layer with PyMuPDF and runs no OCR at all, so those documents are
    `manual_pdf` / `pdf_text_layer`. Only a scan actually processed through OCR
    by this pipeline would qualify as `manual_ocr`. Both labels count toward
    the >=20% requirement, so the stronger claim offers no advantage, and the
    distinction keeps the dataset statistics accurate.
    """
    is_pdf = src["method"] == "pdf"
    return Document(
        text=text, language=LANG, source_name=src["name"],
        source_type="manual",
        collection_method=method or ("pdf_text_layer" if is_pdf else "scrape"),
        manual_or_public=MANUAL_PDF if is_pdf else MANUAL_WEB,
        source_url=url, title=title or None, license=src.get("license", ""),
        extra={"method": src["method"], **(extra or {})})
