"""Gujarati preprocessing: Unicode normalisation, filtering, boilerplate removal.

Reads every raw shard under `data/raw/` (public and manual alike, provenance
preserved) and writes cleaned documents to `data/clean/`.

The Gujarati-specific choices made here, and why:

* NFC, not NFKC. NFC composes the nukta letters (જ઼ ડ઼ ઢ઼) into single code
  points so the tokenizer sees one spelling; NFKC would additionally destroy
  distinctions we want to keep.
* ZWJ (U+200D) removed, ZWNJ (U+200C) kept. Gujarati never needs an explicit
  joiner to render correctly, so stray ZWJ only creates duplicate spellings of
  identical words. ZWNJ can be orthographically meaningful (it blocks a
  conjunct), so it is preserved.
* Gujarati digits ૦-૯ folded to ASCII. They are rare in modern prose and
  would otherwise occupy vocabulary slots as a high-variance token class.
* Danda ।/॥ kept. They are shared punctuation used in Gujarati and are the
  main sentence terminator in literary text.
* Script filter on the Gujarati block (U+0A80-U+0AFF) over *letter*
  characters only. Gujarati and Devanagari are different Unicode blocks, so this
  cleanly rejects Hindi/Marathi pages that leak into a Gujarati crawl - no
  pretrained language-ID model needed.

Run:

    python data/preprocessing/preprocess.py
    python data/preprocessing/preprocess.py --limit 5000     # smoke test
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from corpus_io import JsonlWriter, human, lang_root, load_config, read_docs  # noqa: E402

ROOT = lang_root(Path(__file__))
CFG = load_config(ROOT)
Q = CFG["quality"]
N = CFG["normalize"]

# Normalisation

GUJARATI_BLOCK = (0x0A80, 0x0AFF)
DEVANAGARI_BLOCK = (0x0900, 0x097F)
LATIN_BLOCKS = ((0x41, 0x5A), (0x61, 0x7A), (0xC0, 0x24F))

INVISIBLES = ["\u200b", "\ufeff", "\u2060", "\u180e", "\u00ad"]
CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f\u0080-\u009f]")
SPACES = re.compile(r"[ \t\u00a0\u1680\u2000-\u200a\u202f\u205f\u3000]+")
MULTI_NL = re.compile(r"\n{3,}")
URL = re.compile(r"https?://\S+|www\.\S+")
EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")
PUNCT_FOLD = {"\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"',
              "\u2013": "-", "\u2014": "-", "\u2026": "..."}
GU_DIGITS = {chr(0x0AE6 + i): str(i) for i in range(10)}

#: Very frequent Gujarati function words. Genuine prose contains several;
#: keyword spam and machine-generated pages do not.
STOPWORDS = ["અને", "છે", "માટે", "થી", "તે", "કે", "પણ", "નથી", "હતું", "હતી",
             "એક", "આ", "તેમણે", "કરવા", "સાથે", "તથા", "જે", "હોય", "શકે", "ના"]

BOILERPLATE = ["javascript", "enable cookies", "lorem ipsum", "all rights reserved",
               "terms of service", "click here", "your browser", "advertisement"]


def normalize(text: str) -> str:
    """Apply the full Gujarati normalisation chain to one document."""
    if not text:
        return ""
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = CONTROL.sub(" ", text)
    for ch in INVISIBLES:
        text = text.replace(ch, "")
    if N.get("strip_urls", True):
        text = URL.sub(" ", text)
    if N.get("strip_emails", True):
        text = EMAIL.sub(" ", text)
    for a, b in PUNCT_FOLD.items():
        text = text.replace(a, b)

    if N.get("strip_zwj", True):
        text = text.replace("\u200d", "")          # joiner: never needed in Gujarati
    if not N.get("keep_zwnj", True):
        text = text.replace("\u200c", "")          # non-joiner: kept by default
    if N.get("map_digits_to_ascii", True):
        text = text.translate(str.maketrans(GU_DIGITS))

    text = re.sub(r"\s*([\u0964\u0965])\s*", r"\1 ", text)   # consistent danda spacing
    text = SPACES.sub(" ", text)
    text = "\n".join(l.strip() for l in text.split("\n"))
    return MULTI_NL.sub("\n\n", text).strip()


# Filtering


def _ratio(text: str, blocks) -> float:
    """Share of *letter* characters falling inside the given Unicode blocks."""
    total = hit = 0
    for ch in text:
        if not unicodedata.category(ch).startswith("L"):
            continue
        total += 1
        cp = ord(ch)
        if any(lo <= cp <= hi for lo, hi in blocks):
            hit += 1
    return hit / total if total else 0.0


def _dup_line_fraction(text: str) -> float:
    lines = [l for l in text.split("\n") if l.strip()]
    if len(lines) < 2:
        return 0.0
    return 1 - len(set(lines)) / len(lines)


def _top_ngram_fraction(words: list, n: int) -> float:
    """Character share of the single most frequent word n-gram (repetition test)."""
    if len(words) < n:
        return 0.0
    grams = [" ".join(words[i:i + n]) for i in range(len(words) - n + 1)]
    top, cnt = Counter(grams).most_common(1)[0]
    total = sum(len(w) for w in words)
    return (len(top) * cnt) / total if total else 0.0


def check(text: str):
    """Return (keep, reason); reason is None when the document is kept.

    Filters are ordered cheap-first so most junk is rejected before the
    expensive n-gram statistics run.
    """
    # --- length: cheapest checks, so they run first ---
    n_chars = len(text)
    if n_chars < Q["min_chars"]:                    # 1. Too short?
        return False, "too_short"
    if n_chars > Q["max_chars"]:                    # 2. Too long?
        return False, "too_long"

    words = text.split()
    if len(words) < Q["min_words"]:                 # 3. Too few words?
        return False, "too_few_words"
    mean_wl = sum(len(w) for w in words) / len(words)
    if not (Q["min_mean_word_len"] <= mean_wl <= Q["max_mean_word_len"]):
        return False, "bad_word_length"             # 4. Strange average word length?

    # --- script: is this actually Gujarati prose? ---
    if _ratio(text, [GUJARATI_BLOCK]) < Q["min_script_ratio"]:
        return False, "not_gujarati"                # 5. Enough Gujarati?
    if _ratio(text, [DEVANAGARI_BLOCK]) > Q["max_devanagari_ratio"]:
        return False, "devanagari_leakage"          # 6. Too much Hindi/Marathi?
    if _ratio(text, LATIN_BLOCKS) > Q["max_latin_ratio"]:
        return False, "too_much_latin"              # 7. Too much English?
    if sum(1 for w in STOPWORDS if w in text) < Q["min_stopword_hits"]:
        return False, "no_stopwords"                # 8. Real prose, or just a menu?

    # --- content quality: expensive checks last ---
    if sum(1 for c in text if c.isdigit()) / n_chars > Q["max_digit_ratio"]:
        return False, "digit_heavy"                 # 9. Mostly numbers (a table)?
    if sum(1 for m in BOILERPLATE if m in text.lower()) >= 2:
        return False, "boilerplate"                 # 10. Cookie/subscribe notice?
    if _dup_line_fraction(text) > Q["max_duplicate_line_fraction"]:
        return False, "repeated_lines"              # 11. Same line over and over?
    for n, key in ((2, "max_top_2gram"), (3, "max_top_3gram"), (4, "max_top_4gram")):
        if _top_ngram_fraction(words, n) > Q[key]:
            return False, f"repetitive_{n}gram"     # 12. One phrase dominates?
    return True, None


def main() -> None:
    ap = argparse.ArgumentParser(description="Normalise and filter the raw Gujarati corpus.")
    ap.add_argument("--limit", type=int, default=None, help="Process only N documents")
    ap.add_argument("--no-line-pass", action="store_true",
                    help="Skip the repeated-line boilerplate pass (saves RAM)")
    args = ap.parse_args()

    raw_dir = CFG["paths"]["raw"]
    out = CFG["paths"]["clean"] / "corpus.jsonl.gz"
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        out.unlink()

    # Pass A: find lines repeated across many documents (nav bars, OCR headers).
    #
    # Counting every distinct line over a 900k-document corpus needs several
    # gigabytes, because almost every line is unique and never becomes
    # boilerplate. So the counter is pruned periodically: singletons seen
    # before the previous checkpoint are dropped. This is lossy counting, and
    # it is safe for this purpose - genuine boilerplate appears in a large
    # fraction of documents and is re-seen long before every prune, whereas the
    # entries being discarded are ordinary sentences that would never have
    # reached the threshold.
    blocked = set()
    if not args.no_line_pass:
        df = Counter()
        min_df = int(CFG.get("repeated_line_df", 50))
        prune_every = int(CFG.get("repeated_line_prune_every", 100_000))
        print("pass A: counting repeated lines ...")
        for i, doc in enumerate(read_docs(raw_dir, limit=args.limit), 1):
            for line in {l.strip() for l in doc.text.split("\n")
                         if 8 <= len(l.strip()) <= 200}:
                df[line] += 1
            if i % prune_every == 0:
                before = len(df)
                df = Counter({l: c for l, c in df.items() if c > 1})
                print(f"  {i:,} documents | tracking {len(df):,} repeated lines "
                      f"(pruned {before - len(df):,} singletons)", flush=True)
        blocked = {l for l, c in df.items() if c >= min_df}
        print(f"  blocked {len(blocked):,} boilerplate line patterns")

    # Pass B: normalise, strip boilerplate lines, filter, write.
    print("pass B: normalising and filtering ...")
    writer = JsonlWriter(out)
    reasons = Counter()
    by_source = {}
    seen = raw_chars = 0

    for i, doc in enumerate(read_docs(raw_dir, limit=args.limit), 1):
        seen += 1
        raw_chars += doc.n_chars
        text = normalize(doc.text)
        if blocked:
            text = "\n".join(l for l in text.split("\n") if l.strip() not in blocked).strip()

        if not text:
            keep, reason = False, "empty_after_normalisation"
        else:
            keep, reason = check(text)

        if keep:
            doc.text = text
            # The cleaned document is a new artifact with different text, so
            # its id is recomputed here rather than carried over from the raw
            # shard. This is also what guarantees the id is unique per
            # document: ids assigned at collection time were derived from the
            # source URL, which is identical for every document of a packaged
            # dataset. Deduplication (LSH keys) and split assignment (hashing)
            # both require uniqueness.
            doc.recompute_id()
            writer.write(doc)
        else:
            reasons[reason] += 1
            by_source.setdefault(doc.source_name, Counter())[reason] += 1

        if i % 50000 == 0:
            print(f"  {i:,} seen | {writer.n:,} kept ({writer.n/i*100:.1f}%)", flush=True)

    writer.close()

    summary = {
        "language": CFG["language"], "documents_in": seen, "documents_kept": writer.n,
        "keep_rate": round(writer.n / max(seen, 1), 4),
        "chars_in": raw_chars, "chars_kept": writer.chars,
        "rejections": dict(reasons),
        "rejections_by_source": {k: dict(v) for k, v in by_source.items()},
        "quality_thresholds": Q, "normalisation": N,
        "boilerplate_lines_blocked": len(blocked),
    }
    rep = CFG["paths"]["reports"]
    rep.mkdir(parents=True, exist_ok=True)
    (rep / "cleaning.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n==== CLEAN [{CFG['language']}] ====")
    print(f"  {seen:,} in -> {writer.n:,} kept ({writer.n/max(seen,1)*100:.1f}% of docs, "
          f"{human(writer.chars)} chars)")
    for r, c in reasons.most_common():
        print(f"    {r:<28} {c:>10,}")
    print(f"  summary -> {rep / 'cleaning.json'}")


if __name__ == "__main__":
    main()
