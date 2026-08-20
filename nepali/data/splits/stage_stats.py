#!/usr/bin/env python3
"""Documents / characters / words at every pipeline stage, split manual vs public.

Answers the question the per-stage logs cannot: how much of the corpus is lost
at each step, and whether cleaning and deduplication hit the manual and public
buckets differently. `preprocess.py` and `deduplicate.py` each report their
own totals, but neither breaks the numbers down by provenance, and neither
counts words.

Words here are whitespace-separated tokens - the unit the corpus has *before*
subword tokenization. They are not the tokens the 500M target is measured in;
those only exist once the tokenizer is trained, and are reported by
`corpus_stats.py --tokenizer`. Both are shown so the difference is visible:
subword tokenization increases the count by roughly 50% for these languages.

    python3 data/splits/stage_stats.py            # every stage that exists
    python3 data/splits/stage_stats.py --stage raw
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from corpus_io import lang_root, load_config, read_docs  # noqa: E402

ROOT = lang_root(Path(__file__))
CFG = load_config(ROOT)


def scan(path: Path) -> dict:
    """Count documents, characters and words, separated by provenance.

    Tolerates a truncated tail: this is often run while a later pipeline stage
    is still writing its output, and a half-written gzip member would otherwise
    abort the whole report instead of just that stage.
    """
    import gzip

    acc = {"manual": [0, 0, 0], "public": [0, 0, 0], "partial": False}
    try:
        for doc in read_docs(path):
            row = acc["manual" if doc.manual else "public"]
            row[0] += 1
            row[1] += doc.n_chars
            row[2] += len(doc.text.split())
    except (OSError, EOFError, gzip.BadGzipFile):
        acc["partial"] = True
    return acc


def stages() -> list:
    """(label, path) for each stage that has been produced so far."""
    p = CFG["paths"]
    out = [("raw (collected)", p["raw"]),
           ("clean (preprocessed)", p["clean"] / "corpus.jsonl.gz"),
           ("dedup (deduplicated)", p["dedup"] / "corpus.jsonl.gz"),
           ("train split", p["splits"] / "train.jsonl.gz")]
    return [(label, path) for label, path in out
            if path.exists() and (path.is_dir() or path.stat().st_size > 0)]


def main() -> None:
    """Print documents, characters and words at every pipeline stage."""
    ap = argparse.ArgumentParser(description="Per-stage corpus statistics.")
    ap.add_argument("--stage", help="only this stage (substring match)")
    args = ap.parse_args()

    todo = [(l, p) for l, p in stages()
            if not args.stage or args.stage.lower() in l.lower()]
    if not todo:
        print("no stages found - run the pipeline first")
        return

    print(f"\n{'stage':<24}{'bucket':<9}{'documents':>12}{'characters':>15}"
          f"{'words':>14}")
    print("-" * 74)
    prev = None
    for label, path in todo:
        acc = scan(path)
        tot = [sum(x) for x in zip(acc["manual"], acc["public"])]
        for bucket in ("manual", "public"):
            d, c, w = acc[bucket]
            print(f"{label if bucket == 'manual' else '':<24}{bucket:<9}"
                  f"{d:>12,}{c:>15,}{w:>14,}")
        share = 100 * acc["manual"][2] / tot[2] if tot[2] else 0
        if acc["partial"]:
            label += " (INCOMPLETE - still being written)"
        line = (f"{'':<24}{'TOTAL':<9}{tot[0]:>12,}{tot[1]:>15,}{tot[2]:>14,}"
                f"   manual {share:.1f}% of words")
        if prev:
            line += f"   (kept {100 * tot[2] / prev:.1f}% of words)"
        print(line)
        print("-" * 74)
        prev = tot[2]

    print("\nNote: 'words' are whitespace-separated and are NOT the tokens the\n"
          "500M target is measured in. Subword tokenization raises the count by\n"
          "roughly 50%; run corpus_stats.py --tokenizer for the real figure.")


if __name__ == "__main__":
    main()
