"""Create deterministic document-level train / validation / test splits.

Design decisions worth defending:

* Document level, never sentence level. Splitting inside a document would
  put half an article in train and half in test.
* Deterministic hashing of the document id with a fixed salt, so the split
  is reproducible and Phase 2/3 stay comparable across re-runs.
* 90/5/5. The character caps in the config sit above what 5% comes to, so
  the fraction is what binds, not the cap.
* Provenance survives the split, so the manual/public breakdown can be
  reported per split.

Run after `deduplicate.py`.

    python3 data/splits/build_splits.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from corpus_io import JsonlWriter, human, lang_root, load_config, read_docs  # noqa: E402

ROOT = lang_root(Path(__file__))
CFG = load_config(ROOT)
S = CFG["splits"]


def bucket(doc_id: str, salt: str) -> float:
    """Map a document id to a stable pseudo-random value in [0, 1)."""
    return int(hashlib.md5(f"{salt}:{doc_id}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF


def main() -> None:
    """Assign every deduplicated document to train, validation or test."""
    # argparse is here so that --help prints help instead of silently starting
    # a full rebuild. Without it an interrupted "--help" once truncated
    # train.jsonl.gz, and the only reason that was recoverable is that the
    # split is deterministic.
    ap = argparse.ArgumentParser(
        description="Build deterministic Nepali train/validation/test splits.")
    ap.add_argument("--dry-run", action="store_true",
                    help="report the split sizes without writing any file")
    args = ap.parse_args()

    salt = str(S.get("seed", 1337))
    val_frac, test_frac = float(S["val_fraction"]), float(S["test_fraction"])
    val_cap, test_cap = int(float(S["val_max_chars"])), int(float(S["test_max_chars"]))
    heldout = set(S.get("heldout_sources", []))

    # The existing splits are deleted up front, so an interrupted run leaves a
    # truncated file rather than a stale one. That is safe only because the
    # split is deterministic and a rerun reproduces it exactly.
    out_root = CFG["paths"]["splits"]
    writers = {}
    if not args.dry_run:
        for name in ("train", "validation", "test"):
            path = out_root / f"{name}.jsonl.gz"
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.exists():
                path.unlink()
            writers[name] = JsonlWriter(path)

    counts = defaultdict(lambda: {"docs": 0, "chars": 0, "manual_docs": 0, "manual_chars": 0})
    val_chars = test_chars = 0

    for doc in read_docs(CFG["paths"]["dedup"]):
        if doc.source_name in heldout:
            split = "test"                      # whole-source held out as a domain-shift probe
        else:
            b = bucket(doc.id, salt)
            if b < val_frac and val_chars < val_cap:
                split = "validation"
            elif b < val_frac + test_frac and test_chars < test_cap:
                split = "test"
            else:
                split = "train"

        if not args.dry_run:
            writers[split].write(doc)
        c = counts[split]
        c["docs"] += 1
        c["chars"] += doc.n_chars
        if doc.manual:
            c["manual_docs"] += 1
            c["manual_chars"] += doc.n_chars
        if split == "validation":
            val_chars += doc.n_chars
        elif split == "test":
            test_chars += doc.n_chars

    for w in writers.values():
        w.close()

    summary = {
        "language": CFG["language"],
        "config": {"seed": salt, "val_fraction": val_frac, "test_fraction": test_frac,
                   "val_max_chars": val_cap, "test_max_chars": test_cap,
                   "heldout_sources": sorted(heldout)},
        "splits": {k: dict(v, manual_char_fraction=round(v["manual_chars"] / max(v["chars"], 1), 4))
                   for k, v in counts.items()},
        "leakage_control": "document-level split applied AFTER exact + MinHash deduplication",
        "note": "token counts per split are reported by corpus_stats.py using the trained tokenizer",
    }
    if args.dry_run:
        print("dry run: nothing written")

    reports = CFG["paths"]["reports"] / "json"
    reports.mkdir(parents=True, exist_ok=True)
    path = reports / "splits.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"\n==== SPLITS [{CFG['language']}] ====")
    for name in ("train", "validation", "test"):
        c = counts[name]
        print(f"  {name:<11} {c['docs']:>9,} docs | {human(c['chars']):>8} chars "
              f"| manual {c['manual_chars']/max(c['chars'],1)*100:5.1f}%")
    print(f"  summary -> {path}")


if __name__ == "__main__":
    main()
