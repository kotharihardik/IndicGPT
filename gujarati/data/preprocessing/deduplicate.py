"""Remove duplicate and near-duplicate documents.

Run AFTER preprocessing and BEFORE splitting. That ordering matters: if a
near-duplicate of a test document survives into train, the Phase-2 perplexity
numbers are meaningless.

Two passes over the cleaned corpus:

1. Exact - SHA1 of the whitespace-collapsed text. Public web corpora overlap
   heavily (CulturaX, OSCAR and CC-100 all re-derive from Common Crawl), so this
   removes the bulk cheaply.
2. Near-duplicate - MinHash + LSH over 5-word shingles at Jaccard ~0.8,
   which catches syndicated news reprinted with a changed headline.

Manual documents are streamed first, so when a manual and a public document
collide the manual copy is the one kept and the 20% quota is not eroded by dedup.

    python3 data/preprocessing/deduplicate.py
    python3 data/preprocessing/deduplicate.py --no-minhash    # exact only, low RAM
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from corpus_io import JsonlWriter, human, lang_root, load_config, read_docs  # noqa: E402

ROOT = lang_root(Path(__file__))
CFG = load_config(ROOT)
WS = re.compile(r"\s+")


def content_hash(text: str) -> str:
    """Exact-duplicate key: SHA1 of whitespace-collapsed, case-folded text."""
    return hashlib.sha1(WS.sub(" ", text).strip().lower().encode("utf-8", "ignore")).hexdigest()


def shingles(text: str, k: int = 5, cap: int = 20000) -> set:
    """Word k-shingles, used as the MinHash input set."""
    words = WS.sub(" ", text).strip().split(" ")
    if len(words) < k:
        return {" ".join(words).encode("utf-8", "ignore")} if words else set()
    out = set()
    for i in range(len(words) - k + 1):
        out.add(" ".join(words[i:i + k]).encode("utf-8", "ignore"))
        if len(out) >= cap:
            break
    return out


def main() -> None:
    """Remove exact then near-duplicate documents into data/dedup/."""
    ap = argparse.ArgumentParser(description="Exact + near-duplicate removal.")
    ap.add_argument("--no-minhash", action="store_true", help="Skip near-duplicate detection")
    ap.add_argument("--threshold", type=float, default=0.8, help="MinHash Jaccard threshold")
    # 128 permutations is the usual floor for MinHash-LSH near-duplicate
    # detection (Dolma uses 128; RefinedWeb uses far more in buckets). The
    # estimator's standard error is ~1/sqrt(num_perm), so 64 gives ~12.5% -
    # noisy enough to miss genuine duplicates near the threshold.
    # AUDITED on this corpus: re-running detection over a 60,000-document
    # sample of the final split at num_perm=256 found 10 surviving near
    # duplicates (0.017%) and 0 exact duplicates, so 64 cost very little here.
    # 128 is still the right default for a defensible pipeline.
    ap.add_argument("--num-perm", type=int, default=128,
                    help="MinHash permutations (higher = better recall, more RAM)")
    args = ap.parse_args()

    clean_dir = CFG["paths"]["clean"]
    out = CFG["paths"]["dedup"] / "corpus.jsonl.gz"
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        out.unlink()   # dedup is not incremental; rebuild from scratch

    lsh = mh_cls = None
    if not args.no_minhash:
        from datasketch import MinHash, MinHashLSH
        mh_cls = MinHash
        lsh = MinHashLSH(threshold=args.threshold, num_perm=args.num_perm)

    def ordered():
        """Manual documents first, so they win every collision."""
        for d in read_docs(clean_dir):
            if d.manual:
                yield d
        for d in read_docs(clean_dir):
            if not d.manual:
                yield d

    seen, writer = set(), JsonlWriter(out)
    n_in = n_exact = n_near = 0
    try:
        for doc in ordered():
            n_in += 1
            h = content_hash(doc.text)
            if h in seen:
                n_exact += 1
                continue
            if lsh is not None:
                sig = mh_cls(num_perm=args.num_perm)
                for sh in shingles(doc.text):
                    sig.update(sh)
                if lsh.query(sig):
                    n_near += 1
                    continue
                # Key the index by position, not by doc.id. LSH keys are
                # internal to this pass and never used again, so they only need
                # to be unique within it. Relying on doc.id here made the whole
                # stage crash on the first id collision, and it silently wrote
                # a truncated corpus that the next stage then split.
                lsh.insert(n_in, sig)
            seen.add(h)
            writer.write(doc)
            if n_in % 50000 == 0:
                print(f"  {n_in:,} in | {writer.n:,} kept | exact={n_exact:,} near={n_near:,}",
                      flush=True)
    finally:
        writer.close()

    summary = {
        "language": CFG["language"], "documents_in": n_in, "documents_out": writer.n,
        "exact_duplicates": n_exact, "near_duplicates": n_near,
        "removal_rate": round(1 - writer.n / max(n_in, 1), 4),
        "chars_kept": writer.chars,
        "minhash": None if args.no_minhash else {"threshold": args.threshold,
                                                 "num_perm": args.num_perm, "shingle_k": 5},
    }
    reports = CFG["paths"]["reports"] / "json"
    reports.mkdir(parents=True, exist_ok=True)
    path = reports / "dedup.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"\n==== DEDUP [{CFG['language']}] ====")
    print(f"  in={n_in:,} out={writer.n:,} | exact={n_exact:,} | near={n_near:,} "
          f"| removed {summary['removal_rate']*100:.1f}% | {human(writer.chars)} chars kept")
    print(f"  summary -> {path}")


if __name__ == "__main__":
    main()
