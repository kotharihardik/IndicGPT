"""Turn the Phase 1 splits into flat token streams the trainer can memory-map.

Each split becomes one `uint16` file: every document is encoded with this
language's tokenizer, wrapped `<s> ... </s>`, and appended to one stream.

`meta.json` records documents, characters, bytes and tokens per split. The byte
totals are the denominator of bits-per-byte.

    python3 train/prepare_data.py                       # all three splits
    python3 train/prepare_data.py --splits validation   # just one
    python3 train/prepare_data.py --limit-docs 500 --out-dir /tmp/tiny   # test
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path
from typing import Dict, Iterator, List, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "model"))
from corpus_io import human, lang_root, load_config, read_docs, split_file  # noqa: E402
from config import load_stage, tokenizer_path  # noqa: E402

ROOT = lang_root(Path(__file__))
PIPELINE = load_config(ROOT)
LANG = PIPELINE["language"]

SPLITS = ("train", "validation", "test")

#: Documents handed to one worker at a time.
CHUNK_DOCS = 1000

#: Filled once per worker process by `_init_worker`.
_SP = None


def _init_worker(model_path: str) -> None:
    """Load the tokenizer once per worker instead of once per chunk."""
    global _SP
    import sentencepiece as spm

    _SP = spm.SentencePieceProcessor()
    _SP.load(model_path)


def _encode_chunk(args: Tuple[List[str], int, int]) -> Tuple[np.ndarray, Dict[str, int]]:
    """Encode one chunk of documents: (uint16 token ids, counts)."""
    texts, bos_id, eos_id = args
    encoded = _SP.encode(texts)                     # list of list[int]
    n_content = sum(len(ids) for ids in encoded)
    stream = np.empty(n_content + 2 * len(encoded), dtype=np.uint16)
    at = 0
    for ids in encoded:
        stream[at] = bos_id
        stream[at + 1:at + 1 + len(ids)] = ids
        stream[at + 1 + len(ids)] = eos_id
        at += len(ids) + 2
    counts = {
        "docs": len(texts),
        "content_tokens": n_content,
        "special_tokens": 2 * len(texts),
        "chars": sum(len(t) for t in texts),
        "bytes": sum(len(t.encode("utf-8")) for t in texts),
    }
    return stream, counts


def iter_chunks(path: Path, limit: int | None, bos_id: int,
                eos_id: int) -> Iterator[Tuple[List[str], int, int]]:
    """Stream documents from a split, grouped into worker-sized chunks."""
    chunk: List[str] = []
    for n, doc in enumerate(read_docs(path)):
        if limit and n >= limit:
            break
        chunk.append(doc.text)
        if len(chunk) >= CHUNK_DOCS:
            yield chunk, bos_id, eos_id
            chunk = []
    if chunk:
        yield chunk, bos_id, eos_id


def encode_split(split: str, src: Path, dst: Path, model_path: Path,
                 bos_id: int, eos_id: int, workers: int,
                 limit: int | None) -> Dict[str, int]:
    """Encode one split into `dst` and return its totals.

    Appended chunk by chunk, so peak memory is one chunk, not the corpus.
    """
    totals = {"docs": 0, "content_tokens": 0, "special_tokens": 0,
              "chars": 0, "bytes": 0}
    started = time.time()
    dst.parent.mkdir(parents=True, exist_ok=True)

    with open(dst, "wb") as out:
        with mp.Pool(workers, initializer=_init_worker,
                     initargs=(str(model_path),)) as pool:
            work = iter_chunks(src, limit, bos_id, eos_id)
            for stream, counts in pool.imap(_encode_chunk, work, chunksize=1):
                out.write(stream.tobytes())
                for key, value in counts.items():
                    totals[key] += value
                if totals["docs"] % (50 * CHUNK_DOCS) == 0:
                    rate = totals["docs"] / max(time.time() - started, 1e-9)
                    print(f"    {split:<11}{totals['docs']:>10,} docs  "
                          f"{human(totals['content_tokens']):>9} tokens  "
                          f"{rate:>7.0f} docs/s", flush=True)

    totals["tokens"] = totals["content_tokens"] + totals["special_tokens"]
    totals["seconds"] = round(time.time() - started, 1)
    totals["bin"] = dst.name
    totals["bytes_on_disk"] = dst.stat().st_size
    return totals


def summarise(split: str, totals: Dict[str, int]) -> str:
    """One human-readable line per split for the run log."""
    cpt = totals["chars"] / max(totals["content_tokens"], 1)
    return (f"  {split:<11}{totals['docs']:>10,} docs  "
            f"{totals['tokens']:>13,} tokens  "
            f"{human(totals['bytes']):>9} bytes  "
            f"{cpt:>5.3f} chars/token  {totals['seconds']:>7.1f}s")


def main() -> None:
    """Tokenize the requested splits and write the streams plus meta.json."""
    ap = argparse.ArgumentParser(description="Tokenize the splits for pretraining.")
    ap.add_argument("--splits", nargs="+", default=list(SPLITS), choices=SPLITS)
    ap.add_argument("--out-dir", default=None,
                    help="where to write the .bin files (default: data/tokenized)")
    ap.add_argument("--limit-docs", type=int, default=None,
                    help="stop after this many documents per split (testing)")
    ap.add_argument("--workers", type=int, default=max(1, mp.cpu_count() - 1))
    ap.add_argument("--force", action="store_true",
                    help="re-encode splits whose .bin already exists")
    args = ap.parse_args()

    cfg = load_stage(ROOT, "pretrain")
    out_dir = Path(args.out_dir) if args.out_dir else Path(cfg["data_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    model_path = tokenizer_path(ROOT)
    import sentencepiece as spm
    sp = spm.SentencePieceProcessor()
    sp.load(str(model_path))
    vocab_size = sp.get_piece_size()
    if vocab_size > np.iinfo(np.uint16).max + 1:
        raise SystemExit(f"vocab {vocab_size} does not fit in uint16")

    print(f"\nTokenizing {PIPELINE['language_name']} with {model_path.name} "
          f"({vocab_size:,} pieces), {args.workers} workers\n")

    meta_path = out_dir / "meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    meta.update({
        "language": LANG,
        "tokenizer": str(model_path.relative_to(ROOT)),
        "vocab_size": vocab_size,
        "dtype": "uint16",
        "document_wrapping": "<s> text </s>",
        "special_ids": {"pad": sp.pad_id(), "unk": sp.unk_id(),
                        "bos": sp.bos_id(), "eos": sp.eos_id()},
    })
    meta.setdefault("splits", {})

    for split in args.splits:
        try:
            src = split_file(PIPELINE["paths"]["splits"], split)
        except FileNotFoundError as exc:
            raise SystemExit(f"{exc} - run the Phase 1 pipeline first")
        dst = out_dir / f"{split}.bin"
        if dst.exists() and not args.force:
            print(f"  {split:<11}already encoded, skipping (--force to redo)")
            continue
        totals = encode_split(split, src, dst, model_path,
                              sp.bos_id(), sp.eos_id(), args.workers,
                              args.limit_docs)
        meta["splits"][split] = totals
        print(summarise(split, totals), flush=True)

    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False),
                         encoding="utf-8")
    print(f"\n  wrote {meta_path}\n")


if __name__ == "__main__":
    main()
