"""Train the Gujarati tokenizer from scratch (SentencePiece Unigram).

Nothing pretrained is used. The model is fitted on this project's own Gujarati
training split only, and its vocabulary is not shared with the other language.

Unigram, not BPE. Gujarati is agglutinative: one stem takes a chain of case and
postposition suffixes (ઘર, ઘરમાં, ઘરોમાં, ઘરોમાંથી). BPE merges greedily, so
whether a suffix boundary survives depends on which merges fired first, and the
same stem can come out as `ઘર|ોમાં` in one word and `ઘરો|માં` in another.
Unigram fits a probabilistic model over the whole corpus instead, so a stem is
spelled the same way wherever it appears. At 25M parameters that matters:
every inconsistent spelling splits one stem's evidence across two embedding
rows.

The settings that matter here, and why:

    normalization_rule_name="identity"   preprocess.py already normalised the
        text. SentencePiece's default nmt_nfkc rule would normalise it a second
        time, differently, so the tokenizer would no longer match the corpus.
    required_chars=GUJARATI_CHARS        pins every Gujarati code point into the
        vocabulary. Without it a rare matra or the virama can be dropped and
        every word containing it falls back to bytes.
    byte_fallback=True                   nothing is unrepresentable, so UNK is
        ~0 by construction and byte-fallback rate is the real coverage metric.
    split_by_unicode_script=True         stops pieces straddling Gujarati and
        Latin, which is common in news copy and generalises to nothing.
    split_digits=True                    digits become single pieces instead of
        memorised numbers like "2024".

If the training text has to be capped, whole documents are subsampled at a
fixed probability rather than truncating the file. Shards are read in filename
order, so stopping early would train the tokenizer almost entirely on
abtakmedia and leave the encyclopedia register out of the vocabulary.

    python tokenizer/train_tokenizer.py --sweep    # candidate vocab sizes
    python tokenizer/tokenizer_stats.py --compare-sweep
    # set tokenizer.vocab_size in configs/pipeline.yaml, then:
    python tokenizer/train_tokenizer.py            # final model
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
import unicodedata
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data"))
from corpus_io import human, lang_root, load_config, read_docs  # noqa: E402

ROOT = lang_root(Path(__file__))
CFG = load_config(ROOT)
LANG = CFG["language"]
TCFG = CFG["tokenizer"]

#: Short algorithm name for filenames: "sentencepiece_unigram" -> "unigram".
ALGO = TCFG.get("algorithm", "sentencepiece_unigram").split("_")[-1]


def model_stem(vocab_size: int) -> str:
    """Filename stem for one tokenizer: size first, language last.

    32,000 -> "32k_unigram_guj". A size that is not a whole number of
    thousands keeps its full digits, so nothing is silently rounded.
    """
    size = f"{vocab_size // 1000}k" if vocab_size % 1000 == 0 else str(vocab_size)
    return f"{size}_{ALGO}_{LANG}"


#: Every assigned code point in the Gujarati block U+0A80-U+0AFF: independent
#: vowels, consonants, dependent vowel signs (matras), anusvara, visarga,
#: candrabindu, nukta, virama, avagraha, the Gujarati digits and the currency
#: sign. Pinned into the vocabulary so no Gujarati character can ever fall
#: back to bytes, however rare it is in the sample.
GUJARATI_CHARS = "".join(
    chr(cp) for cp in range(0x0A80, 0x0B00)
    if unicodedata.name(chr(cp), "")
)

#: Minimum line length worth training on. Shorter lines are navigation
#: fragments and headline stubs that survived cleaning; they bias the model
#: toward very short pieces.
MIN_LINE_CHARS = 20


# Training text


def train_split_path() -> Path:
    """Path to the training split, with a clear error if it is missing."""
    path = CFG["paths"]["splits"] / "train.jsonl.gz"
    if not path.exists():
        raise SystemExit(
            f"{path} not found.\n"
            f"Run the pipeline in order first:\n"
            f"  python data/preprocessing/preprocess.py\n"
            f"  python data/preprocessing/deduplicate.py\n"
            f"  python data/splits/build_splits.py")
    return path


def estimate_train_chars() -> int:
    """Total characters in the training split, for unbiased subsampling.

    Prefers the count `build_splits.py` already wrote to
    `report/splits_gu.json` so we do not pay a full extra pass over a
    multi-gigabyte corpus just to decide a sampling probability.
    """
    report = CFG["paths"]["reports"] / "splits.json"
    if report.exists():
        try:
            blob = json.loads(report.read_text(encoding="utf-8"))
            for key in ("train_chars", "chars_train"):
                if key in blob:
                    return int(blob[key])
            train = blob.get("splits", {}).get("train", {})
            if "chars" in train:
                return int(train["chars"])
        except (ValueError, KeyError, TypeError):
            pass

    print("  counting training characters (no splits report to read) ...")
    total = sum(len(doc.text) for doc in read_docs(train_split_path()))
    print(f"  training split holds {human(total)} characters")
    return total


def export_training_text(max_chars: int, seed: int) -> tuple:
    """Write the training split to a flat UTF-8 file for SentencePiece.

    Uses the *train* split only. Validation and test text must never reach the
    tokenizer, or the fertility and coverage numbers measured on held-out text
    in `tokenizer_stats.py` would be measured on data it had already seen.

    Args:
        max_chars: Cap on characters written. When the split is larger, whole
            documents are kept with probability `max_chars/total` so every
            source stays represented in proportion.
        seed: Fixes the subsampling so a rerun produces the same training text.

    Returns:
        `(path, chars_written, per_source_chars)`.
    """
    src = train_split_path()
    out = CFG["paths"]["tokenizer"] / "spm_input.txt"
    out.parent.mkdir(parents=True, exist_ok=True)

    keep_prob = 1.0
    if max_chars:
        total = estimate_train_chars()
        if total > max_chars:
            keep_prob = max_chars / total
            print(f"  subsampling documents at p={keep_prob:.4f} so all "
                  f"sources stay proportionally represented")

    rng = random.Random(seed)
    by_source: Counter = Counter()
    written = n_docs = 0
    t0 = time.time()

    with open(out, "w", encoding="utf-8") as fh:
        for doc in read_docs(src):
            if keep_prob < 1.0 and rng.random() > keep_prob:
                continue
            kept = 0
            for line in doc.text.split("\n"):
                line = line.strip()
                if len(line) < MIN_LINE_CHARS:
                    continue
                fh.write(line + "\n")
                kept += len(line)
            if kept:
                written += kept
                n_docs += 1
                by_source[doc.source_name] += kept

    print(f"  training text: {human(written)} characters from {n_docs:,} "
          f"documents in {time.time() - t0:.0f}s -> {out}")
    if by_source:
        print("  composition by source:")
        for name, chars in by_source.most_common():
            print(f"    {name:<24}{human(chars):>10} chars "
                  f"({100 * chars / written:5.1f}%)")
    return out, written, dict(by_source)


# Training


def train(input_path: Path, vocab_size: int, sweep: bool = False) -> Path:
    """Train one SentencePiece Unigram model. Returns the `.model` path.

    SentencePiece writes `<prefix>.model` and `<prefix>.vocab` side by side,
    so training happens in a staging directory and the two artifacts are then
    filed separately: `tokenizer/model/` is what Phase 2 loads, while
    `tokenizer/vocab/` holds the human-readable piece list used for the
    report. Keeping them apart stops the loadable model from being lost among
    sweep by-products.
    """
    import sentencepiece as spm

    stem = model_stem(vocab_size)
    staging = CFG["paths"]["tokenizer"] / "_staging"
    staging.mkdir(parents=True, exist_ok=True)
    out_prefix = staging / stem
    spm.SentencePieceTrainer.train(
        input=str(input_path),
        model_prefix=str(out_prefix),
        model_type="unigram",
        vocab_size=vocab_size,
        character_coverage=float(TCFG.get("character_coverage", 0.9995)),

        # Preprocessing already normalised this text. Do not touch it again.
        normalization_rule_name="identity",
        remove_extra_whitespaces=False,

        # Coverage: no Gujarati character may ever become byte fallback.
        byte_fallback=True,
        required_chars=GUJARATI_CHARS,

        # Segmentation behaviour.
        split_by_unicode_script=True,
        split_by_whitespace=True,
        split_digits=True,
        allow_whitespace_only_pieces=True,
        max_sentencepiece_length=int(TCFG.get("max_piece_length", 16)),

        # Fixed special-token ids so Phase 2 can rely on them.
        pad_id=0, unk_id=1, bos_id=2, eos_id=3,
        pad_piece="<pad>", unk_piece="<unk>", bos_piece="<s>", eos_piece="</s>",

        input_sentence_size=int(TCFG.get("input_sentence_size", 5_000_000)),
        shuffle_input_sentence=True,
        num_threads=int(TCFG.get("num_threads", 8)),
        train_extremely_large_corpus=bool(TCFG.get("large_corpus", False)),
    )

    model_dir = CFG["paths"]["tokenizer_model"]
    vocab_dir = CFG["paths"]["tokenizer_vocab"]
    if sweep:
        # Candidates go in their own subfolder, so the final model is never
        # lost among them and a candidate of the same size cannot overwrite it.
        model_dir, vocab_dir = model_dir / "sweep", vocab_dir / "sweep"
    model_dir.mkdir(parents=True, exist_ok=True)
    vocab_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_dir / f"{stem}.model"
    Path(f"{out_prefix}.model").replace(model_path)
    Path(f"{out_prefix}.vocab").replace(vocab_dir / f"{stem}.vocab")
    return model_path


def embedding_params(vocab_size: int) -> tuple:
    """(parameters, share of budget) spent on the embedding at this vocab size.

    The input embedding and the output projection are tied in Phase 2, so the
    cost is counted once: `vocab_size x model_dim`.
    """
    d_model = int(TCFG.get("model_dim", 384))
    budget = float(TCFG.get("target_params", 25e6))
    params = vocab_size * d_model
    return params, params / budget


def write_config(vocab_size: int, chars: int, by_source: dict) -> Path:
    """Record every setting and provenance fact needed to reproduce this model."""
    params, share = embedding_params(vocab_size)
    meta = {
        "language": CFG["language"],
        "algorithm": "sentencepiece_unigram",
        "vocab_size": vocab_size,
        "trained_on": "train split only (validation/test never seen)",
        "training_chars": chars,
        "training_composition_by_source": by_source,
        "pretrained_tokenizer_used": False,
        "shared_with_other_language": False,
        "script": "Gujarati (U+0A80-U+0AFF)",
        "required_chars_count": len(GUJARATI_CHARS),
        "settings": {
            "normalization_rule_name": "identity",
            "byte_fallback": True,
            "split_by_unicode_script": True,
            "split_by_whitespace": True,
            "split_digits": True,
            "character_coverage": TCFG.get("character_coverage", 0.9995),
            "max_sentencepiece_length": TCFG.get("max_piece_length", 16),
        },
        "special_tokens": {"<pad>": 0, "<unk>": 1, "<s>": 2, "</s>": 3},
        "phase2_cost": {
            "model_dim": int(TCFG.get("model_dim", 384)),
            "embedding_parameters": params,
            "share_of_target_budget": round(share, 4),
            "note": "input and output embeddings are tied, so counted once",
        },
    }
    path = CFG["paths"]["tokenizer"] / "tokenizer_config.json"
    path.write_text(json.dumps(meta, indent=2, ensure_ascii=False),
                    encoding="utf-8")
    return path


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Train the Gujarati tokenizer from scratch.")
    ap.add_argument("--sweep", action="store_true",
                    help="train every candidate vocab size for comparison")
    ap.add_argument("--vocab-size", type=int, default=None,
                    help="override the configured vocabulary size")
    ap.add_argument("--max-chars", type=int, default=None,
                    help="cap characters exported for training")
    ap.add_argument("--reuse-input", action="store_true",
                    help="reuse an existing spm_input.txt instead of rebuilding")
    args = ap.parse_args()

    max_chars = args.max_chars or int(float(TCFG.get("train_text_max_chars", 2e9)))
    seed = int(TCFG.get("seed", 1337))
    existing = CFG["paths"]["tokenizer"] / "spm_input.txt"

    if args.reuse_input and existing.exists():
        inp, chars, by_source = existing, existing.stat().st_size, {}
        print(f"reusing {inp} ({human(chars)} bytes)")
    else:
        print("exporting training text ...")
        inp, chars, by_source = export_training_text(max_chars, seed)

    if args.sweep:
        sizes = TCFG.get("sweep_vocab_sizes", [8000, 16000, 24000, 32000, 48000])
        print(f"\nsweeping {len(sizes)} vocabulary sizes: "
              f"{', '.join(f'{v:,}' for v in sizes)}")
        for vocab in sizes:
            params, share = embedding_params(vocab)
            print(f"\n--- vocab_size={vocab:,}  "
                  f"(embedding {human(params)} params, "
                  f"{100 * share:.0f}% of the {human(float(TCFG.get('target_params', 25e6)))} "
                  f"budget) ---")
            try:
                train(inp, vocab, sweep=True)
            except RuntimeError as exc:
                # SentencePiece refuses a vocabulary the corpus cannot support.
                print(f"  ! vocab_size={vocab:,} not achievable here: {exc}")
        print("\nsweep done. Compare and get a recommendation with:\n"
              "  python tokenizer/tokenizer_stats.py --compare-sweep")
        return

    vocab = args.vocab_size or int(TCFG["vocab_size"])
    params, share = embedding_params(vocab)
    print(f"\ntraining final Gujarati Unigram model, vocab_size={vocab:,}")
    print(f"  embedding cost: {human(params)} parameters "
          f"({100 * share:.0f}% of the target budget)")
    model = train(inp, vocab)
    config = write_config(vocab, chars, by_source)

    vocab = CFG["paths"]["tokenizer_vocab"] / f"{model_stem(vocab)}.vocab"
    print(f"\n  model  -> {model}")
    print(f"  vocab  -> {vocab}")
    print(f"  config -> {config}")
    print("\nNext:  python tokenizer/tokenizer_stats.py")


if __name__ == "__main__":
    main()
