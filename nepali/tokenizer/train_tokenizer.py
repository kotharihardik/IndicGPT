"""Train the Nepali tokenizer from scratch (SentencePiece Unigram).

Nothing pretrained is used. The model is fitted on this project's own Nepali
training split only, and the vocabulary is not shared with the other language.

Unigram rather than BPE because Nepali is agglutinative: one stem takes a chain
of suffixes (घर, घरमा, घरहरूमा, घरहरूबाट). BPE merges greedily, so the same stem can come out split two
different ways in two different words. Unigram scores a whole segmentation, so
the stem stays spelled the same way. At 25M parameters that matters - two
spellings of one stem split its evidence across two embedding rows.

Settings that matter:

    normalization_rule_name="identity"  preprocess.py already normalised the
        text; letting SentencePiece do it again would not match the corpus.
    required_chars=NEPALI_CHARS         pins every Nepali code point into the
        vocabulary so no word can fall back to bytes on its own letters.
    byte_fallback=True                  nothing is unrepresentable, so UNK is 0
        by construction and byte-fallback rate is the real coverage metric.
    split_by_unicode_script=True        stops pieces straddling two scripts.
    split_digits=True                   digits become single pieces instead of
        memorised years.

When the training text is capped, whole documents are subsampled at a fixed
probability rather than truncating the file. Shards are read in filename order,
so stopping early would fit the tokenizer to whichever source sorts first.

    python3 tokenizer/train_tokenizer.py --sweep    # candidate vocab sizes
    python3 tokenizer/tokenizer_stats.py --compare-sweep
    # set tokenizer.vocab_size in configs/pipeline.yaml, then:
    python3 tokenizer/train_tokenizer.py            # final model
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

#: Overridable from the CLI. A single-element list so `train()` reads the
#: value set by `main()` without threading it through every call site.
INPUT_SENTENCE_SIZE = [int(TCFG.get("input_sentence_size", 5_000_000))]


def model_stem(vocab_size: int, algo: str = "") -> str:
    """Build the filename stem for one tokenizer: size, algorithm, language.

    Args:
        vocab_size: Target vocabulary size. A size that is not a whole number
            of thousands keeps its full digits, so nothing is silently rounded.
        algo: Algorithm short name, "unigram" or "bpe". Defaults to the value
            configured in pipeline.yaml.

    Returns:
        A stem such as "16k_unigram_nep" or "16k_bpe_nep".
    """
    size = f"{vocab_size // 1000}k" if vocab_size % 1000 == 0 else str(vocab_size)
    return f"{size}_{algo or ALGO}_{LANG}"


#: Every assigned code point in the Devanagari block U+0900-U+097F:
#: independent vowels, consonants, dependent vowel signs (matras), anusvara,
#: visarga, candrabindu, nukta, virama, avagraha and the Devanagari digits.
#: Pinned into the vocabulary so no Devanagari character can ever fall back to
#: bytes, however rare it is in the sample.
NEPALI_CHARS = "".join(
    chr(cp) for cp in range(0x0900, 0x0980)
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
    `report/phase1/nepali/json/splits.json` so we do not pay a full extra pass over a
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


def present_required_chars(input_path: Path, wanted: str) -> str:
    """Return the subset of `wanted` that actually occurs in the training text.

    `required_chars` forces code points into the vocabulary so that no character
    of the script can degrade to byte fallback. BPE's trainer asserts that every
    required character has non-zero frequency and aborts otherwise, while
    Unigram tolerates absent characters. Filtering to the characters actually
    present keeps the guarantee wherever it can be honoured and makes the option
    safe for both algorithms.

    Args:
        input_path: The exported training text.
        wanted: Every code point of the language's script.

    Returns:
        The characters from `wanted` observed at least once, in original order.
    """
    seen: set = set()
    target = set(wanted)
    # errors="ignore": a shard truncated mid-character by an interrupted run
    # must not abort tokenizer training over a character-presence scan.
    with open(input_path, encoding="utf-8", errors="ignore") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), ""):
            seen |= target & set(chunk)
            if len(seen) == len(target):
                break
    return "".join(c for c in wanted if c in seen)


def train(input_path: Path, vocab_size: int, algo: str = "",
          sweep: bool = False) -> Path:
    """Train one SentencePiece Unigram model. Returns the `.model` path.

    SentencePiece writes `<prefix>.model` and `<prefix>.vocab` side by side,
    so training happens in a staging directory and the two artifacts are then
    filed separately: `tokenizer/model/` is what Phase 2 loads, while
    `tokenizer/vocab/` holds the human-readable piece list used for the
    report. Keeping them apart stops the loadable model from being lost among
    sweep by-products.
    """
    import sentencepiece as spm

    algo = algo or ALGO
    stem = model_stem(vocab_size, algo)

    # The two algorithms need different character handling, and this is forced
    # by SentencePiece rather than chosen. Unigram accepts `required_chars` and
    # uses it to pin every Devanagari code point into the vocabulary. BPE's
    # trainer asserts freq > 0 for each required character and aborts
    # (bpe_model_trainer.cc:45) at any character_coverage, so BPE instead uses
    # character_coverage=1.0, which retains every character observed in the
    # training text and achieves the same practical guarantee: no Devanagari
    # character present in the corpus can degrade to byte fallback.
    if algo == "unigram":
        char_kwargs = {
            "required_chars": present_required_chars(input_path, NEPALI_CHARS),
            "character_coverage": float(TCFG.get("character_coverage", 0.9995)),
        }
    else:
        char_kwargs = {"character_coverage": 1.0}
    staging = CFG["paths"]["tokenizer"] / "_staging"
    staging.mkdir(parents=True, exist_ok=True)
    out_prefix = staging / stem
    spm.SentencePieceTrainer.train(
        input=str(input_path),
        model_prefix=str(out_prefix),
        model_type=algo,
        vocab_size=vocab_size,
        **char_kwargs,

        # Preprocessing already normalised this text. Do not touch it again.
        normalization_rule_name="identity",
        remove_extra_whitespaces=False,

        # Coverage: no Devanagari character may ever become byte fallback.
        byte_fallback=True,

        # Segmentation behaviour.
        split_by_unicode_script=True,
        split_by_whitespace=True,
        split_digits=True,
        allow_whitespace_only_pieces=True,
        max_sentencepiece_length=int(TCFG.get("max_piece_length", 16)),

        # Fixed special-token ids so Phase 2 can rely on them.
        pad_id=0, unk_id=1, bos_id=2, eos_id=3,
        pad_piece="<pad>", unk_piece="<unk>", bos_piece="<s>", eos_piece="</s>",

        # Unigram peak memory is dominated by the seed vocabulary built before
        # pruning (SentencePiece default: 1,000,000 pieces). Lowering it is the
        # main lever for fitting a larger text sample into a fixed RAM budget.
        # BPE builds no seed vocabulary, so the option is only passed for
        # Unigram.
        **({"seed_sentencepiece_size": int(TCFG["seed_sentencepiece_size"])}
           if algo == "unigram" and TCFG.get("seed_sentencepiece_size") else {}),

        input_sentence_size=INPUT_SENTENCE_SIZE[0],
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
        "script": "Devanagari (U+0900-U+097F)",
        "required_chars_count": len(NEPALI_CHARS),
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
    """Export the training text and train one model, or the whole sweep."""
    ap = argparse.ArgumentParser(
        description="Train the Nepali tokenizer from scratch.")
    ap.add_argument("--sweep", action="store_true",
                    help="train every candidate vocab size for comparison")
    ap.add_argument("--vocab-size", type=int, default=None,
                    help="override the configured vocabulary size")
    ap.add_argument("--max-chars", type=int, default=None,
                    help="cap characters exported for training")
    ap.add_argument("--reuse-input", action="store_true",
                    help="reuse an existing spm_input.txt instead of rebuilding")
    ap.add_argument("--input-sentence-size", type=int, default=None,
                    help="lines SentencePiece loads. Must exceed the sample's "
                         "line count so nothing is sub-sampled, but the trainer "
                         "reserves against this value, so do not set it far "
                         "above (see configs/pipeline.yaml).")
    ap.add_argument("--algo", choices=["unigram", "bpe"], default=None,
                    help="segmentation algorithm; defaults to the configured "
                         "value. Both are trained during the comparison so the "
                         "choice can be made on measured evidence.")
    args = ap.parse_args()
    if args.input_sentence_size:
        INPUT_SENTENCE_SIZE[0] = args.input_sentence_size

    # `--max-chars 0` means no cap. `or` would treat 0 as unset and fall
    # back to the configured value, so test for None explicitly.
    max_chars = (args.max_chars if args.max_chars is not None
                 else int(float(TCFG.get("train_text_max_chars", 2e9))))
    seed = int(TCFG.get("seed", 1337))
    existing = CFG["paths"]["tokenizer"] / "spm_input.txt"

    if args.reuse_input and existing.exists():
        inp, chars, by_source = existing, existing.stat().st_size, {}
        print(f"reusing {inp} ({human(chars)} bytes)")
    else:
        print("exporting training text ...")
        inp, chars, by_source = export_training_text(max_chars, seed)

    if args.sweep:
        sizes = TCFG.get("sweep_vocab_sizes", [8000, 12000, 16000, 24000, 32000])
        # With no --algo given, sweep both algorithms so Unigram and BPE can be
        # compared on identical training text and identical vocabulary sizes.
        algos = [args.algo] if args.algo else ["unigram", "bpe"]
        print(f"\nsweeping {len(sizes)} vocabulary sizes x "
              f"{len(algos)} algorithm(s) = {len(sizes) * len(algos)} models")
        print(f"  sizes: {', '.join(f'{v:,}' for v in sizes)}")
        print(f"  algos: {', '.join(algos)}")
        for algo in algos:
            for vocab in sizes:
                params, share = embedding_params(vocab)
                print(f"\n--- {algo}  vocab_size={vocab:,}  "
                      f"(embedding {human(params)} params, "
                      f"{100 * share:.0f}% of the "
                      f"{human(float(TCFG.get('target_params', 25e6)))} budget) ---")
                try:
                    train(inp, vocab, algo=algo, sweep=True)
                except RuntimeError as exc:
                    # SentencePiece refuses a vocabulary the corpus cannot
                    # support, and raises here rather than producing a model.
                    print(f"  ! {algo} vocab_size={vocab:,} not achievable "
                          f"here: {exc}")
        print("\nsweep done. Compare and get a recommendation with:\n"
              "  python tokenizer/tokenizer_stats.py --compare-sweep")
        return

    vocab = args.vocab_size or int(TCFG["vocab_size"])
    algo = args.algo or ALGO
    params, share = embedding_params(vocab)
    print(f"\ntraining final Nepali {algo} model, vocab_size={vocab:,}")
    print(f"  embedding cost: {human(params)} parameters "
          f"({100 * share:.0f}% of the target budget)")
    model = train(inp, vocab, algo=algo)
    config = write_config(vocab, chars, by_source)

    vocab = CFG["paths"]["tokenizer_vocab"] / f"{model_stem(vocab, algo)}.vocab"
    print(f"\n  model  -> {model}")
    print(f"  vocab  -> {vocab}")
    print(f"  config -> {config}")
    print("\nNext:  python tokenizer/tokenizer_stats.py")


if __name__ == "__main__":
    main()
