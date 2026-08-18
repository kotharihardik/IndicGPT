"""Nepali tokenizer statistics, and the evidence for the vocabulary size.

Covers what Phase 1 asks for (vocabulary size, token-frequency statistics,
average characters per token, tokenization examples, unknown-token statistics)
plus the numbers that actually drive the decision:

    byte-fallback rate  The tokenizer uses byte_fallback=True, so an unknown
        character comes out as <0xNN> pieces instead of <unk>. That makes the
        literal UNK rate ~0 by construction and dishonest to quote on its own.
        Byte fallback is what really measures script coverage.
    fertility           Tokens per whitespace word. The standard comparison
        metric for Indic tokenizers and the main input to the vocab choice.
    embedding cost      Input and output embeddings are tied in Phase 2, so the
        vocabulary costs vocab_size x model_dim of a ~25M budget. A 32k
        vocabulary spends nearly half the budget on a lookup table.
    fertility by source Shows whether the tokenizer fits the literary and legal
        registers as well as it fits news copy.

Everything is measured on the validation split. Measuring fertility on training
text would flatter the tokenizer with text it was fitted to.

    python tokenizer/tokenizer_stats.py                   # the final model
    python tokenizer/tokenizer_stats.py --compare-sweep   # all sweep sizes
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
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


BYTE_PIECE = re.compile(r"^<0x[0-9A-F]{2}>$")
DEVANAGARI = re.compile(r"[\u0900-\u097F]")

#: Above this byte-fallback rate the vocabulary is not really covering the
#: held-out text, and its compression numbers should not be trusted. Note that
#: most byte fallback here is Latin characters and punctuation embedded in
#: Nepali articles, not Devanagari itself: every Devanagari code point is
#: pinned into the vocabulary by `required_chars` in train_tokenizer.py.
MAX_ACCEPTABLE_BYTE_FALLBACK = 0.01

#: Below this relative gain in chars/token, a larger vocabulary is not buying
#: enough compression to justify its embedding parameters. This is the knee.
KNEE_RELATIVE_GAIN = 0.02

#: Hard ceiling on the share of the Phase 2 parameter budget the embedding
#: table may consume. Input and output embeddings are tied, so this is a lookup
#: table competing directly with Transformer blocks: at d_model=384 each block
#: costs ~12*d^2 = 1.77M parameters, so every 16k of vocabulary is worth about
#: 3.5 layers. Letting the embedding take most of the budget buys a little
#: compression and pays for it with a model too shallow to use it.
MAX_EMBEDDING_SHARE = 0.34

#: A vocabulary this under-used on held-out text is too large for the corpus:
#: the unused pieces are embedding rows that would receive almost no gradient
#: in Phase 2.
MIN_VOCAB_UTILISATION = 0.95


# Computation


def heldout_docs(max_docs: int) -> list:
    """Load validation documents as `(text, source_name)` pairs."""
    path = CFG["paths"]["splits"] / "validation.jsonl.gz"
    if not path.exists():
        raise SystemExit(f"{path} not found - run data/splits/build_splits.py first")
    return [(d.text, d.source_name) for d in read_docs(path, limit=max_docs)]


def evaluate(model_path: Path, docs: list) -> dict:
    """Compute coverage, fertility and frequency statistics for one model."""
    import sentencepiece as spm

    sp = spm.SentencePieceProcessor(model_file=str(model_path))
    freq: Counter = Counter()
    per_source: dict = defaultdict(lambda: {"chars": 0, "tokens": 0})
    n_tokens = n_chars = n_words = n_unk = n_byte = 0

    for text, source in docs:
        ids = sp.encode(text)
        freq.update(ids)
        n_tokens += len(ids)
        n_chars += len(text)
        n_words += len(text.split())
        n_unk += sum(1 for i in ids if i == sp.unk_id())
        n_byte += sum(1 for i in ids if BYTE_PIECE.match(sp.id_to_piece(i)))
        per_source[source]["chars"] += len(text)
        per_source[source]["tokens"] += len(ids)

    vocab = sp.get_piece_size()
    d_model = int(TCFG.get("model_dim", 384))
    budget = float(TCFG.get("target_params", 25e6))

    # Piece-length distribution, weighted by how often pieces are actually
    # used: an unweighted histogram is dominated by rare long pieces.
    length_weighted: Counter = Counter()
    length_types: Counter = Counter()
    for piece_id, count in freq.items():
        piece = sp.id_to_piece(piece_id)
        if BYTE_PIECE.match(piece):
            continue
        length = len(piece.replace("▁", ""))     # strip the word-start mark
        length_weighted[length] += count
        length_types[length] += 1

    return {
        "model": model_path.name,
        "vocab_size": vocab,
        "vocab_used_on_heldout": len(freq),
        "vocab_utilisation": round(len(freq) / vocab, 4),
        "tokens": n_tokens,
        "chars": n_chars,
        "words": n_words,
        "chars_per_token": round(n_chars / max(n_tokens, 1), 3),
        "fertility_tokens_per_word": round(n_tokens / max(n_words, 1), 3),
        "unk_rate": round(n_unk / max(n_tokens, 1), 8),
        "byte_fallback_rate": round(n_byte / max(n_tokens, 1), 6),
        "hapax_pieces_on_heldout": sum(1 for c in freq.values() if c == 1),
        "top_pieces": [(sp.id_to_piece(i), c) for i, c in freq.most_common(30)],
        "piece_length_weighted": dict(sorted(length_weighted.items())),
        "piece_length_types": dict(sorted(length_types.items())),
        "frequency_ranks": [c for _, c in freq.most_common(5000)],
        "embedding_parameters": vocab * d_model,
        "embedding_share_of_budget": round(vocab * d_model / budget, 4),
        "chars_per_token_by_source": {
            name: round(v["chars"] / max(v["tokens"], 1), 3)
            for name, v in sorted(per_source.items())
        },
    }


def normalize_like_corpus(text: str) -> str:
    """Apply the corpus normalisation chain to a raw example sentence.

    The example sentences live in `configs/pipeline.yaml` as hand-typed
    strings, but every document the tokenizer was trained on went through
    `preprocess.py` first. Tokenizing a raw string would therefore show
    segmentations the model will never actually meet - an em dash, for
    instance, is folded to a hyphen by preprocessing, so tokenizing the raw
    form reports a byte-fallback that does not occur in the real corpus.

    Falls back to returning the text unchanged if preprocessing cannot be
    imported, so statistics never fail because of a demonstration feature.
    """
    try:
        sys.path.insert(0, str(ROOT / "data" / "preprocessing"))
        from preprocess import normalize            # noqa: WPS433
        return normalize(text)
    except Exception:
        return text


def sample_tokenizations(model_path: Path) -> list:
    """Tokenize the Nepali example sentences from configs/pipeline.yaml.

    Sentences are normalised exactly as corpus documents were, so the
    segmentations shown are the ones Phase 2 will really see.
    """
    import sentencepiece as spm

    sp = spm.SentencePieceProcessor(model_file=str(model_path))
    out = []
    for raw in TCFG.get("eval_examples", []):
        sentence = normalize_like_corpus(raw)
        pieces = sp.encode(sentence, out_type=str)
        entry = {
            "text": sentence,
            "n_tokens": len(pieces),
            "pieces": pieces,
            "chars_per_token": round(len(sentence) / max(len(pieces), 1), 2),
        }
        if sentence != raw:
            entry["raw_text"] = raw
        out.append(entry)
    return out


def recommend_vocab(results: list) -> dict:
    """Pick a vocabulary size from the sweep, and say why.

    Two rules, applied in order:

    1. Coverage first. Any candidate whose byte-fallback rate exceeds
       `MAX_ACCEPTABLE_BYTE_FALLBACK` is not representing Nepali and is
       rejected however good its compression looks.
    2. Then the knee. Walk the surviving candidates from smallest upward
       and stop at the first one where doubling down on vocabulary buys less
       than `KNEE_RELATIVE_GAIN` more characters per token. Past that
       point the extra vocabulary is spending Phase 2 parameters on an
       embedding table for very little compression.
    """
    usable = [r for r in results
              if r["byte_fallback_rate"] <= MAX_ACCEPTABLE_BYTE_FALLBACK]
    rejected = [r["vocab_size"] for r in results if r not in usable]
    warning = ""
    if not usable:
        # Still make a recommendation rather than refusing to answer: a high
        # byte-fallback rate across every candidate means the corpus carries a
        # lot of non-Nepali text, which is a preprocessing finding, not a
        # reason to leave the vocabulary size unchosen.
        usable, rejected = results, []
        warning = (f"every candidate exceeds the {MAX_ACCEPTABLE_BYTE_FALLBACK:.1%} "
                   f"byte-fallback bar - check how much Latin text survived "
                   f"preprocessing before trusting these compression numbers")

    usable.sort(key=lambda r: r["vocab_size"])

    # Rule 2: the vocabulary must actually be used. Pieces never seen on
    # held-out text are embedding rows that will not train in Phase 2.
    underused = [r["vocab_size"] for r in usable
                 if r["vocab_utilisation"] < MIN_VOCAB_UTILISATION]
    usable = [r for r in usable
              if r["vocab_utilisation"] >= MIN_VOCAB_UTILISATION] or usable

    # Rule 3: the embedding table must not crowd out the Transformer. This is
    # the rule that matters most at ~25M parameters, and the one a
    # fertility-only criterion misses: fertility keeps improving with vocabulary
    # long after the model has no capacity left to exploit it.
    affordable = [r for r in usable
                  if r["embedding_share_of_budget"] <= MAX_EMBEDDING_SHARE]
    too_expensive = [r["vocab_size"] for r in usable if r not in affordable]
    if affordable:
        usable = affordable

    choice = usable[-1]
    reason = (f"largest vocabulary whose embedding stays within "
              f"{100 * MAX_EMBEDDING_SHARE:.0f}% of the parameter budget")

    # Rule 4: within what is affordable and used, stop at the knee.
    for current, nxt in zip(usable, usable[1:]):
        gain = (nxt["chars_per_token"] - current["chars_per_token"]) \
            / max(current["chars_per_token"], 1e-9)
        if gain < KNEE_RELATIVE_GAIN:
            choice = current
            reason = (f"vocab {nxt['vocab_size']:,} improves chars/token by only "
                      f"{100 * gain:.1f}% over {current['vocab_size']:,} "
                      f"(< {100 * KNEE_RELATIVE_GAIN:.0f}%), while costing "
                      f"{human(nxt['embedding_parameters'] - current['embedding_parameters'])} "
                      f"more embedding parameters")
            break

    d_model = int(TCFG.get("model_dim", 384))
    budget = float(TCFG.get("target_params", 25e6))
    per_layer = 12 * d_model ** 2          # attention 4d^2 + FFN 8d^2
    layers = (budget - choice["embedding_parameters"]) / per_layer

    return {
        "recommended": choice["vocab_size"],
        "reason": reason,
        "warning": warning,
        "chars_per_token": choice["chars_per_token"],
        "fertility_tokens_per_word": choice["fertility_tokens_per_word"],
        "byte_fallback_rate": choice["byte_fallback_rate"],
        "vocab_utilisation": choice["vocab_utilisation"],
        "embedding_parameters": choice["embedding_parameters"],
        "embedding_share_of_budget": choice["embedding_share_of_budget"],
        "transformer_budget_left": budget - choice["embedding_parameters"],
        "approx_layers_affordable": round(layers, 1),
        "rejected_for_coverage": rejected,
        "rejected_for_underuse": underused,
        "rejected_as_too_expensive": too_expensive,
    }


# Visualisation  (kept separate from computation)


def plot_sweep(results: list, path: Path) -> None:
    """Fertility against vocabulary size, with the embedding cost alongside."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sizes = [r["vocab_size"] for r in results]
    cpt = [r["chars_per_token"] for r in results]
    share = [100 * r["embedding_share_of_budget"] for r in results]

    fig, ax = plt.subplots(figsize=(8, 5))
    line1 = ax.plot(sizes, cpt, "o-", color="#1f77b4",
                    label="Characters per token (higher = better compression)")
    ax.set_xlabel("Vocabulary size (pieces)")
    ax.set_ylabel("Characters per token", color="#1f77b4")
    ax.tick_params(axis="y", labelcolor="#1f77b4")
    ax.grid(alpha=0.3)

    ax2 = ax.twinx()
    line2 = ax2.plot(sizes, share, "s--", color="#d62728",
                     label="Embedding cost (% of 25M parameter budget)")
    ax2.set_ylabel("Embedding share of parameter budget (%)", color="#d62728")
    ax2.tick_params(axis="y", labelcolor="#d62728")

    ax.set_title("Nepali tokenizer: compression vs. parameter cost\n"
                 "measured on the held-out validation split")
    lines = line1 + line2
    ax.legend(lines, [ln.get_label() for ln in lines], loc="lower right",
              fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_frequency(result: dict, path: Path) -> None:
    """Zipf plot of piece frequency against rank."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    counts = result["frequency_ranks"]
    ranks = range(1, len(counts) + 1)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.loglog(ranks, counts, color="#2ca02c",
              label=f"Vocabulary size {result['vocab_size']:,}")
    ax.set_xlabel("Piece rank (log scale)")
    ax.set_ylabel("Frequency on held-out text (log scale)")
    ax.set_title("Nepali tokenizer: token frequency distribution\n"
                 "a straight line indicates Zipfian behaviour")
    ax.grid(alpha=0.3, which="both")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_piece_lengths(result: dict, path: Path) -> None:
    """How long the pieces are, by type and weighted by usage."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    weighted = result["piece_length_weighted"]
    types = result["piece_length_types"]
    lengths = sorted(set(map(int, weighted)) | set(map(int, types)))
    w_total = sum(weighted.values()) or 1
    t_total = sum(types.values()) or 1
    w = [100 * weighted.get(str(n), weighted.get(n, 0)) / w_total for n in lengths]
    t = [100 * types.get(str(n), types.get(n, 0)) / t_total for n in lengths]

    fig, ax = plt.subplots(figsize=(8, 5))
    width = 0.4
    ax.bar([n - width / 2 for n in lengths], t, width,
           label="Share of vocabulary (types)", color="#9467bd")
    ax.bar([n + width / 2 for n in lengths], w, width,
           label="Share of tokens used (weighted by frequency)", color="#ff7f0e")
    ax.set_xlabel("Piece length (Devanagari characters)")
    ax.set_ylabel("Percentage (%)")
    ax.set_title("Nepali tokenizer: piece-length distribution\n"
                 "short pieces dominate usage even when long pieces "
                 "dominate the vocabulary")
    ax.set_xticks(lengths)
    ax.grid(alpha=0.3, axis="y")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def print_table(results: list) -> None:
    """Print the sweep comparison table."""
    print(f"{'vocab':>8} {'chars/tok':>10} {'fertility':>10} {'byte-fb':>10} "
          f"{'unk':>9} {'vocab used':>11} {'emb params':>12} {'% budget':>9}")
    print("-" * 84)
    for r in results:
        print(f"{r['vocab_size']:>8,} {r['chars_per_token']:>10.3f} "
              f"{r['fertility_tokens_per_word']:>10.3f} "
              f"{r['byte_fallback_rate']:>10.6f} {r['unk_rate']:>9.6f} "
              f"{r['vocab_utilisation'] * 100:>10.1f}% "
              f"{human(r['embedding_parameters']):>12} "
              f"{100 * r['embedding_share_of_budget']:>8.0f}%")


def write_markdown(path: Path, results: list, samples: list,
                   recommendation: dict, n_docs: int, gu_share: float) -> None:
    """Write the human-readable report table."""
    lines = [
        "# Nepali tokenizer statistics (Model L)",
        "",
        f"SentencePiece Unigram, trained from scratch on the Nepali "
        f"training split only. Measured on {n_docs:,} held-out validation "
        f"documents ({100 * gu_share:.1f}% Devanagari script).",
        "",
        "## Vocabulary size comparison",
        "",
        "| Vocab | Chars/token | Fertility (tok/word) | Byte-fallback | UNK | "
        "Vocab used | Embedding params | % of 25M budget |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        lines.append(
            f"| {r['vocab_size']:,} | {r['chars_per_token']:.3f} | "
            f"{r['fertility_tokens_per_word']:.3f} | "
            f"{r['byte_fallback_rate']:.6f} | {r['unk_rate']:.6f} | "
            f"{100 * r['vocab_utilisation']:.1f}% | "
            f"{human(r['embedding_parameters'])} | "
            f"{100 * r['embedding_share_of_budget']:.0f}% |")

    if recommendation.get("recommended"):
        lines += ["", "## Recommended vocabulary size", "",
                  f"{recommendation['recommended']:,} — {recommendation['reason']}.",
                  "",
                  "UNK rate is ~0 by construction because `byte_fallback=True`; "
                  "the byte-fallback rate is the real coverage metric."]

    best = results[-1]
    if best.get("chars_per_token_by_source"):
        lines += ["", "## Compression by source register", "",
                  "| Source | Chars/token |", "|---|---|"]
        for name, value in sorted(best["chars_per_token_by_source"].items(),
                                  key=lambda kv: -kv[1]):
            lines.append(f"| {name} | {value:.3f} |")

    if samples:
        lines += ["", "## Tokenization examples", ""]
        for s in samples:
            lines += [f"- `{s['text']}`",
                      f"  - {s['n_tokens']} tokens, "
                      f"{s['chars_per_token']} chars/token",
                      f"  - `{' | '.join(s['pieces'])}`"]

    lines += ["", "## Most frequent pieces", "",
              "| Piece | Count |", "|---|---|"]
    for piece, count in best["top_pieces"][:20]:
        lines.append(f"| `{piece}` | {count:,} |")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Nepali tokenizer statistics on held-out text.")
    ap.add_argument("--compare-sweep", action="store_true",
                    help="evaluate every vocab size from the training sweep")
    ap.add_argument("--max-docs", type=int, default=3000,
                    help="held-out documents to evaluate on")
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args()

    tok_dir = CFG["paths"]["tokenizer_model"]
    docs = heldout_docs(args.max_docs)
    total_chars = sum(len(t) for t, _ in docs)
    gu_share = sum(len(DEVANAGARI.findall(t)) for t, _ in docs) / max(total_chars, 1)
    print(f"evaluating on {len(docs):,} held-out documents "
          f"({total_chars:,} characters, {100 * gu_share:.1f}% Devanagari script)\n")

    if args.compare_sweep:
        models = sorted((tok_dir / "sweep").glob("*.model"))
        if not models:
            raise SystemExit("no sweep models found - "
                             "run 'train_tokenizer.py --sweep' first")
    else:
        model = tok_dir / f"{model_stem(int(TCFG['vocab_size']))}.model"
        if not model.exists():
            raise SystemExit(f"{model} not found - "
                             f"run 'tokenizer/train_tokenizer.py' first")
        models = [model]

    results = sorted((evaluate(m, docs) for m in models),
                     key=lambda r: r["vocab_size"])
    print_table(results)

    recommendation = recommend_vocab(results) if len(results) > 1 else {}
    if recommendation.get("recommended"):
        # Show what each extra step of vocabulary actually costs, because the
        # raw fertility column always favours the largest candidate and hides
        # the fact that the gains get steadily more expensive.
        print("\nmarginal cost of each step up in vocabulary:")
        print(f"  {'step':<20}{'+chars/tok':>12}{'+params':>11}{'params per 1% gain':>21}")
        for cur, nxt in zip(results, results[1:]):
            gain = (nxt["chars_per_token"] - cur["chars_per_token"]) \
                / max(cur["chars_per_token"], 1e-9)
            extra = nxt["embedding_parameters"] - cur["embedding_parameters"]
            cost = human(extra / (100 * gain)) if gain > 0 else "-"
            step = f"{cur['vocab_size']:,} -> {nxt['vocab_size']:,}"
            print(f"  {step:<20}{100 * gain:>11.1f}%{human(extra):>11}{cost:>21}")

        print(f"\nRECOMMENDED vocab_size = {recommendation['recommended']:,}")
        print(f"  {recommendation['reason']}")
        print(f"  leaves {human(recommendation['transformer_budget_left'])} for "
              f"Transformer blocks = ~{recommendation['approx_layers_affordable']} "
              f"layers at d_model={TCFG.get('model_dim', 384)}")
        if recommendation.get("rejected_as_too_expensive"):
            print(f"  rejected - embedding would exceed "
                  f"{100 * MAX_EMBEDDING_SHARE:.0f}% of the budget: "
                  f"{recommendation['rejected_as_too_expensive']}")
        if recommendation.get("rejected_for_underuse"):
            print(f"  rejected - under {100 * MIN_VOCAB_UTILISATION:.0f}% of the "
                  f"vocabulary is used on held-out text: "
                  f"{recommendation['rejected_for_underuse']}")
        if recommendation["rejected_for_coverage"]:
            print(f"  rejected for weak script coverage: "
                  f"{recommendation['rejected_for_coverage']}")
        print(f"  set tokenizer.vocab_size in configs/pipeline.yaml, then "
              f"rerun train_tokenizer.py without --sweep")

    final = tok_dir / f"{model_stem(int(TCFG['vocab_size']))}.model"
    samples = sample_tokenizations(final) if final.exists() else []
    if samples:
        print("\nExample tokenizations:")
        for s in samples:
            print(f"\n  {s['text']}")
            print(f"  -> {s['n_tokens']} tokens, {s['chars_per_token']} chars/token")
            print(f"  -> {' | '.join(s['pieces'])}")

    by_source = results[-1].get("chars_per_token_by_source") or {}
    if len(by_source) > 1:
        print("\nCompression by source register (chars/token):")
        for name, value in sorted(by_source.items(), key=lambda kv: -kv[1]):
            print(f"  {name:<24}{value:>7.3f}")

    reports = CFG["paths"]["reports"]
    figures = reports / "figures"
    reports.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)

    payload = {
        "language": LANG,
        "algorithm": "sentencepiece_unigram",
        "heldout_documents": len(docs),
        "heldout_chars": total_chars,
        "heldout_devanagari_share": round(gu_share, 4),
        "results": results,
        "recommendation": recommendation,
        "examples": samples,
        "note": ("byte_fallback=True means characters outside the vocabulary "
                 "become <0xNN> pieces rather than <unk>, so unk_rate is ~0 by "
                 "construction and byte_fallback_rate is the meaningful "
                 "coverage metric"),
    }
    # The sweep comparison and the final-model report are different documents
    # and must not share a filename: whichever ran last used to silently
    # overwrite the other, which cost the Gujarati sweep table once already.
    stem = "tokenizer_sweep" if args.compare_sweep else "tokenizer_stats"
    json_path = reports / f"{stem}.json"
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False),
                         encoding="utf-8")
    md_path = reports / f"{stem}.md"
    write_markdown(md_path, results, samples, recommendation, len(docs), gu_share)
    print(f"\nsaved -> {json_path}\nsaved -> {md_path}")

    if not args.no_figures:
        if len(results) > 1:
            plot_sweep(results, figures / "tokenizer_vocab_sweep.png")
            print(f"saved -> {figures / 'tokenizer_vocab_sweep.png'}")
        plot_frequency(results[-1], figures / "tokenizer_frequency.png")
        plot_piece_lengths(results[-1],
                           figures / "tokenizer_piece_length.png")
        print(f"saved -> {figures / 'tokenizer_frequency.png'}")
        print(f"saved -> {figures / 'tokenizer_piece_length.png'}")


if __name__ == "__main__":
    main()
