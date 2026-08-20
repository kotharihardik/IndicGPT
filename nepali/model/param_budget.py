"""Exact parameter count for the Phase 2 decoder-only Transformer.

Phase 2 asks for ~25M trainable parameters per model, and the vocabulary chosen
in Phase 1 spends part of that before a single Transformer block exists. This
makes the trade-off checkable before training instead of after.

For a GPT-style block of width d:

    token embedding                    vocab x d
    learned positional embedding       context x d
    LayerNorm x2, per block            4d
    attention Wq,Wk,Wv,Wo, per block   4d^2 + 4d
    FFN d -> 4d -> d, per block        8d^2 + 5d
    final LayerNorm                    2d
    output head (tied)                 0

So a block costs 12d^2 + 11d, and the whole model:

    vocab*d + context*d + L*(12d^2 + 11d) + 2d

Weight tying is assumed, so the output projection reuses the token-embedding
matrix. That saves vocab x d - at 16k x 384 that is 6.1M parameters, a quarter
of the budget.

    python3 model/param_budget.py           # check the configured setup
    python3 model/param_budget.py --sweep   # layer counts per vocabulary
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data"))
from corpus_io import lang_root, load_config  # noqa: E402

ROOT = lang_root(Path(__file__))
CFG = load_config(ROOT)
TCFG = CFG["tokenizer"]


def count_params(vocab: int, d_model: int, n_layers: int, context: int = 512,
                 ffn_mult: int = 4, tie_embeddings: bool = True) -> dict:
    """Exact trainable-parameter breakdown for one configuration."""
    tok_emb = vocab * d_model
    pos_emb = context * d_model
    attn = 4 * d_model ** 2 + 4 * d_model
    ffn = 2 * ffn_mult * d_model ** 2 + (ffn_mult + 1) * d_model
    norms = 4 * d_model                      # two LayerNorms per block
    per_block = attn + ffn + norms
    blocks = n_layers * per_block
    final_ln = 2 * d_model
    head = 0 if tie_embeddings else vocab * d_model

    total = tok_emb + pos_emb + blocks + final_ln + head
    return {
        "vocab": vocab, "d_model": d_model, "n_layers": n_layers,
        "context": context,
        "token_embedding": tok_emb,
        "positional_embedding": pos_emb,
        "per_block": per_block,
        "all_blocks": blocks,
        "final_layernorm": final_ln,
        "output_head": head,
        "total": total,
        "embedding_share": (tok_emb + pos_emb + head) / total,
        "non_embedding": blocks + final_ln,
    }


def fmt(n: float) -> str:
    """Format a byte count for the progress line."""
    return f"{n / 1e6:.2f}M"


def report(cfg: dict) -> None:
    """Print a full breakdown for one configuration."""
    print(f"\n  vocab {cfg['vocab']:,}  d_model {cfg['d_model']}  "
          f"layers {cfg['n_layers']}  context {cfg['context']}")
    print("  " + "-" * 56)
    rows = [
        ("token embedding (tied with output head)", cfg["token_embedding"]),
        ("positional embedding", cfg["positional_embedding"]),
        (f"{cfg['n_layers']} blocks x {fmt(cfg['per_block'])}", cfg["all_blocks"]),
        ("final LayerNorm", cfg["final_layernorm"]),
    ]
    if cfg["output_head"]:
        rows.append(("output head (UNTIED)", cfg["output_head"]))
    for label, value in rows:
        print(f"  {label:<44}{fmt(value):>12}")
    print("  " + "-" * 56)
    print(f"  {'TOTAL':<44}{fmt(cfg['total']):>12}")
    print(f"  {'of which embedding':<44}"
          f"{100 * cfg['embedding_share']:>11.1f}%")

    target = float(TCFG.get("target_params", 25e6))
    delta = 100 * (cfg["total"] - target) / target
    verdict = "OK" if abs(delta) <= 10 else "OUT OF RANGE"
    print(f"  {'vs ~25M target':<44}{delta:>+10.1f}%  {verdict}")


def best_layers(vocab: int, d_model: int, target: float, context: int = 512) -> int:
    """Layer count landing closest to the parameter target."""
    best, best_err = 1, float("inf")
    for layers in range(1, 65):
        err = abs(count_params(vocab, d_model, layers, context)["total"] - target)
        if err < best_err:
            best, best_err = layers, err
    return best


def sweep(target: float, context: int) -> None:
    """Show what each candidate vocabulary leaves for the Transformer."""
    print(f"\nLayers affordable at ~{fmt(target)} parameters "
          f"(context {context}, tied embeddings)\n")
    print(f"  {'vocab':>7}{'d_model':>9}{'layers':>8}{'total':>10}"
          f"{'embed %':>9}{'non-embed':>11}")
    print("  " + "-" * 54)
    for vocab in TCFG.get("sweep_vocab_sizes", [8000, 16000, 24000, 32000, 48000]):
        for d_model in (256, 320, 384, 448):
            layers = best_layers(vocab, d_model, target, context)
            cfg = count_params(vocab, d_model, layers, context)
            mark = "  <-- configured" if (
                vocab == int(TCFG["vocab_size"])
                and d_model == int(TCFG.get("model_dim", 384))) else ""
            print(f"  {vocab:>7,}{d_model:>9}{layers:>8}{fmt(cfg['total']):>10}"
                  f"{100 * cfg['embedding_share']:>8.0f}%"
                  f"{fmt(cfg['non_embedding']):>11}{mark}")
    print("\n  Deeper is generally better than a bigger lookup table at this "
          "scale:\n  every token passes through every block, but touches only "
          "one embedding row.")


def main() -> None:
    """Print the parameter breakdown, or sweep layer counts per vocabulary."""
    ap = argparse.ArgumentParser(description="Phase 2 parameter budget check.")
    ap.add_argument("--vocab", type=int, default=None)
    ap.add_argument("--d-model", type=int, default=None)
    ap.add_argument("--layers", type=int, default=None)
    ap.add_argument("--context", type=int, default=512)
    ap.add_argument("--untied", action="store_true",
                    help="do NOT tie the output head to the token embedding")
    ap.add_argument("--sweep", action="store_true",
                    help="layer counts affordable for every candidate vocabulary")
    args = ap.parse_args()

    target = float(TCFG.get("target_params", 25e6))
    vocab = args.vocab or int(TCFG["vocab_size"])
    d_model = args.d_model or int(TCFG.get("model_dim", 384))

    if args.sweep:
        sweep(target, args.context)
        return

    if args.layers:
        report(count_params(vocab, d_model, args.layers, args.context,
                            tie_embeddings=not args.untied))
        return

    # No layer count given: show the two that bracket the target, so the
    # trade-off is visible rather than decided silently.
    best = best_layers(vocab, d_model, target, args.context)
    print(f"\nClosest configurations to the ~{fmt(target)} target "
          f"with the Phase 1 vocabulary:")
    for layers in (best - 1, best, best + 1):
        if layers >= 1:
            report(count_params(vocab, d_model, layers, args.context,
                                tie_embeddings=not args.untied))


if __name__ == "__main__":
    main()
