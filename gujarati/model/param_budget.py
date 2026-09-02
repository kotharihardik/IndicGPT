"""Parameter budget for the Phase 2 decoder-only Transformer.

Phase 1 used this file to price each candidate vocabulary against the ~25M
target before the model existed. Now the model does exist, so the analytic
formula is checked against the real thing: `count_params` computes the budget
from a config, `measured()` builds `DecoderLM` and counts its tensors, and the
two must agree. That way a number quoted in the report can never drift away
from the network that was actually trained.

For a pre-norm block of width d with h heads, SwiGLU width d_ff and no biases:

    attention  W_Q, W_K, W_V, W_O        4 d^2
    SwiGLU     W_gate, W_up, W_down      3 d d_ff
    LayerNorm  x2 (weight + bias each)   4 d

and the whole model, with the output head tied to the token embedding:

    vocab d + L (4 d^2 + 3 d d_ff + 4 d) + 2 d

RoPE contributes nothing: it is computed from the position, not learned. A
learned positional table would add `context x d` on top.

    python3 model/param_budget.py            # the configured model
    python3 model/param_budget.py --sweep    # depths affordable per vocabulary
    python3 model/param_budget.py --layers 9 --d-model 448
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data"))
from corpus_io import lang_root, load_config  # noqa: E402
from config import load_model_config, read_yaml  # noqa: E402
from transformer import ModelConfig  # noqa: E402

ROOT = lang_root(Path(__file__))
PIPELINE = load_config(ROOT)
TCFG = PIPELINE["tokenizer"]

#: The brief's target. Everything is reported as a distance from it.
TARGET = float(TCFG.get("target_params", 25.0e6))


def count_params(cfg: ModelConfig) -> Dict[str, int]:
    """Analytic parameter breakdown for one configuration."""
    d, ff, L = cfg.d_model, cfg.d_ff, cfg.n_layers
    bias = 1 if cfg.bias else 0

    tok_emb = cfg.vocab_size * d
    pos_emb = cfg.context_length * d if cfg.positional == "learned" else 0
    attn = 4 * d * d + 4 * bias * d
    ffn_mats = 3 if cfg.ffn_activation == "swiglu" else 2
    ffn = ffn_mats * d * ff + bias * (ff * (ffn_mats - 1) + d)
    norms = 4 * d                                    # two LayerNorms per block
    per_block = attn + ffn + norms
    head = 0 if cfg.tie_embeddings else cfg.vocab_size * d

    total = tok_emb + pos_emb + L * per_block + 2 * d + head
    return {
        "token_embedding": tok_emb,
        "positional_embedding": pos_emb,
        "per_block": per_block,
        "attention_per_block": attn,
        "ffn_per_block": ffn,
        "all_blocks": L * per_block,
        "final_layernorm": 2 * d,
        "output_head": head,
        "total": total,
        "non_embedding": L * per_block + 2 * d,
        "embedding_share": (tok_emb + pos_emb + head) / total,
    }


def measured(cfg: ModelConfig) -> int:
    """Build the real model and count its trainable tensors."""
    from transformer import DecoderLM

    return DecoderLM(cfg).num_parameters()


def fmt(n: float) -> str:
    """Compact parameter count, e.g. 25.63M."""
    return f"{n / 1e6:.2f}M"


def report(cfg: ModelConfig, verify: bool = True) -> None:
    """Print the breakdown for one configuration."""
    counts = count_params(cfg)
    print(f"\n  vocab {cfg.vocab_size:,}  d_model {cfg.d_model}  "
          f"layers {cfg.n_layers}  heads {cfg.n_heads}  d_ff {cfg.d_ff}")
    print(f"  context {cfg.context_length}  positional {cfg.positional}  "
          f"ffn {cfg.ffn_activation}  tied {cfg.tie_embeddings}")
    print("  " + "-" * 60)

    rows = [("token embedding" + (" (tied with output head)" if cfg.tie_embeddings
                                  else ""), counts["token_embedding"])]
    if counts["positional_embedding"]:
        rows.append(("learned positional embedding", counts["positional_embedding"]))
    rows += [
        (f"  attention  x{cfg.n_layers}", cfg.n_layers * counts["attention_per_block"]),
        (f"  {cfg.ffn_activation:<9} x{cfg.n_layers}", cfg.n_layers * counts["ffn_per_block"]),
        (f"  LayerNorms x{cfg.n_layers}", cfg.n_layers * 4 * cfg.d_model),
        ("final LayerNorm", counts["final_layernorm"]),
    ]
    if counts["output_head"]:
        rows.append(("output head (UNTIED)", counts["output_head"]))
    for label, value in rows:
        print(f"  {label:<44}{fmt(value):>14}")

    print("  " + "-" * 60)
    print(f"  {'TOTAL':<44}{counts['total']:>14,}")
    print(f"  {'of which embedding':<44}{100 * counts['embedding_share']:>13.1f}%")
    delta = 100 * (counts["total"] - TARGET) / TARGET
    verdict = "OK" if abs(delta) <= 10 else "OUT OF RANGE"
    print(f"  {'vs the ~25M target':<44}{delta:>+13.1f}%  {verdict}")

    if verify:
        real = measured(cfg)
        agree = "matches" if real == counts["total"] else "MISMATCH"
        print(f"  {'built and counted':<44}{real:>14,}  {agree}")


def best_layers(cfg: ModelConfig, target: float) -> int:
    """Depth landing closest to the parameter target at this width."""
    best, best_err = 1, float("inf")
    for layers in range(1, 65):
        trial = ModelConfig.from_dict({**cfg.to_dict(), "n_layers": layers})
        err = abs(count_params(trial)["total"] - target)
        if err < best_err:
            best, best_err = layers, err
    return best


def sweep(base: ModelConfig) -> None:
    """Show what each candidate vocabulary leaves for the Transformer blocks."""
    print(f"\nDepth affordable at ~{fmt(TARGET)} parameters "
          f"(context {base.context_length}, {base.positional}, tied embeddings)\n")
    print(f"  {'vocab':>7}{'d_model':>9}{'d_ff':>7}{'layers':>8}{'total':>10}"
          f"{'embed %':>9}{'non-embed':>11}")
    print("  " + "-" * 61)
    for vocab in TCFG.get("sweep_vocab_sizes", [8000, 12000, 16000, 24000, 32000]):
        for d_model in (256, 320, 384, 448):
            # Keep the SwiGLU width at 8/3 of the model dimension, rounded to a
            # multiple of 64, so every row is the same architecture rescaled.
            d_ff = int(round(d_model * 8 / 3 / 64)) * 64
            trial = ModelConfig.from_dict({**base.to_dict(), "vocab_size": vocab,
                                           "d_model": d_model, "d_ff": d_ff,
                                           "n_heads": d_model // 64})
            layers = best_layers(trial, TARGET)
            trial = ModelConfig.from_dict({**trial.to_dict(), "n_layers": layers})
            counts = count_params(trial)
            mark = ("  <-- configured"
                    if vocab == base.vocab_size and d_model == base.d_model else "")
            print(f"  {vocab:>7,}{d_model:>9}{d_ff:>7}{layers:>8}"
                  f"{fmt(counts['total']):>10}{100 * counts['embedding_share']:>8.0f}%"
                  f"{fmt(counts['non_embedding']):>11}{mark}")
    print("\n  Every token passes through every block but touches exactly one\n"
          "  embedding row, so at a fixed budget depth generally beats a larger\n"
          "  lookup table - which is why Phase 1 capped the vocabulary at 16,000.")


def main() -> None:
    """Print the configured budget, or sweep widths and vocabularies."""
    ap = argparse.ArgumentParser(description="Phase 2 parameter budget.")
    ap.add_argument("--vocab", type=int, default=None)
    ap.add_argument("--d-model", type=int, default=None)
    ap.add_argument("--d-ff", type=int, default=None)
    ap.add_argument("--layers", type=int, default=None)
    ap.add_argument("--context", type=int, default=None)
    ap.add_argument("--untied", action="store_true",
                    help="do NOT tie the output head to the token embedding")
    ap.add_argument("--sweep", action="store_true",
                    help="depths affordable for every candidate vocabulary")
    ap.add_argument("--no-verify", action="store_true",
                    help="skip building the model to cross-check the formula")
    args = ap.parse_args()

    # Start from configs/model.yaml so this always prices the real model. The
    # vocabulary size is whatever the trained tokenizer actually holds.
    stated = read_yaml(ROOT / "configs" / "model.yaml").get("vocab_size")
    cfg = load_model_config(ROOT, vocab_size=args.vocab or stated)

    override = {}
    if args.d_model:
        override["d_model"] = args.d_model
        override["n_heads"] = max(1, args.d_model // 64)
        override["d_ff"] = args.d_ff or int(round(args.d_model * 8 / 3 / 64)) * 64
    if args.d_ff:
        override["d_ff"] = args.d_ff
    if args.layers:
        override["n_layers"] = args.layers
    if args.context:
        override["context_length"] = args.context
    if args.untied:
        override["tie_embeddings"] = False
    if override:
        cfg = ModelConfig.from_dict({**cfg.to_dict(), **override})

    if args.sweep:
        sweep(cfg)
        return
    report(cfg, verify=not args.no_verify)


if __name__ == "__main__":
    main()
