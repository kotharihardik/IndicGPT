"""Intrinsic language-modelling metrics on the held-out splits.

Three numbers per split:

    cross-entropy   mean negative log-likelihood per predicted token, in nats
    perplexity      exp(cross-entropy) - the effective number of tokens the
                    model is choosing between at each step
    bits-per-byte   the same total likelihood divided by the UTF-8 bytes of the
                    text instead of by tokens

Bits-per-byte is the only one of the three that compares two models fairly when
their tokenizers differ. Perplexity is per *token*, and a tokenizer that cuts
text into fewer, longer pieces makes each prediction harder. Dividing by bytes
removes the tokenizer, because bytes are a property of the text. This applies
between Model H and Model L, and between the 16k and 12k runs of one model.

Evaluation walks each split front to back in non-overlapping windows of
`context_length` tokens. Every token is predicted exactly once, except a final
remainder shorter than one window; the byte total is scaled by the same fraction
so the two sides of the division cover the same text.

    python3 eval/perplexity.py                       # validation and test
    python3 eval/perplexity.py --splits validation --max-windows 200
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path
from typing import Dict

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (LANGUAGE, add_common_args, eval_config, load_model,  # noqa: E402
                    load_split, pick_device, resolve_checkpoint, save_json)
from dataset import sequential_windows, to_batch  # noqa: E402

LN2 = math.log(2.0)


@torch.no_grad()
def score_split(model, stream, split_meta: Dict, block_size: int,
                batch_size: int, device: torch.device,
                max_windows: int | None = None) -> Dict[str, float]:
    """Total likelihood of one split, and the three metrics derived from it.

    Args:
        model: The trained language model, already in eval mode.
        stream: The tokenized split.
        split_meta: The `meta.json` entry for this split, holding its byte and
            token totals.
        block_size: Window length in tokens, the model's context.
        batch_size: Windows per forward pass.
        device: Where to run.
        max_windows: Stop early; the byte total is scaled to match.

    Returns:
        Cross-entropy, perplexity, bits-per-byte and the counts behind them.
    """
    model.eval()
    total_nll, predicted = 0.0, 0
    started = time.time()

    for window in sequential_windows(stream, block_size, batch_size, max_windows):
        x, y = to_batch(window, device)
        # float32 throughout: this is the reported number, and skipping mixed
        # precision here costs only a few seconds.
        _, loss, _ = model(x, targets=y)
        total_nll += float(loss) * y.numel()
        predicted += y.numel()

    stream_tokens = len(stream)
    covered = predicted / stream_tokens
    # `bytes` counts the UTF-8 text only. The <s>/</s> wrappers add tokens with
    # no bytes, so the ratio here is not the tokenizer's chars-per-token.
    bytes_covered = split_meta["bytes"] * covered

    cross_entropy = total_nll / max(predicted, 1)
    return {
        "cross_entropy_nats": cross_entropy,
        "perplexity": math.exp(min(cross_entropy, 20.0)),
        "bits_per_byte": total_nll / (LN2 * bytes_covered) if bytes_covered else 0.0,
        "bits_per_token": cross_entropy / LN2,
        "predicted_tokens": predicted,
        "split_tokens": stream_tokens,
        "coverage": covered,
        "bytes_evaluated": bytes_covered,
        "bytes_per_token": bytes_covered / max(predicted, 1),
        "documents": split_meta["docs"],
        "seconds": round(time.time() - started, 1),
    }


def print_table(results: Dict[str, Dict]) -> None:
    """Show every split on one line each."""
    print(f"\n  {LANGUAGE}")
    print(f"  {'split':<12}{'CE (nats)':>12}{'perplexity':>13}{'bits/byte':>12}"
          f"{'tokens':>14}{'bytes':>14}")
    print("  " + "-" * 77)
    for split, row in results.items():
        print(f"  {split:<12}{row['cross_entropy_nats']:>12.4f}"
              f"{row['perplexity']:>13.2f}{row['bits_per_byte']:>12.4f}"
              f"{row['predicted_tokens']:>14,}{int(row['bytes_evaluated']):>14,}")
    print()


def main() -> None:
    """Score the requested splits and write the results into the report."""
    ap = argparse.ArgumentParser(description="Perplexity and bits-per-byte.")
    add_common_args(ap)
    ap.add_argument("--splits", nargs="+", default=["validation", "test"])
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-windows", type=int, default=None,
                    help="evaluate only this many windows (testing)")
    ap.add_argument("--out", default="perplexity")
    args = ap.parse_args()

    cfg = eval_config(args)
    device = pick_device(args.device)
    checkpoint = resolve_checkpoint(args.checkpoint, cfg)
    model, model_cfg, meta = load_model(checkpoint, device)
    print(f"\n  checkpoint {checkpoint.name} at step "
          f"{meta['state']['step']:,}  device {device}")

    results = {}
    for split in args.splits:
        stream, split_meta = load_split(cfg, split)
        results[split] = score_split(model, stream, split_meta,
                                     model_cfg.context_length, args.batch_size,
                                     device, args.max_windows)
    print_table(results)

    save_json(cfg, args.out, {
        "language": LANGUAGE,
        "checkpoint": str(checkpoint),
        "step": meta["state"]["step"],
        "tokens_seen": meta["state"]["tokens_seen"],
        "context_length": model_cfg.context_length,
        "parameters": sum(p.numel() for p in model.parameters()),
        "evaluation": "non-overlapping windows, float32, whole split",
        "splits": results,
    })


if __name__ == "__main__":
    main()
