"""Verify empirically that the model cannot see the future.

Four independent checks:

  1. Future edit     change every token after t; logits at or before t must be
                     identical, and logits after t must change
  2. Prefix equality feeding only the first t+1 tokens gives the same logits
  3. Mask shape      no attention weight above the diagonal, in any head
  4. Gradient reach  d(logit at t)/d(embedding at s) is zero for every s > t,
                     which tests the graph rather than one set of values

    python3 eval/causal_check.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "model"))
from common import (LANGUAGE, add_common_args, eval_config, load_model,  # noqa: E402
                    pick_device, resolve_checkpoint, save_json)
from transformer import DecoderLM  # noqa: E402


@torch.no_grad()
def future_edit_check(model, seq_len: int, cut: int, device) -> Dict:
    """Change everything after `cut` and confirm earlier logits do not move."""
    vocab = model.cfg.vocab_size
    generator = torch.Generator(device="cpu").manual_seed(0)
    original = torch.randint(0, vocab, (1, seq_len), generator=generator)
    edited = original.clone()
    # Shift every future token, so no position stays the same by chance.
    edited[0, cut + 1:] = (edited[0, cut + 1:] + 1 + torch.randint(
        0, vocab - 1, (seq_len - cut - 1,), generator=generator)) % vocab

    before, _, _ = model(original.to(device))
    after, _, _ = model(edited.to(device))
    past = (before[0, :cut + 1] - after[0, :cut + 1]).abs().max().item()
    future = (before[0, cut + 1:] - after[0, cut + 1:]).abs().max().item()
    return {"max_change_at_or_before_cut": past,
            "max_change_after_cut": future,
            "cut_position": cut,
            "passed": past == 0.0 and future > 0.0}


@torch.no_grad()
def prefix_check(model, seq_len: int, cut: int, device) -> Dict:
    """Feeding a prefix must reproduce the logits the full sequence gave."""
    generator = torch.Generator(device="cpu").manual_seed(1)
    ids = torch.randint(0, model.cfg.vocab_size, (1, seq_len), generator=generator)
    full, _, _ = model(ids.to(device))
    prefix, _, _ = model(ids[:, :cut + 1].to(device))
    delta = (full[0, :cut + 1] - prefix[0]).abs().max().item()
    # Not exactly zero: the two runs reduce over different shapes.
    return {"max_difference": delta, "cut_position": cut,
            "passed": delta < 1e-4}


@torch.no_grad()
def mask_check(model, seq_len: int, device) -> Dict:
    """Every attention weight above the diagonal must be exactly zero."""
    generator = torch.Generator(device="cpu").manual_seed(2)
    ids = torch.randint(0, model.cfg.vocab_size, (1, seq_len), generator=generator)
    _, _, attentions = model(ids.to(device), need_weights=True)
    worst = max(float(w[0].triu(diagonal=1).abs().max()) for w in attentions)
    rows = max(float((w[0].sum(-1) - 1).abs().max()) for w in attentions)
    return {"max_weight_above_diagonal": worst,
            "max_row_sum_error": rows,
            "layers_checked": len(attentions),
            "passed": worst == 0.0 and rows < 1e-4}


def gradient_check(model, seq_len: int, cut: int, device) -> Dict:
    """No gradient may flow from a logit at `cut` back to a later position."""
    embeddings = model.token_embedding(
        torch.randint(0, model.cfg.vocab_size, (1, seq_len), device=device))
    embeddings = embeddings.detach().requires_grad_(True)

    # Run the stack directly on the embeddings so the input is differentiable.
    hidden = model.embed_dropout(embeddings)
    for block in model.blocks:
        hidden, _ = block(hidden)
    logits = model.lm_head(model.ln_f(hidden))
    logits[0, cut].sum().backward()

    grad = embeddings.grad[0]
    past = grad[:cut + 1].abs().max().item()
    future = grad[cut + 1:].abs().max().item()
    return {"max_gradient_at_or_before_cut": past,
            "max_gradient_after_cut": future,
            "cut_position": cut,
            "passed": future == 0.0 and past > 0.0}


def run_all(model, seq_len: int, device) -> Dict:
    """Run all four checks at a cut in the middle of the sequence."""
    cut = seq_len // 2
    checks = {
        "future_edit": future_edit_check(model, seq_len, cut, device),
        "prefix_equality": prefix_check(model, seq_len, cut, device),
        "mask_shape": mask_check(model, seq_len, device),
        "gradient_reach": gradient_check(model, seq_len, cut, device),
    }
    checks["all_passed"] = all(c["passed"] for c in checks.values())
    return checks


def print_report(checks: Dict) -> None:
    """Print each check with the number that decided it."""
    detail = {
        "future_edit": ("logits at or before the cut move by",
                        "max_change_at_or_before_cut"),
        "prefix_equality": ("prefix vs full-sequence logits differ by",
                            "max_difference"),
        "mask_shape": ("largest attention weight above the diagonal",
                       "max_weight_above_diagonal"),
        "gradient_reach": ("gradient reaching positions after the cut",
                           "max_gradient_after_cut"),
    }
    print(f"\n  {LANGUAGE}: causal masking checks")
    print("  " + "-" * 66)
    for name, (label, key) in detail.items():
        check = checks[name]
        print(f"  {name:<18}{label:<44}{check[key]:.3e}  "
              f"{'PASS' if check['passed'] else 'FAIL'}")
    print(f"\n  overall: {'PASS' if checks['all_passed'] else 'FAIL'}\n")


def main() -> None:
    """Check a trained checkpoint, or a freshly built model if none exists."""
    ap = argparse.ArgumentParser(description="Verify the causal mask.")
    add_common_args(ap)
    ap.set_defaults(device="cpu")               # the checks compare exact values
    ap.add_argument("--untrained", action="store_true",
                    help="check a freshly initialised model instead")
    ap.add_argument("--seq-len", type=int, default=64)
    ap.add_argument("--out", default="causal_check")
    args = ap.parse_args()

    cfg = eval_config(args)
    device = pick_device(args.device)

    if args.untrained:
        from config import load_model_config
        from common import ROOT

        model_cfg = load_model_config(ROOT)
        model = DecoderLM(model_cfg).to(device).eval()
        source = "freshly initialised model (no checkpoint)"
    else:
        checkpoint = resolve_checkpoint(args.checkpoint, cfg)
        model, model_cfg, meta = load_model(checkpoint, device)
        source = f"{Path(checkpoint).name} at step {meta['state']['step']:,}"

    seq_len = min(args.seq_len, model_cfg.context_length)
    print(f"\n  {source}, sequence length {seq_len}, device {device}")
    checks = run_all(model, seq_len, device)
    print_report(checks)

    save_json(cfg, args.out, {"language": LANGUAGE, "source": source,
                              "sequence_length": seq_len, "checks": checks})
    raise SystemExit(0 if checks["all_passed"] else 1)


if __name__ == "__main__":
    main()
