"""Pretrain this language's decoder-only Transformer.

    python3 train/pretrain.py                          # start or continue a run
    python3 train/pretrain.py --resume auto            # explicit resume
    python3 train/pretrain.py --max-hours 10.0         # stop before a session dies
    python3 train/pretrain.py --smoke                  # 20 steps, tiny model

`--resume auto` is the default: if the run directory already holds a `latest.pt`
it is picked up, otherwise a fresh run starts. That makes the same command safe
to re-run after a hosted session is cut off, which is the whole point of the
checkpointing requirement.

Everything about the run - both configs, the exact parameter count, the device,
the step budget - is written to `train/runs/<name>/config.json` next to the
checkpoints, so a result can be traced back to what produced it.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "model"))
from corpus_io import lang_root, load_config  # noqa: E402
from config import load_model_config, load_stage  # noqa: E402
from transformer import ModelConfig, build_model  # noqa: E402
from dataset import TokenStream, load_meta  # noqa: E402
from trainer import Trainer  # noqa: E402

ROOT = lang_root(Path(__file__))
PIPELINE = load_config(ROOT)


def pick_device(requested: str) -> torch.device:
    """Resolve `auto` to the best device available here."""
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def seed_everything(seed: int) -> None:
    """Make a run reproducible from its seed."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def describe_model(model, model_cfg: ModelConfig) -> dict:
    """Print and return the parameter breakdown for the run log."""
    total = model.num_parameters()
    non_embed = model.num_parameters(non_embedding=True)
    print(f"\n  {PIPELINE['language_name']} model")
    print(f"  vocab {model_cfg.vocab_size:,}  d_model {model_cfg.d_model}  "
          f"layers {model_cfg.n_layers}  heads {model_cfg.n_heads}  "
          f"d_ff {model_cfg.d_ff}  context {model_cfg.context_length}")
    print(f"  positional {model_cfg.positional}  ffn {model_cfg.ffn_activation}  "
          f"tied {model_cfg.tie_embeddings}  dropout {model_cfg.dropout}")
    for name, count in model.parameter_table():
        print(f"    {name:<34}{count:>12,}")
    print(f"    {'TOTAL':<34}{total:>12,}   "
          f"({100 * (total - 25e6) / 25e6:+.2f}% of the ~25M target)")
    print(f"    {'non-embedding':<34}{non_embed:>12,}\n")
    return {"total": total, "non_embedding": non_embed,
            "table": model.parameter_table()}


def smoke_overrides(model_cfg: ModelConfig, cfg: dict) -> tuple:
    """Shrink a run to something that finishes in seconds, for testing."""
    model_cfg = ModelConfig.from_dict({**model_cfg.to_dict(),
                                       "d_model": 128, "n_layers": 2,
                                       "n_heads": 4, "d_ff": 320,
                                       "context_length": 128})
    cfg = {**cfg, "micro_batch_size": 4, "gradient_accumulation_steps": 2,
           "max_steps": 20, "warmup_steps": 5, "log_interval": 5,
           "eval_interval": 10, "checkpoint_interval": 10, "eval_windows": 4}
    return model_cfg, cfg


def main() -> None:
    """Build the model, restore any previous run, and train."""
    ap = argparse.ArgumentParser(description="Pretrain the Phase 2 model.")
    ap.add_argument("--run-name", default=None,
                    help="subdirectory under train/runs (default: from the config)")
    ap.add_argument("--resume", default="auto",
                    help="'auto', 'none', or a path to a checkpoint")
    ap.add_argument("--data-dir", default=None,
                    help="override the tokenized data directory")
    ap.add_argument("--runs-dir", default=None,
                    help="override where checkpoints and logs are written")
    ap.add_argument("--reports-dir", default=None,
                    help="override where report assets are written")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--max-hours", type=float, default=None,
                    help="stop and checkpoint after this many hours")
    ap.add_argument("--precision", default=None, choices=["auto", "bf16", "fp16", "fp32"])
    ap.add_argument("--micro-batch-size", type=int, default=None)
    ap.add_argument("--eval-windows", type=int, default=None,
                    help="validation windows per in-training check (default: config)")
    ap.add_argument("--smoke", action="store_true",
                    help="tiny model, 20 steps: checks the pipeline, not the model")
    args = ap.parse_args()

    cfg = load_stage(ROOT, "pretrain")
    model_cfg = load_model_config(ROOT)
    if args.smoke:
        model_cfg, cfg = smoke_overrides(model_cfg, cfg)
    for key, value in (("max_steps", args.max_steps),
                       ("precision", args.precision),
                       ("micro_batch_size", args.micro_batch_size),
                       ("eval_windows", args.eval_windows)):
        if value is not None:
            cfg[key] = value
    for key in ("runs_dir", "reports_dir"):
        if getattr(args, key):
            cfg[key] = Path(getattr(args, key)).resolve()

    data_dir = Path(args.data_dir) if args.data_dir else Path(cfg["data_dir"])
    meta = load_meta(data_dir)
    if meta["vocab_size"] != model_cfg.vocab_size:
        raise SystemExit(f"tokenized data has vocab {meta['vocab_size']} but the "
                         f"model is built for {model_cfg.vocab_size}")

    run_name = args.run_name or cfg.get("run_name") or "pretrain"
    if args.smoke:
        run_name = f"{run_name}-smoke"
    run_dir = Path(cfg["runs_dir"]) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    device = pick_device(args.device)
    seed_everything(int(cfg["seed"]))
    if device.type == "cuda":
        # TF32 costs a little precision in the matmuls and buys a lot of speed;
        # it does not affect the fp16/bf16 autocast path, only fp32 fallbacks.
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    train_stream = TokenStream(data_dir / "train.bin",
                               preload=bool(cfg.get("preload_train", True)))
    val_stream = TokenStream(data_dir / "validation.bin", preload=True)

    model = build_model(model_cfg)
    params = describe_model(model, model_cfg)

    trainer = Trainer(model, model_cfg, cfg, train_stream, val_stream,
                      run_dir, device)

    print(f"  device {device}  precision "
          f"{'fp32' if not trainer.amp_on else str(trainer.amp_dtype).split('.')[-1]}")
    print(f"  train stream {len(train_stream):,} tokens  "
          f"validation {len(val_stream):,} tokens")
    print(f"  batch {trainer.micro_batch} x {trainer.accum_steps} accum x "
          f"{trainer.block_size} = {trainer.tokens_per_step:,} tokens/step\n")

    (run_dir / "config.json").write_text(json.dumps({
        "language": PIPELINE["language"],
        "language_name": PIPELINE["language_name"],
        "run_name": run_name,
        "model_config": model_cfg.to_dict(),
        "train_config": {k: (str(v) if isinstance(v, Path) else v)
                         for k, v in cfg.items()},
        "parameters": {"total": params["total"],
                       "non_embedding": params["non_embedding"]},
        "data": {"dir": str(data_dir), "meta": meta},
        "max_steps": trainer.max_steps,
        "tokens_per_step": trainer.tokens_per_step,
        "device": str(device),
        "torch": torch.__version__,
        "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    if args.resume != "none":
        path = (trainer.checkpoint_path("latest") if args.resume == "auto"
                else Path(args.resume))
        if path.exists():
            trainer.load_checkpoint(path)
            # The stored threshold may come from a different evaluation set.
            trainer.recalibrate_best()
        elif args.resume != "auto":
            raise SystemExit(f"checkpoint {path} not found")
        else:
            trainer.log("no checkpoint found, starting from scratch")

    if trainer.state.step == 0:
        # Starting over in a directory that already holds a history: keep the
        # old metrics for reference, but do not let two runs share one file,
        # because the step axis would fold back on itself in the plots.
        old = trainer.metrics_path
        if old.exists():
            old.replace(old.with_name(f"metrics.{int(time.time())}.jsonl"))

    trainer.fit(max_hours=args.max_hours)


if __name__ == "__main__":
    main()
