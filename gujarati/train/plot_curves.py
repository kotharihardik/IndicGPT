"""Loss curves and throughput from a run's metrics.jsonl.

The trainer appends one JSON object per logged step, so this script only reads
and draws. Nothing is recomputed, which keeps the figures and the report tables
consistent.

Two figures:

    loss_curve.png        training and validation loss against tokens seen
    schedule.png          learning rate and gradient norm, the two things to
                          check first when a run misbehaves

    python3 train/plot_curves.py
    python3 train/plot_curves.py --run-name gujarati-25m
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "model"))
from corpus_io import lang_root, load_config  # noqa: E402
from config import load_stage  # noqa: E402

ROOT = lang_root(Path(__file__))
PIPELINE = load_config(ROOT)
LANGUAGE = PIPELINE["language_name"]
COLOUR = {"guj": "#2c7fb8", "nep": "#d95f02"}[PIPELINE["language"]]


def read_metrics(path: Path) -> Dict[str, List[Dict]]:
    """Split metrics.jsonl into its training and validation rows."""
    rows = {"train": [], "validation": []}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                row = json.loads(line)
                rows.setdefault(row["split"], []).append(row)
    return rows


def plot_loss(rows: Dict[str, List[Dict]], path: Path) -> None:
    """Training and validation cross-entropy against tokens seen."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    train, val = rows["train"], rows["validation"]
    ax.plot([r["tokens"] / 1e6 for r in train], [r["loss"] for r in train],
            color=COLOUR, linewidth=1.0, alpha=0.75, label="Training loss")
    ax.plot([r["tokens"] / 1e6 for r in val], [r["loss"] for r in val],
            color="#d62728", marker="o", markersize=3, linewidth=1.4,
            label="Validation loss")
    ax.set_xlabel("Training tokens seen (millions)")
    ax.set_ylabel("Cross-entropy loss (nats per token)")
    ax.set_title(f"{LANGUAGE}: pretraining loss")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)



def plot_schedule(rows: Dict[str, List[Dict]], path: Path) -> None:
    """Learning rate and gradient norm on a shared step axis."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    train = rows["train"]
    fig, ax = plt.subplots(figsize=(8, 5))
    line1 = ax.plot([r["step"] for r in train], [r["lr"] for r in train],
                    color="#1f77b4", label="Learning rate")
    ax.set_xlabel("Optimizer step")
    ax.set_ylabel("Learning rate", color="#1f77b4")
    ax.tick_params(axis="y", labelcolor="#1f77b4")
    ax.grid(alpha=0.3)

    ax2 = ax.twinx()
    line2 = ax2.plot([r["step"] for r in train],
                     [r.get("grad_norm", 0.0) for r in train],
                     color="#7f7f7f", alpha=0.6, linewidth=0.8,
                     label="Gradient norm (pre-clip)")
    ax2.set_ylabel("Gradient norm", color="#7f7f7f")
    ax2.tick_params(axis="y", labelcolor="#7f7f7f")

    ax.set_title(f"{LANGUAGE}: learning-rate schedule and gradient norm")
    lines = line1 + line2
    ax.legend(lines, [ln.get_label() for ln in lines], loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def summarise(rows: Dict[str, List[Dict]]) -> Dict:
    """Headline numbers about the run, for the report tables.

    The step and token totals are the maximum over every row, not the last
    training row: training rows are written every `log_interval` steps, so the
    final partial interval never produces one and the run's true end is only
    recorded by the closing validation row.
    """
    train, val = rows["train"], rows["validation"]
    speeds = [r["tokens_per_second"] for r in train if r.get("tokens_per_second")]
    best = min(val, key=lambda r: r["loss"]) if val else {}
    every = train + val
    return {
        "steps": max((r["step"] for r in every), default=0),
        "tokens": max((r["tokens"] for r in every), default=0),
        "final_train_loss": train[-1]["loss"] if train else None,
        "final_val_loss": val[-1]["loss"] if val else None,
        "final_val_perplexity": val[-1]["perplexity"] if val else None,
        "best_val_loss": best.get("loss"),
        "best_val_perplexity": best.get("perplexity"),
        "best_val_step": best.get("step"),
        "median_tokens_per_second": sorted(speeds)[len(speeds) // 2] if speeds else None,
        "wall_hours": (train[-1].get("wall_seconds", 0) / 3600) if train else 0,
    }


def main() -> None:
    """Draw the curves and write a one-line summary of the run."""
    ap = argparse.ArgumentParser(description="Plot the pretraining curves.")
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--runs-dir", default=None)
    ap.add_argument("--reports-dir", default=None)
    ap.add_argument("--logs-dir", default=None)
    ap.add_argument("--metrics", default=None, help="path to a metrics.jsonl")
    args = ap.parse_args()

    cfg = load_stage(ROOT, "pretrain")
    cfg.setdefault("logs_dir", PIPELINE["paths"]["logs"])
    for key in ("runs_dir", "reports_dir", "logs_dir"):
        if getattr(args, key):
            cfg[key] = Path(getattr(args, key)).resolve()
    run_name = args.run_name or cfg["run_name"]
    metrics = Path(args.metrics) if args.metrics else \
        Path(cfg["runs_dir"]) / run_name / "metrics.jsonl"
    if not metrics.exists():
        raise SystemExit(f"{metrics} not found - train the model first")

    rows = read_metrics(metrics)
    # The brief asks for training logs in the repository, but train/runs/ is
    # gitignored because it holds ~300 MB of checkpoints. Copy the small text
    # files into <language>/logs/pretrain/<run>/, alongside the Phase 1 logs.
    # One directory per run, named by the config: the 12k and 16k runs write the
    # same four filenames and would otherwise overwrite each other.
    logs = Path(cfg["logs_dir"]) / "pretrain" / cfg.get("logs_subdir", "pretrain")
    logs.mkdir(parents=True, exist_ok=True)
    for name in ("metrics.jsonl", "train.log", "config.json", "COMPLETED"):
        source = metrics.parent / name
        if source.exists():
            shutil.copy2(source, logs / name)

    figures = Path(cfg["reports_dir"]) / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    plot_loss(rows, figures / "loss_curve.png")
    plot_schedule(rows, figures / "schedule.png")

    summary = summarise(rows)
    out = Path(cfg["reports_dir"]) / "json" / "training.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"language": LANGUAGE, "run_name": run_name,
                               "metrics_file": str(metrics), **summary},
                              indent=2), encoding="utf-8")

    print(f"\n  {LANGUAGE}: {summary['steps']:,} steps, "
          f"{summary['tokens'] / 1e6:.1f}M tokens, "
          f"{summary['wall_hours']:.2f} h")
    print(f"  best validation loss {summary['best_val_loss']:.4f} "
          f"(perplexity {summary['best_val_perplexity']:.2f}) "
          f"at step {summary['best_val_step']:,}")
    print(f"  figures in {figures}")
    print(f"  logs copied to {logs}\n")


if __name__ == "__main__":
    main()
