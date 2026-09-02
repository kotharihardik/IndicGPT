"""Run the whole Phase 2 evaluation suite for this language, in order.

    python3 eval/run_eval.py                 # everything, on the best checkpoint
    python3 eval/run_eval.py --quick         # a small version, for a smoke test
    python3 eval/run_eval.py --only perplexity attention

Each stage is a separate script and is run as its own process, so one failing
stage cannot leave a half-written result behind in another, and any stage can be
re-run on its own afterwards. Everything lands in `report/phase2/<language>/`.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List

HERE = Path(__file__).resolve().parent
LANG_ROOT = HERE.parent

#: Stage name -> script, in the order they should run. The causal check goes
#: first: if the mask is broken, every number after it is meaningless.
STAGES: Dict[str, Path] = {
    "causal_check": HERE / "causal_check.py",
    "perplexity": HERE / "perplexity.py",
    "generation": HERE / "generate.py",
    "attention": HERE / "attention.py",
    "curves": LANG_ROOT / "train" / "plot_curves.py",
}

#: Cut-down arguments so the suite can be exercised in a minute.
QUICK: Dict[str, List[str]] = {
    "perplexity": ["--max-windows", "20"],
    "generation": ["--prompts", "16", "--continuation-tokens", "32"],
    "attention": ["--batches", "2", "--batch-size", "2"],
}


def run_stage(name: str, script: Path, extra: List[str]) -> bool:
    """Run one stage and report whether it succeeded."""
    print("\n" + "=" * 72, flush=True)
    print(f"  {name}", flush=True)
    print("=" * 72, flush=True)
    started = time.time()
    result = subprocess.run([sys.executable, str(script), *extra],
                            cwd=str(LANG_ROOT))
    ok = result.returncode == 0
    print(f"  {name}: {'ok' if ok else 'FAILED'} in {time.time() - started:.1f}s")
    return ok


def main() -> None:
    """Run every stage, then report which ones passed."""
    ap = argparse.ArgumentParser(description="Full Phase 2 evaluation suite.")
    ap.add_argument("--checkpoint", default=None, help="path, or 'best'/'latest'")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--runs-dir", default=None)
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--reports-dir", default=None)
    ap.add_argument("--logs-dir", default=None)
    ap.add_argument("--only", nargs="+", choices=list(STAGES),
                    help="run only these stages")
    ap.add_argument("--quick", action="store_true",
                    help="tiny sample sizes, for checking the plumbing")
    args = ap.parse_args()

    wanted = args.only or list(STAGES)
    outcomes = {}
    for name in wanted:
        extra: List[str] = []
        if name == "curves":
            # plot_curves.py reads a metrics file, not a checkpoint.
            for flag in ("runs_dir", "run_name", "reports_dir", "logs_dir"):
                if getattr(args, flag):
                    extra += [f"--{flag.replace('_', '-')}", getattr(args, flag)]
        else:
            if args.checkpoint:
                extra += ["--checkpoint", args.checkpoint]
            # causal_check keeps its own default of cpu. Three of its four tests
            # compare values for EXACT equality, and it runs on one 64-token
            # sequence, so there is nothing to gain from a GPU and something to
            # lose if a fused kernel ever reorders a reduction.
            if name != "causal_check":
                extra += ["--device", args.device]
            for flag in ("data_dir", "runs_dir", "run_name", "reports_dir"):
                if getattr(args, flag):
                    extra += [f"--{flag.replace('_', '-')}", getattr(args, flag)]
        if args.quick:
            extra += QUICK.get(name, [])
        outcomes[name] = run_stage(name, STAGES[name], extra)

    print("\n" + "=" * 72)
    for name, ok in outcomes.items():
        print(f"  {name:<16}{'ok' if ok else 'FAILED'}")
    print("=" * 72 + "\n")
    raise SystemExit(0 if all(outcomes.values()) else 1)


if __name__ == "__main__":
    main()
