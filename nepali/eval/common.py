"""Shared loading and plotting helpers for the evaluation scripts.

Resolves the four things every stage needs: a checkpoint, the model it
describes, the tokenizer, and where to write. Output goes to `reports_dir` from
configs/eval.yaml.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "model"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "train"))
from corpus_io import lang_root, load_config  # noqa: E402
from config import load_stage, load_tokenizer as _load_tokenizer  # noqa: E402
from transformer import ModelConfig, DecoderLM  # noqa: E402
from dataset import TokenStream, load_meta  # noqa: E402

ROOT = lang_root(Path(__file__))
PIPELINE = load_config(ROOT)
LANGUAGE = PIPELINE["language_name"]

#: One colour per language, reused by every figure.
COLOUR = {"guj": "#2c7fb8", "nep": "#d95f02"}[PIPELINE["language"]]


def pick_device(requested: str = "auto") -> torch.device:
    """Resolve `auto` to the best device available here."""
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_checkpoint(spec: Optional[str], cfg: Dict[str, Any]) -> Path:
    """Turn a checkpoint argument into a path.

    Accepts a full path, or a tag such as `best` / `latest` resolved inside the
    configured run directory.
    """
    if spec and Path(spec).exists():
        return Path(spec)
    tag = spec or cfg.get("checkpoint_tag", "best")
    run_dir = Path(cfg["runs_dir"]) / cfg["run_name"]
    path = run_dir / "checkpoints" / f"{tag}.pt"
    if not path.exists():
        raise SystemExit(f"no checkpoint at {path} - train the model first")
    return path


def load_model(checkpoint: Path, device: torch.device
               ) -> Tuple[DecoderLM, ModelConfig, Dict[str, Any]]:
    """Rebuild the model a checkpoint describes and load its weights.

    Architecture comes from the checkpoint, not the YAML, so evaluating an
    earlier run cannot pick up a newer config.
    """
    blob = torch.load(checkpoint, map_location=device, weights_only=False)
    model_cfg = ModelConfig.from_dict(blob["model_config"])
    model = DecoderLM(model_cfg)
    model.load_state_dict(blob["model"])
    model.to(device).eval()
    meta = {k: v for k, v in blob.items()
            if k in ("state", "val_loss", "saved_at", "max_steps", "version")}
    meta["checkpoint"] = str(checkpoint)
    return model, model_cfg, meta


def load_tokenizer():
    """This language's Phase 1 SentencePiece model."""
    return _load_tokenizer(ROOT)


def add_common_args(parser) -> None:
    """Arguments every evaluation script accepts (path overrides are for
    hosted workers, where data and runs live under different roots)."""
    parser.add_argument("--checkpoint", default=None, help="path, or 'best'/'latest'")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--data-dir", default=None, help="tokenized .bin directory")
    parser.add_argument("--runs-dir", default=None, help="directory holding runs")
    parser.add_argument("--run-name", default=None, help="run subdirectory name")
    parser.add_argument("--reports-dir", default=None, help="where results are written")


def eval_config(args=None) -> Dict[str, Any]:
    """`configs/eval.yaml` merged with the run settings it needs.

    Command-line path overrides win over both files.
    """
    train_cfg = load_stage(ROOT, "pretrain")
    cfg = load_stage(ROOT, "eval")
    for key in ("runs_dir", "run_name", "data_dir", "reports_dir"):
        cfg.setdefault(key, train_cfg[key])
    for key in ("data_dir", "runs_dir", "reports_dir"):
        value = getattr(args, key, None) if args else None
        if value:
            cfg[key] = Path(value).resolve()
    if args is not None and getattr(args, "run_name", None):
        cfg["run_name"] = args.run_name
    return cfg


def load_split(cfg: Dict[str, Any], split: str) -> Tuple[TokenStream, Dict]:
    """Load one tokenized split and the metadata prepare_data.py wrote for it."""
    data_dir = Path(cfg["data_dir"])
    meta = load_meta(data_dir)
    return TokenStream(data_dir / f"{split}.bin"), meta["splits"][split]


def report_dir(cfg: Dict[str, Any], kind: str) -> Path:
    """`report/phase2/<language>/<kind>/`, created if missing."""
    path = Path(cfg["reports_dir"]) / kind
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(cfg: Dict[str, Any], name: str, blob: Dict[str, Any]) -> Path:
    """Write one result file and say where it went."""
    path = report_dir(cfg, "json") / f"{name}.json"
    path.write_text(json.dumps(blob, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  wrote {path}")
    return path


def get_plt():
    """matplotlib.pyplot with the headless Agg backend already selected."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def new_figure(size=(8, 5)):
    """A (pyplot, figure, axes) triple, ready to draw on."""
    plt = get_plt()
    fig, ax = plt.subplots(figsize=size)
    return plt, fig, ax
