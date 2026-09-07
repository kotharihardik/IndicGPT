"""Load the Phase 2 YAML configs and turn them into objects.

    configs/model.yaml      what the network is  -> ModelConfig
    configs/pretrain.yaml   how it is trained    -> dict
    configs/eval.yaml       how it is measured   -> dict

Paths are written relative to the language directory and resolved here. The
vocabulary size comes from the trained tokenizer, not the YAML.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data"))
from corpus_io import load_config as load_pipeline_config  # noqa: E402
from transformer import ModelConfig  # noqa: E402


#: Keys whose values are paths and must be resolved against the language root.
PATH_KEYS = ("tokenizer_model", "data_dir", "runs_dir", "reports_dir",
             "logs_dir", "splits_dir", "checkpoint")


def read_yaml(path: Path) -> Dict[str, Any]:
    """Parse one YAML file into a dict."""
    return yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}


def resolve_paths(blob: Dict[str, Any], root: Path) -> Dict[str, Any]:
    """Turn every known path key in `blob` into an absolute Path."""
    out = dict(blob)
    for key in PATH_KEYS:
        if isinstance(out.get(key), str):
            out[key] = (Path(root) / out[key]).resolve()
    return out


def load_stage(root: Path, stage: str) -> Dict[str, Any]:
    """Load `configs/<stage>.yaml` with its paths resolved.

    Args:
        root: The language directory, e.g. `<repo>/gujarati`.
        stage: `pretrain` or `eval`.
    """
    blob = read_yaml(Path(root) / "configs" / f"{stage}.yaml")
    return resolve_paths(blob, root)


def tokenizer_path(root: Path, cfg: Optional[Dict[str, Any]] = None) -> Path:
    """Path of this language's SentencePiece model, from Phase 1.

    Named by `tokenizer_model` in configs/pretrain.yaml, never guessed: picking
    the wrong one would tokenize against one table and train an embedding sized
    for another.
    """
    if cfg is None:
        cfg = read_yaml(Path(root) / "configs" / "pretrain.yaml")
    stated = cfg.get("tokenizer_model")
    if stated:
        # load_stage() has already resolved this one; a raw read has not.
        path = Path(stated)
        return path if path.is_absolute() else (Path(root) / path).resolve()
    pipeline = load_pipeline_config(root)
    model_dir = pipeline["paths"]["tokenizer_model"]
    models = sorted(model_dir.glob("*.model"))
    if not models:
        raise FileNotFoundError(f"no tokenizer .model in {model_dir}")
    return models[0]


def load_tokenizer(root: Path, cfg: Optional[Dict[str, Any]] = None):
    """Load the Phase 1 SentencePiece processor for this language."""
    import sentencepiece as spm

    sp = spm.SentencePieceProcessor()
    sp.load(str(tokenizer_path(root, cfg)))
    return sp


def load_model_config(root: Path, vocab_size: Optional[int] = None) -> ModelConfig:
    """Build a ModelConfig from `configs/model.yaml`.

    Vocabulary size comes from the tokenizer unless passed in (a checkpoint
    knows its own). If the YAML also states one, the two must agree.
    """
    blob = read_yaml(Path(root) / "configs" / "model.yaml")
    stated = blob.pop("vocab_size", None)
    if vocab_size is None:
        vocab_size = load_tokenizer(root).get_piece_size()
    if stated is not None and int(stated) != int(vocab_size):
        raise ValueError(
            f"configs/model.yaml says vocab_size={stated} but the tokenizer has "
            f"{vocab_size} pieces")
    blob["vocab_size"] = int(vocab_size)
    return ModelConfig.from_dict(blob)
