"""Reading fixed-length windows out of a tokenized split.

Two access patterns, kept apart because they answer different questions:

    BlockSampler        training. Cuts the stream into non-overlapping blocks
                        and visits them in a seeded random order, so one epoch
                        sees every token exactly once and a resumed run picks up
                        the same order it would have followed uninterrupted.

    sequential_windows  evaluation. Walks the stream front to back, so the
                        perplexity and bits-per-byte numbers cover a defined,
                        reproducible portion of the split rather than a sample.

A window is `block_size + 1` tokens long: the first `block_size` are the input
and the last `block_size` are the targets, which is the one-position shift the
causal language-modelling objective needs.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterator, Optional, Tuple

import numpy as np
import torch


class TokenStream:
    """One tokenized split, held as a flat uint16 array.

    Attributes:
        path: The `.bin` file backing this split.
        tokens: The token ids, either memory-mapped or read into RAM.
    """

    def __init__(self, path: Path, preload: bool = True) -> None:
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(
                f"{self.path} not found - run train/prepare_data.py first")
        mapped = np.memmap(self.path, dtype=np.uint16, mode="r")
        # A ~1 GB training stream fits in RAM on any machine that can train on
        # it, and random block access is much cheaper once resident.
        self.tokens = np.asarray(mapped) if preload else mapped

    def __len__(self) -> int:
        """Number of tokens in the split."""
        return int(self.tokens.shape[0])

    def windows(self, starts: np.ndarray, block_size: int) -> np.ndarray:
        """Gather `block_size + 1` tokens from each start offset.

        Returns:
            An int64 array of shape (len(starts), block_size + 1).
        """
        idx = starts[:, None] + np.arange(block_size + 1)[None, :]
        return self.tokens[idx].astype(np.int64)


def to_batch(window: np.ndarray, device: torch.device
             ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Split a window array into (inputs, targets) tensors on `device`.

    Position t of the input predicts position t of the target, which is token
    t+1 of the original stream.
    """
    data = torch.from_numpy(window)
    x, y = data[:, :-1].contiguous(), data[:, 1:].contiguous()
    if device.type == "cuda":
        return (x.pin_memory().to(device, non_blocking=True),
                y.pin_memory().to(device, non_blocking=True))
    return x.to(device), y.to(device)


class BlockSampler:
    """Deterministic shuffled order over non-overlapping training blocks.

    The stream is cut into `n_blocks` windows of `block_size` tokens, and each
    epoch is a fresh permutation seeded by `seed + epoch`. Three properties
    follow:

      - one epoch covers every token exactly once, unlike random-offset
        resampling, which over-visits some spans and skips others;
      - a run that stops and restarts continues with the same blocks in the same
        order, because the order is a function of the seed and the step alone;
      - two runs with the same seed see identical data, which is what makes the
        comparison between the two languages meaningful.
    """

    def __init__(self, n_tokens: int, block_size: int, batch_size: int,
                 seed: int = 1337) -> None:
        self.block_size = block_size
        self.batch_size = batch_size
        self.seed = seed
        # -1 because the last window needs one extra token for its target.
        self.n_blocks = max(1, (n_tokens - 1) // block_size)
        self.batches_per_epoch = max(1, self.n_blocks // batch_size)
        self._epoch = -1
        self._order: Optional[np.ndarray] = None

    def _permutation(self, epoch: int) -> np.ndarray:
        """Block order for one epoch, cached because it is a few MB."""
        if epoch != self._epoch:
            rng = np.random.default_rng(self.seed + epoch)
            self._order = rng.permutation(self.n_blocks)
            self._epoch = epoch
        return self._order

    def starts(self, batch_index: int) -> np.ndarray:
        """Token offsets for the `batch_index`-th batch of the whole run."""
        epoch, within = divmod(batch_index, self.batches_per_epoch)
        order = self._permutation(epoch)
        chosen = order[within * self.batch_size:(within + 1) * self.batch_size]
        return chosen.astype(np.int64) * self.block_size

    def tokens_per_batch(self) -> int:
        """How many training tokens one batch covers."""
        return self.batch_size * self.block_size


def sequential_windows(stream: TokenStream, block_size: int, batch_size: int,
                       max_windows: Optional[int] = None
                       ) -> Iterator[np.ndarray]:
    """Walk the split in non-overlapping windows, front to back.

    With `max_windows` unset - the default, and what every reported metric uses
    - this covers the entire split: every token except a final remainder shorter
    than one window is predicted exactly once. Each window starts with no left
    context, the standard non-overlapping evaluation. A sliding window would
    report a slightly lower perplexity by giving early positions more history;
    this convention is cheaper, more common, and applied identically to both
    models.

    With `max_windows` set, the windows are spread evenly across the whole split
    rather than taken from the front. Phase 1 writes manual sources first, so
    the head of every split file is manual text with no public documents in it;
    reading the first N windows would evaluate one register only.
    """
    total = (len(stream) - 1) // block_size
    if max_windows and max_windows < total:
        # linspace, not random: the same subset every time, so two checkpoints
        # and two languages are compared on identical text.
        chosen = np.linspace(0, total - 1, max_windows).astype(np.int64)
    else:
        chosen = np.arange(total, dtype=np.int64)
    for first in range(0, len(chosen), batch_size):
        starts = chosen[first:first + batch_size] * block_size
        yield stream.windows(starts.astype(np.int64), block_size)


def load_meta(data_dir: Path) -> Dict:
    """Read the `meta.json` written by prepare_data.py."""
    path = Path(data_dir) / "meta.json"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found - run train/prepare_data.py first")
    return json.loads(path.read_text(encoding="utf-8"))
