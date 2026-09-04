"""Training loop for the decoder-only language model.

One class holds the whole run: build the optimizer and schedule, step with
gradient accumulation and mixed precision, evaluate on held-out data at a fixed
interval, write metrics, and checkpoint often enough that losing a session costs
minutes rather than hours.

Every checkpoint contains model weights, optimizer state, scheduler state,
gradient-scaler state, the training step, the number of tokens seen, the model
and training configuration, and the random-number states of Python, NumPy and
PyTorch. A resumed run continues on the data it would have seen anyway, because
the batch order is a function of (seed, step) alone.

SIGINT and SIGTERM finish the current step, write a checkpoint and exit cleanly;
`--max-hours` does the same before a hosted session is killed on its own
schedule.
"""

from __future__ import annotations

import json
import math
import os
import random
import signal
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn

from dataset import BlockSampler, TokenStream, sequential_windows, to_batch


#: Bumped if the checkpoint layout ever changes incompatibly.
CHECKPOINT_VERSION = 1


@dataclass
class TrainState:
    """Everything that has to survive a restart, apart from tensors."""

    step: int = 0
    tokens_seen: int = 0
    best_val_loss: float = float("inf")
    wall_seconds: float = 0.0

    def to_dict(self) -> Dict:
        """Plain dict for the checkpoint."""
        return asdict(self)


def make_scaler(enabled: bool):
    """GradScaler across PyTorch versions that moved the constructor."""
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):        # pragma: no cover - old torch
        return torch.cuda.amp.GradScaler(enabled=enabled)


def pick_precision(device: torch.device, requested: str) -> tuple:
    """Choose the autocast dtype for this device.

    `auto` picks bf16 on Ampere or newer and fp16 everywhere else. The test is
    the compute capability, not `torch.cuda.is_bf16_supported()`: that helper
    defaults to `including_emulation=True`, so on a Turing card such as the T4
    it only checks that a bf16 tensor can be created and answers yes. There are
    no bf16 tensor cores behind it, so the maths falls back to fp32 and the card
    runs at a quarter of its fp16 speed. fp16 uses the tensor cores and needs
    only a loss scaler. An explicit `bf16` is still honoured.

    Returns:
        (autocast enabled, dtype, whether a gradient scaler is needed).
    """
    if device.type != "cuda" or requested == "fp32":
        return False, torch.float32, False
    if requested == "bf16":
        return True, torch.bfloat16, False
    if requested == "auto" and torch.cuda.get_device_capability(device)[0] >= 8:
        # Ampere and newer: real bf16 tensor cores, and bf16 keeps fp32's
        # exponent range, so no loss scaling is needed.
        return True, torch.bfloat16, False
    # Turing (T4) and Pascal (P100): fp16 is the fast path. Gradients underflow
    # in fp16's narrow exponent range, so a scaler multiplies the loss up before
    # the backward pass and divides it out before the optimizer step.
    return True, torch.float16, True


def cosine_schedule(warmup_steps: int, max_steps: int, min_ratio: float):
    """Learning-rate multiplier: linear warmup, then cosine decay to `min_ratio`.

    Warmup is needed because Adam's second-moment estimate is meaningless for
    the first few dozen steps, so a full learning rate there takes large, badly
    scaled steps. Cosine decay keeps most of the run near the peak and anneals
    smoothly at the end, which lets the final loss settle.
    """

    def multiplier(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        if step >= max_steps:
            return min_ratio
        progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
        return min_ratio + 0.5 * (1 - min_ratio) * (1 + math.cos(math.pi * progress))

    return multiplier


class Trainer:
    """Pretrains one monolingual language model on one tokenized corpus."""

    def __init__(self, model: nn.Module, model_cfg, cfg: Dict,
                 train_stream: TokenStream, val_stream: TokenStream,
                 run_dir: Path, device: torch.device) -> None:
        self.model = model.to(device)
        self.model_cfg = model_cfg
        self.cfg = cfg
        self.device = device
        self.run_dir = Path(run_dir)
        (self.run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

        self.block_size = int(model_cfg.context_length)
        self.micro_batch = int(cfg["micro_batch_size"])
        self.accum_steps = int(cfg["gradient_accumulation_steps"])
        self.tokens_per_step = self.micro_batch * self.accum_steps * self.block_size

        self.train_stream = train_stream
        self.val_stream = val_stream
        self.sampler = BlockSampler(len(train_stream), self.block_size,
                                    self.micro_batch, seed=int(cfg["seed"]))

        self.max_steps = self._resolve_max_steps()
        self.optimizer = self._build_optimizer()
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            cosine_schedule(int(cfg["warmup_steps"]), self.max_steps,
                            float(cfg["min_lr"]) / float(cfg["learning_rate"])))

        self.amp_on, self.amp_dtype, needs_scaler = pick_precision(
            device, str(cfg.get("precision", "auto")))
        self.scaler = make_scaler(needs_scaler)

        self.state = TrainState()
        self.metrics_path = self.run_dir / "metrics.jsonl"
        self.log_path = self.run_dir / "train.log"
        self._stop_reason: Optional[str] = None
        self._install_signal_handlers()

    # Setup

    def _resolve_max_steps(self) -> int:
        """Total optimizer steps, from an explicit count or a token budget."""
        if self.cfg.get("max_steps"):
            return int(self.cfg["max_steps"])
        epochs = float(self.cfg.get("epochs", 1.0))
        return max(1, int(epochs * len(self.train_stream) / self.tokens_per_step))

    def _build_optimizer(self) -> torch.optim.Optimizer:
        """AdamW with weight decay on matrices only.

        LayerNorm gains and biases have too few parameters to overfit, and
        shrinking them only fights the normalisation, so they are excluded.
        """
        decay, no_decay = [], []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            (decay if param.dim() >= 2 else no_decay).append(param)
        groups = [
            {"params": decay, "weight_decay": float(self.cfg["weight_decay"])},
            {"params": no_decay, "weight_decay": 0.0},
        ]
        fused = self.device.type == "cuda"
        try:
            return torch.optim.AdamW(
                groups, lr=float(self.cfg["learning_rate"]),
                betas=tuple(self.cfg["betas"]), eps=float(self.cfg["eps"]),
                fused=fused)
        except (RuntimeError, TypeError):       # pragma: no cover - old torch/CPU
            return torch.optim.AdamW(
                groups, lr=float(self.cfg["learning_rate"]),
                betas=tuple(self.cfg["betas"]), eps=float(self.cfg["eps"]))

    def _install_signal_handlers(self) -> None:
        """Turn Ctrl-C and a session kill into a clean, checkpointed stop."""

        def handler(signum, _frame):
            if self._stop_reason is None:
                self._stop_reason = f"signal {signal.Signals(signum).name}"
                self.log("stop requested, finishing the current step first")

        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                signal.signal(sig, handler)
            except ValueError:                  # not the main thread
                pass

    # Logging

    def log(self, message: str) -> None:
        """Print a line and append it to the run log."""
        line = f"[{time.strftime('%H:%M:%S')}] {message}"
        print(line, flush=True)
        with open(self.log_path, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")

    def record(self, row: Dict) -> None:
        """Append one metrics row as JSON, for the loss-curve plots."""
        with open(self.metrics_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(row) + "\n")

    # Training

    def _micro_batch(self, batch_index: int):
        """Fetch one micro-batch of training windows."""
        starts = self.sampler.starts(batch_index)
        return to_batch(self.train_stream.windows(starts, self.block_size),
                        self.device)

    def train_step(self) -> Dict[str, float]:
        """One optimizer step: `accum_steps` micro-batches, then an update."""
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0

        base = self.state.step * self.accum_steps
        for micro in range(self.accum_steps):
            x, y = self._micro_batch(base + micro)
            with torch.autocast(device_type=self.device.type,
                                dtype=self.amp_dtype, enabled=self.amp_on):
                _, loss, _ = self.model(x, targets=y)
            # Gradients accumulate as a sum, so each micro-batch contributes
            # its share of the mean over the full batch.
            self.scaler.scale(loss / self.accum_steps).backward()
            total_loss += loss.item()

        # Unscale before clipping, or the threshold would apply to gradients
        # that are still multiplied by the loss scale.
        self.scaler.unscale_(self.optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), float(self.cfg["grad_clip"]))
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.scheduler.step()

        self.state.step += 1
        self.state.tokens_seen += self.tokens_per_step
        return {"loss": total_loss / self.accum_steps,
                "grad_norm": float(grad_norm)}

    @torch.no_grad()
    def evaluate(self, stream: TokenStream, max_windows: Optional[int] = None
                 ) -> Dict[str, float]:
        """Mean cross-entropy over held-out windows, and its perplexity."""
        self.model.eval()
        total_nll, total_tokens = 0.0, 0
        for window in sequential_windows(stream, self.block_size,
                                         self.micro_batch, max_windows):
            x, y = to_batch(window, self.device)
            with torch.autocast(device_type=self.device.type,
                                dtype=self.amp_dtype, enabled=self.amp_on):
                _, loss, _ = self.model(x, targets=y)
            total_nll += float(loss) * y.numel()
            total_tokens += y.numel()
        self.model.train()
        mean = total_nll / max(total_tokens, 1)
        return {"loss": mean, "perplexity": math.exp(min(mean, 20.0)),
                "tokens": total_tokens}

    def fit(self, max_hours: Optional[float] = None) -> None:
        """Run the training loop until the step budget, a limit or a signal."""
        cfg = self.cfg
        log_every = int(cfg["log_interval"])
        eval_every = int(cfg["eval_interval"])
        ckpt_every = int(cfg["checkpoint_interval"])
        eval_windows = int(cfg["eval_windows"])
        deadline = time.time() + max_hours * 3600 if max_hours else None

        self.log(f"training to step {self.max_steps:,} "
                 f"({self.tokens_per_step:,} tokens/step, "
                 f"{self.max_steps * self.tokens_per_step / 1e6:.1f}M tokens)")
        started = time.time()
        window_loss: List[float] = []
        last_report = started

        while self.state.step < self.max_steps and self._stop_reason is None:
            stats = self.train_step()
            window_loss.append(stats["loss"])
            step = self.state.step

            if step % log_every == 0:
                now = time.time()
                elapsed = now - last_report
                tok_per_s = log_every * self.tokens_per_step / max(elapsed, 1e-9)
                remaining = (self.max_steps - step) * self.tokens_per_step / max(tok_per_s, 1e-9)
                mean_loss = sum(window_loss) / len(window_loss)
                lr = self.scheduler.get_last_lr()[0]
                self.record({"step": step, "split": "train", "loss": mean_loss,
                             "perplexity": math.exp(min(mean_loss, 20.0)),
                             "lr": lr, "grad_norm": stats["grad_norm"],
                             "tokens": self.state.tokens_seen,
                             "tokens_per_second": tok_per_s,
                             "wall_seconds": self.state.wall_seconds + now - started})
                self.log(f"step {step:>7,}/{self.max_steps:,}  "
                         f"loss {mean_loss:6.4f}  ppl {math.exp(min(mean_loss, 20.0)):8.2f}  "
                         f"lr {lr:.2e}  |g| {stats['grad_norm']:5.2f}  "
                         f"{tok_per_s / 1e3:6.1f}k tok/s  eta {remaining / 3600:5.2f}h")
                window_loss.clear()
                last_report = now

            if step % eval_every == 0 or step == self.max_steps:
                val = self.evaluate(self.val_stream, eval_windows)
                self.record({"step": step, "split": "validation", **val,
                             "tokens": self.state.tokens_seen})
                self.log(f"  validation  loss {val['loss']:6.4f}  "
                         f"ppl {val['perplexity']:8.2f}  "
                         f"({val['tokens']:,} tokens)")
                if val["loss"] < self.state.best_val_loss:
                    self.state.best_val_loss = val["loss"]
                    self.save_checkpoint("best", val_loss=val["loss"])

            if step % ckpt_every == 0:
                self.save_checkpoint("latest")
                self.save_checkpoint(f"step{step:07d}")
                self.prune_checkpoints(int(cfg.get("keep_checkpoints", 2)))

            if deadline and time.time() > deadline:
                self._stop_reason = f"time limit of {max_hours}h reached"

        self.state.wall_seconds += time.time() - started
        final = self.evaluate(self.val_stream, eval_windows)
        self.record({"step": self.state.step, "split": "validation", **final,
                     "tokens": self.state.tokens_seen, "final": True})
        if final["loss"] < self.state.best_val_loss:
            self.state.best_val_loss = final["loss"]
            self.save_checkpoint("best", val_loss=final["loss"])
        self.save_checkpoint("latest")

        done = self.state.step >= self.max_steps
        self.log(f"stopped at step {self.state.step:,}/{self.max_steps:,} "
                 f"({'complete' if done else self._stop_reason}); "
                 f"validation loss {final['loss']:.4f}, "
                 f"perplexity {final['perplexity']:.2f}")

        # Marker file, so a script can tell "finished" from "ran out of time"
        # without reopening a 300 MB checkpoint to compare step counts.
        marker = self.run_dir / "COMPLETED"
        if done:
            marker.write_text(
                f"steps {self.state.step}\ntokens {self.state.tokens_seen}\n"
                f"best_val_loss {self.state.best_val_loss:.6f}\n"
                f"finished {time.strftime('%Y-%m-%dT%H:%M:%S')}\n", encoding="utf-8")
        else:
            marker.unlink(missing_ok=True)
            self.log("resume with:  python3 train/pretrain.py --resume auto")

    # Checkpoints

    def checkpoint_path(self, tag: str) -> Path:
        """Where a checkpoint with this tag lives."""
        return self.run_dir / "checkpoints" / f"{tag}.pt"

    def save_checkpoint(self, tag: str, val_loss: Optional[float] = None) -> Path:
        """Write a fully resumable checkpoint.

        Contains everything needed to continue: weights, optimizer moments,
        scheduler position, loss scale, step, token count, both configs and the
        random-number states.
        """
        payload = {
            "version": CHECKPOINT_VERSION,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "scaler": self.scaler.state_dict(),
            "state": self.state.to_dict(),
            "model_config": self.model_cfg.to_dict(),
            "train_config": self.cfg,
            "max_steps": self.max_steps,
            "val_loss": val_loss,
            "rng": {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch": torch.get_rng_state(),
                "cuda": (torch.cuda.get_rng_state_all()
                         if torch.cuda.is_available() else None),
            },
            "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        path = self.checkpoint_path(tag)
        tmp = path.with_suffix(".tmp")
        torch.save(payload, tmp)
        os.replace(tmp, path)                   # atomic: never a half-written file
        return path

    def prune_checkpoints(self, keep: int) -> None:
        """Delete old numbered checkpoints, keeping `latest` and `best`."""
        numbered = sorted((self.run_dir / "checkpoints").glob("step*.pt"))
        for path in numbered[:-keep] if keep > 0 else numbered:
            path.unlink(missing_ok=True)

    def recalibrate_best(self) -> float:
        """Re-measure the best-checkpoint threshold under the current protocol.

        `best_val_loss` is a threshold: a checkpoint is written as `best` only
        when validation beats it, which is meaningful only while the number is
        measured the same way. A run resumed after the evaluation set changed -
        a different `eval_windows`, or a change in how windows are selected -
        carries a threshold from a different measurement, and `best.pt` can then
        sit stale for thousands of steps.

        A resumed run therefore re-evaluates once, adopts that as the threshold
        and writes `best` immediately. It costs one validation pass and keeps
        `best` from ever being worse than `latest`.
        """
        current = self.evaluate(self.val_stream, int(self.cfg["eval_windows"]))
        previous = self.state.best_val_loss
        self.state.best_val_loss = current["loss"]
        self.save_checkpoint("best", val_loss=current["loss"])
        self.log(f"re-measured the best-checkpoint threshold: "
                 f"{previous:.4f} -> {current['loss']:.4f} "
                 f"({current['tokens']:,} validation tokens)")
        return current["loss"]

    def load_checkpoint(self, path: Path) -> None:
        """Restore a run from a checkpoint written by `save_checkpoint`."""
        blob = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(blob["model"])
        self.optimizer.load_state_dict(blob["optimizer"])
        self.scheduler.load_state_dict(blob["scheduler"])
        # The loss scaler is the one piece of state that does not carry across a
        # precision change: a bf16 run has no scaler and saves an empty dict,
        # which an fp16 scaler refuses to load. Starting it fresh costs only a
        # few hundred steps of rescaling, so a run may resume at any precision.
        try:
            self.scaler.load_state_dict(blob["scaler"])
        except (RuntimeError, KeyError) as exc:
            self.log(f"starting the gradient scaler fresh ({exc})")
        self.state = TrainState(**blob["state"])

        self._restore_rng(blob.get("rng") or {})

        self.log(f"resumed from {path.name} at step {self.state.step:,} "
                 f"({self.state.tokens_seen / 1e6:.1f}M tokens seen, "
                 f"best validation loss {self.state.best_val_loss:.4f})")

    def _restore_rng(self, rng: Dict) -> None:
        """Put the random-number generators back where they were.

        Best effort by design. A run started on a two-GPU machine and resumed on
        a one-GPU machine has the wrong number of CUDA generator states, which
        warrants a warning rather than a failed resume: the data order comes
        from (seed, step), not from these generators, so training continues on
        the right batches either way.
        """
        try:
            if rng.get("python"):
                random.setstate(rng["python"])
            if rng.get("numpy") is not None:
                np.random.set_state(rng["numpy"])
            if rng.get("torch") is not None:
                torch.set_rng_state(rng["torch"].cpu().to(torch.uint8))
            cuda_states = rng.get("cuda")
            if cuda_states and torch.cuda.is_available():
                if len(cuda_states) == torch.cuda.device_count():
                    torch.cuda.set_rng_state_all(
                        [s.cpu().to(torch.uint8) for s in cuda_states])
                else:
                    self.log(f"checkpoint holds {len(cuda_states)} CUDA generator "
                             f"states but this machine has "
                             f"{torch.cuda.device_count()}; keeping the current ones")
        except (RuntimeError, ValueError, TypeError) as exc:
            self.log(f"could not restore RNG state ({exc}); continuing anyway")
