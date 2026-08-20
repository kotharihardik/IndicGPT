"""Progress reporting and the run log.

Separated from the adapters so that reporting and collection logic remain
independent: adapters call `prog.keep()` and `prog.reject()` and perform no
formatting themselves.

The reporting is functional rather than cosmetic. A stalled collector and a
merely slow one are indistinguishable without a reported rate; the archive.org
adapter ran at 3.7 K chars/s for an hour before the rate line revealed that it
was fetching books sequentially rather than in batches. The reject counters
are also the source of the cleaning-funnel figures reported in Phase 1.
"""

from __future__ import annotations

import time

from .base import CHARS_PER_TOKEN, TARGET_TOKENS
from corpus_io import human


def fmt_dur(seconds: float) -> str:
    """Format a duration as '45s', '7m20s' or '2h14m'."""
    # NaN and inf both show up here when the rate is still zero.
    if seconds != seconds or seconds in (float("inf"), float("-inf")) or seconds < 0:
        return "?"
    s = int(seconds)
    if s < 60:
        return f"{s}s"
    if s < 3600:
        return f"{s // 60}m{s % 60:02d}s"
    return f"{s // 3600}h{(s % 3600) // 60:02d}m"


class Progress:
    """Counters and the periodic status line for one source.

    `total_chars` and `log_file` are class attributes on purpose: every source
    in a run shares one running total (so the ETA is for the whole run, not
    just the current source) and one log file handle.
    """

    total_chars = 0        # every source this run, including resumed chars
    log_file = None

    def __init__(self, name: str, already: int = 0, budget: int = 0,
                 every: float = 10.0):
        self.name = name
        self.already = already      # chars from previous runs of this source
        self.budget = budget        # max_chars from the source YAML, 0 = none
        self.every = every          # seconds between status lines
        self.start = time.time()
        self.last = 0.0
        self.docs = 0
        self.chars = 0
        self.requests = 0
        self.rejected: dict = {}

    def keep(self, n_chars: int) -> None:
        """Record one document that passed the quality gate."""
        self.docs += 1
        self.chars += n_chars
        Progress.total_chars += n_chars

    def reject(self, reason: str) -> None:
        """Record one dropped document, tallied by reason."""
        self.rejected[reason] = self.rejected.get(reason, 0) + 1

    @property
    def source_chars(self) -> int:
        """This source's total including what earlier runs already collected."""
        return self.already + self.chars

    @property
    def elapsed(self) -> float:
        """Seconds since this source started collecting."""
        return time.time() - self.start

    def line(self, tail: str = "") -> str:
        """Build one status line. Pure formatting, no side effects."""
        rate = self.chars / self.elapsed if self.elapsed else 0
        eta = ((TARGET_TOKENS * CHARS_PER_TOKEN - Progress.total_chars) / rate
               if rate else 0)
        parts = [f"[{self.name}]", f"{self.docs:,} docs",
                 f"{human(self.source_chars)} ch",
                 f"~{human(self.source_chars / CHARS_PER_TOKEN)} tok",
                 f"{human(rate)} ch/s", f"{self.requests:,} req"]
        if self.budget:
            parts.append(f"{100 * self.source_chars / self.budget:.0f}% of budget")
        parts += [f"ETA {fmt_dur(eta)}", fmt_dur(self.elapsed)]
        if tail:
            parts.append(tail)
        return "  |  ".join(parts)

    def tick(self, force: bool = False) -> None:
        """Print a status line, but at most once every `self.every` seconds."""
        now = time.time()
        if force or now - self.last >= self.every:
            self.last = now
            log("  " + self.line())

    def done(self, why: str = "") -> None:
        """Print the final line for this source plus its top reject reasons."""
        log("  " + self.line("DONE" + (f" ({why})" if why else "")))
        if self.rejected:
            top = sorted(self.rejected.items(), key=lambda kv: -kv[1])[:5]
            log("       rejected: " + ", ".join(f"{r} x{n:,}" for r, n in top))


def log(message: str) -> None:
    """Print to the terminal and append to logs/collection/collect_manual.log."""
    print(message, flush=True)
    if Progress.log_file:
        Progress.log_file.write(f"{time.strftime('%H:%M:%S')} {message}\n")
        Progress.log_file.flush()
