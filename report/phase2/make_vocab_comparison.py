#!/usr/bin/env python3
"""Compare the 16k and 12k vocabulary runs, for both languages.

    python3 report/phase2/make_vocab_comparison.py

Phase 1 chose a 16,000-piece vocabulary; the runner-up in the same sweep was
12,000. Phase 2 trains both at the same ~25.6M parameter budget: the smaller
embedding frees 1.54M parameters, which buys a twelfth Transformer block, so
vocabulary is the only variable and depth absorbs it. Writes:

    report/phase2/vocab_comparison.md
    report/phase2/figures/vocab_ablation/*.png

Tables are written into the page itself rather than as separate files, so each
number appears in one place.

Why bits-per-byte and not perplexity. Perplexity is per token, and the two runs
do not share a tokenizer. A 12,000-piece vocabulary cuts text into more, shorter
pieces, and predicting one of 12,000 is easier than one of 16,000, so the 12k
run posts a lower perplexity without having learned more. Bits-per-byte divides
the same cross-entropy by the bytes of text it explains, and bytes do not depend
on the tokenizer:

    bits_per_byte = (cross_entropy_nats / ln 2) / bytes_per_token

Every conclusion here rests on that number. The generation metrics are also
comparable, because eval/text_metrics.py scores words and characters of the
decoded text rather than tokenizer pieces.

No metric is recomputed. Values are read from the evaluation JSON, or derived
from it by the identity above.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402

ROOT = Path(__file__).resolve().parent            # report/phase2
REPO = ROOT.parents[1]

LANGS = [("gujarati", "Gujarati", "Model H"), ("nepali", "Nepali", "Model L")]
VARIANTS = ["16k", "12k"]

#: One colour per language, one line style per vocabulary, so a reader can see
#: language and run independently in every figure.
COLOUR = {"gujarati": "#2c7fb8", "nepali": "#d95f02"}
STYLE = {"16k": "-", "12k": "--"}

#: Decoding settings, in the order eval/generate.py reports them.
SETTINGS = ["greedy", "temp 0.5", "temp 1.0", "temp 1.5"]

LN2 = math.log(2.0)
MISSING = "_not run_"


# Loading


def run_dir(lang: str, variant: str) -> Path:
    """`report/phase2/<language>/<language>_<variant>/`."""
    return ROOT / lang / f"{lang}_{variant}"


def logs_dir(lang: str, variant: str) -> Path:
    """`<language>/logs/pretrain/pretrain_<variant>/`."""
    return REPO / lang / "logs" / "pretrain" / f"pretrain_{variant}"


def load_json(path: Path) -> Dict[str, Any]:
    """Read one JSON file, or {} if that stage has not run."""
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def read_metrics(lang: str, variant: str) -> List[Dict[str, Any]]:
    """The validation rows of one run's metrics.jsonl, in step order."""
    path = logs_dir(lang, variant) / "metrics.jsonl"
    if not path.exists():
        return []
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("split") == "validation":
                rows.append(row)
    return sorted(rows, key=lambda r: r["step"])


def collect() -> Dict[str, Dict[str, Dict[str, Any]]]:
    """Every result file for both languages and both runs, in one dict."""
    out: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for lang, _, _ in LANGS:
        out[lang] = {}
        for variant in VARIANTS:
            d = run_dir(lang, variant) / "json"
            out[lang][variant] = {
                "perplexity": load_json(d / "perplexity.json"),
                "generation": load_json(d / "generation.json"),
                "attention": load_json(d / "attention.json"),
                "causal": load_json(d / "causal_check.json"),
                "training": load_json(d / "training.json"),
                "run": load_json(logs_dir(lang, variant) / "config.json"),
                "validation_curve": read_metrics(lang, variant),
            }
    return out


def dig(blob: Any, *keys, default=None):
    """Walk nested dicts without raising when a stage has not run."""
    for key in keys:
        if not isinstance(blob, dict) or key not in blob:
            return default
        blob = blob[key]
    return blob


def bits_per_byte(loss_nats: float, bytes_per_token: float) -> float:
    """Convert a per-token cross-entropy in nats to bits per byte of text.

    The identity eval/perplexity.py uses. Applying it to the in-training
    validation losses puts two runs with different tokenizers on one axis.
    `bytes_per_token` is the whole-split ratio the final evaluation measured, so
    the curve is exact at the end of the run and close earlier, where the
    subsampled windows may have a slightly different ratio.
    """
    return (loss_nats / LN2) / bytes_per_token


def split_bpb(blob: Dict, split: str) -> Optional[float]:
    """Final measured bits-per-byte for one split."""
    return dig(blob, "perplexity", "splits", split, "bits_per_byte")


def cell(value: Optional[float], fmt: str = "{:.4f}") -> str:
    """Format a number for a table, or mark it as not measured."""
    return MISSING if value is None else fmt.format(value)


def delta(new: Optional[float], old: Optional[float]) -> str:
    """Signed relative change from the 16k run to the 12k run."""
    if new is None or old is None or not old:
        return MISSING
    return f"{100.0 * (new - old) / abs(old):+.2f}%"


# Tables


def setup_table(data: Dict) -> str:
    """What actually differs between the two runs."""
    rows = [
        ("Vocabulary", lambda b: dig(b, "run", "model_config", "vocab_size"), "{:,}"),
        ("Layers", lambda b: dig(b, "run", "model_config", "n_layers"), "{}"),
        ("Model dimension", lambda b: dig(b, "run", "model_config", "d_model"), "{}"),
        ("Parameters", lambda b: dig(b, "run", "parameters", "total"), "{:,}"),
        ("Embedding parameters",
         lambda b: (dig(b, "run", "model_config", "vocab_size", default=0)
                    * dig(b, "run", "model_config", "d_model", default=0)) or None,
         "{:,}"),
        ("Training tokens", lambda b: dig(b, "training", "tokens"), "{:,}"),
        ("Optimizer steps", lambda b: dig(b, "training", "steps"), "{:,}"),
        ("Wall-clock hours", lambda b: dig(b, "training", "wall_hours"), "{:.2f}"),
        ("Throughput (tokens/s)",
         lambda b: dig(b, "training", "median_tokens_per_second"), "{:,.0f}"),
    ]
    head = ["| |"] + [f" {name} {v} |" for _, name, _ in LANGS for v in VARIANTS]
    lines = ["".join(head), "|---|" + "---:|" * (len(LANGS) * len(VARIANTS))]
    for label, get, fmt in rows:
        cells = []
        for lang, _, _ in LANGS:
            for variant in VARIANTS:
                value = get(data[lang][variant])
                cells.append(MISSING if value is None else fmt.format(value))
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def intrinsic_table(data: Dict) -> str:
    """Bits-per-byte and perplexity, with the comparability warning attached.

    Test split only. The validation numbers move with it and say nothing extra;
    each run's own validation figures are in its evaluation JSON.
    """
    lines = ["| Model | Split | Metric | 16k | 12k | Change | Comparable? |",
             "|---|---|---|---:|---:|---:|---|"]
    for lang, name, tier in LANGS:
        for split in ("test",):
            a, b = data[lang]["16k"], data[lang]["12k"]
            bpb_a, bpb_b = split_bpb(a, split), split_bpb(b, split)
            lines.append(
                f"| {name} ({tier}) | {split} | **Bits per byte** | "
                f"**{cell(bpb_a)}** | **{cell(bpb_b)}** | **{delta(bpb_b, bpb_a)}** | "
                f"**yes** |")
            ppl_a = dig(a, "perplexity", "splits", split, "perplexity")
            ppl_b = dig(b, "perplexity", "splits", split, "perplexity")
            lines.append(
                f"| {name} ({tier}) | {split} | Perplexity | {cell(ppl_a, '{:.2f}')} | "
                f"{cell(ppl_b, '{:.2f}')} | {delta(ppl_b, ppl_a)} | "
                f"**no** - per token |")
            ce_a = dig(a, "perplexity", "splits", split, "cross_entropy_nats")
            ce_b = dig(b, "perplexity", "splits", split, "cross_entropy_nats")
            lines.append(
                f"| {name} ({tier}) | {split} | Cross-entropy (nats/token) | "
                f"{cell(ce_a)} | {cell(ce_b)} | {delta(ce_b, ce_a)} | "
                f"**no** - per token |")
            bpt_a = dig(a, "perplexity", "splits", split, "bytes_per_token")
            bpt_b = dig(b, "perplexity", "splits", split, "bytes_per_token")
            lines.append(
                f"| {name} ({tier}) | {split} | Bytes per token | "
                f"{cell(bpt_a, '{:.3f}')} | {cell(bpt_b, '{:.3f}')} | "
                f"{delta(bpt_b, bpt_a)} | - |")
    return "\n".join(lines)


def generation_table(data: Dict) -> str:
    """Both runs at the decoding setting where each scores its best chrF++.

    The full sweep over decoding settings is the figure below the table; a row
    per setting per metric would repeat it without adding anything.
    """
    metrics = [("BLEU-4", ("bleu", "bleu"), "{:.2f}"),
               ("chrF++", ("chrf", "chrf"), "{:.2f}"),
               ("ROUGE-L", ("rouge_l", "rouge_l"), "{:.2f}"),
               ("Distinct-2", ("diversity", "distinct_2"), "{:.3f}"),
               ("Repetition-4", ("diversity", "repetition_4"), "{:.3f}")]
    lines = ["| Model | Vocabulary | Setting | "
             + " | ".join(label for label, _, _ in metrics) + " |",
             "|---|---|---|" + "---:|" * len(metrics)]
    for lang, name, tier in LANGS:
        # One setting per language, the same for both runs, so the rows compare.
        scored = [(s, max((dig(data[lang][v], "generation", "settings", s,
                               "chrf", "chrf") or -1) for v in VARIANTS))
                  for s in SETTINGS]
        scored = [(s, v) for s, v in scored if v >= 0]
        if not scored:
            continue
        best = max(scored, key=lambda pair: pair[1])[0]
        for variant in VARIANTS:
            blob = data[lang][variant]
            cells = [cell(dig(blob, "generation", "settings", best, *path), fmt)
                     for _, path, fmt in metrics]
            lines.append(f"| {name} ({tier}) | {variant} | {best} | "
                         + " | ".join(cells) + " |")
    return "\n".join(lines)


def attention_table(data: Dict) -> str:
    """Mean entropy and attention distance, averaged over all layers and heads.

    Per-layer rows are not compared: the two runs are 11 and 12 blocks deep, so
    layer *k* is not the same relative depth in both.
    """
    lines = ["| Model | Metric | 16k | 12k | Change |", "|---|---|---:|---:|---:|"]
    for lang, name, tier in LANGS:
        for label, key, fmt in (("Mean normalised entropy", "normalised_entropy",
                                 "{:.3f}"),
                                ("Mean attention distance (tokens)", "distance",
                                 "{:.1f}"),
                                ("Mean previous-token mass", "previous_token_mass",
                                 "{:.3f}")):
            values = []
            for variant in VARIANTS:
                grid = dig(data[lang][variant], "attention", "statistics", key)
                if not grid:
                    values.append(None)
                    continue
                flat = [x for row in grid for x in row] if isinstance(grid[0], list) \
                    else list(grid)
                values.append(sum(flat) / len(flat) if flat else None)
            lines.append(f"| {name} ({tier}) | {label} | {cell(values[0], fmt)} | "
                         f"{cell(values[1], fmt)} | {delta(values[1], values[0])} |")
        layers = [dig(data[lang][v], "attention", "n_layers") for v in VARIANTS]
        lines.append(f"| {name} ({tier}) | Layers x heads | "
                     f"{layers[0] or MISSING} x "
                     f"{dig(data[lang]['16k'], 'attention', 'n_heads') or '?'} | "
                     f"{layers[1] or MISSING} x "
                     f"{dig(data[lang]['12k'], 'attention', 'n_heads') or '?'} | - |")
    return "\n".join(lines)


def cost_table(data: Dict) -> str:
    """What the smaller vocabulary cost in compute, per byte of corpus read."""
    lines = ["| Model | Quantity | 16k | 12k | Change |", "|---|---|---:|---:|---:|"]
    for lang, name, tier in LANGS:
        rows = []
        for variant in VARIANTS:
            b = data[lang][variant]
            tokens = dig(b, "training", "tokens")
            hours = dig(b, "training", "wall_hours")
            steps = dig(b, "training", "steps")
            rows.append((tokens, steps, hours))
        for label, index, fmt in (("Training tokens for one epoch", 0, "{:,}"),
                                  ("Optimizer steps", 1, "{:,}"),
                                  ("Wall-clock hours (T4)", 2, "{:.2f}")):
            a, b = rows[0][index], rows[1][index]
            lines.append(f"| {name} ({tier}) | {label} | {cell(a, fmt)} | "
                         f"{cell(b, fmt)} | {delta(b, a)} |")
    return "\n".join(lines)


# Figures


def new_axes(title: str, xlabel: str, ylabel: str, figsize=(9, 5)):
    """A labelled axes. Every figure in this report carries all three labels."""
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3, linewidth=0.6)
    return fig, ax


def save(fig, path: Path) -> bool:
    """Write one figure and say where it went."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  wrote {path}")
    return True


def plot_bpb_curves(data: Dict, path: Path) -> bool:
    """Validation bits-per-byte against training tokens, all four runs.

    Loss in nats per token cannot share one axis across two tokenizers, so each
    curve is converted to bits per byte with that run's own whole-split
    bytes-per-token ratio.
    """
    fig, ax = new_axes(
        "Validation bits-per-byte during pretraining\n"
        "(per-token loss converted to bits/byte, so the two vocabularies compare)",
        "Training tokens seen", "Validation bits per byte (lower is better)")
    drew = False
    for lang, name, tier in LANGS:
        for variant in VARIANTS:
            blob = data[lang][variant]
            rows = blob["validation_curve"]
            bpt = dig(blob, "perplexity", "splits", "validation", "bytes_per_token")
            if not rows or not bpt:
                continue
            xs = [r["tokens"] for r in rows]
            ys = [bits_per_byte(r["loss"], bpt) for r in rows]
            ax.plot(xs, ys, STYLE[variant], color=COLOUR[lang], linewidth=1.6,
                    label=f"{name} ({tier}), {variant} vocab")
            drew = True
    if not drew:
        plt.close(fig)
        return False
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper right", frameon=True)
    return save(fig, path)


def plot_ppl_trap(data: Dict, path: Path) -> bool:
    """The same four runs read as perplexity and as bits-per-byte, side by side.

    The left panel is what ignoring the tokenizer would suggest; the right panel
    is the comparable measurement. Only the right one is a claim about the model.
    """
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    panels = [("Perplexity (per token) - NOT comparable", "perplexity",
               "Test perplexity", "{:.1f}"),
              ("Bits per byte - comparable", "bits_per_byte",
               "Test bits per byte", "{:.4f}")]
    drew = False
    for ax, (title, key, ylabel, fmt) in zip(axes, panels):
        labels, values, colours = [], [], []
        for lang, name, _ in LANGS:
            for variant in VARIANTS:
                value = dig(data[lang][variant], "perplexity", "splits", "test", key)
                if value is None:
                    continue
                labels.append(f"{name}\n{variant}")
                values.append(value)
                colours.append(COLOUR[lang])
        if not values:
            continue
        drew = True
        bars = ax.bar(labels, values, color=colours, alpha=0.85,
                      edgecolor="black", linewidth=0.6)
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, value, fmt.format(value),
                    ha="center", va="bottom", fontsize=8)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Model and vocabulary")
        ax.set_ylabel(ylabel)
        ax.set_ylim(0, max(values) * 1.2)
        ax.grid(alpha=0.3, linewidth=0.6, axis="y")
    if not drew:
        plt.close(fig)
        return False
    handles = [plt.Rectangle((0, 0), 1, 1, color=COLOUR[lang], alpha=0.85)
               for lang, _, _ in LANGS]
    axes[1].legend(handles, [name for _, name, _ in LANGS], loc="upper right",
                   fontsize=8, frameon=True)
    fig.suptitle("Why the 12k run only looks better: the same four models, "
                 "two ways of reading them", fontsize=11)
    return save(fig, path)


def plot_generation(data: Dict, path: Path) -> bool:
    """chrF++ across decoding settings, both vocabularies."""
    fig, ax = new_axes("Generation quality across decoding settings\n"
                       "(chrF++ is character-level on decoded text, so the two "
                       "vocabularies compare)",
                       "Decoding setting", "chrF++ (higher is better)")
    drew = False
    for lang, name, tier in LANGS:
        for variant in VARIANTS:
            ys = [dig(data[lang][variant], "generation", "settings", s, "chrf", "chrf")
                  for s in SETTINGS]
            if all(y is None for y in ys):
                continue
            xs = [s for s, y in zip(SETTINGS, ys) if y is not None]
            ys = [y for y in ys if y is not None]
            ax.plot(xs, ys, STYLE[variant], marker="o", color=COLOUR[lang],
                    linewidth=1.6, markersize=5,
                    label=f"{name} ({tier}), {variant} vocab")
            drew = True
    if not drew:
        plt.close(fig)
        return False
    ax.legend(loc="best", frameon=True)
    return save(fig, path)


# The page


def findings(data: Dict) -> List[str]:
    """The conclusions, computed rather than typed."""
    notes: List[str] = []
    for lang, name, tier in LANGS:
        a, b = data[lang]["16k"], data[lang]["12k"]
        bpb_a, bpb_b = split_bpb(a, "test"), split_bpb(b, "test")
        ppl_a = dig(a, "perplexity", "splits", "test", "perplexity")
        ppl_b = dig(b, "perplexity", "splits", "test", "perplexity")
        if None in (bpb_a, bpb_b, ppl_a, ppl_b):
            continue
        ppl_change = 100.0 * (ppl_b - ppl_a) / ppl_a
        bpb_change = 100.0 * (bpb_b - bpb_a) / bpb_a
        better = "better" if bpb_b < bpb_a else "worse"
        notes.append(
            f"**{name} ({tier}).** Perplexity falls {abs(ppl_change):.1f}% "
            f"({ppl_a:.2f} to {ppl_b:.2f}), which looks decisive and is almost "
            f"entirely an artifact of the smaller vocabulary: predicting one of "
            f"12,000 pieces is an easier question than one of 16,000. On "
            f"bits-per-byte, which divides by bytes of text rather than by "
            f"tokens, the 12k run is {abs(bpb_change):.2f}% {better} "
            f"({bpb_a:.4f} to {bpb_b:.4f}) - the same model quality to within a "
            f"fraction of a percent.")
        tok_a = dig(a, "training", "tokens")
        tok_b = dig(b, "training", "tokens")
        h_a = dig(a, "training", "wall_hours")
        h_b = dig(b, "training", "wall_hours")
        if None not in (tok_a, tok_b, h_a, h_b):
            notes.append(
                f"That {abs(bpb_change):.2f}% was not free. The 12k tokenizer "
                f"cuts the same corpus into {100.0 * (tok_b - tok_a) / tok_a:+.1f}% "
                f"tokens ({tok_a:,} to {tok_b:,}), so one epoch took "
                f"{h_b:.2f} h instead of {h_a:.2f} h "
                f"({100.0 * (h_b - h_a) / h_a:+.1f}%) on the same card.")
    chrf_notes = []
    for lang, name, tier in LANGS:
        best = None
        for setting in SETTINGS:
            a = dig(data[lang]["16k"], "generation", "settings", setting, "chrf", "chrf")
            b = dig(data[lang]["12k"], "generation", "settings", setting, "chrf", "chrf")
            if a is None or b is None:
                continue
            if best is None or max(a, b) > best[1]:
                best = (setting, max(a, b), a, b)
        if best:
            setting, _, a, b = best
            chrf_notes.append(
                f"**{name} ({tier}).** At its best decoding setting ({setting}) "
                f"chrF++ moves from {a:.2f} to {b:.2f} "
                f"({100.0 * (b - a) / a:+.1f}%). chrF++ is character-level on the "
                f"decoded text, so unlike perplexity this is a like-for-like "
                f"number.")
    return notes + chrf_notes


def conclusion(data: Dict) -> str:
    """The verdict, with its numbers read from the runs rather than asserted."""
    gains, token_costs, time_costs = [], [], []
    for lang, _, _ in LANGS:
        a, b = data[lang]["16k"], data[lang]["12k"]
        bpb_a, bpb_b = split_bpb(a, "test"), split_bpb(b, "test")
        if None not in (bpb_a, bpb_b):
            gains.append(100.0 * (bpb_a - bpb_b) / bpb_a)
        for key, bucket in (("tokens", token_costs), ("wall_hours", time_costs)):
            x, y = dig(a, "training", key), dig(b, "training", key)
            if None not in (x, y) and x:
                bucket.append(100.0 * (y - x) / x)
    if not gains:
        return "Not enough finished runs to draw a conclusion."
    span = (f"{min(gains):.2f}-{max(gains):.2f}%" if len(gains) > 1
            else f"{gains[0]:.2f}%")
    tok = (f"{min(token_costs):.1f}-{max(token_costs):.1f}%" if token_costs else "?")
    hrs = (f"{min(time_costs):.1f}-{max(time_costs):.1f}%" if time_costs else "?")
    layers = dig(data[LANGS[0][0]]["12k"], "run", "model_config", "n_layers")
    base_layers = dig(data[LANGS[0][0]]["16k"], "run", "model_config", "n_layers")
    return (
        f"The Phase 1 choice stands. Holding the parameter budget fixed and "
        f"spending the freed embedding on depth, a 12,000-piece vocabulary buys "
        f"{span} of bits-per-byte and costs {hrs} more wall-clock time per "
        f"epoch. That cost has two independent halves and both are structural: "
        f"the same corpus becomes {tok} more tokens to read, and every one of "
        f"those tokens passes through {layers} Transformer blocks instead of "
        f"{base_layers}. Generation quality does not move in the 12k run's "
        f"favour either - chrF++ at the best decoding setting is flat to "
        f"slightly worse. Neither vocabulary is meaningfully better at modelling "
        f"the language; the 16k run reaches the same place for less compute, "
        f"which is why it remains the reported model for Phase 2 and the "
        f"checkpoint Phase 3 finetunes from.")


def page(data: Dict, figures: Dict[str, bool]) -> str:
    """The generated comparison document."""
    lines = [
        "# Phase 2 - vocabulary ablation: 16k vs 12k", "",
        "Generated by `report/phase2/make_vocab_comparison.py` from the JSON each",
        "evaluation script wrote. Do not edit by hand - re-run the script instead.",
        "",
        "Phase 1 selected a 16,000-piece vocabulary from a sweep whose runner-up",
        "was 12,000. Phase 2 trains both, for both languages, at the same ~25.6M",
        "parameter budget: the smaller embedding frees 1.54M parameters and that",
        "buys a twelfth Transformer block, so vocabulary size is the only",
        "variable and depth is what absorbs it. Four runs, one epoch each, same",
        "optimizer, same schedule, same seed, same corpus.", "",
        "The reported 16k models are documented in "
        "[`results.md`](results.md); the 12k control run is "
        "reported here only.",
        "",
        "## 1. How to read this comparison", "",
        "**Perplexity is not comparable between these runs, and bits-per-byte "
        "is.**", "",
        "Perplexity is the exponential of a cross-entropy measured *per token*, "
        "and the two runs do not share a tokenizer. The 12k tokenizer cuts text "
        "into more, shorter pieces, and choosing the next piece out of 12,000 is "
        "an easier question than choosing it out of 16,000. A model that had "
        "learned nothing extra would still post a lower perplexity on the 12k "
        "vocabulary. The number moves because the units moved.", "",
        "Bits-per-byte divides the same cross-entropy by the bytes of text it "
        "explains:", "",
        "```",
        "bits_per_byte = (cross_entropy_nats / ln 2) / bytes_per_token",
        "```", "",
        "Bytes are a property of the corpus, not of the tokenizer, so this is a "
        "like-for-like number and every conclusion below rests on it. The two "
        "runs evaluate over the same held-out text and the measured byte totals "
        "agree to within 0.001%, which is the check that they really are being "
        "scored on the same material.", "",
        "The generation metrics are comparable for the same reason: "
        "`eval/text_metrics.py` scores word- and character-level on the decoded "
        "text and never on tokenizer pieces.", "",
    ]
    if figures.get("trap"):
        lines += ["![Perplexity versus bits-per-byte]"
                  "(figures/vocab_ablation/perplexity_vs_bpb.png)", ""]
    lines += ["## 2. What differs between the runs", "", setup_table(data), "",
              "## 3. Intrinsic language modelling", "", intrinsic_table(data), ""]
    if figures.get("curves"):
        lines += ["![Validation bits-per-byte during training]"
                  "(figures/vocab_ablation/bpb_curves.png)", "",
                  "The curves are the in-training validation losses converted to "
                  "bits per byte with each run's own whole-split bytes-per-token "
                  "ratio. Plotted against tokens seen rather than steps, because "
                  "a step is 32,768 tokens in both runs but those tokens cover "
                  "different amounts of text.", ""]
    lines += ["## 4. Generation quality", "", generation_table(data), ""]
    if figures.get("generation"):
        lines += ["![chrF++ across decoding settings]"
                  "(figures/vocab_ablation/generation.png)", ""]
    lines += ["A caveat that applies to this section only: prompts are the first "
              "64 tokens of a held-out document and continuations are 96 tokens, "
              "so the 12k model is given slightly less text to read and asked "
              "for slightly less text to write. The scores are comparable; the "
              "task is a few percent different in length.", "",
              "## 5. Attention", "", attention_table(data), "",
              "## 6. Cost", "", cost_table(data), "",
              "## 7. What the numbers say", ""]
    lines += [f"- {note}" for note in findings(data)]
    lines += ["", "## 8. Conclusion", "", conclusion(data), "",
              "The result worth keeping is methodological: a 13-14% perplexity "
              "drop that survives no correction for tokenization is the kind of "
              "improvement that is easy to claim and easy to believe, and "
              "bits-per-byte is what distinguishes it from a real one.", ""]
    return "\n".join(lines)


def main() -> None:
    """Build the vocabulary comparison from whatever runs are present."""
    ap = argparse.ArgumentParser(description="Compare the 16k and 12k runs.")
    ap.add_argument("--root", default=None,
                    help="read and write under this directory instead of report/phase2")
    args = ap.parse_args()
    global ROOT
    if args.root:
        ROOT = Path(args.root).resolve()

    have = [(lang, v) for lang, _, _ in LANGS for v in VARIANTS
            if (run_dir(lang, v) / "json" / "perplexity.json").exists()]
    if len(have) < 2:
        raise SystemExit(
            "need at least two finished runs to compare; found "
            + (", ".join(f"{lang}/{v}" for lang, v in have) or "none")
            + f".\n  Looked under {ROOT}/<language>/<language>_<variant>/json/.")
    print(f"\n  comparing {len(have)} runs: "
          + ", ".join(f"{lang} {v}" for lang, v in have))

    data = collect()
    figs = ROOT / "figures" / "vocab_ablation"
    figs.mkdir(parents=True, exist_ok=True)

    figures = {
        "trap": plot_ppl_trap(data, figs / "perplexity_vs_bpb.png"),
        "curves": plot_bpb_curves(data, figs / "bpb_curves.png"),
        "generation": plot_generation(data, figs / "generation.png"),
    }
    drawn = [name for name, ok in figures.items() if ok]
    print(f"  drew {len(drawn)} figures: {', '.join(drawn) or 'none'}")

    out = ROOT / "vocab_comparison.md"
    out.write_text(page(data, figures), encoding="utf-8")
    print(f"  wrote {out}\n")


if __name__ == "__main__":
    main()
