#!/usr/bin/env python3
"""Build the Phase 2 results page and cross-language figures from the run output.

Run after both models have been trained and evaluated:

    python3 report/phase2/make_report_assets.py

It reads the JSON the evaluation scripts wrote under
`report/phase2/<language>/<language>_16k/json/` and writes:

    report/phase2/results.md                            all measured numbers
    report/phase2/<language>/<language>_16k/README.md    one model's figures
    report/phase2/figures/models/*.png                   cross-language figures

`--variant 12k` selects the 12,000-piece control run instead. That run is
reported only through `vocab_comparison.md`, so this script writes no pages for
it; use `make_vocab_comparison.py` after it has been evaluated.

Every number is read from the evaluation JSON and nothing is recomputed here, so
a figure and the table beside it cannot disagree. Tables live in the results
page rather than in separate files, so each number appears in one place, and a
figure is drawn only where a plot shows something a table cannot.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402

ROOT = Path(__file__).resolve().parent            # report/phase2
REPO = ROOT.parents[1]
LANGS = [("gujarati", "Gujarati", "Model H"), ("nepali", "Nepali", "Model L")]

#: The vocabulary chosen in Phase 1, and the run Phase 2 reports on.
REPORTED = "16k"

#: Which run to build the report for. Set by --variant; every input and output
#: path is keyed on it.
VARIANT = REPORTED
COLOUR = {"gujarati": "#2c7fb8", "nepali": "#d95f02"}
SETTINGS = ["greedy", "temp 0.5", "temp 1.0", "temp 1.5"]

MISSING = "_not run yet_"


# Loading


def lang_dir(lang: str) -> Path:
    """`report/phase2/<language>/<language>_<variant>/` for this run."""
    return ROOT / lang / f"{lang}_{VARIANT}"


def load(lang: str, name: str) -> Dict[str, Any]:
    """Read one JSON result, returning {} if that stage has not run."""
    path = lang_dir(lang) / "json" / f"{name}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


#: Where plot_curves.py leaves a finished run's logs: alongside the Phase 1
#: logs, under the language directory, not inside the report.
def logs_dir(lang: str) -> Path:
    """`<language>/logs/pretrain/pretrain_<variant>/` for this language."""
    return REPO / lang / "logs" / "pretrain" / f"pretrain_{VARIANT}"


def load_run_config(lang: str) -> Dict[str, Any]:
    """The config.json the trainer wrote, copied out by plot_curves.py."""
    path = logs_dir(lang) / "config.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def results() -> Dict[str, Dict[str, Any]]:
    """Every result file for both languages, in one dict."""
    return {lang: {"training": load(lang, "training"),
                   "perplexity": load(lang, "perplexity"),
                   "generation": load(lang, "generation"),
                   "attention": load(lang, "attention"),
                   "causal": load(lang, "causal_check"),
                   "run": load_run_config(lang)}
            for lang, _, _ in LANGS}


def corpus_facts(lang: str) -> Dict[str, float]:
    """Composition of one training split, read from the Phase 1 statistics.

    The resource-level comparison explains the gap between the two models by
    what their corpora contain, so those figures are derived here rather than
    typed into the prose, where they would drift the next time Phase 1 is re-run.
    """
    path = REPO / "report" / "phase1" / lang / "json" / "corpus_stats.json"
    if not path.exists():
        return {}
    train = json.loads(path.read_text(encoding="utf-8"))["splits"]["train"]
    by_source = train["by_source"]
    total = train["totals"]["tokens"]

    # archive.org book OCR, against everything else that is not the one large
    # public corpus: books compress very differently from web text.
    books = {k: v for k, v in by_source.items() if k.startswith("ia_")}
    biggest = max(by_source.items(), key=lambda kv: kv[1]["tokens"])
    rest = [v for k, v in by_source.items()
            if k not in books and k != biggest[0]]

    facts = {"public_source": biggest[0],
             "public_share": 100.0 * biggest[1]["tokens"] / total}
    if books:
        book_tokens = sum(v["tokens"] for v in books.values())
        facts["book_share"] = 100.0 * book_tokens / total
        facts["book_chars_per_token"] = sum(v["chars"] for v in books.values()) / book_tokens
    if rest:
        rest_tokens = sum(v["tokens"] for v in rest)
        facts["other_chars_per_token"] = sum(v["chars"] for v in rest) / rest_tokens
    return facts


def cell(value: Optional[float], fmt: str = "{:.4f}") -> str:
    """Format a number for a table, or mark it as not yet measured."""
    return MISSING if value is None else fmt.format(value)


def dig(blob: Dict, *keys, default=None):
    """Walk nested dicts without raising when a stage has not run."""
    for key in keys:
        if not isinstance(blob, dict) or key not in blob:
            return default
        blob = blob[key]
    return blob


# Tables


def architecture_table(data: Dict) -> str:
    """The two model configurations side by side, with exact parameter counts."""
    rows = [("Parameters (total)", "parameters", "total", "{:,}"),
            ("Parameters (non-embedding)", "parameters", "non_embedding", "{:,}")]
    fields = [("Model dimension", "d_model"), ("Layers", "n_layers"),
              ("Attention heads", "n_heads"), ("Head dimension", None),
              ("Feed-forward width", "d_ff"), ("Feed-forward", "ffn_activation"),
              ("Context length", "context_length"), ("Positional scheme", "positional"),
              ("Tied embeddings", "tie_embeddings"), ("Dropout", "dropout"),
              ("Vocabulary", "vocab_size")]

    lines = ["| | Gujarati (Model H) | Nepali (Model L) |", "|---|---:|---:|"]
    for label, key in fields:
        values = []
        for lang, _, _ in LANGS:
            cfg = dig(data[lang], "run", "model_config", default={})
            if key is None:
                d, h = cfg.get("d_model"), cfg.get("n_heads")
                values.append(str(d // h) if d and h else MISSING)
            else:
                value = cfg.get(key)
                if value is None:
                    values.append(MISSING)
                elif isinstance(value, bool):
                    values.append("yes" if value else "no")
                elif isinstance(value, int):
                    values.append(f"{value:,}")
                else:
                    values.append(str(value))
        lines.append(f"| {label} | {values[0]} | {values[1]} |")
    for label, group, key, fmt in rows:
        values = [cell(dig(data[lang], "run", group, key), fmt) for lang, _, _ in LANGS]
        lines.append(f"| **{label}** | **{values[0]}** | **{values[1]}** |")
    return "\n".join(lines)


def training_table(data: Dict) -> str:
    """How each run went: steps, tokens, throughput, wall time, best loss."""
    fields = [("Optimizer steps", "steps", "{:,}"),
              ("Training tokens seen", "tokens", "{:,}"),
              ("Wall-clock hours", "wall_hours", "{:.2f}"),
              ("Median throughput (tokens/s)", "median_tokens_per_second", "{:,.0f}"),
              ("Final training loss", "final_train_loss", "{:.4f}"),
              ("Best validation loss", "best_val_loss", "{:.4f}"),
              ("Best validation perplexity", "best_val_perplexity", "{:.2f}"),
              ("Step of best checkpoint", "best_val_step", "{:,}")]
    lines = ["| | Gujarati (Model H) | Nepali (Model L) |", "|---|---:|---:|"]
    for label, key, fmt in fields:
        values = [cell(data[lang]["training"].get(key), fmt) for lang, _, _ in LANGS]
        lines.append(f"| {label} | {values[0]} | {values[1]} |")
    return "\n".join(lines)


def language_modelling_table(data: Dict) -> str:
    """Cross-entropy, perplexity and bits-per-byte on both held-out splits."""
    lines = ["| Split | Metric | Gujarati (Model H) | Nepali (Model L) |",
             "|---|---|---:|---:|"]
    for split in ("validation", "test"):
        for label, key, fmt in (("Cross-entropy (nats/token)", "cross_entropy_nats", "{:.4f}"),
                                ("Perplexity", "perplexity", "{:.2f}"),
                                ("**Bits per byte**", "bits_per_byte", "**{:.4f}**"),
                                ("Tokens predicted", "predicted_tokens", "{:,}")):
            values = [cell(dig(data[lang], "perplexity", "splits", split, key), fmt)
                      for lang, _, _ in LANGS]
            lines.append(f"| {split} | {label} | {values[0]} | {values[1]} |")
    return "\n".join(lines)


def generation_table(data: Dict) -> str:
    """Every generation metric, for every decoding setting, for both models."""
    metrics = [("BLEU-4", ("bleu", "bleu"), "{:.2f}"),
               ("chrF++", ("chrf", "chrf"), "{:.2f}"),
               ("ROUGE-L", ("rouge_l", "rouge_l"), "{:.2f}"),
               ("Distinct-1", ("diversity", "distinct_1"), "{:.3f}"),
               ("Distinct-2", ("diversity", "distinct_2"), "{:.3f}"),
               ("Repetition-4", ("diversity", "repetition_4"), "{:.3f}")]
    lines = ["| Model | Decoding | " + " | ".join(m[0] for m in metrics) + " |",
             "|---|---|" + "---:|" * len(metrics)]
    for lang, name, tier in LANGS:
        for setting in SETTINGS:
            cells = [cell(dig(data[lang], "generation", "settings", setting, *keys), fmt)
                     for _, keys, fmt in metrics]
            lines.append(f"| {name} ({tier}) | {setting} | " + " | ".join(cells) + " |")
        ref = dig(data[lang], "generation", "settings", "greedy", "reference_diversity",
                  default={})
        if ref:
            lines.append(f"| {name} ({tier}) | _held-out reference_ | - | - | - | "
                         f"{ref['distinct_1']:.3f} | {ref['distinct_2']:.3f} | "
                         f"{ref['repetition_4']:.3f} |")
    return "\n".join(lines)


def attention_table(data: Dict) -> str:
    """Attention summary for both models.

    Summary rows, not one row per layer: the per-layer profile is the figure
    below the table, and the full per-head grid is in `json/attention.json`.
    """
    stats = {lang: dig(data[lang], "attention", "statistics", default={})
             for lang, _, _ in LANGS}
    if not any(s.get("normalised_entropy") for s in stats.values()):
        return MISSING

    def layer_means(lang: str, key: str):
        """Mean over heads, one value per layer."""
        grid = stats[lang].get(key) or []
        return [float(np.mean(row)) for row in grid]

    rows = [
        ("Layers x heads", lambda l: (
            f"{dig(data[l], 'attention', 'n_layers')} x "
            f"{dig(data[l], 'attention', 'n_heads')}")),
        ("Mean normalised entropy", lambda l: (
            f"{np.mean(layer_means(l, 'normalised_entropy')):.3f}")),
        ("Most diffuse layer", lambda l: (
            f"L{int(np.argmax(layer_means(l, 'normalised_entropy')))} "
            f"({max(layer_means(l, 'normalised_entropy')):.3f})")),
        ("Most focused layer", lambda l: (
            f"L{int(np.argmin(layer_means(l, 'normalised_entropy')))} "
            f"({min(layer_means(l, 'normalised_entropy')):.3f})")),
        ("Mean attention distance (tokens)", lambda l: (
            f"{np.mean(layer_means(l, 'distance')):.1f}")),
        ("Longest-range layer", lambda l: (
            f"L{int(np.argmax(layer_means(l, 'distance')))} "
            f"({max(layer_means(l, 'distance')):.1f})")),
        ("Most local layer", lambda l: (
            f"L{int(np.argmin(layer_means(l, 'distance')))} "
            f"({min(layer_means(l, 'distance')):.1f})")),
        ("Mean previous-token mass", lambda l: (
            f"{np.mean(layer_means(l, 'previous_token_mass')):.3f}")),
    ]
    lines = ["| | Gujarati (Model H) | Nepali (Model L) |", "|---|---:|---:|"]
    for label, get in rows:
        try:
            cells = [get(lang) for lang, _, _ in LANGS]
        except (ValueError, IndexError, KeyError):
            continue
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def causal_table(data: Dict) -> str:
    """The four causal-masking checks, for both models."""
    checks = [("Future edit moves earlier logits by", "future_edit",
               "max_change_at_or_before_cut"),
              ("Prefix vs full-sequence logits differ by", "prefix_equality",
               "max_difference"),
              ("Largest weight above the diagonal", "mask_shape",
               "max_weight_above_diagonal"),
              ("Gradient reaching future positions", "gradient_reach",
               "max_gradient_after_cut")]
    lines = ["| Check | Gujarati | Nepali |", "|---|---:|---:|"]
    for label, name, key in checks:
        values = []
        for lang, _, _ in LANGS:
            value = dig(data[lang], "causal", "checks", name, key)
            passed = dig(data[lang], "causal", "checks", name, "passed")
            values.append(MISSING if value is None
                          else f"{value:.1e} {'PASS' if passed else 'FAIL'}")
        lines.append(f"| {label} | {values[0]} | {values[1]} |")
    return "\n".join(lines)


# Figures


def read_metrics(lang: str) -> Dict[str, List[Dict]]:
    """Reload one run's metrics.jsonl from the copy kept in the report."""
    path = logs_dir(lang) / "metrics.jsonl"
    rows: Dict[str, List[Dict]] = {"train": [], "validation": []}
    if not path.exists():
        return rows
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                row = json.loads(line)
                rows.setdefault(row["split"], []).append(row)
    return rows


def plot_loss_comparison(path: Path) -> bool:
    """Both validation curves on one axis, against tokens seen."""
    curves = {lang: read_metrics(lang) for lang, _, _ in LANGS}
    if not any(c["validation"] for c in curves.values()):
        return False
    fig, ax = plt.subplots(figsize=(8, 5))
    for lang, name, tier in LANGS:
        val = curves[lang]["validation"]
        if val:
            ax.plot([r["tokens"] / 1e6 for r in val], [r["loss"] for r in val],
                    color=COLOUR[lang], marker="o", markersize=3, linewidth=1.4,
                    label=f"{name} ({tier})")
    ax.set_xlabel("Training tokens seen (millions)")
    ax.set_ylabel("Validation cross-entropy (nats per token)")
    ax.set_title("Validation loss, Model H against Model L\n"
                 "identical architecture, optimizer and schedule; separate corpora")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return True




def plot_attention_profile(data: Dict, path: Path) -> bool:
    """Entropy and attention distance by depth, both models on one axis."""
    stats = {lang: dig(data[lang], "attention", "statistics", default={})
             for lang, _, _ in LANGS}
    if not all(s.get("normalised_entropy") for s in stats.values()):
        return False
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for lang, name, tier in LANGS:
        entropy = np.array(stats[lang]["normalised_entropy"])
        distance = np.array(stats[lang]["distance"])
        layers = np.arange(entropy.shape[0])
        axes[0].plot(layers, entropy.mean(axis=1), "o-", color=COLOUR[lang],
                     label=f"{name} ({tier})")
        axes[0].fill_between(layers, entropy.min(axis=1), entropy.max(axis=1),
                             color=COLOUR[lang], alpha=0.15)
        axes[1].plot(layers, distance.mean(axis=1), "o-", color=COLOUR[lang],
                     label=f"{name} ({tier})")
        axes[1].fill_between(layers, distance.min(axis=1), distance.max(axis=1),
                             color=COLOUR[lang], alpha=0.15)
    axes[0].set_ylabel("Normalised attention entropy")
    axes[0].set_title("How diffuse attention is, by depth\n"
                      "1.0 = uniform over every allowed key")
    axes[1].set_ylabel("Mean attention distance (tokens)")
    axes[1].set_title("How far heads look back, by depth")
    depth = len(stats["gujarati"]["normalised_entropy"])
    for ax in axes:
        ax.set_xlabel("Layer (0 = closest to the embedding)")
        ax.set_xticks(range(depth))
        ax.grid(alpha=0.3)
        ax.legend(fontsize=9)
    fig.suptitle("Attention behaviour, Model H against Model L "
                 "(line = mean over heads, band = min to max)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return True


# Pages


def comparison_notes(data: Dict) -> List[str]:
    """Comparative sentences, computed rather than typed."""
    notes: List[str] = []
    guj = dig(data["gujarati"], "perplexity", "splits", "test", default={})
    nep = dig(data["nepali"], "perplexity", "splits", "test", default={})
    if guj and nep:
        ppl_gap = 100 * (nep["perplexity"] - guj["perplexity"]) / guj["perplexity"]
        bpb_gap = 100 * (nep["bits_per_byte"] - guj["bits_per_byte"]) / guj["bits_per_byte"]
        density = nep["bytes_per_token"] / guj["bytes_per_token"]
        agree = (ppl_gap < 0) == (bpb_gap < 0)
        better = "Nepali" if bpb_gap < 0 else "Gujarati"
        notes.append(
            f"On the test split Nepali's perplexity is {ppl_gap:+.1f}% relative to "
            f"Gujarati's and its bits-per-byte is {bpb_gap:+.1f}%. The two "
            + ("point the same way, which is the stronger result: a Nepali token "
               f"carries {density:.2f}x as many bytes as a Gujarati one, so Nepali "
               "is making the harder prediction at each step and still wins on both "
               "measures. "
               if agree else
               f"disagree because perplexity is per token and a Nepali token carries "
               f"{density:.2f}x as many bytes, so it is the harder prediction. ")
            + f"Bits-per-byte is the comparable number, and {better} is ahead on it.")
    tokens = {lang: data[lang]["training"].get("tokens") for lang, _, _ in LANGS}
    if all(tokens.values()):
        gap = 100 * (tokens["gujarati"] - tokens["nepali"]) / tokens["gujarati"]
        notes.append(
            f"Model H saw {tokens['gujarati'] / 1e6:.1f}M training tokens and "
            f"Model L {tokens['nepali'] / 1e6:.1f}M, a {gap:.1f}% difference. "
            f"Everything else about the two runs - architecture, optimizer, "
            f"schedule, batch size, seed - is identical, so the corpus and the "
            f"tokenizer are the only variables.")
    if guj and nep and bpb_gap < 0:
        h, l = corpus_facts("gujarati"), corpus_facts("nepali")
        if "book_share" in h and "public_share" in l:
            notes.append(
                f"Model H having the *worse* bits-per-byte while seeing more "
                f"tokens points at the corpora, not the models - the two "
                f"networks, optimizers, schedules and seeds are identical. "
                f"Phase 1 measured the difference: {h['book_share']:.1f}% of the "
                f"Gujarati training split is archive.org OCR of books, "
                f"compressing at {h['book_chars_per_token']:.2f} characters per "
                f"token against {h['other_chars_per_token']:.2f} for its other "
                f"manual sources, while the Nepali split has no comparable "
                f"source and is {l['public_share']:.1f}% one homogeneous public "
                f"corpus. Model H is predicting a more varied and noisier "
                f"distribution.")

    for lang, name, tier in LANGS:
        settings = dig(data[lang], "generation", "settings", default={})
        if settings:
            best = max(SETTINGS, key=lambda s: dig(settings, s, "chrf", "chrf", default=-1))
            notes.append(
                f"{name} ({tier}) scores highest on chrF++ at **{best}** "
                f"({dig(settings, best, 'chrf', 'chrf'):.2f}), against a "
                f"reference Distinct-2 of "
                f"{dig(settings, 'greedy', 'reference_diversity', 'distinct_2'):.3f} "
                f"and a generated Distinct-2 of "
                f"{dig(settings, best, 'diversity', 'distinct_2'):.3f}.")
    return notes


def results_page(data: Dict, figures: Dict[str, bool]) -> str:
    """The generated results document: every number, both models."""
    lines = [
        "# Phase 2 results", "",
        "Generated by `report/phase2/make_report_assets.py` from the JSON each",
        "evaluation script wrote. Do not edit by hand - re-run the script instead.",
        "", "The narrative and the design decisions are in "
        "[`README.md`](README.md); this page is the numbers.", "",
        "## 1. Architecture and parameter count", "", architecture_table(data), "",
        "Both models are the same network. The vocabularies are separate and were",
        "built independently in Phase 1, and the two sweeps ranked the candidate",
        f"sizes the same way, so both models use {VARIANT} pieces and the parameter",
        "counts match as well. A 12,000-piece vocabulary was also trained at the",
        "same parameter budget as a control; the two are compared in",
        "[`vocab_comparison.md`](vocab_comparison.md).", "",
        "## 2. Pretraining", "", training_table(data), "",
    ]
    if figures.get("loss"):
        lines += [f"![Validation loss, both models]"
                  '(figures/models/loss_comparison.png)', ""]
    lines += ["Per-model loss, perplexity and learning-rate curves are in each",
              "language's own page.", "",
              "## 3. Language modelling", "", language_modelling_table(data), ""]
    lines += ["## 4. Generation quality", "", generation_table(data), ""]
    lines += [f"Sample continuations: "
              f"[Gujarati](gujarati/gujarati_{VARIANT}/samples.md) - "
              f"[Nepali](nepali/nepali_{VARIANT}/samples.md).", "",
              "## 5. Attention", "", attention_table(data), ""]
    if figures.get("attention"):
        lines += [f"![Attention profile]"
                  '(figures/models/attention_comparison.png)', ""]
    lines += ["Heatmaps and the per-head grids are in each language's own page.", "",
              "## 6. Causal masking", "", causal_table(data), "",
              "All four checks are described in `eval/causal_check.py`.", ""]

    notes = comparison_notes(data)
    if notes:
        lines += ["## 7. What the numbers say", ""]
        lines += [f"- {note}" for note in notes]
        lines.append("")
    return "\n".join(lines)


def language_page(lang: str, name: str, tier: str, blob: Dict) -> str:
    """One model's figure page.

    The numbers live in `results/results_<variant>.md`, which reports both
    models side by side. This page carries the per-model figures and an index
    of the raw JSON.
    """
    cfg = dig(blob, "run", "model_config", default={})
    lines = [f"# {name} ({tier}) - Phase 2 figures, {VARIANT} vocabulary", "",
             f"{cfg.get('n_layers', '?')} layers, {cfg.get('d_model', '?')} wide, "
             f"{cfg.get('n_heads', '?')} heads, vocabulary "
             f"{cfg.get('vocab_size', 0):,}. All measured numbers for this model "
             f"are in "
             f"[`results.md`](../../results.md); "
             f"this page is the figures.", ""]

    captions = [
        ("loss_curve.png", "Training and validation loss",
         "Validation is measured every 500 steps on held-out text."),
        ("schedule.png", "Learning-rate schedule",
         "Linear warmup over 500 steps, then cosine decay to 10% of peak."),
        ("attention_entropy.png", "Attention entropy per head and layer",
         "Normalised by log(keys visible), so 1.0 is uniform attention at any "
         "depth. Low rows are focused heads."),
        ("attention_distance.png", "Mean attention distance per head and layer",
         "Mass-weighted mean of (query position - key position). Separates "
         "local heads from heads that gather across the prefix."),
        ("attention_heads.png", "Per-head profile",
         "Every head placed by entropy against distance, which is where local, "
         "diffuse and long-range heads separate."),
    ]
    heatmaps = sorted((lang_dir(lang) / "figures").glob("attention_example*.png"))

    figures_dir = lang_dir(lang) / "figures"
    body = []
    for filename, title, note in captions:
        if (figures_dir / filename).exists():
            body += [f"### {title}", "", f"![{title}](figures/{filename})", "",
                     note, ""]
    if heatmaps:
        body += ["### Attention heatmaps", "",
                 "One early layer and one late layer, all heads, on example "
                 "sentences in this model's language. Axes are token indices; "
                 "the sentences and their numbered pieces are in "
                 "[`attention_examples.md`](attention_examples.md).", ""]
        for figure in heatmaps:
            title = figure.stem.replace("attention_", "").replace("_", ", ")
            body += [f"![{title}](figures/{figure.name})", ""]
    if body:
        lines += body

    lines += ["## Raw results", "",
              "`json/` holds the output of each evaluation stage: "
              "`perplexity.json`, `generation.json`, `attention.json` "
              "(per-head entropy and distance grids), `causal_check.json` and "
              "`training.json`. Sample continuations are in "
              "[`samples.md`](samples.md).", ""]
    return "\n".join(lines)


def main() -> None:
    """Build every table, figure and page from whatever has been run."""
    global ROOT, VARIANT
    ap = argparse.ArgumentParser(description="Build the Phase 2 report assets.")
    ap.add_argument("--root", default=None,
                    help="read and write under this directory instead of report/phase2")
    ap.add_argument("--variant", default=VARIANT,
                    help="which vocabulary run to report on: 16k or 12k")
    args = ap.parse_args()
    if args.root:
        ROOT = Path(args.root).resolve()
    VARIANT = args.variant

    missing = [str(lang_dir(lang).relative_to(ROOT)) for lang, _, _ in LANGS
               if not lang_dir(lang).exists()]
    if len(missing) == len(LANGS):
        present = sorted({d.name.rsplit("_", 1)[-1] for lang, _, _ in LANGS
                          for d in (ROOT / lang).glob(f"{lang}_*") if d.is_dir()})
        raise SystemExit(
            f"no results for variant {VARIANT!r}: neither "
            + " nor ".join(missing) + " exists under " + str(ROOT)
            + ".\n  Variants present: " + (", ".join(present) or "(none)"))

    print(f"\n  variant {VARIANT}")
    if VARIANT != REPORTED:
        # The control run is reported only as a comparison against the chosen
        # vocabulary, so its evaluation JSON is all that is kept.
        print(f"  {VARIANT} is the control run; its results are reported in "
              f"vocab_comparison.md.\n  Run: python3 "
              f"report/phase2/make_vocab_comparison.py\n")
        return

    data = results()
    figs = ROOT / "figures" / "models"
    figs.mkdir(parents=True, exist_ok=True)

    # Only two cross-language figures are drawn. Perplexity, bits-per-byte and
    # the generation metrics are reported as numbers in the results page; a bar
    # chart of the same four values would repeat it without adding anything.
    figures = {
        "loss": plot_loss_comparison(figs / "loss_comparison.png"),
        "attention": plot_attention_profile(data, figs / "attention_comparison.png"),
    }
    drawn = [name for name, ok in figures.items() if ok]
    print(f"  drew {len(drawn)} cross-language figures: {', '.join(drawn) or 'none'}")

    # Named by role, not by vocabulary: only the reported run gets a page, so a
    # `_16k` suffix would imply a `_12k` sibling that is not written.
    page_path = ROOT / "results.md"
    page_path.write_text(results_page(data, figures), encoding="utf-8")
    print(f"  wrote {page_path}")
    for lang, name, tier in LANGS:
        page = language_page(lang, name, tier, data[lang])
        out = lang_dir(lang) / "README.md"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(page, encoding="utf-8")
        print(f"  wrote {out}")

    pending = [f"{lang}/{lang}_{VARIANT}/{stage}" for lang, _, _ in LANGS
               for stage in ("training", "perplexity", "generation", "attention")
               if not data[lang][stage]]
    if pending:
        print("\n  still to run: " + ", ".join(pending))
    print()


if __name__ == "__main__":
    main()
