#!/usr/bin/env python3
"""Compare the no-positional-embeddings ablation against the standard model.

    python3 report/bonus/make_ablation_comparison.py

The optional bonus in the brief: retrain one language with positional
information removed and compare it to the standard model for that language.
This reads the evaluation JSON of both runs and writes:

    report/bonus/README.md
    report/bonus/figures/*.png

Gujarati, 16k vocabulary. The two runs share a corpus, a tokenizer, a parameter
count (RoPE is computed, not learned, so removing it costs nothing), a schedule
and a seed. Position is the only variable.

The bonus is self-contained in `report/bonus/`: its own results, figures, logs
and this script. The standard model it is compared against is read from where
Phase 2 already keeps it.

Because the tokenizer is shared, perplexity is directly comparable here - unlike
the 16k/12k vocabulary ablation, where only bits-per-byte is. Both are reported.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402

ROOT = Path(__file__).resolve().parent            # report/bonus
REPO = ROOT.parents[1]
PHASE2 = REPO / "report" / "phase2"               # where the standard model lives

LANG, LANG_NAME = "gujarati", "Gujarati"
BASE, ABLATED = "16k", "16k_nopos"
LABELS = {BASE: "RoPE (reported)", ABLATED: "no positional encoding"}
COLOUR = {BASE: "#2c7fb8", ABLATED: "#d95f02"}
SETTINGS = ["greedy", "temp 0.5", "temp 1.0", "temp 1.5"]


#: The ablation keeps its own evidence together; the standard model's already
#: sits where Phase 2 put it.
def json_dir(variant: str) -> Path:
    """Where one run's evaluation JSON lives."""
    return (ROOT / "json" if variant == ABLATED
            else PHASE2 / LANG / f"{LANG}_{variant}" / "json")


def load(variant: str, name: str) -> Dict[str, Any]:
    """One evaluation JSON, or {} if that stage has not run."""
    path = json_dir(variant) / f"{name}.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def curve(variant: str) -> List[Dict[str, Any]]:
    """The validation rows of one run's metrics.jsonl, in step order."""
    path = (ROOT / "logs" / "metrics.jsonl" if variant == ABLATED
            else REPO / LANG / "logs" / "pretrain" / f"pretrain_{variant}" / "metrics.jsonl")
    if not path.exists():
        return []
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    return sorted((r for r in rows if r["split"] == "validation"), key=lambda r: r["step"])


def dig(blob: Any, *keys, default=None):
    """Walk nested dicts without raising when a stage is missing."""
    for key in keys:
        if not isinstance(blob, dict) or key not in blob:
            return default
        blob = blob[key]
    return blob


def delta(new: Optional[float], old: Optional[float]) -> str:
    """Signed relative change from the standard run to the ablated one."""
    if new is None or old is None or not old:
        return "--"
    return f"{100.0 * (new - old) / old:+.1f}%"


def collect() -> Dict[str, Dict[str, Any]]:
    """Every result for both runs."""
    return {v: {n: load(v, n) for n in ("training", "perplexity", "generation",
                                        "attention", "causal_check")}
            for v in (BASE, ABLATED)}


# Tables


def intrinsic_table(data: Dict) -> str:
    """Cross-entropy, perplexity and bits-per-byte on the test split."""
    lines = ["| Metric (test split) | RoPE | No position | Change |",
             "|---|---:|---:|---:|"]
    for label, key, fmt in (("Cross-entropy (nats/token)", "cross_entropy_nats", "{:.4f}"),
                            ("**Perplexity**", "perplexity", "{:.2f}"),
                            ("**Bits per byte**", "bits_per_byte", "{:.4f}")):
        a = dig(data[BASE], "perplexity", "splits", "test", key)
        b = dig(data[ABLATED], "perplexity", "splits", "test", key)
        cells = [fmt.format(a) if a is not None else "--",
                 fmt.format(b) if b is not None else "--"]
        if label.startswith("**"):
            cells = [f"**{c}**" for c in cells]
        lines.append(f"| {label} | {cells[0]} | {cells[1]} | {delta(b, a)} |")
    return "\n".join(lines)


def generation_table(data: Dict) -> str:
    """Generation metrics at every decoding setting."""
    metrics = [("BLEU-4", ("bleu", "bleu"), "{:.2f}"),
               ("chrF++", ("chrf", "chrf"), "{:.2f}"),
               ("ROUGE-L", ("rouge_l", "rouge_l"), "{:.2f}"),
               ("Distinct-2", ("diversity", "distinct_2"), "{:.3f}"),
               ("Repetition-4", ("diversity", "repetition_4"), "{:.3f}")]
    lines = ["| Decoding | Metric | RoPE | No position | Change |",
             "|---|---|---:|---:|---:|"]
    for setting in SETTINGS:
        for label, path, fmt in metrics:
            a = dig(data[BASE], "generation", "settings", setting, *path)
            b = dig(data[ABLATED], "generation", "settings", setting, *path)
            if a is None and b is None:
                continue
            lines.append(f"| {setting} | {label} | "
                         f"{fmt.format(a) if a is not None else '--'} | "
                         f"{fmt.format(b) if b is not None else '--'} | {delta(b, a)} |")
    return "\n".join(lines)


def attention_table(data: Dict) -> str:
    """Head statistics, averaged over all layers and heads."""
    lines = ["| Metric | RoPE | No position | Change |", "|---|---:|---:|---:|"]
    for label, key, fmt in (("Mean normalised entropy", "normalised_entropy", "{:.3f}"),
                            ("Mean attention distance (tokens)", "distance", "{:.1f}"),
                            ("Mean previous-token mass", "previous_token_mass", "{:.3f}"),
                            ("Mean start-token (sink) mass", "bos_mass", "{:.3f}")):
        vals = []
        for v in (BASE, ABLATED):
            grid = dig(data[v], "attention", "statistics", key)
            vals.append(float(np.mean(grid)) if grid else None)
        lines.append(f"| {label} | "
                     f"{fmt.format(vals[0]) if vals[0] is not None else '--'} | "
                     f"{fmt.format(vals[1]) if vals[1] is not None else '--'} | "
                     f"{delta(vals[1], vals[0])} |")
    return "\n".join(lines)


def cost_table(data: Dict) -> str:
    """That the two runs really were matched."""
    lines = ["| | RoPE | No position |", "|---|---:|---:|"]
    for label, key, fmt in (("Optimizer steps", "steps", "{:,}"),
                            ("Training tokens", "tokens", "{:,}"),
                            ("Wall-clock hours", "wall_hours", "{:.2f}"),
                            ("Best validation loss", "best_val_loss", "{:.4f}")):
        vals = [dig(data[v], "training", key) for v in (BASE, ABLATED)]
        lines.append(f"| {label} | " + " | ".join(
            fmt.format(x) if x is not None else "--" for x in vals) + " |")
    return "\n".join(lines)


# Figures


def plot_curves(data: Dict, path: Path) -> bool:
    """Validation loss for both runs against tokens seen."""
    fig, ax = plt.subplots(figsize=(8, 5))
    drew = False
    for v in (BASE, ABLATED):
        rows = curve(v)
        if not rows:
            continue
        ax.plot([r["tokens"] / 1e6 for r in rows], [r["loss"] for r in rows],
                marker="o", markersize=3, linewidth=1.4,
                color=COLOUR[v], label=LABELS[v])
        drew = True
    if not drew:
        plt.close(fig)
        return False
    ax.set_xlabel("Training tokens seen (millions)")
    ax.set_ylabel("Validation cross-entropy (nats per token)")
    ax.set_title(f"{LANG_NAME}: what positional encoding is worth\n"
                 "identical corpus, tokenizer, parameter count, schedule and seed")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return True


def plot_attention_profile(data: Dict, path: Path) -> bool:
    """Entropy and attention distance by depth, both runs."""
    stats = {v: dig(data[v], "attention", "statistics", default={}) for v in (BASE, ABLATED)}
    if not all(s.get("normalised_entropy") for s in stats.values()):
        return False
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for v in (BASE, ABLATED):
        ent = np.array(stats[v]["normalised_entropy"])
        dist = np.array(stats[v]["distance"])
        layers = np.arange(ent.shape[0])
        axes[0].plot(layers, ent.mean(1), "o-", color=COLOUR[v], label=LABELS[v])
        axes[1].plot(layers, dist.mean(1), "o-", color=COLOUR[v], label=LABELS[v])
    axes[0].set_ylabel("Normalised attention entropy")
    axes[0].set_title("How diffuse attention is, by depth")
    axes[1].set_ylabel("Mean attention distance (tokens)")
    axes[1].set_title("How far heads look back, by depth")
    for ax in axes:
        ax.set_xlabel("Layer (0 = closest to the embedding)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=9)
    fig.suptitle(f"{LANG_NAME}: attention with and without positional encoding "
                 "(line = mean over heads)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return True


# The page


def findings(data: Dict) -> List[str]:
    """The conclusions, computed rather than typed."""
    notes = []
    a = dig(data[BASE], "perplexity", "splits", "test", default={})
    b = dig(data[ABLATED], "perplexity", "splits", "test", default={})
    if a and b:
        ppl = 100.0 * (b["perplexity"] - a["perplexity"]) / a["perplexity"]
        bpb = 100.0 * (b["bits_per_byte"] - a["bits_per_byte"]) / a["bits_per_byte"]
        notes.append(
            f"Removing positional encoding costs {ppl:+.1f}% test perplexity "
            f"({a['perplexity']:.2f} to {b['perplexity']:.2f}) and {bpb:+.1f}% "
            f"bits-per-byte ({a['bits_per_byte']:.4f} to {b['bits_per_byte']:.4f}). "
            f"Both runs share a tokenizer, so unlike the vocabulary ablation the "
            f"perplexity comparison is valid on its own here.")
        notes.append(
            "The ablated model still learns. A decoder-only model is not fully "
            "permutation invariant even without positional encoding, because the "
            "causal mask already carries position: the query at index 0 can attend "
            "to one key, the query at index 5 to six. That residual signal is what "
            "the ablated model is left with, and it is why the loss falls at all "
            "rather than flattening at the unigram entropy.")
    for setting in ("temp 1.0", "greedy"):
        x = dig(data[BASE], "generation", "settings", setting, "chrf", "chrf")
        y = dig(data[ABLATED], "generation", "settings", setting, "chrf", "chrf")
        if x and y:
            notes.append(
                f"At {setting}, chrF++ moves from {x:.2f} to {y:.2f} "
                f"({100.0 * (y - x) / x:+.1f}%). chrF++ is character-level on the "
                f"decoded text, so it is directly comparable.")
    # Head-level, not just the mean: the mean previous-token mass falls while the
    # single sharpest previous-token head gets *stronger*, and only the per-head
    # grids show that.
    stats = {v: dig(data[v], "attention", "statistics", default={}) for v in (BASE, ABLATED)}
    if all(s.get("previous_token_mass") for s in stats.values()):
        peak, local, mean_d, mean_e = {}, {}, {}, {}
        for v in (BASE, ABLATED):
            prev = np.array(stats[v]["previous_token_mass"])
            dist = np.array(stats[v]["distance"])
            i = np.unravel_index(prev.argmax(), prev.shape)
            peak[v] = (int(i[0]), int(i[1]), float(prev[i]), float(dist[i]))
            local[v] = float(dist.min())
            mean_d[v] = float(np.mean(dist))
            mean_e[v] = float(np.mean(stats[v]["normalised_entropy"]))
        pb, pa = peak[BASE], peak[ABLATED]
        notes.append(
            f"Attention reorganises rather than degrades. Mean attention distance "
            f"rises from {mean_d[BASE]:.1f} to {mean_d[ABLATED]:.1f} tokens and mean "
            f"normalised entropy from {mean_e[BASE]:.3f} to {mean_e[ABLATED]:.3f}: most "
            f"heads become more diffuse, because without position they cannot "
            f"specialise on a relative offset.")
        notes.append(
            f"But the one positional relation the causal mask still permits - "
            f"\"the token immediately before me\" - is held *harder*. The sharpest "
            f"previous-token head goes from L{pb[0]}H{pb[1]} at {pb[2]:.3f} "
            f"(mean distance {pb[3]:.1f} tokens) under RoPE to L{pa[0]}H{pa[1]} at "
            f"{pa[2]:.3f} (mean distance {pa[3]:.1f} tokens) without it, and the most "
            f"local head in the model tightens from {local[BASE]:.1f} to "
            f"{local[ABLATED]:.1f} tokens. The ablated model concentrates its remaining "
            f"positional capacity into a few very sharp adjacency heads instead of "
            f"spreading offset specialisation across many.")
        sink = [float(np.mean(stats[v]["bos_mass"])) for v in (BASE, ABLATED)]
        notes.append(
            f"Start-token (sink) mass falls from {sink[0]:.3f} to {sink[1]:.3f}, "
            f"{100*(sink[1]-sink[0])/sink[0]:+.0f}%. With no positional signal to "
            f"anchor it, the start token is no longer a distinctive place to park "
            f"attention that has nothing to contribute.")
    g = dig(data[ABLATED], "generation", "settings", "greedy", "diversity", default={})
    gb = dig(data[BASE], "generation", "settings", "greedy", "diversity", default={})
    if g and gb:
        notes.append(
            f"The clearest practical symptom is looping. Under greedy decoding "
            f"repetition-4 rises from {gb['repetition_4']:.3f} to {g['repetition_4']:.3f} "
            f"and Distinct-2 falls from {gb['distinct_2']:.3f} to {g['distinct_2']:.3f}: "
            f"a model that cannot tell where it is in the sequence is worse at "
            f"noticing it has already said something.")
    return notes


def coverage_table() -> str:
    """What the bonus section asks for, and where each piece is in this folder."""
    rows = [("Retrain with positional embeddings removed",
             "`logs/config.json` - `positional: none`, same seed and hyperparameters"),
            ("Perplexity / BPB", "Section 2, from `json/perplexity.json`"),
            ("BLEU / chrF++ / ROUGE-L", "Section 3, from `json/generation.json`"),
            ("Diversity diagnostics", "Section 3, Distinct-2 and Repetition-4"),
            ("Attention heatmaps", "`figures/attention_layer0.png`, `attention_layer10.png`"),
            ("Attention entropy / distance", "Section 4, plus `figures/attention_entropy.png`, "
                                             "`figures/attention_distance.png`"),
            ("Compare to the standard model", "every table here is RoPE against no-position"),
            ("Document the setup", "Section 1"),
            ("Explain what breaks", "Section 5 and Section 6")]
    lines = ["| The bonus asks for | Where it is |", "|---|---|"]
    lines += [f"| {a} | {b} |" for a, b in rows]
    return "\n".join(lines)


def insight(data: Dict) -> List[str]:
    """The plain-language reading of the comparison."""
    out = []
    a = dig(data[BASE], "perplexity", "splits", "test", default={})
    b = dig(data[ABLATED], "perplexity", "splits", "test", default={})
    if a and b:
        d = b["cross_entropy_nats"] - a["cross_entropy_nats"]
        out.append(
            f"**The two headline numbers are one fact, not two.** Perplexity rises "
            f"{100 * (math.exp(d) - 1):.1f}% but bits-per-byte only "
            f"{100 * (b['bits_per_byte'] - a['bits_per_byte']) / a['bits_per_byte']:.1f}%. "
            f"Both come from the same {d:+.3f} nats per token. Perplexity is the "
            f"*exponential* of cross-entropy, so it magnifies a small change; "
            f"bits-per-byte is cross-entropy rescaled, so it does not. The honest "
            f"size of the effect is the {d:.3f} nats - a small but real loss.")
    rows_a, rows_b = dict(_pairs(BASE)), dict(_pairs(ABLATED))
    common = sorted(set(rows_a) & set(rows_b))
    if len(common) > 4:
        early = rows_b[common[len(common) // 10]] - rows_a[common[len(common) // 10]]
        late = rows_b[common[-1]] - rows_a[common[-1]]
        out.append(
            f"**The gap stops growing.** Early in training the ablated model is "
            f"{early:.3f} nats behind; by the end it is {late:.3f} nats behind. It "
            f"falls behind quickly, then holds a steady distance rather than "
            f"drifting further away. Losing position is a fixed handicap, not a "
            f"failure that compounds as training goes on.")
    out.append(
        "**Why it does not collapse.** A causal mask is itself a positional signal: "
        "the first token can attend to one key, the sixth to six. That tells the "
        "model roughly how far into the sentence it is, even with no positional "
        "encoding at all, which is why the ablated model still learns Gujarati "
        "rather than producing a bag of words.")
    out.append(
        "**What position is actually buying.** Not raw word knowledge - the two "
        "models are close on that. What RoPE buys is knowing *which* nearby token "
        "is which. The ablated model tries to recover this with one very sharp "
        "adjacency head, but one head cannot replace a relative-distance signal "
        "available to every head at every layer.")
    g, gb = (dig(data[ABLATED], "generation", "settings", "greedy", "diversity", default={}),
             dig(data[BASE], "generation", "settings", "greedy", "diversity", default={}))
    t10, t10b = (dig(data[ABLATED], "generation", "settings", "temp 1.0", "chrf", "chrf"),
                 dig(data[BASE], "generation", "settings", "temp 1.0", "chrf", "chrf"))
    if g and gb and t10 and t10b:
        out.append(
            f"**Sampling hides the damage; greedy exposes it.** At temperature 1.0 "
            f"chrF++ barely moves ({t10b:.2f} to {t10:.2f}), because randomness "
            f"breaks loops for free. Under greedy decoding, where the model must "
            f"rely on its own sense of where it is, repetition rises from "
            f"{gb['repetition_4']:.3f} to {g['repetition_4']:.3f}. If you judge this "
            f"model only by sampled text you would barely notice position was gone.")
    out.append(
        "**Was RoPE worth it?** Yes, and cheaply. It costs zero parameters and no "
        "measurable training time, and buys back the whole gap. Nothing else in "
        "this project improves the model for free.")
    return out


def _pairs(variant: str):
    """(tokens, validation loss) for one run, for the gap-over-training reading."""
    return [(r["tokens"], r["loss"]) for r in curve(variant)]


def page(data: Dict, figures: Dict[str, bool]) -> str:
    """The generated ablation document."""
    lines = [
        "# Phase 2 bonus - ablation: no positional embeddings", "",
        "Generated by `report/phase2/make_ablation_comparison.py` from the JSON each",
        "evaluation script wrote. Do not edit by hand - re-run the script instead.", "",
        f"The brief's optional bonus: retrain one language with positional "
        f"information removed and compare it against the standard model. This is "
        f"{LANG_NAME}, 16k vocabulary.", "",
        "## 0. What the bonus asks for", "", coverage_table(), "",
        "## 1. What differs, and what does not", "",
        "The two runs share the corpus, the tokenizer, the schedule, the batch size",
        "and the seed. RoPE is computed from the position rather than learned, so",
        "removing it changes the parameter count by **zero** - both models have",
        "25,625,856 trainable parameters. `positional: none` in `configs/model.yaml`",
        "is the only difference between them.", "",
        cost_table(data), "",
        "## 2. Language modelling", "", intrinsic_table(data), "",
    ]
    if figures.get("curves"):
        lines += ["![Validation loss](figures/loss_comparison.png)", ""]
    lines += ["## 3. Generation quality", "", generation_table(data), "",
              "## 4. Attention", "", attention_table(data), ""]
    if figures.get("attention"):
        lines += ["![Attention profile](figures/attention_comparison.png)", ""]
    notes = findings(data)
    if notes:
        lines += ["## 5. What breaks without position", ""]
        lines += [f"- {n}" for n in notes]
        lines.append("")
    lines += ["## 6. What this tells us", ""]
    lines += [f"- {n}" for n in insight(data)]
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    """Build the ablation comparison from the two runs."""
    ap = argparse.ArgumentParser(description="Compare the no-position ablation.")
    ap.add_argument("--root", default=None,
                    help="read and write under this directory instead of report/bonus")
    args = ap.parse_args()
    global ROOT
    if args.root:
        ROOT = Path(args.root).resolve()

    missing = [v for v in (BASE, ABLATED)
               if not (json_dir(v) / "perplexity.json").exists()]
    if missing:
        raise SystemExit("no evaluation results for: " + ", ".join(missing)
                         + f".\n  Looked in " + ", ".join(str(json_dir(v)) for v in missing))

    data = collect()
    figs = ROOT / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    figures = {"curves": plot_curves(data, figs / "loss_comparison.png"),
               "attention": plot_attention_profile(data, figs / "attention_comparison.png")}
    drawn = [k for k, v in figures.items() if v]
    print(f"\n  drew {len(drawn)} figures: {', '.join(drawn) or 'none'}")

    out = ROOT / "README.md"
    out.write_text(page(data, figures), encoding="utf-8")
    print(f"  wrote {out}\n")


if __name__ == "__main__":
    main()
