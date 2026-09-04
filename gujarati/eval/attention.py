"""What the attention heads learned: heatmaps, entropy and attention distance.

Three things are produced, all from the same forward passes:

  heatmaps    query-by-key attention for an early layer and a late layer, every
              head, on an example sentence in this language.
  entropy     how spread out each head's attention is. A head that puts all its
              mass on one key has entropy 0; one that spreads evenly over t
              allowed keys has entropy log(t). Because the number of allowed
              keys grows with position, the normalised form - entropy divided by
              log(t+1) - is the comparable number across heads and models.
  distance    how far back each head looks, as the mass-weighted mean of
              (query position - key position). This is what separates a local
              head, which reads the previous word or two, from a long-range head
              that gathers from the whole prefix.

Two further diagnostics are computed alongside them: `previous_token_mass`
(attention on exactly t-1, the signature of an induction or positional head) and
`bos_mass` (attention parked on the start token, the sink a head uses when it
has nothing to contribute).

Axis labels are token *indices*, not the pieces themselves: rendering Gujarati
or Devanagari inside a figure needs a script font a hosted worker may not have,
and the labels would come out as empty boxes. The pieces are listed beside the
figures in `attention_examples.md`, indexed to match.

    python3 eval/attention.py
    python3 eval/attention.py --batches 5        # quick
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (LANGUAGE, PIPELINE, add_common_args, eval_config, get_plt,  # noqa: E402
                    load_model, load_split, load_tokenizer, new_figure,
                    pick_device, report_dir, resolve_checkpoint, save_json)
from dataset import sequential_windows, to_batch  # noqa: E402


@torch.no_grad()
def head_statistics(model, stream, block_size: int, batch_size: int,
                    n_batches: int, device: torch.device) -> Dict:
    """Entropy, distance and mass diagnostics for every layer and head.

    Averaged over held-out text rather than one sentence, so the numbers
    describe the head rather than the example.
    """
    n_layers, n_heads = model.cfg.n_layers, model.cfg.n_heads
    keys = ("entropy", "normalised_entropy", "distance", "normalised_distance",
            "previous_token_mass", "self_mass", "bos_mass")
    sums = {key: np.zeros((n_layers, n_heads)) for key in keys}
    batches_seen = 0

    positions = torch.arange(block_size, device=device).float()
    allowed = positions + 1.0                          # keys visible at query t
    log_allowed = torch.log(allowed.clamp(min=1.0)).clamp(min=1e-6)

    def head_mean(values: torch.Tensor) -> np.ndarray:
        """Average a (B, h, T) tensor over the batch and query positions 1..T-1.

        Query 0 is dropped: it can only attend to itself, so its entropy is 0
        and its distance is 0 for every head, which would flatten the averages.
        """
        return values[:, :, 1:].mean(dim=(0, 2)).cpu().numpy()

    # sequential_windows counts windows, not batches, so convert.
    for window in sequential_windows(stream, block_size, batch_size,
                                     n_batches * batch_size):
        x, _ = to_batch(window, device)
        _, _, attentions = model(x, need_weights=True)
        for layer, weights in enumerate(attentions):           # (B, h, T, T)
            probs = weights.float()
            entropy = -(probs * torch.log(probs.clamp(min=1e-12))).sum(-1)
            offsets = positions[None, None, :, None] - positions[None, None, None, :]
            distance = (probs * offsets).sum(-1)
            # p[t, t-1] for every query t, padded so index 0 lines up.
            previous = torch.nn.functional.pad(
                torch.diagonal(probs, offset=-1, dim1=-2, dim2=-1), (1, 0))

            sums["entropy"][layer] += head_mean(entropy)
            sums["normalised_entropy"][layer] += head_mean(entropy / log_allowed)
            sums["distance"][layer] += head_mean(distance)
            sums["normalised_distance"][layer] += head_mean(distance / allowed)
            sums["previous_token_mass"][layer] += head_mean(previous)
            sums["self_mass"][layer] += head_mean(
                torch.diagonal(probs, offset=0, dim1=-2, dim2=-1))
            sums["bos_mass"][layer] += head_mean(probs[:, :, :, 0])
        batches_seen += 1

    result = {key: (value / max(batches_seen, 1)).tolist()
              for key, value in sums.items()}
    result["batches"] = batches_seen
    result["tokens"] = batches_seen * batch_size * block_size
    return result


@torch.no_grad()
def sentence_attention(model, sp, sentence: str, device: torch.device
                       ) -> Dict:
    """Attention weights for one example sentence, plus its token pieces."""
    ids = [sp.bos_id()] + sp.encode(sentence)
    ids = ids[:model.cfg.context_length]
    x = torch.tensor([ids], dtype=torch.long, device=device)
    _, _, attentions = model(x, need_weights=True)
    return {
        "sentence": sentence,
        "pieces": [sp.id_to_piece(i) for i in ids],
        "ids": ids,
        # (layers, heads, T, T) as plain lists so it can live in JSON
        "weights": [w[0].float().cpu().numpy() for w in attentions],
    }


# Plots


def plot_heatmaps(example: Dict, layer: int, path: Path, title_note: str) -> None:
    """One panel per head: query position against key position.

    All heads share one colour scale so the panels can be compared, and the
    scale is square-root rather than linear. Attention is strongly peaked - many
    heads park a large share of their mass on the start token - and on a linear
    scale everything except that cell renders as flat black.
    """
    import matplotlib.colors as mcolors

    plt = get_plt()
    weights = example["weights"][layer]                      # (h, T, T)
    n_heads, seq_len, _ = weights.shape
    cols = min(3, n_heads)
    rows = math.ceil(n_heads / cols)

    # Blank out the future so the causal mask is visible as absence, not as a
    # very dark colour that could be confused with a small weight.
    future = np.triu(np.ones((seq_len, seq_len), dtype=bool), k=1)
    shown = np.where(future[None, :, :], np.nan, weights)
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("white")
    norm = mcolors.PowerNorm(gamma=0.5, vmin=0.0, vmax=float(np.nanmax(shown)))

    fig, axes = plt.subplots(rows, cols, figsize=(3.8 * cols, 3.6 * rows),
                             squeeze=False, constrained_layout=True)
    ticks = np.arange(0, seq_len, max(1, seq_len // 8))
    image = None
    for head in range(n_heads):
        ax = axes[head // cols][head % cols]
        image = ax.imshow(shown[head], cmap=cmap, norm=norm, aspect="auto",
                          interpolation="nearest")
        ax.set_title(f"head {head}", fontsize=10)
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.tick_params(labelsize=7)
        if head // cols == rows - 1:
            ax.set_xlabel("Key position (token index)", fontsize=8)
        if head % cols == 0:
            ax.set_ylabel("Query position (token index)", fontsize=8)
    for spare in range(n_heads, rows * cols):
        axes[spare // cols][spare % cols].axis("off")

    fig.suptitle(f"{LANGUAGE}: attention weights, layer {layer} ({title_note})\n"
                 f"one example sentence, {seq_len} tokens; "
                 f"white above the diagonal is the causal mask", fontsize=11)
    bar = fig.colorbar(image, ax=axes, shrink=0.85, pad=0.02)
    bar.set_label("Attention weight (square-root colour scale)", fontsize=9)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_grid(values: List[List[float]], path: Path, title: str,
              bar_label: str) -> None:
    """Layer-by-head grid, used for both entropy and distance."""
    plt, fig, ax = new_figure((7, 4.5))
    data = np.array(values)
    image = ax.imshow(data, cmap="magma", aspect="auto")
    ax.set_xlabel("Attention head")
    ax.set_ylabel("Layer (0 = closest to the embedding)")
    ax.set_xticks(range(data.shape[1]))
    ax.set_yticks(range(data.shape[0]))
    ax.set_title(title)
    for layer in range(data.shape[0]):
        for head in range(data.shape[1]):
            ax.text(head, layer, f"{data[layer, head]:.2f}", ha="center",
                    va="center", fontsize=7,
                    color="white" if data[layer, head] < data.max() * 0.6 else "black")
    bar = fig.colorbar(image, ax=ax)
    bar.set_label(bar_label)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_entropy_vs_distance(stats: Dict, path: Path) -> None:
    """Every head as a point: how diffuse it is against how far it looks."""
    plt, fig, ax = new_figure((7, 5))
    entropy = np.array(stats["normalised_entropy"])
    distance = np.array(stats["distance"])
    n_layers = entropy.shape[0]
    colours = plt.cm.viridis(np.linspace(0, 1, n_layers))
    for layer in range(n_layers):
        ax.scatter(distance[layer], entropy[layer], color=colours[layer],
                   s=45, label=f"layer {layer}", edgecolor="black", linewidth=0.3)
    ax.set_xlabel("Mean attention distance (tokens back)")
    ax.set_ylabel("Normalised entropy (1.0 = uniform over allowed keys)")
    ax.set_title(f"{LANGUAGE}: attention heads, focus against reach\n"
                 "bottom left = local and sharp, top right = diffuse and distant")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, ncol=2, title="Layer")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def examples_markdown(examples: Sequence[Dict], layers: Sequence[int]) -> str:
    """The example sentences and their pieces, indexed to match the figures."""
    lines = [f"# {LANGUAGE} — attention example sentences", "",
             "The heatmap axes are token indices. The pieces below are numbered "
             "to match, so a bright cell can be read back to actual words.", ""]
    for n, example in enumerate(examples):
        lines += [f"## Example {n + 1}", "", f"> {example['sentence']}", "",
                  f"{len(example['pieces'])} tokens (index 0 is `<s>`):", "",
                  "| # | piece |", "|---:|---|"]
        lines += [f"| {i} | `{piece}` |"
                  for i, piece in enumerate(example["pieces"])]
        lines += ["", "Heatmaps: " + ", ".join(
            f"`figures/attention_example{n + 1}_layer{layer}.png`"
            for layer in layers), ""]
    return "\n".join(lines)


def print_table(stats: Dict) -> None:
    """Per-layer summary of the head statistics."""
    entropy = np.array(stats["normalised_entropy"])
    distance = np.array(stats["distance"])
    prev = np.array(stats["previous_token_mass"])
    bos = np.array(stats["bos_mass"])
    print(f"\n  {LANGUAGE}: attention summary, averaged over held-out text")
    print(f"  {'layer':<7}{'norm. entropy':>16}{'distance':>12}"
          f"{'prev-token':>13}{'BOS mass':>11}")
    print("  " + "-" * 59)
    for layer in range(entropy.shape[0]):
        print(f"  {layer:<7}{entropy[layer].mean():>16.3f}"
              f"{distance[layer].mean():>12.1f}{prev[layer].mean():>13.3f}"
              f"{bos[layer].mean():>11.3f}")
    print()


def main() -> None:
    """Compute the statistics, draw the figures, and write the JSON."""
    ap = argparse.ArgumentParser(description="Attention analysis.")
    add_common_args(ap)
    ap.add_argument("--split", default="validation")
    ap.add_argument("--batches", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--out", default="attention")
    args = ap.parse_args()

    cfg = eval_config(args)
    attn_cfg = cfg.get("attention", {})
    n_batches = args.batches or int(attn_cfg.get("batches", 20))

    device = pick_device(args.device)
    checkpoint = resolve_checkpoint(args.checkpoint, cfg)
    model, model_cfg, meta = load_model(checkpoint, device)
    sp = load_tokenizer()
    print(f"\n  checkpoint {checkpoint.name} at step {meta['state']['step']:,}  "
          f"device {device}")

    stream, _ = load_split(cfg, args.split)
    started = time.time()
    stats = head_statistics(model, stream, model_cfg.context_length,
                            args.batch_size, n_batches, device)
    print(f"  head statistics over {stats['batches']} batches "
          f"({time.time() - started:.1f}s)")
    print_table(stats)

    figures = report_dir(cfg, "figures")
    early, late = 0, model_cfg.n_layers - 1
    layers = attn_cfg.get("heatmap_layers") or [early, late]

    sentences = attn_cfg.get("examples") or PIPELINE["tokenizer"]["eval_examples"]
    examples = [sentence_attention(model, sp, s, device)
                for s in sentences[:int(attn_cfg.get("example_count", 2))]]
    for n, example in enumerate(examples):
        for layer in layers:
            note = ("early: near the raw embedding" if layer == early
                    else "late: near the output head" if layer == late
                    else "middle of the stack")
            plot_heatmaps(example, layer, figures /
                          f"attention_example{n + 1}_layer{layer}.png", note)
    print(f"  wrote {len(examples) * len(layers)} heatmap figures")

    plot_grid(stats["normalised_entropy"],
              figures / "attention_entropy.png",
              f"{LANGUAGE}: attention entropy per layer and head\n"
              "normalised by log(keys visible), so 1.0 is uniform attention",
              "Normalised entropy")
    plot_grid(stats["distance"], figures / "attention_distance.png",
              f"{LANGUAGE}: mean attention distance per layer and head\n"
              "mass-weighted mean of (query position - key position)",
              "Distance (tokens)")
    plot_entropy_vs_distance(stats, figures / "attention_heads.png")

    (Path(cfg["reports_dir"]) / "attention_examples.md").write_text(
        examples_markdown(examples, layers), encoding="utf-8")

    save_json(cfg, args.out, {
        "language": LANGUAGE,
        "checkpoint": str(checkpoint),
        "step": meta["state"]["step"],
        "split": args.split,
        "context_length": model_cfg.context_length,
        "n_layers": model_cfg.n_layers,
        "n_heads": model_cfg.n_heads,
        "heatmap_layers": layers,
        "statistics": {k: v for k, v in stats.items()},
        "examples": [{"sentence": e["sentence"], "pieces": e["pieces"]}
                     for e in examples],
    })


if __name__ == "__main__":
    main()
