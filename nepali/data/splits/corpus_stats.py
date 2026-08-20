"""Corpus statistics, including the manual vs public token accounting.

The 500M-token target is defined on tokens after tokenization, not bytes,
file size or words. So this script measures token counts with the trained
SentencePiece model whenever one exists:

    python3 data/splits/corpus_stats.py                  # before the tokenizer exists
    python3 data/splits/corpus_stats.py --tokenizer      # exact counts (run this for the report)

Without `--tokenizer` it reports characters/bytes/words only and marks token
counts as unavailable - it never guesses a token count from file size.

Outputs `report/corpus_stats_<lang>.{json,md}` plus two labelled figures.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from corpus_io import lang_root, load_config, read_docs  # noqa: E402

ROOT = lang_root(Path(__file__))
CFG = load_config(ROOT)
LANG = CFG["language"]
TARGET_TOKENS = 500_000_000


def load_tokenizer():
    """Load this language's SentencePiece model, or None if not trained yet.

    The name is derived the same way train_tokenizer.py builds it:
    <size>_<algo>_<lang>, e.g. 32k_unigram_guj.model. Looks in
    tokenizer/model/ first and falls back to the tokenizer/ root. Sweep
    candidates live in tokenizer/model/sweep/ and are deliberately not
    considered here. Without this the script silently reports zero tokens and
    falls back to measuring the manual share by characters.
    """
    tcfg = CFG["tokenizer"]
    size = int(tcfg["vocab_size"])
    algo = tcfg.get("algorithm", "sentencepiece_unigram").split("_")[-1]
    stem = f"{size // 1000}k" if size % 1000 == 0 else str(size)
    stem = f"{stem}_{algo}_{CFG['language']}"

    candidates = [CFG["paths"].get("tokenizer_model"), CFG["paths"]["tokenizer"]]
    model = None
    for base in candidates:
        if base is None:
            continue
        p = base / f"{stem}.model"
        if p.exists():
            model = p
            break
    if model is None:
        return None, (candidates[0] or candidates[1]) / f"{stem}.model"
    import sentencepiece as spm
    return spm.SentencePieceProcessor(model_file=str(model)), model


def scan(split_path: Path, sp) -> dict:
    """Single streaming pass: totals, per-source and per-provenance breakdowns."""
    blank = lambda: {"docs": 0, "chars": 0, "bytes": 0, "words": 0, "tokens": 0}
    totals = blank()
    by_source, by_prov = defaultdict(blank), defaultdict(blank)
    manual = blank()
    lengths = []
    # Provenance per source, recorded during this pass. It used to be collected
    # by a second, limited pass over the first 200k documents - but the split
    # is written manual-first, so that sample never reached the public source
    # and reported its provenance as unknown.
    prov_of_source = {}

    for d in read_docs(split_path):
        prov_of_source.setdefault(d.source_name, d.manual_or_public)
        rec = {"docs": 1, "chars": d.n_chars, "bytes": d.n_bytes,
               "words": len(d.text.split()),
               "tokens": len(sp.encode(d.text)) if sp else 0}
        for k, v in rec.items():
            totals[k] += v
            by_source[d.source_name][k] += v
            by_prov[d.manual_or_public][k] += v
        if d.manual:
            for k, v in rec.items():
                manual[k] += v
        if len(lengths) < 300000:
            lengths.append(d.n_chars)

    return {"totals": totals, "manual": manual,
            "by_source": {k: dict(v) for k, v in by_source.items()},
            "by_provenance": {k: dict(v) for k, v in by_prov.items()},
            "provenance_of_source": prov_of_source,
            "lengths": lengths}


def figures(train: dict, split_totals: dict) -> None:
    """Draw the two report figures (title, axis labels and legend on both)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    fig_dir = CFG["paths"]["reports"] / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    prov = train["by_provenance"]
    labels = list(prov.keys())
    unit = "tokens" if train["totals"]["tokens"] else "chars"
    sizes = [prov[k][unit] for k in labels]
    frac = train["manual"][unit] / max(train["totals"][unit], 1)
    fig, ax = plt.subplots(figsize=(7, 6))
    wedges, _, _ = ax.pie(sizes, autopct="%1.1f%%", startangle=120)
    ax.set_title(f"Manual vs public {unit} - {LANG} train (manual = {frac*100:.1f}%)")
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.legend(wedges, labels, title="provenance", loc="center left", bbox_to_anchor=(1.0, 0.5))
    fig.savefig(fig_dir / "provenance.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    names = list(split_totals.keys())
    docs = [split_totals[s]["docs"] for s in names]
    vals = [split_totals[s][unit] / 1e6 for s in names]
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - 0.2, docs, 0.4, label="documents")
    ax2 = ax.twinx()
    ax2.bar(x + 0.2, vals, 0.4, color="darkorange", label=f"{unit} (millions)")
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_title(f"Split sizes - {LANG}")
    ax.set_xlabel("Split")
    ax.set_ylabel("Number of documents")
    ax2.set_ylabel(f"{unit.capitalize()} (millions)")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper right")
    fig.savefig(fig_dir / "splits.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    """Report per-split, per-source and per-provenance corpus statistics."""
    ap = argparse.ArgumentParser(description="Corpus statistics + manual/public token accounting.")
    ap.add_argument("--tokenizer", action="store_true",
                    help="Count tokens with the trained SentencePiece model (exact)")
    args = ap.parse_args()

    sp, model_path = (load_tokenizer() if args.tokenizer else (None, None))
    if args.tokenizer and sp is None:
        print(f"! tokenizer not found at {model_path} - train it first; "
              f"reporting character counts only\n")

    stats, split_totals = {}, {}
    for split in ("train", "validation", "test"):
        path = CFG["paths"]["splits"] / f"{split}.jsonl.gz"
        if not path.exists():
            continue
        print(f"scanning {split} ...")
        stats[split] = scan(path, sp)
        split_totals[split] = stats[split]["totals"]

    if not stats:
        print("no splits found - run build_splits.py first")
        return

    train = stats["train"]
    unit = "tokens" if sp else "chars"
    total_u = train["totals"][unit]
    manual_u = train["manual"][unit]
    pct = manual_u / max(total_u, 1) * 100

    # ---- Markdown report ---- #
    md = [f"# Corpus statistics - {CFG['language_name']} (`{LANG}`)", ""]
    md.append("Token counts: exact (SentencePiece)" if sp else
              "Token counts: not available (tokenizer not trained yet; "
              "re-run with `--tokenizer`). Character counts are exact.")
    md += ["", "## Manual vs public (train split)", "",
           "| Metric | Value |", "|---|---:|",
           f"| Manual documents | {train['manual']['docs']:,} |",
           f"| Manual tokens | {train['manual']['tokens']:,} |",
           f"| Public documents | {train['totals']['docs'] - train['manual']['docs']:,} |",
           f"| Public tokens | {train['totals']['tokens'] - train['manual']['tokens']:,} |",
           f"| Total tokens | {train['totals']['tokens']:,} |",
           f"| Manual percentage (by {unit}) | {pct:.2f}% "
           f"({'PASS >=20%' if pct >= 20 else 'BELOW 20% - collect more manual data'}) |",
           f"| Progress to 500M-token target | "
           f"{train['totals']['tokens']/TARGET_TOKENS*100:.2f}% |" if sp else "", ""]

    md += ["## Per split", "", "| Split | Docs | Chars | Words | Tokens | Manual % |",
           "|---|---:|---:|---:|---:|---:|"]
    for s, st in stats.items():
        t, m = st["totals"], st["manual"]
        md.append(f"| {s} | {t['docs']:,} | {t['chars']:,} | {t['words']:,} | "
                  f"{t['tokens']:,} | {m[unit]/max(t[unit],1)*100:.1f}% |")

    md += ["", "## Per source (train split)", "",
           "| Source | Provenance | Docs | Chars | Tokens | Share |", "|---|---|---:|---:|---:|---:|"]
    prov_of = train["provenance_of_source"]
    for name, v in sorted(train["by_source"].items(), key=lambda x: -x[1]["chars"]):
        md.append(f"| `{name}` | {prov_of.get(name, '?')} | {v['docs']:,} | {v['chars']:,} | "
                  f"{v['tokens']:,} | {v[unit]/max(total_u,1)*100:.1f}% |")

    md += ["", "## Document length (train, characters)", ""]
    L = sorted(train["lengths"])
    if L:
        q = lambda p: L[min(len(L) - 1, int(p / 100 * (len(L) - 1)))]
        md += ["| mean | median | p90 | p99 | max |", "|---:|---:|---:|---:|---:|",
               f"| {sum(L)/len(L):.0f} | {q(50):,} | {q(90):,} | {q(99):,} | {L[-1]:,} |"]

    rep = CFG["paths"]["reports"]
    rep.mkdir(parents=True, exist_ok=True)
    # No markdown here: report/tables/corpus_summary.md presents these
    # figures for both languages together, which is the more useful form.
    (rep / "json").mkdir(parents=True, exist_ok=True)
    dumpable = {s: {k: v for k, v in st.items() if k != "lengths"} for s, st in stats.items()}
    (rep / "json" / "corpus_stats.json").write_text(
        json.dumps({"language": LANG, "token_counts_exact": bool(sp), "splits": dumpable},
                   indent=2, ensure_ascii=False), encoding="utf-8")
    figures(train, split_totals)

    print(f"\n==== {CFG['language_name']} train ====")
    print(f"  Manual documents:  {train['manual']['docs']:,}")
    print(f"  Manual tokens:     {train['manual']['tokens']:,}")
    print(f"  Public documents:  {train['totals']['docs'] - train['manual']['docs']:,}")
    print(f"  Public tokens:     {train['totals']['tokens'] - train['manual']['tokens']:,}")
    print(f"  Total tokens:      {train['totals']['tokens']:,}")
    print(f"  Manual percentage: {pct:.2f}%  (measured by {unit})")
    if sp:
        print(f"  500M target:       {train['totals']['tokens']/TARGET_TOKENS*100:.2f}%")
    print(f"  report -> {rep / 'json' / 'corpus_stats.json'}")


if __name__ == "__main__":
    main()
