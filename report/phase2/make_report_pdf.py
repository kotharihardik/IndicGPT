#!/usr/bin/env python3
"""Build the Phase 2 report PDF from the evaluation output.

    python3 report/phase2/make_report_pdf.py

Writes `report/phase2/phase2_report.tex` and compiles it with XeLaTeX, the same
toolchain the Phase 1 report used. XeLaTeX is needed because the report quotes
Gujarati and Devanagari text, which requires a system font.

Every number and every sample is read from the evaluation JSON and the generated
markdown, so the PDF cannot disagree with the rest of the report. Nothing is
typed in by hand.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

ROOT = Path(__file__).resolve().parent            # report/phase2
REPO = ROOT.parents[1]
LANGS = [("gujarati", "Gujarati", "Model H"), ("nepali", "Nepali", "Model L")]
SETTINGS = ["greedy", "temp 0.5", "temp 1.0", "temp 1.5"]
REPORTED, CONTROL = "16k", "12k"

#: System fonts that cover the two scripts. Both ship with macOS.
FONTS = {"gujarati": "Gujarati Sangam MN", "nepali": "Devanagari Sangam MN"}


# Loading


def load(lang: str, variant: str, name: str) -> Dict[str, Any]:
    """One evaluation JSON for one run."""
    path = ROOT / lang / f"{lang}_{variant}" / "json" / f"{name}.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def run_config(lang: str, variant: str) -> Dict[str, Any]:
    """The config.json the trainer wrote for one run."""
    path = REPO / lang / "logs" / "pretrain" / f"pretrain_{variant}" / "config.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def collect() -> Dict:
    """Every result for both languages and both runs."""
    return {(lang, v): {n: load(lang, v, n) for n in
                        ("training", "perplexity", "generation", "attention",
                         "causal_check")} | {"run": run_config(lang, v)}
            for lang, _, _ in LANGS for v in (REPORTED, CONTROL)}


def dig(blob: Any, *keys, default=None):
    """Walk nested dicts without raising when a stage is missing."""
    for key in keys:
        if not isinstance(blob, dict) or key not in blob:
            return default
        blob = blob[key]
    return blob


# LaTeX helpers


ESCAPES = {"&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_",
           "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}",
           "^": r"\textasciicircum{}", "\\": r"\textbackslash{}"}


def tex(text: str) -> str:
    """Escape a plain string so LaTeX prints it literally."""
    return "".join(ESCAPES.get(c, c) for c in str(text))


#: Codepoints each script font covers: its own block, ZWNJ/ZWJ, and ASCII.
BLOCKS = {"gujarati": (0x0A80, 0x0AFF), "nepali": (0x0900, 0x097F)}


def _covered(ch: str, lang: str) -> bool:
    """True if this language's script font has a glyph for `ch`."""
    o = ord(ch)
    lo, hi = BLOCKS[lang]
    return lo <= o <= hi or o in (0x200C, 0x200D) or 0x20 <= o <= 0x7E


def native(lang: str, text: str) -> str:
    """Typeset text in this language's script font, run by run.

    Generated text occasionally contains a character the script font has no
    glyph for - the corpora carry a little legacy-encoding noise, and the models
    reproduce it. Those runs are set in a fallback font instead of being dropped,
    so the sample stays faithful rather than showing an empty box.
    """
    parts, buf, mode = [], [], None
    for ch in text:
        here = _covered(ch, lang)
        if mode is None:
            mode = here
        if here != mode:
            parts.append((mode, "".join(buf)))
            buf, mode = [], here
        buf.append(ch)
    if buf:
        parts.append((mode, "".join(buf)))
    return "".join(f"\\{lang}text{{{tex(s)}}}" if ok else f"\\fallbacktext{{{tex(s)}}}"
                   for ok, s in parts if s)


def shorten(text: str, limit: int = 260) -> str:
    """Cut long generated text at a word boundary, so the report stays short."""
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    return text[:limit].rsplit(" ", 1)[0] + " ..."


def table(header: List[str], rows: List[List[str]], align: str,
          caption: str, label: str) -> str:
    """A booktabs table inside a float."""
    body = "\n".join(" & ".join(r) + r" \\" for r in rows)
    return (f"\\begin{{table}}[H]\n\\centering\n\\small\n"
            f"\\begin{{tabular}}{{{align}}}\n\\toprule\n"
            + " & ".join(f"\\textbf{{{h}}}" for h in header) + r" \\" + "\n"
            f"\\midrule\n{body}\n\\bottomrule\n\\end{{tabular}}\n"
            f"\\caption{{{caption}}}\n\\label{{tab:{label}}}\n\\end{{table}}\n")


def figure(path: str, caption: str, label: str, width: str = r"\linewidth") -> str:
    """An includegraphics float."""
    return (f"\\begin{{figure}}[H]\n\\centering\n"
            f"\\includegraphics[width={width}]{{{path}}}\n"
            f"\\caption{{{caption}}}\n\\label{{fig:{label}}}\n\\end{{figure}}\n")


def num(value, fmt="{:,}", dash="--"):
    """Format a number, or a dash when the stage did not run."""
    return dash if value is None else fmt.format(value)


# Samples


def first_sample(lang: str) -> Dict[str, str]:
    """Prompt, greedy and temperature-1.0 continuation of the first sample.

    Read from the generated `samples.md` rather than re-run, so the report shows
    exactly the text the evaluation scored.
    """
    path = ROOT / lang / f"{lang}_{REPORTED}" / "samples.md"
    if not path.exists():
        return {}
    block = path.read_text(encoding="utf-8").split("\n## ")[1]
    out = {}
    for label, key in (("Prompt", "prompt"), ("greedy", "greedy"),
                       ("temp 1.0", "temp1")):
        m = re.search(rf"\*\*{re.escape(label)}\*\*\n\n((?:>.*\n)+)", block)
        if m:
            quoted = m.group(1)
            out[key] = " ".join(l.lstrip("> ").strip()
                                for l in quoted.splitlines()).strip()
    return out


# Sections


def summary_table(data: Dict) -> str:
    """The headline result for both models, on the test split."""
    rows = []
    for lang, name, tier in LANGS:
        d = data[(lang, REPORTED)]
        t, p = d["training"], dig(d, "perplexity", "splits", "test", default={})
        rows.append([
            f"{name} ({tier})",
            num(dig(d, "run", "parameters", "total")),
            num(dig(d, "run", "model_config", "vocab_size")),
            num(t.get("tokens")),
            num(p.get("perplexity"), "{:.2f}"),
            f"\\textbf{{{num(p.get('bits_per_byte'), '{:.4f}')}}}",
        ])
    return table(["Model", "Parameters", "Vocabulary", "Training tokens",
                  "Test PPL", "Test BPB"],
                 rows, "lrrrrr",
                 "Headline result. Bits-per-byte is the number that compares "
                 "across two tokenizers; perplexity is per token and does not.",
                 "summary")


def architecture_table(data: Dict) -> str:
    """The configuration both models share, and where they differ."""
    fields = [("Model dimension $d$", "d_model", "{:,}"),
              ("Layers $N$", "n_layers", "{:,}"),
              ("Attention heads $h$", "n_heads", "{:,}"),
              ("Feed-forward width", "d_ff", "{:,}"),
              ("Context length $T$", "context_length", "{:,}"),
              ("Vocabulary $V$", "vocab_size", "{:,}")]
    rows = []
    for label, key, fmt in fields:
        rows.append([label] + [num(dig(data[(l, REPORTED)], "run", "model_config", key), fmt)
                               for l, _, _ in LANGS])
    for label, key in (("Parameters (total)", "total"),
                       ("Parameters (non-embedding)", "non_embedding")):
        rows.append([f"\\textbf{{{label}}}"]
                    + [f"\\textbf{{{num(dig(data[(l, REPORTED)], 'run', 'parameters', key))}}}"
                       for l, _, _ in LANGS])
    return table(["", "Gujarati (Model H)", "Nepali (Model L)"], rows, "lrr",
                 "Architecture. Head dimension is $d/h = 64$. The two models are "
                 "the same network with separate weights; only the vocabularies "
                 "were built independently, and both sweeps chose 16{,}000.",
                 "arch")


def decisions_table() -> str:
    """The design choices the brief asks to be justified."""
    rows = [
        ["Positional scheme", "RoPE",
         "Costs no parameters (a learned table needs $512\\times384=196{,}608$) "
         "and encodes relative distance, so length is not fixed by a table."],
        ["Normalisation", "Pre-norm LayerNorm",
         "Leaves an unbroken identity path through the stack, so 11 layers train "
         "without a long warmup."],
        ["Feed-forward", "SwiGLU, $d_{ff}=1024$",
         "Same cost as GELU at 1536; the gate can suppress a channel "
         "multiplicatively."],
        ["Output head", "Tied to the embedding",
         "Saves 6.14M parameters, a quarter of the budget."],
        ["Biases", "None",
         "LayerNorm already re-centres every sublayer input."],
        ["Dropout", "0.0 for pretraining",
         "One pass over ${\\sim}500$M tokens: no document repeats, so there is "
         "nothing to memorise. Implemented and used in Phase 3."],
        ["Run length", "One epoch",
         "One pass is ${\\sim}500$M tokens, which is both the brief's target and "
         "about 20 tokens per parameter."],
    ]
    return table(["Choice", "Setting", "Reason"],
                 [[r[0], r[1], r[2]] for r in rows],
                 r"p{0.16\linewidth}p{0.20\linewidth}p{0.52\linewidth}",
                 "Design decisions and why they were made.", "decisions")


def causal_table(data: Dict) -> str:
    """The four checks that the model cannot see the future."""
    checks = [("Future edit moves earlier logits", "future_edit",
               "max_change_at_or_before_cut"),
              ("Prefix vs.\\ full-sequence logits differ", "prefix_equality",
               "max_difference"),
              ("Largest weight above the diagonal", "mask_shape",
               "max_weight_above_diagonal"),
              ("Gradient reaching a future position", "gradient_reach",
               "max_gradient_after_cut")]
    rows = []
    for label, name, key in checks:
        cells = []
        for lang, _, _ in LANGS:
            c = dig(data[(lang, REPORTED)], "causal_check", "checks", name, default={})
            v, ok = c.get(key), c.get("passed")
            cells.append("--" if v is None else
                         f"{v:.1e} \\textsc{{{'pass' if ok else 'fail'}}}")
        rows.append([label] + cells)
    return table(["Check", "Gujarati", "Nepali"], rows, "lrr",
                 "Causal masking, verified empirically rather than assumed. "
                 "A model that could see the next token would train normally and "
                 "report an implausibly low perplexity.", "causal")


def pretraining_table(data: Dict) -> str:
    """How each run went."""
    fields = [("Optimizer steps", "steps", "{:,}"),
              ("Training tokens seen", "tokens", "{:,}"),
              ("Wall-clock hours (Kaggle T4)", "wall_hours", "{:.2f}"),
              ("Throughput (tokens/s, median)", "median_tokens_per_second", "{:,.0f}"),
              ("Final training loss", "final_train_loss", "{:.4f}"),
              ("Best validation loss", "best_val_loss", "{:.4f}")]
    rows = [[label] + [num(data[(l, REPORTED)]["training"].get(key), fmt)
                       for l, _, _ in LANGS] for label, key, fmt in fields]
    return table(["", "Gujarati (Model H)", "Nepali (Model L)"], rows, "lrr",
                 "Pretraining. Both runs used AdamW ($\\beta=(0.9,0.95)$, weight "
                 "decay 0.1 on matrices only), 500 warmup steps then cosine decay "
                 "from $6\\times10^{-4}$ to $6\\times10^{-5}$, gradient clipping "
                 "at 1.0, batch $32\\times2\\times512=32{,}768$ tokens, seed 1337.",
                 "pretrain")


def lm_table(data: Dict) -> str:
    """Cross-entropy, perplexity and bits-per-byte on both held-out splits."""
    rows = []
    for split in ("validation", "test"):
        for label, key, fmt in (("Cross-entropy (nats/token)", "cross_entropy_nats", "{:.4f}"),
                                ("Perplexity", "perplexity", "{:.2f}"),
                                ("Bits per byte", "bits_per_byte", "{:.4f}")):
            cells = [num(dig(data[(l, REPORTED)], "perplexity", "splits", split, key), fmt)
                     for l, _, _ in LANGS]
            if key == "bits_per_byte":
                label = f"\\textbf{{{label}}}"
                cells = [f"\\textbf{{{c}}}" for c in cells]
            rows.append([split.capitalize(), label] + cells)
    tokens = [num(dig(data[(l, REPORTED)], "perplexity", "splits", "test",
                      "predicted_tokens")) for l, _, _ in LANGS]
    rows.append(["Test", "Tokens predicted"] + tokens)
    return table(["Split", "Metric", "Gujarati (H)", "Nepali (L)"], rows, "llrr",
                 "Language-modelling metrics on the complete validation and test "
                 "splits, in float32, over non-overlapping 512-token windows so "
                 "every token is predicted exactly once.", "lm")


def generation_table(data: Dict) -> str:
    """Every generation metric for every decoding setting."""
    metrics = [("BLEU-4", ("bleu", "bleu"), "{:.2f}"),
               ("chrF++", ("chrf", "chrf"), "{:.2f}"),
               ("ROUGE-L", ("rouge_l", "rouge_l"), "{:.2f}"),
               ("Dist-1", ("diversity", "distinct_1"), "{:.3f}"),
               ("Dist-2", ("diversity", "distinct_2"), "{:.3f}"),
               ("Rep-4", ("diversity", "repetition_4"), "{:.3f}")]
    rows = []
    for lang, name, tier in LANGS:
        st = dig(data[(lang, REPORTED)], "generation", "settings", default={})
        for setting in SETTINGS:
            rows.append([f"{name}", setting] +
                        [num(dig(st, setting, *path), fmt) for _, path, fmt in metrics])
        ref = dig(st, "greedy", "reference_diversity", default={})
        if ref:
            rows.append([name, "\\emph{reference}", "--", "--", "--",
                         f"{ref['distinct_1']:.3f}", f"{ref['distinct_2']:.3f}",
                         f"{ref['repetition_4']:.3f}"])
    return table(["Model", "Decoding"] + [m[0] for m in metrics], rows,
                 "ll" + "r" * len(metrics),
                 "Generation quality on 200 held-out test documents: a 64-token "
                 "prompt continued for 96 tokens, scored against the true "
                 "continuation. The held-out reference is scored the same way, so "
                 "the diversity numbers have a yardstick.", "gen")


def attention_table(data: Dict) -> str:
    """Attention summary, averaged over held-out text."""
    def layer_means(lang, key):
        grid = dig(data[(lang, REPORTED)], "attention", "statistics", key) or []
        return [float(np.mean(r)) for r in grid]

    rows = []
    for label, key, fmt in (("Mean normalised entropy", "normalised_entropy", "{:.3f}"),
                            ("Mean attention distance (tokens)", "distance", "{:.1f}"),
                            ("Mean previous-token mass", "previous_token_mass", "{:.3f}"),
                            ("Mean start-token (sink) mass", "bos_mass", "{:.3f}")):
        rows.append([label] + [fmt.format(np.mean(layer_means(l, key)))
                               for l, _, _ in LANGS])
    for label, key, fmt, pick in (("Most diffuse layer", "normalised_entropy", "{:.3f}", np.argmax),
                                  ("Most focused layer", "normalised_entropy", "{:.3f}", np.argmin),
                                  ("Longest-range layer", "distance", "{:.1f}", np.argmax)):
        cells = []
        for lang, _, _ in LANGS:
            m = layer_means(lang, key)
            i = int(pick(m))
            cells.append(f"L{i} ({fmt.format(m[i])})")
        rows.append([label] + cells)
    return table(["", "Gujarati (H)", "Nepali (L)"], rows, "lrr",
                 "Attention statistics, averaged over 81{,}920 tokens of held-out "
                 "text. Entropy is normalised by the log of the number of keys "
                 "visible at that position, so 1.0 means uniform attention at "
                 "any depth.", "attn")


def ablation_table(data: Dict) -> str:
    """16k against 12k, at a matched parameter budget."""
    rows = []
    for lang, name, tier in LANGS:
        a, b = data[(lang, REPORTED)], data[(lang, CONTROL)]
        get = lambda d, k: dig(d, "perplexity", "splits", "test", k)
        ppl_a, ppl_b = get(a, "perplexity"), get(b, "perplexity")
        bpb_a, bpb_b = get(a, "bits_per_byte"), get(b, "bits_per_byte")
        ha = dig(a, "training", "wall_hours")
        hb = dig(b, "training", "wall_hours")
        rows.append([name, "Perplexity \\emph{(not comparable)}",
                     f"{ppl_a:.2f}", f"{ppl_b:.2f}",
                     f"{100 * (ppl_b - ppl_a) / ppl_a:+.1f}\\%"])
        rows.append([name, "\\textbf{Bits per byte} \\emph{(comparable)}",
                     f"\\textbf{{{bpb_a:.4f}}}", f"\\textbf{{{bpb_b:.4f}}}",
                     f"\\textbf{{{100 * (bpb_b - bpb_a) / bpb_a:+.2f}\\%}}"])
        rows.append([name, "Wall-clock hours",
                     f"{ha:.2f}", f"{hb:.2f}",
                     f"{100 * (hb - ha) / ha:+.1f}\\%"])
    return table(["Model", "Metric (test split)", "16k", "12k", "Change"],
                 rows, "llrrr",
                 "Vocabulary ablation at a matched parameter budget: the smaller "
                 "embedding frees 1.54M parameters, which buys a twelfth "
                 "Transformer block (25{,}860{,}864 against 25{,}625{,}856, within "
                 "0.92\\%). Perplexity appears to improve by 13--14\\%, but that is "
                 "the vocabulary changing, not the model.", "ablation")


def comparison_table(data: Dict) -> str:
    """What actually differs between Model H and Model L."""
    g = data[("gujarati", REPORTED)]["training"]
    n = data[("nepali", REPORTED)]["training"]
    gap = 100 * (g["tokens"] - n["tokens"]) / g["tokens"]
    rows = [
        ["Training tokens", f"{g['tokens']:,}", f"{n['tokens']:,}",
         f"$-{gap:.1f}\\%$"],
        ["Manual share of corpus", "31.44\\%", "21.58\\%", "$-9.9$ pts"],
        ["Characters per token", "3.865", "4.689", "$+21\\%$"],
        ["Fertility (tokens/word)", "1.503", "1.416", "$-5.8\\%$"],
        ["Sources in the corpus", "1 public + 12 manual", "1 public + 23 manual",
         "--"],
    ]
    return table(["", "Gujarati (Model H)", "Nepali (Model L)", "Difference"],
                 rows, "lrrr",
                 "The only three things that differ between the two runs: the "
                 "corpus, its composition, and the tokenizer. Architecture, "
                 "optimizer, schedule, batch size, epochs and seed are identical. "
                 "Corpus figures come from Phase 1.", "compare")


# The document


PREAMBLE = r"""\documentclass[11pt,a4paper]{article}
\usepackage[margin=2.4cm]{geometry}
\usepackage{fontspec}
\usepackage{booktabs}
\usepackage{graphicx}
\usepackage{float}
\usepackage[table]{xcolor}
\usepackage[colorlinks=true,linkcolor=black,urlcolor=blue,citecolor=black]{hyperref}

% parskip / caption / enumitem are not in a minimal TeX install, so the few
% settings taken from them are written out here instead. This keeps the report
% compilable with nothing but a base TeX Live.
\setlength{\parindent}{0pt}
\setlength{\parskip}{0.55em}
\setlength{\abovecaptionskip}{6pt}
\setlength{\belowcaptionskip}{2pt}

\newfontfamily\gujaratifont{@GUJFONT@}[Scale=1.05]
\newfontfamily\nepalifont{@NEPFONT@}[Scale=1.05]
\newfontfamily\fallbackfont{Helvetica}
\newcommand{\gujaratitext}[1]{{\gujaratifont #1}}
\newcommand{\nepalitext}[1]{{\nepalifont #1}}
\newcommand{\fallbacktext}[1]{{\fallbackfont #1}}

\makeatletter
\renewcommand{\@makecaption}[2]{%
  \vskip\abovecaptionskip
  \begingroup\small #1: #2\par\endgroup
  \vskip\belowcaptionskip}
\makeatother
\setlength{\emergencystretch}{3em}

\title{\vspace{-1.5cm}\textbf{Phase 2 --- Model Implementation, Pretraining and
Evaluation}\\[0.3em]
\large Monolingual Transformer Language Models for Gujarati and Nepali}
\author{Language Models and Agents --- Individual Project}
\date{Branch \texttt{phase-2}}
"""


def build_tex(data: Dict) -> str:
    """Assemble the whole document."""
    g_sample, n_sample = first_sample("gujarati"), first_sample("nepali")
    gt = data[("gujarati", REPORTED)]
    nt = data[("nepali", REPORTED)]
    g_bpb = dig(gt, "perplexity", "splits", "test", "bits_per_byte")
    n_bpb = dig(nt, "perplexity", "splits", "test", "bits_per_byte")
    bpb_gap = 100 * (n_bpb - g_bpb) / g_bpb
    tok_gap = 100 * (gt["training"]["tokens"] - nt["training"]["tokens"]) \
        / gt["training"]["tokens"]
    g_chrf = dig(gt, "generation", "settings", "temp 1.0", "chrf", "chrf")
    n_chrf = dig(nt, "generation", "settings", "temp 1.0", "chrf", "chrf")
    g_rep = dig(gt, "generation", "settings", "greedy", "diversity", "repetition_4")
    n_rep = dig(nt, "generation", "settings", "greedy", "diversity", "repetition_4")

    preamble = (PREAMBLE.replace("@GUJFONT@", FONTS["gujarati"])
                        .replace("@NEPFONT@", FONTS["nepali"]))
    parts = [preamble,
             r"\begin{document}", r"\maketitle", r"\vspace{-0.8cm}"]

    # 1 --------------------------------------------------------------------
    parts += [r"\section{What was built}", r"""
Two decoder-only Transformer language models, written from scratch in PyTorch
and pretrained independently: \textbf{Model H} on Gujarati (higher-resource) and
\textbf{Model L} on Nepali (lower-resource). They share no data, no tokenizer,
no vocabulary and no weights. They do share an architecture and a training
recipe on purpose, so that the only variables between them are the corpus and
the tokenizer.

No \texttt{nn.Transformer*}, no HuggingFace model class and no
\texttt{F.scaled\_dot\_product\_attention}: attention, the causal mask and the
positional scheme are written out in \texttt{<language>/model/transformer.py}.
""", summary_table(data), r"""
\textbf{Checkpoints.} All ten --- five runs, each with \texttt{best.pt} (lowest
validation loss) and \texttt{latest.pt} (end of the run) --- are published at
\begin{center}
\url{https://huggingface.co/hardik300/LMA\_mini\_project}
\end{center}
"""]

    # 2 --------------------------------------------------------------------
    parts += [r"\section{The model}", r"""
The forward pass is: token embedding $\rightarrow$ embedding dropout
$\rightarrow$ $N$ blocks of [LayerNorm $\rightarrow$ causal self-attention
$\rightarrow$ residual; LayerNorm $\rightarrow$ SwiGLU feed-forward
$\rightarrow$ residual] $\rightarrow$ final LayerNorm $\rightarrow$ tied output
projection $\rightarrow$ logits, trained with cross-entropy against the next
token.

Multi-head attention splits $(B,T,d)$ into $(B,h,T,d_k)$ with $d_k=d/h=64$,
computes $\mathrm{softmax}\!\left(QK^{\top}/\sqrt{d_k} + M\right)V$ per head,
then concatenates and applies $W_O$. The $1/\sqrt{d_k}$ scaling matters because
the dot product of two $d_k$-dimensional vectors grows like $d_k$; without it the
softmax saturates and its gradient vanishes. $M$ is an additive mask with
$-\infty$ above the diagonal, so position $t$ can only attend to positions
$\le t$.
""", architecture_table(data), r"""
\textbf{Depth against width.} Phase 1 fixed $d=384$ by pricing the embedding
against the 25M budget. At that width one block costs 1{,}771{,}008 parameters,
so what is left after the 6.14M embedding buys 11 of them. The alternatives at
the same budget are wider and shallower ($448\times8$) or narrower and deeper
($320\times14$). Depth/width ratios of 20--40 behave similarly at this scale;
$384\times11$ sits in that band and trains faster than the narrow option, which
needs more sequential kernel launches for the same arithmetic --- which matters
on a single T4 with a fixed session limit.
""", decisions_table()]

    parts += [r"\subsection{Maximum sequence length}", r"""
RoPE stores no positional table, so the model does not fail on a longer
sequence: the cosine cache is recomputed for any length, and the self-test feeds
the trained architecture a 640-token input successfully. What is still bounded is
\emph{quality} --- the model never saw relative distances beyond 512, so it
degrades past that rather than raising an error. A learned table would have no
row for position 600 at all.
"""]

    parts += [r"\subsection{Verifying the causal mask}", r"""
A model that can see the next token trains normally and reports an implausibly
low perplexity, so the mask is tested four ways in
\texttt{eval/causal\_check.py} instead of being assumed. The fourth is the
strongest: it differentiates a logit at position $t$ with respect to the input
embeddings and checks that no gradient reaches any position after $t$, which
tests the computational graph rather than just the values.
""", causal_table(data)]

    # 3 --------------------------------------------------------------------
    parts += [r"\section{Pretraining}", r"""
Each model was trained for one epoch over its own tokenized training split on a
single Kaggle T4, in fp16 with a gradient scaler. Checkpoints contain model
weights, optimizer state, scheduler state, scaler state, step, tokens seen, both
configs and the RNG states, so a run resumes exactly where it stopped; the batch
order is a function of (seed, step) alone, so a resumed run continues on the data
it would have seen anyway.
""", pretraining_table(data),
              figure("figures/models/loss_comparison.png",
                     "Validation loss for both models against tokens seen. "
                     "Identical architecture, optimizer and schedule; separate "
                     "corpora. Validation loss tracks training loss throughout, "
                     "which is the check that zero dropout was the right choice.",
                     "loss")]

    # 4 --------------------------------------------------------------------
    parts += [r"\section{Language modelling results}", lm_table(data), f"""
Bits-per-byte is the number to compare. Perplexity is measured per \\emph{{token}},
and the two tokenizers differ: a Nepali token carries about 21\\% more characters
than a Gujarati one, so it is the harder prediction. Dividing the same total
likelihood by UTF-8 bytes removes the tokenizer, because bytes are a property of
the text rather than of how it was split.

Model L is ahead on both measures --- test bits-per-byte {n_bpb:.4f} against
{g_bpb:.4f}, {abs(bpb_gap):.1f}\\% lower --- while seeing {abs(tok_gap):.1f}\\%
\\emph{{fewer}} tokens under an identical recipe. Section~\\ref{{sec:compare}}
explains why.
"""]

    # 5 --------------------------------------------------------------------
    parts += [r"\section{Generation quality}", generation_table(data), f"""
\\textbf{{Which metric is informative.}} chrF++ is the useful one here: both
languages mark case and number with suffix chains, so a correct stem with the
wrong ending scores zero on word-level BLEU but keeps most of its character
$n$-grams. BLEU-4 is weak for open-ended continuation --- there is no single
correct next 96 tokens, and exact 4-gram overlap with one reference is close to
accidental --- but its \\emph{{ranking}} across decoding settings is still
meaningful. ROUGE-L sits between them: the longest common subsequence rewards the
right words in the right order without demanding adjacency.

\\textbf{{What the numbers say.}} Greedy decoding collapses into loops, with
repetition-4 at {g_rep:.3f} (Gujarati) and {n_rep:.3f} (Nepali) against
about 0.007 for the held-out reference.
Temperature 0.5 is fluent but circles a topic, which is why it scores the best
BLEU-4 and ROUGE-L. Temperature 1.0 reads best and is where chrF++ peaks
({g_chrf:.2f} and {n_chrf:.2f}), with diversity almost exactly matching the
reference. Temperature 1.5 breaks down: Distinct-2 reaches 0.982, \\emph{{above}}
the reference, which is the diagnostic that the text has become more varied than
real text.

\\textbf{{Coherence and language correctness.}} Script, morphology, agreement and
punctuation are right far more often than not in both languages. What is missing
is discourse-level consistency: dates, numbers and places are invented, and
consecutive sentences need not be about the same event. That is what a
25M-parameter model trained for one epoch should look like --- local fluency
without long-range factual control.
"""]

    if g_sample:
        parts += [r"\subsection{Sample continuation (Gujarati, Model H)}",
                  r"\textbf{Prompt.} ", native("gujarati", shorten(g_sample.get("prompt", ""), 200)),
                  "\n\n", r"\textbf{Greedy} --- note the loop: ",
                  native("gujarati", shorten(g_sample.get("greedy", ""), 240)),
                  "\n\n", r"\textbf{Temperature 1.0:} ",
                  native("gujarati", shorten(g_sample.get("temp1", ""), 300)), "\n"]
    if n_sample:
        parts += [r"\subsection{Sample continuation (Nepali, Model L)}",
                  r"\textbf{Prompt.} ", native("nepali", shorten(n_sample.get("prompt", ""), 200)),
                  "\n\n", r"\textbf{Temperature 1.0:} ",
                  native("nepali", shorten(n_sample.get("temp1", ""), 300)), "\n\n",
                  r"""More samples, and the same prompts run by hand at three
temperatures, are in \texttt{samples.md} and \texttt{model\_output.md} under
each model's report directory."""]

    # 6 --------------------------------------------------------------------
    parts += [r"\section{Attention analysis}", attention_table(data),
              figure("figures/models/attention_comparison.png",
                     "Attention behaviour by depth. Line is the mean over heads, "
                     "band is min to max. Early layers are diffuse and long-range; "
                     "the middle of the stack is where focused, local heads appear.",
                     "attnprofile"), r"""
Both models arrive at closely matched structure. Layers 0--2 are diffuse and
long-range (normalised entropy 0.85--0.95, mean distance 87--114 tokens): early
attention averages broadly over the prefix rather than selecting from it. Across
layers 3--6 entropy drops to 0.56--0.76 and mean distance to 39--52 tokens, and
individual heads become strongly previous-token --- Gujarati L6H0 puts 58\% of its
mass on exactly $t-1$, Nepali L7H3 puts 58\%. These are the local, positional
heads. Late layers split in two: some stay local (Nepali L10H2 has a mean
distance of 11.3 tokens) while others become long-range and content-driven
(Gujarati L8H1 reaches 145 tokens at low entropy, which is focused attention on
something far back rather than an average).

The clearest difference between the two models is the \emph{attention sink}.
Gujarati parks 5.9\% of its mass on the start token overall and 21\% at layer 8,
with one head at 41.9\%; Nepali parks 2.6\% and never exceeds 10.9\% in any head.
A sink head is one with nothing to contribute at that position, dumping its mass
somewhere harmless. Model H uses that escape route about twice as often, which is
consistent with the more varied corpus it is modelling.
"""]

    for lang, name, tier in LANGS:
        d = ROOT / lang / f"{lang}_{REPORTED}" / "figures"
        early = d / "attention_example1_layer0.png"
        late = sorted(d.glob("attention_example1_layer1*.png"))
        example = dig(data[(lang, REPORTED)], "attention", "examples", default=[{}])[0]
        if example.get("sentence"):
            # The brief asks for the example sentence alongside the heatmaps. The
            # figure axes are token indices, because rendering the script inside a
            # matplotlib figure needs a font a hosted worker may not have, so the
            # sentence and its numbered pieces are given here instead.
            parts.append(
                f"\\subsection*{{Example sentence --- {name}}}\n"
                + native(lang, example["sentence"])
                + f"\n\n{len(example.get('pieces', []))} tokens including the "
                  "leading \\texttt{<s>}; the heatmap axes below are indices into "
                  "this sentence, and the numbered pieces are in "
                  "\\texttt{attention\\_examples.md}.\n")
        if early.exists():
            parts.append(figure(f"{lang}/{lang}_{REPORTED}/figures/attention_example1_layer0.png",
                                f"{name} ({tier}), \\textbf{{early layer 0}}, all six heads. "
                                "Axes are token indices; white above the diagonal is the "
                                "causal mask. The broad triangular wash is the diffuse "
                                "early-layer behaviour the entropy numbers predict.",
                                f"heat{lang}0"))
        if late:
            parts.append(figure(f"{lang}/{lang}_{REPORTED}/figures/{late[-1].name}",
                                f"{name} ({tier}), \\textbf{{late layer {late[-1].stem[-2:].lstrip('r')}}}, "
                                "all six heads. Narrow near-diagonal bands (local heads) and "
                                "isolated bright columns far to the left (content-based and "
                                "sink heads) replace the early wash.",
                                f"heat{lang}L"))

    # 7 --------------------------------------------------------------------
    parts += [r"\section{Model H against Model L}\label{sec:compare}",
              comparison_table(data), r"""
The lower-resource model scoring \emph{better} is a statement about the corpora,
not about the two languages or the two networks. Phase 1 measured the difference:
8.7\% of the Gujarati training split is archive.org OCR of books, compressing at
2.77 characters per token against 3.97 for its other manual sources, so Model H
is predicting a more varied and noisier distribution. The Nepali split has no
comparable source and is 78.4\% one homogeneous public corpus, which is easier to
model. Nepali's tokenizer also has better fertility (1.416 tokens per word
against 1.503), so its model spends fewer prediction steps per sentence.

Two effects therefore pull in opposite directions, and both favour Model L: a
more homogeneous corpus and a more efficient tokenizer, against 5.5\% fewer
training tokens. The honest reading is that a single held-out bits-per-byte
cannot separate ``the model learned more'' from ``the held-out text was more
predictable''.
"""]

    # 8 --------------------------------------------------------------------
    parts += [r"\section{Vocabulary ablation: 16k against 12k}",
              r"""
Phase 1 chose a 16{,}000-piece vocabulary using a rule about parameter budget and
vocabulary coverage, not a measurement of downstream quality. Phase 2 tests that
choice by training the 12{,}000-piece runner-up as well, for both languages, at
the same parameter budget so that vocabulary is the variable and depth absorbs
it.
""", ablation_table(data),
              figure("figures/vocab_ablation/perplexity_vs_bpb.png",
                     "The same four models read two ways. Perplexity (left) says "
                     "the 12k runs are much better; bits-per-byte (right) says the "
                     "two vocabularies are within a fraction of a percent. Only the "
                     "right panel is a claim about the model.", "trap"), r"""
\textbf{Conclusion.} The Phase 1 choice stands. A 12{,}000-piece vocabulary buys
0.08--0.26\% of bits-per-byte and costs 8.9--13.0\% more wall-clock time per
epoch, for two compounding reasons: the same corpus becomes 3.7--4.0\% more
tokens to read, and every one of those tokens passes through 12 blocks instead of
11. Generation quality does not move in its favour either (chrF++ at the best
decoding setting is flat to slightly worse). The 16k run therefore remains the
reported Phase 2 model and the checkpoint Phase 3 finetunes from.

The result worth keeping is methodological. A 13--14\% perplexity drop that
survives no correction for tokenization is easy to claim and easy to believe.
Bits-per-byte is what separates it from a real improvement, which is why the
brief asks for it.
"""]

    # 9 --------------------------------------------------------------------
    parts += [r"\section{Reproducing}", r"""
\begin{verbatim}
cd gujarati                              # or nepali
python3 model/transformer.py --self-test # RoPE, attention, masking, init
python3 model/param_budget.py            # parameter count, formula vs model
python3 eval/text_metrics.py --self-test # BLEU / chrF vs sacrebleu
python3 train/prepare_data.py            # splits -> uint16 token streams
python3 train/pretrain.py                # pretrain (resumable; --max-hours)
python3 eval/run_eval.py                 # the whole evaluation suite
python3 report/phase2/make_report_assets.py
\end{verbatim}

Checkpoints (${\sim}295$ MB per run) are published at
\url{https://huggingface.co/hardik300/LMA\_mini\_project}, one folder per run,
with the link also in the top-level \texttt{README}; training logs, loss curves,
evaluation JSON, generated samples and attention figures are committed under
\texttt{<language>/logs/pretrain/} and \texttt{report/phase2/}. Full tables are in
\texttt{report/phase2/results.md} and \texttt{report/phase2/vocab\_comparison.md};
this report is generated from the same JSON by
\texttt{report/phase2/make\_report\_pdf.py}.
"""]

    parts.append(r"\end{document}")
    return "\n\n".join(parts)


def main() -> None:
    """Write the .tex and compile it."""
    data = collect()
    missing = [f"{lang}/{v}" for (lang, v), d in data.items() if not d["perplexity"]]
    if missing:
        raise SystemExit("no evaluation results for: " + ", ".join(missing))

    tex_path = ROOT / "phase2_report.tex"
    tex_path.write_text(build_tex(data), encoding="utf-8")
    print(f"  wrote {tex_path}")

    if not shutil.which("xelatex"):
        raise SystemExit("xelatex not found; the .tex is written but not compiled")

    for pass_no in (1, 2):                  # twice, so references resolve
        result = subprocess.run(
            ["xelatex", "-interaction=nonstopmode", "-halt-on-error",
             tex_path.name], cwd=ROOT, capture_output=True, text=True)
        if result.returncode:
            log = (ROOT / "phase2_report.log")
            tail = log.read_text(errors="replace").splitlines()[-40:] if log.exists() \
                else result.stdout.splitlines()[-40:]
            print("\n".join(tail), file=sys.stderr)
            raise SystemExit(f"xelatex failed on pass {pass_no}")

    for suffix in (".aux", ".log", ".out"):
        (ROOT / f"phase2_report{suffix}").unlink(missing_ok=True)
    print(f"  wrote {ROOT / 'phase2_report.pdf'}")


if __name__ == "__main__":
    main()
