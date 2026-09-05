# Monolingual Transformer LMs - Gujarati and Nepali

Individual project for Language Models and Agents, Monsoon 2026.

Two completely independent decoder-only Transformer language models, built from
scratch:

- **Model H - Gujarati**, the higher-resource language
- **Model L - Nepali**, from the allowed lower-resource list

The two models share no data, no tokenizer, no vocabulary and no weights. Each
language lives in its own top-level directory and can be run on its own.

> **Corpus download** (raw + preprocessed, both languages):
> <https://iiithydstudents-my.sharepoint.com/:f:/g/personal/hardik_k_students_iiit_ac_in/IgDZ46BkCoOhT74gAOPIOBrCAVnN7uIm5sKRY9rQiqDukj0?e=0hlbPM>
>
> Plain readable `.jsonl` - no archives to unpack. Tokenizer models and
> vocabularies are committed to this repository directly.

> **Pretrained checkpoints** (all five runs, ten files):
> <https://huggingface.co/hardik300/LMA_mini_project>
>
> A public Hugging Face model repository - it opens without a sign-in or an
> access request, which is what the brief asks a checkpoint link to do.

## Status

| Phase | Contents | Branch | State |
|---|---|---|---|
| 1 | Corpus collection and tokenizer construction | `phase-1` | complete |
| 2 | Transformer implementation, pretraining, evaluation | `phase-2` | complete |
| 3 | Reasoning finetuning, attention analysis, final report | `phase-3` | not started |

## Phase 1 results

| | Gujarati (Model H) | Nepali (Model L) |
|---|---:|---:|
| **Total corpus tokens** | **573,667,143** | **541,960,704** |
| against the ~500M target | **114.7%** | **108.4%** |
| manual tokens | 180,350,586 | 116,938,372 |
| **manual share** | **31.44%** | **21.58%** |
| train / validation / test tokens | 517,108,008 / 28,416,795 / 28,142,340 | 488,046,826 / 26,814,777 / 27,099,101 |
| train / validation / test documents | 747,765 / 41,171 / 41,474 | 1,116,564 / 61,604 / 61,484 |
| split ratio | 90 / 5 / 5 | 90 / 5 / 5 |
| manual sources | 13 | 23 |
| tokenizer | SentencePiece Unigram | SentencePiece Unigram |
| vocabulary | 16,000 | 16,000 |
| characters per token | 3.865 | 4.689 |
| fertility (tokens/word) | 1.503 | 1.416 |
| byte-fallback rate | 0.449% | 0.356% |

**Both corpora exceed the 500M token target and both clear the >=20% manual
requirement.**

Read strictly as the *training split alone*, Gujarati reaches 517.1M and Nepali
488.0M - 11.95M short. The Nepali corpus itself clears the target; the
difference is that 10% is held out for validation and test. That ratio was
chosen so Phase 2 gets a validation set spanning all 23 sources (61,604
documents rather than ~3,900), and Section 1.2 of the brief allows the lower-resource
language to report the maximum feasible amount with a justification.

Per-language detail: [Gujarati](report/phase1/gujarati/README.md) | [Nepali](report/phase1/nepali/README.md). Cross-language tables and figures: [`report/phase1/`](report/phase1/).

## Phase 2 results

Two independent decoder-only Transformers, written from scratch in PyTorch and
pretrained on the corpora above. Same architecture and same training recipe for
both, so the only variables between them are the corpus and the tokenizer.

| | Gujarati (Model H) | Nepali (Model L) |
|---|---:|---:|
| **Trainable parameters** | **25,625,856** | **25,625,856** |
| against the ~25M target | +2.5% | +2.5% |
| d_model / layers / heads | 384 / 11 / 6 | 384 / 11 / 6 |
| context length / positional | 512 / RoPE | 512 / RoPE |
| vocabulary (Phase 1, separate) | 16,000 | 16,000 |
| pretraining tokens (one epoch) | 518,586,368 | 490,274,816 |
| optimizer steps | 15,826 | 14,962 |
| wall-clock on a Kaggle T4 | 4.57 h | 4.51 h |
| throughput | 31,738 tok/s | 30,497 tok/s |
| **test cross-entropy** | **3.9439** | **3.6618** |
| **test perplexity** | **51.62** | **38.93** |
| **test bits per byte** | **0.5764** | **0.4247** |
| causal-mask checks | 4 / 4 pass | 4 / 4 pass |

Nothing is shared between the two: separate corpora, separate tokenizers,
separate vocabularies, separate weights, separate runs. Both were trained in one
pass over their own training split, on one Kaggle T4 each, at the same time.

**Model L scores better than Model H on both perplexity and bits-per-byte**,
despite Model H seeing 5.5% more tokens under an identical recipe. That is a
property of the corpora rather than the models: 8.7% of the Gujarati training
split is archive.org OCR of books, compressing at 2.77 characters per token
against 3.97 for its other manual sources, while the Nepali split has no
comparable source and is 78.4% one homogeneous public corpus. Model H is predicting a more varied and
noisier distribution. [`results.md`](report/phase2/results.md) Section 7
develops this.

**The Phase 2 report is [`report/phase2/phase2_report.pdf`](report/phase2/phase2_report.pdf)**
(10 pages: architecture and decisions, results, samples, attention, comparison).
It is generated from the evaluation JSON by `make_report_pdf.py`.

**Measured results** - loss curves, perplexity, bits-per-byte, BLEU / chrF++ /
ROUGE-L, diversity, attention heatmaps and the causal-masking checks - are in
[`results.md`](report/phase2/results.md). The design decisions
and the written analysis are in
[`report/phase2/README.md`](report/phase2/README.md), and the per-model figure
pages are [Gujarati](report/phase2/gujarati/gujarati_16k/README.md) and
[Nepali](report/phase2/nepali/nepali_16k/README.md).

### Two vocabularies per language

Phase 1's sweep left 12,000 as the runner-up to the 16,000-piece vocabulary, so
each language is trained twice - once with each - and the two are compared. Both
runs sit at the same ~25.6M parameter budget: the 12k tokenizer frees 1.54M
parameters of embedding, and that buys a twelfth Transformer block, so
vocabulary is the only variable and depth is what absorbs it.

| | vocab | layers | embedding | total parameters |
|---|---:|---:|---:|---:|
| `_16k` | 16,000 | 11 | 6.14M | 25,625,856 |
| `_12k` | 12,000 | 12 | 4.61M | 25,860,864 |

Every directory keeps its plain name and holds one subdirectory per run, so
nothing fans out into four siblings:

```
report/phase2/<lang>/<lang>_16k/     <lang>/logs/pretrain/pretrain_16k/
report/phase2/<lang>/<lang>_12k/     <lang>/logs/pretrain/pretrain_12k/
```

Which vocabulary a run uses is set by `tokenizer_model` in
`<language>/configs/pretrain.yaml`, which ships set to the reported 16k run.
**All four runs are complete.** The 12k control keeps only its evaluation JSON
and its training log, since the comparison is the only thing it is reported
for.

**The result:** measured on bits-per-byte - the only intrinsic metric that is
comparable when the tokenizers differ - the two vocabularies are within 0.1-0.3%
of each other, while the 12k run costs 9-13% more wall-clock time per epoch.
Perplexity falls 13-14% for the 12k models, but that is an artifact of the
smaller vocabulary rather than a better model, and
[`vocab_comparison.md`](report/phase2/vocab_comparison.md) shows why. The 16k
run stays the reported Phase 2 model and the checkpoint Phase 3 finetunes
from.

### Bonus: no-positional-embeddings ablation

Gujarati 16k retrained with `positional: none`, matched on everything else
(same corpus, tokenizer, parameter count, schedule, seed, and the same 15,826
steps). Position costs **+8.6% test perplexity** and **+2.1% bits-per-byte**.
Results, figures and the write-up:
[`report/bonus/`](report/bonus/README.md).

## Data

| Contents | Where |
|---|---|
| Raw and preprocessed corpora, both languages | [OneDrive](https://iiithydstudents-my.sharepoint.com/:f:/g/personal/hardik_k_students_iiit_ac_in/IgDZ46BkCoOhT74gAOPIOBrCAVnN7uIm5sKRY9rQiqDukj0?e=0hlbPM) |
| Tokenizer models and vocabularies | in this repo, `<language>/tokenizer/` |
| Phase 2 checkpoints, 16k (Model H and Model L) | [Hugging Face](https://huggingface.co/hardik300/LMA_mini_project/tree/main) - `gujarati-model-h/16k-rope/`, `nepali-model-l/16k-rope/` |
| Phase 2 checkpoints, 12k (Model H and Model L) | [Hugging Face](https://huggingface.co/hardik300/LMA_mini_project/tree/main) - `gujarati-model-h/12k-rope/`, `nepali-model-l/12k-rope/` |
| Bonus ablation checkpoint (no positional encoding) | [Hugging Face](https://huggingface.co/hardik300/LMA_mini_project/tree/main) - `gujarati-model-h/16k-nopos/` |
| Phase 2 training logs | in this repo, `<language>/logs/pretrain/pretrain_<vocab>/` |
| Phase 2 loss curves, metrics, samples | in this repo, `report/phase2/<language>/<language>_<vocab>/` |
| Phase 3 finetuned checkpoints | _added in Phase 3_ |

**All ten checkpoints are uploaded.** Five runs, each with `best.pt` (lowest
validation loss) and `latest.pt` (end of the run), ~295 MB apiece:

```
gujarati-model-h/16k-rope/     reported Model H
gujarati-model-h/12k-rope/     vocabulary control
gujarati-model-h/16k-nopos/    bonus ablation, no positional encoding
nepali-model-l/16k-rope/       reported Model L
nepali-model-l/12k-rope/       vocabulary control
```

Every checkpoint is resume-capable: model weights, optimizer state, scheduler
state, gradient-scaler state, step, tokens seen, both configs and the RNG
states. The model card at the link above lists all five runs with their
parameter counts and test scores.

The brief names Google Drive for large artifacts. These are on Hugging Face
instead: it is public, versioned, needs no access request, and serves files
directly. The requirement the link has to meet - a grader can open it without
asking for permission - is met either way.

The corpus folder is laid out as `<language>/raw/{public, manual/<source>}` and
`<language>/preprocessed/{train, validation, test}`. Every line is one JSON
document carrying its provenance, so the manual-vs-public split is verifiable
from the data itself.

## Repository layout

```
gujarati/                       Model H - self-contained
nepali/                         Model L - self-contained, same structure
  configs/pipeline.yaml         every path, threshold and hyperparameter
  data/
    sources/manual_sources.yaml the source registry, with probe results
    collection/                 collect_public.py, collect_manual.py
      adapters/                 one module per collection method
    preprocessing/              preprocess.py, deduplicate.py
    splits/                     build_splits.py, corpus_stats.py, stage_stats.py
    corpus_io.py                shared document record and gzip JSONL I/O
  tokenizer/
    train_tokenizer.py          SentencePiece training + vocabulary sweep
    tokenizer_stats.py          held-out evaluation and the size decision
    model/  vocab/              the committed tokenizer artifacts
  model/
    param_budget.py             parameter budget: formula vs the built model
    transformer.py              the decoder-only Transformer, from scratch
    config.py                   YAML -> ModelConfig, with paths resolved
  train/
    prepare_data.py             splits -> flat uint16 token streams
    dataset.py                  memory-mapped windows, deterministic batch order
    trainer.py                  the training loop, checkpointing and resume
    pretrain.py                 the pretraining entry point
    plot_curves.py              loss / perplexity / schedule figures
    runs/                       checkpoints (gitignored)
  eval/
    common.py                   checkpoint, tokenizer and split loading
    perplexity.py               cross-entropy, perplexity, bits-per-byte
    generate.py                 continuations under four decoding settings
    text_metrics.py             BLEU-4, chrF++, ROUGE-L, distinct-n, repetition
    attention.py                heatmaps, entropy, mean attention distance
    causal_check.py             four proofs the model cannot see the future
    run_eval.py                 runs the whole suite in order
    infer.py                    generate from a prompt, interactively
  logs/                         run logs: collection, preprocessing,
                                tokenizer (Phase 1), and pretrain/ holding
                                pretrain_16k/ and pretrain_12k/ (Phase 2,
                                one directory per run)

report/
  bonus/                        optional ablation: no positional embeddings
                                (self-contained: README, figures, json, logs,
                                 and the script that regenerates it)
  phase1/                       corpus and tokenizer report
  phase2/
    README.md                   Phase 2 report: decisions, methodology, analysis
    results.md                  every measured number, both models (generated)
    vocab_comparison.md         16k vs 12k, on bits-per-byte (generated)
    gujarati/  nepali/          per-model evidence: <lang>_16k/ with json/,
                                figures/ and samples.md, and <lang>_12k/ with
                                the control run's json/
    figures/models/             cross-language figures for results.md
    figures/vocab_ablation/     figures for vocab_comparison.md
    make_report_assets.py       regenerates the results page and figures
    make_vocab_comparison.py    regenerates the 16k-vs-12k comparison
```

Both language directories hold the same modules with the same names. The code is
not shared between them: each is standalone, and the differences (language
identification, normalisation rules, adapters) live inside each copy.

## Requirements

Python 3.12 for Phase 1; Phase 2 also runs on 3.11, which is what Kaggle has.
No GPU is needed for Phase 1. Phase 2 pretraining needs one - about 2-4 hours
per model on a Kaggle T4 in fp16, plus ~30 minutes for the evaluation suite.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

| Package | Used for |
|---|---|
| `pyyaml` | configs and the source registries |
| `datasets`, `huggingface_hub` | streaming the public dataset (never fully downloaded) |
| `aiohttp` | all async fetching - article endpoints, sitemaps, archive.org |
| `requests` | streaming large PDF downloads |
| `trafilatura`, `beautifulsoup4`, `lxml` | article extraction from HTML, sitemap parsing |
| `PyMuPDF` | PDF text-layer extraction |
| `datasketch` | MinHash + LSH near-duplicate detection |
| `sentencepiece` | tokenizer training |
| `matplotlib`, `numpy` | statistics and report figures |
| `torch` | the Transformer, pretraining and evaluation (Phase 2) |

Disk: roughly 12 GB per language for the full pipeline (raw, clean, dedup and
split copies). Memory: 8 GB is enough - the tokenizer training sample is capped
to stay within it.

## Running the pipeline

For either language (`gujarati` or `nepali`):

```bash
cd gujarati

# 1. corpus
python3 data/collection/collect_public.py          # public dataset (streamed)
python3 data/collection/collect_manual.py --check  # robots.txt + reachability
python3 data/collection/collect_manual.py          # manual collection
python3 data/collection/collect_manual.py --status # progress vs the 20% floor

# 2. preprocessing
python3 data/preprocessing/preprocess.py           # normalise, filter, boilerplate
python3 data/preprocessing/deduplicate.py          # exact + MinHash near-duplicates
python3 data/splits/build_splits.py                # 90/5/5, hashed per document

# 3. tokenizer
python3 tokenizer/train_tokenizer.py --sweep       # candidate vocabulary sizes
python3 tokenizer/tokenizer_stats.py --compare-sweep   # pick a size on held-out text
python3 tokenizer/train_tokenizer.py               # final model at the chosen size
python3 tokenizer/tokenizer_stats.py               # final held-out statistics

# 4. statistics
python3 data/splits/stage_stats.py                 # what survived each stage
python3 data/splits/corpus_stats.py --tokenizer    # exact token counts
```

Cross-language tables and figures, after both languages have finished:

```bash
python3 report/phase1/make_report_assets.py
```

### Phase 2 - model, pretraining, evaluation

```bash
cd gujarati                       # or nepali

python3 model/transformer.py --self-test    # RoPE, attention, masking, init
python3 model/param_budget.py     # exact parameter count, formula vs the model
python3 eval/text_metrics.py --self-test    # BLEU / chrF against sacrebleu
python3 train/prepare_data.py     # splits -> data/tokenized/*.bin
python3 train/pretrain.py         # pretrain; --resume auto to continue
python3 eval/run_eval.py          # perplexity, generation, attention, mask checks
```

To try a trained model by hand:

```bash
cd gujarati                       # or nepali
python3 eval/infer.py --prompt "ગુજરાત સરકારે"
python3 eval/infer.py --input eval/input.txt   # a file of prompts, one per line
python3 eval/infer.py                          # interactive
```

`eval/input.txt` holds one prompt per line (`#` comments and blank lines are
skipped). With `--input`, each prompt is answered once at temperature 0.5, 1.0
and 1.5 - the three the evaluation reports - and the result is written to
`model_output.md` in the run's report directory as one block per prompt: the
prompt, then its continuations, and nothing else. That file is rewritten on
every run, so it always matches the current prompt file. The single-prompt and
interactive modes instead append to `manual_samples.md`, recording the decoding
settings, so a hand-typed session accumulates.

Read the output from the file rather than the terminal: a terminal lays out one
character per fixed-width cell and mostly does not shape complex scripts, so
Gujarati matras and Devanagari conjuncts land in the wrong place. The text
itself is correct, and any editor or markdown viewer renders it properly.

In interactive mode each line typed is continued, a blank line quits, and `/new`
has the model start a document by itself. `--greedy` is deterministic and loops,
`--temperature 1.0` reads best on both models, `--top-k 40` trims the nonsense
at higher temperatures, `--samples 3` draws three continuations from one prompt,
and `--seed` makes a run reproducible.

`eval/generate.py` is the automated counterpart: it takes 200 held-out test
documents, continues each under four decoding settings and scores the result
with BLEU / chrF++ / ROUGE-L and the diversity diagnostics. Both call the same
`DecoderLM.generate`.

Then, once both models are done:

```bash
python3 report/phase2/make_report_assets.py
```

Pretraining is resumable by design: `train/pretrain.py --resume auto` restores
the model, optimizer, scheduler, loss scale, step and token count, and continues
on the batches the run would have seen anyway. `--max-hours` stops and
checkpoints before a hosted session is cut off.

Both models were trained this way on Kaggle, one per session on a single T4,
using exactly the commands above with `--data-dir`, `--runs-dir` and
`--logs-dir` pointing into `/kaggle/working`. Nothing about the pipeline is
specific to Kaggle; any machine with a GPU and the corpus runs the same four
commands.

Collection is resumable: each run writes its own shard, and restarting skips
URLs already stored. Every configurable value lives in
`<language>/configs/pipeline.yaml`, not in the code.

## Notes on the implementation

- **No pretrained models or tokenizers.** Language identification is by Unicode
  block (Gujarati) or function-word ratios (Nepali), not by a classifier.
- **Deduplication runs before splitting**, so no near-duplicate of a test
  document can appear in train.
- **Splits are deterministic**: `md5(seed:doc_id)` decides a document's split, so
  the pipeline reproduces exactly and Phase 2/3 stay comparable across re-runs.
- **Provenance is carried per document**, so the manual-vs-public share is
  measured rather than estimated.
- **Nothing pretrained is used anywhere.** No pretrained tokenizer, no
  pretrained model, no `nn.Transformer*`, no HuggingFace model class and no
  `F.scaled_dot_product_attention` - multi-head attention, the causal mask and
  the positional scheme are all written out in
  `<language>/model/transformer.py`.
- **The causal mask is tested, not assumed**: `eval/causal_check.py` verifies it
  four independent ways, including that no gradient reaches a future position.
- **BLEU, chrF++ and ROUGE-L are implemented in the repository** and checked
  against sacrebleu 2.6.0 by `python3 eval/text_metrics.py --self-test`.
