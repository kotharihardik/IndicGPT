# Monolingual Transformer LMs — Gujarati and Nepali

Individual project for Language Models and Agents, Monsoon 2026.

Two completely independent decoder-only Transformer language models, built from
scratch:

- **Model H — Gujarati**, the higher-resource language
- **Model L — Nepali**, from the allowed lower-resource list

The two models share no data, no tokenizer, no vocabulary and no weights. Each
language lives in its own top-level directory and can be run on its own.

> **Corpus download** (raw + preprocessed, both languages):
> <https://iiithydstudents-my.sharepoint.com/:f:/g/personal/hardik_k_students_iiit_ac_in/IgDZ46BkCoOhT74gAOPIOBrCAVnN7uIm5sKRY9rQiqDukj0?e=0hlbPM>
>
> Plain readable `.jsonl` — no archives to unpack. Tokenizer models and
> vocabularies are committed to this repository directly.

## Status

| Phase | Contents | Branch | State |
|---|---|---|---|
| 1 | Corpus collection and tokenizer construction | `phase-1` | complete |
| 2 | Transformer implementation, pretraining, evaluation | `phase-2` | not started |
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

**Both corpora exceed the 500M token target and both clear the ≥20% manual
requirement.**

Read strictly as the *training split alone*, Gujarati reaches 517.1M and Nepali
488.0M — 11.95M short. The Nepali corpus itself clears the target; the
difference is that 10% is held out for validation and test. That ratio was
chosen so Phase 2 gets a validation set spanning all 23 sources (61,604
documents rather than ~3,900), and §1.2 of the brief allows the lower-resource
language to report the maximum feasible amount with a justification.

Per-language detail: [Gujarati](report/phase1/gujarati/README.md) · [Nepali](report/phase1/nepali/README.md). Cross-language tables and figures: [`report/phase1/`](report/phase1/).

## Data

| Contents | Where |
|---|---|
| Raw and preprocessed corpora, both languages | [OneDrive](https://iiithydstudents-my.sharepoint.com/:f:/g/personal/hardik_k_students_iiit_ac_in/IgDZ46BkCoOhT74gAOPIOBrCAVnN7uIm5sKRY9rQiqDukj0?e=0hlbPM) |
| Tokenizer models and vocabularies | in this repo, `<language>/tokenizer/` |
| Phase 2 pretrained checkpoints | _added in Phase 2_ |
| Phase 3 finetuned checkpoints | _added in Phase 3_ |

The corpus folder is laid out as `<language>/raw/{public, manual/<source>}` and
`<language>/preprocessed/{train, validation, test}`. Every line is one JSON
document carrying its provenance, so the manual-vs-public split is verifiable
from the data itself.

## Repository layout

```
gujarati/                       Model H — self-contained
nepali/                         Model L — self-contained, same structure
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
  model/param_budget.py         parameter budget calculator for Phase 2
  logs/                         run logs

report/
  phase1/
    gujarati/  nepali/          README.md (corpus + tokenizer report),
                                json/ summaries, figures/
    tables/                     cross-language tables
    figures/                    cross-language figures
    make_report_assets.py       regenerates tables/ and figures/ from the JSON
```

Both language directories hold the same modules with the same names. The code is
not shared between them: each is standalone, and the differences (language
identification, normalisation rules, adapters) live inside each copy.

## Requirements

Python 3.12. No GPU is needed for Phase 1.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

| Package | Used for |
|---|---|
| `pyyaml` | configs and the source registries |
| `datasets`, `huggingface_hub` | streaming the public dataset (never fully downloaded) |
| `aiohttp` | all async fetching — article endpoints, sitemaps, archive.org |
| `requests` | streaming large PDF downloads |
| `trafilatura`, `beautifulsoup4`, `lxml` | article extraction from HTML, sitemap parsing |
| `PyMuPDF` | PDF text-layer extraction |
| `datasketch` | MinHash + LSH near-duplicate detection |
| `sentencepiece` | tokenizer training |
| `matplotlib`, `numpy` | statistics and report figures |

Disk: roughly 12 GB per language for the full pipeline (raw, clean, dedup and
split copies). Memory: 8 GB is enough — the tokenizer training sample is capped
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
