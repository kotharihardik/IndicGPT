# Phase 2 - Model implementation, pretraining and evaluation

Two independent decoder-only Transformers, written from scratch in PyTorch and
pretrained on the Phase 1 corpora.

| | Gujarati (Model H) | Nepali (Model L) |
|---|---|---|
| Code | [`gujarati/`](../../gujarati/) | [`nepali/`](../../nepali/) |
| Architecture config | [`configs/model.yaml`](../../gujarati/configs/model.yaml) | [`configs/model.yaml`](../../nepali/configs/model.yaml) |
| Training config | [`configs/pretrain.yaml`](../../gujarati/configs/pretrain.yaml) | [`configs/pretrain.yaml`](../../nepali/configs/pretrain.yaml) |
| Evaluation config | [`configs/eval.yaml`](../../gujarati/configs/eval.yaml) | [`configs/eval.yaml`](../../nepali/configs/eval.yaml) |
| Figures | [`gujarati_16k/`](gujarati/gujarati_16k/README.md) | [`nepali_16k/`](nepali/nepali_16k/README.md) |

The submitted report is [`phase2_report.pdf`](phase2_report.pdf).

All measured numbers are in [`results.md`](results.md), generated
from the evaluation output. The 16,000-piece vocabulary chosen in Phase 1 is the
reported model; a 12,000-piece control run at the same parameter budget is
compared against it in [`vocab_comparison.md`](vocab_comparison.md). This page
gives the design decisions and the analysis.

The two models share no data, no tokenizer, no vocabulary and no weights. They
do share an architecture and a training recipe, deliberately: with everything
else fixed, the only differences between them are the corpus and the tokenizer,
which is the comparison the project asks for.

---

## 1. The model

`<language>/model/transformer.py`. No `nn.Transformer*`, no HuggingFace classes,
no `F.scaled_dot_product_attention`; attention, the causal mask and the
positional scheme are written out.

```
ids (B,T) -> token embedding (B,T,C) -> embedding dropout
          -> 11 x [ LayerNorm -> causal self-attention -> + residual
                    LayerNorm -> SwiGLU feed-forward   -> + residual ]
          -> final LayerNorm -> tied output projection -> logits (B,T,V)
          -> cross-entropy against the next token
```

| | Value | Reason |
|---|---|---|
| `d_model` | 384 | fixed in Phase 1: the vocabulary was priced against a 25M budget at this width |
| `n_layers` | 11 | 25M - 6.14M embedding = 18.86M, and one block costs 1.77M |
| `n_heads` | 6 | head dimension 64 |
| `d_ff` | 1024 | SwiGLU; `3 x 384 x 1024` costs what GELU at 1536 costs |
| `context_length` | 512 | attention is quadratic in this, and 512 tokens is about 2,000 characters, most of a news article |
| positional | RoPE | zero parameters, relative by construction |
| normalisation | pre-norm LayerNorm | unbroken identity path through 11 blocks |
| output head | tied to the token embedding | saves 6.14M, a quarter of the budget |
| biases | none | LayerNorm already re-centres every sublayer input |
| dropout | 0.0 for pretraining | one pass over ~519M tokens; implemented and used in Phase 3 |
| **parameters** | **25,625,856** | +2.5% of the ~25M target |

`python3 model/param_budget.py` prints the breakdown and checks the analytic
formula against the model it builds, so a number quoted here cannot drift from
the network that was trained.

`python3 model/transformer.py --self-test` checks, in about ten seconds, that a
RoPE score depends only on `m - n`, that RoPE costs no parameters and preserves
norms, that attention rows sum to 1 with nothing above the diagonal, that an
untrained model's first loss lands near `ln(V)`, and that a sequence longer than
the trained context runs rather than raising.

### Why RoPE and not a learned table

Self-attention is permutation invariant: without position, `softmax(QK^T/sqrt(dk))V`
returns the same answer for a sentence and for its shuffling. RoPE rotates each
`(even, odd)` channel pair of **Q and K** - not V, which carries content - by an
angle proportional to the token's position. The dot product of two rotated
vectors depends only on the *difference* of their angles, so the attention score
between positions `i` and `j` becomes a function of `i - j`. Two consequences
decided the choice:

- **It costs nothing.** A learned table is `context x d_model = 512 x 384 =
  196,608` parameters. RoPE is computed from the position, so that budget goes
  into the blocks instead.
- **The sequence length is not capped by a table.** A learned embedding has no
  row for position 600. RoPE's cosines are computed, and
  `RotaryPositionalEmbedding._cache` rebuilds them for any length, so a longer
  sequence runs rather than raising an index error; the self-test feeds the
  trained architecture a 640-token input. What is still bounded is *quality*:
  the model has never seen relative distances beyond 512, so it degrades past
  that rather than failing. This is a softer constraint than a learned table's,
  not the absence of one.

`positional: learned`, `sinusoidal` and `none` are also implemented and
selectable in `configs/model.yaml`; `none` is what the optional bonus ablation
would need.

### What each component does, and what breaks without it

| Component | Computes | Shapes | Remove it and... |
|---|---|---|---|
| Token embedding | id -> dense vector | `(B,T) -> (B,T,384)` | there is nothing to feed the stack |
| RoPE | rotates Q and K by position | `(B,6,T,64) -> (B,6,T,64)` | attention becomes permutation invariant: word order stops existing |
| Q/K/V projections | `XW_Q`, `XW_K`, `XW_V` | `(B,T,384) -> (B,T,384)` each | no query/key distinction; attention collapses to a fixed similarity |
| Head reshape | splits `d` into `h` subspaces | `(B,T,384) -> (B,6,T,64)` | one attention pattern per layer instead of six |
| `QK^T/sqrt(dk)` | pairwise similarity, scaled | `(B,6,T,64)x(B,6,64,T) -> (B,6,T,T)` | without `1/sqrt(dk)` the scores grow like `dk`, softmax saturates and its gradient vanishes |
| Causal mask | `-inf` above the diagonal | `(T,T)` broadcast over `(B,6,T,T)` | position `t` sees token `t+1`; loss collapses and generation is nonsense |
| Softmax | scores -> a distribution over keys | `(B,6,T,T)` | the weights stop being a weighted average and the residual stream explodes |
| `weights @ V` | mixes value vectors | `(B,6,T,T)x(B,6,T,64) -> (B,6,T,64)` | nothing is retrieved from context |
| `W_O` | mixes the concatenated heads | `(B,T,384) -> (B,T,384)` | heads never interact; each writes into its own slice |
| SwiGLU FFN | `down(silu(gate(x)) * up(x))` | `(B,T,384) -> (B,T,1024) -> (B,T,384)` | the model becomes a stack of linear mixes with no per-token non-linear capacity |
| Residual connections | `x + sublayer(x)` | `(B,T,384)` | gradients through 11 layers vanish and the stack will not train |
| LayerNorm (pre-norm) | normalises each sublayer's input | `(B,T,384)` | activations drift in scale with depth and the run diverges |
| Final LayerNorm | normalises before the head | `(B,T,384)` | logit scale drifts with depth |
| Tied output head | `hidden x E^T` | `(B,T,384) -> (B,T,16000)` | either 6.14M more parameters, or no logits at all |

### Verifying the causal mask

A model that peeks trains normally and reports an implausibly low perplexity, so
`eval/causal_check.py` tests it four ways rather than assuming it:

1. **Future edit** - change every token after position `t`; logits at or before
   `t` must be identical, and logits after `t` must change (otherwise the test
   would also pass on a model that ignores its input).
2. **Prefix equality** - feeding only the first `t+1` tokens gives the same
   logits at those positions as feeding the whole sequence.
3. **Mask shape** - every attention weight strictly above the diagonal is
   exactly zero, in every head of every layer.
4. **Gradient reach** - `d(logit at t)/d(embedding at s)` is exactly zero for
   every `s > t`. This tests the computational graph, not just the values.

Both models pass all four; the measured values are in
[`results.md`](results.md) Section 6.

---

## 2. Data

`<language>/train/prepare_data.py` encodes each Phase 1 split with that
language's own tokenizer and appends it to one flat `uint16` array. Documents are
wrapped `<s> ... </s>`: `<s>` gives generation a defined prompt and `</s>` teaches
the model where text ends instead of running one article into the next. `uint16`
fits a 16,000-piece vocabulary and keeps the training stream near 1 GB instead of
the 4 GB an `int32` file would need.

`meta.json` records documents, characters, **bytes** and tokens per split. The
byte totals are the denominator of bits-per-byte, so they are measured once here
rather than re-derived during evaluation.

Training reads windows through `train/dataset.py`. The stream is cut into
non-overlapping blocks of 512 tokens and visited in a seeded permutation, so one
epoch covers every token exactly once and the batch order is a function of
`(seed, step)` alone - which is what lets a resumed run continue on the data it
would have seen anyway.

---

## 3. Pretraining

`<language>/train/trainer.py`, driven by `train/pretrain.py`.

| | Value |
|---|---|
| Objective | causal LM cross-entropy, mean over all positions |
| Optimizer | AdamW, beta = (0.9, 0.95), epsilon = 1e-8, weight decay 0.1 on matrices only |
| Peak / floor learning rate | 6e-4 / 6e-5 |
| Schedule | 500 steps linear warmup, then cosine decay |
| Gradient clipping | global norm 1.0 |
| Batch | 32 x 2 accumulation x 512 = **32,768 tokens per step** |
| Length | one epoch over the training stream |
| Validation during training | every 500 steps, 512 windows (262,144 tokens) spread evenly across the validation split |
| Precision | fp16 with a gradient scaler on Turing/Pascal (T4, P100), bf16 on Ampere and newer; attention softmax and RoPE always in fp32 |
| Seed | 1337, the seed the Phase 1 splits used |

beta2 is 0.95 rather than Adam's 0.999 default: the second-moment estimate then
adapts over about 20 steps instead of about 1,000, which matters when the whole
run is around 15,000 steps. Weight decay skips LayerNorm gains and biases, which
have too few parameters to overfit and are only fighting the normalisation when
shrunk.

**Dropout is 0.0 by decision, not omission.** 25.6M parameters see about 500M
tokens once, so no document is repeated and there is nothing to memorise;
dropout would only slow learning. Validation loss tracking training loss for the
whole run is the check that the choice was right. All three dropout sites the
brief asks for - embedding output, attention probabilities, both sublayer
outputs - are implemented and wired, and Phase 3 turns them on, where the
reasoning corpus is small and seen many times.

**Why one epoch.** The brief targets ~500M pretraining tokens per model, and one
pass over the training stream is 518.6M for Gujarati and 490.3M for Nepali - the
target met by construction rather than by tuning a step count. It is also close
to the compute-optimal ratio: about 20 tokens per parameter at 25.6M parameters
is 512M tokens. The run length is therefore set by `epochs: 1.0`, and the step
count is derived from the actual token count at startup. A second epoch would
repeat documents rather than add information and double the GPU time.

### What is measured on how much data

| | Data used |
|---|---|
| Pretraining | the **entire** training stream, one pass. 15,826 steps x 32,768 = 518,586,368 of 518,603,534 Gujarati tokens (99.997%); 14,962 x 32,768 = 490,274,816 of 490,279,983 Nepali (99.999%). The remainder is the final partial batch. |
| Validation during training | 512 windows = 262,144 tokens, **spread evenly across the whole split**, every 500 steps. A progress signal and the `best.pt` criterion. |
| **Reported cross-entropy, perplexity, bits-per-byte** | the **complete** validation and test splits - 28.5M and 28.2M tokens for Gujarati, 26.9M and 27.2M for Nepali. No subsampling, and in float32 rather than under autocast. |
| Generation metrics | 200 held-out test documents, taken at a stride through the split |
| Attention statistics | 160 windows = 81,920 tokens, spread across the validation split |

Spreading the sampled windows matters. Phase 1 streams manual documents first
through deduplication, so the head of every split file is manual text: the first
4,000 Gujarati validation documents come from five manual sources and contain no
`ai4bharat_sangraha`, which is 68.6% of the corpus. Taking the first N windows
would have scored one register and called it the validation loss, and selected
`best.pt` on it.

**Precision.** `auto` chooses by compute capability, not by
`torch.cuda.is_bf16_supported()`. That helper defaults to
`including_emulation=True`, so on a T4 it only checks that a bf16 tensor can be
created and answers yes. Turing has no bf16 tensor cores, so the first Kaggle
run silently ran at fp32 speed - 10.8k tokens/s, 24% of the card's fp32 peak and
3% of its fp16 peak - turning a ~3 h run into a projected 12.6 h. The test is now
`torch.cuda.get_device_capability()[0] >= 8`, with `--precision bf16` as an
override. Using fp16 has two consequences: gradients underflow in its narrow
exponent range, so `GradScaler` multiplies the loss up before the backward pass
and divides it out before clipping and the optimizer step; and the attention
softmax and the RoPE rotation are computed in fp32, since both are cheap
elementwise work next to the matmuls and are where low precision would show
first.

### Checkpointing and resume

Every checkpoint holds model weights, optimizer state, scheduler state, gradient
scaler state, training step, tokens seen, both configs, and the RNG states of
Python, NumPy and PyTorch. Writes are atomic (`.tmp` then `os.replace`), so an
interrupted save never leaves a corrupt file.

`SIGINT` and `SIGTERM` finish the current step, checkpoint and exit, and
`--max-hours` does the same before a hosted session is killed on its own
schedule. `python3 train/pretrain.py --resume auto` picks up wherever the run
stopped. Both models were trained this way: one Kaggle T4 session each, about
4.6 hours, finishing inside a single session.

---

## 4. Evaluation

`<language>/eval/`, driven by `eval/run_eval.py`.

### Intrinsic

Cross-entropy, perplexity and **bits-per-byte** on the full validation and test
splits, in float32, over non-overlapping 512-token windows, so every token is
predicted exactly once.

Bits-per-byte is the honest cross-language number. Perplexity is per *token*,
and the two tokenizers differ: Nepali compresses at 4.689 characters per token
against Gujarati's 3.865, so a Nepali token carries more text and is a harder
prediction. Dividing the same total likelihood by UTF-8 bytes removes the
tokenizer, because bytes are a property of the text.

### Generation

200 held-out test documents, cut into a 64-token prompt and a 96-token
reference. The model continues the prompt for 96 tokens under four settings -
greedy, and sampling at temperature 0.5, 1.0 and 1.5 - and the continuations are
scored against the *decoded reference token ids*, which is what the model would
have produced had it predicted perfectly.

BLEU-4, chrF++ and ROUGE-L are implemented in `eval/text_metrics.py` rather than
imported, so the exact variant is visible and a hosted worker needs no extra
package. `python3 eval/text_metrics.py --self-test` reproduces corpus BLEU-4 and
chrF/chrF++ to within 1e-3 of values recorded from sacrebleu 2.6.0, on an ASCII
fixture and one in the model's own script.

That self-test found a real defect. The obvious word tokenizer,
`re.sub(r"([^\w\s])", ...)`, is wrong for both languages: Python's `\w` does not
match Unicode combining marks, so every Gujarati and Devanagari word was split
into consonant-plus-matra fragments and every word-level metric measured those
fragments. The tokenizer now classifies by Unicode category, keeping marks
(Mn, Mc, Me) and format characters (Cf, for ZWNJ) inside the word and detaching
punctuation including the danda.

What each metric is worth here:

- **chrF++** is the informative one. Both languages mark case and number with
  suffix chains, so a correct stem with the wrong ending scores zero on
  word-level BLEU but keeps most of its character n-grams.
- **BLEU-4** is weak for open-ended continuation: there is no single correct
  next 96 tokens, and exact 4-gram overlap with one reference is close to
  accidental. It is reported because the brief asks for it, and because the
  *ranking* across decoding settings is still meaningful.
- **ROUGE-L** sits between them: the longest common subsequence rewards the
  right words in the right order without demanding adjacency, so it tolerates
  the insertions a language model makes.
- **Distinct-1/2 and repetition-4** are the diagnostics that catch failure.
  Greedy decoding on an under-trained model loops on one phrase, which shows up
  as repetition near 1.0 and Distinct-2 near 0. The held-out reference is scored
  the same way and printed alongside, so "diverse" has a yardstick.

All of these are computed on words and characters of the decoded text, never on
tokenizer pieces: piece-level scores would compare the two tokenizers as much as
the two models.

### Attention

Averaged over about 82,000 tokens of held-out text, per layer and per head:

- **normalised entropy** - entropy divided by `log(keys visible at t)`. Raw
  entropy grows with position simply because more keys become available;
  normalising makes 1.0 mean "uniform" at every position and depth.
- **mean attention distance** - the mass-weighted mean of `query position - key
  position`, which separates a local head from one that gathers across the
  prefix.
- **previous-token mass** and **BOS mass** - attention on exactly `t-1`, the
  signature of a positional or induction head, and attention parked on the start
  token, the sink a head uses when it has nothing to contribute.

Heatmaps cover an early layer and a late layer, every head, on example sentences
in each model's own language. Their axes are token **indices**: rendering
Gujarati or Devanagari inside a figure needs a script font a hosted worker may
not have. The sentences and their numbered pieces are in each language's
`attention_examples.md`, indexed to match, so a bright cell can be read back to
the words it covers.

---

## 5. Findings

### 5.1 Resource-level comparison

Model H and Model L have the **same architecture, optimizer, schedule, batch
size, number of epochs and seed**. Three things differ, and only three:

| | Gujarati (Model H) | Nepali (Model L) |
|---|---:|---:|
| Training tokens | 518.6 M | 490.3 M (-5.5%) |
| Manual share of the corpus | 31.44% | 21.58% |
| Tokenizer compression | 3.865 chars/token | 4.689 chars/token |
| Sources in the corpus | 1 public + 12 manual | 1 public + 23 manual |

The third is why **bits-per-byte, not perplexity, is the number to compare**. A
Nepali token covers about 21% more text than a Gujarati one, so it is a harder
prediction: Nepali would score a worse perplexity even if both models understood
their language equally well.

Model L is ahead on both measures - test bits-per-byte 0.4247 against 0.5764,
26.3% lower - while seeing 5.5% fewer tokens. Two effects explain this:

- **Data.** Model L sees fewer tokens, but the *composition* differs more than
  the count. Gujarati carries 44.8M tokens of archive.org book OCR - 8.7% of its
  training split - compressing at 2.77 characters per token against 3.97 for its
  other manual sources, so Model H is predicting a more varied and noisier
  distribution. The Nepali split has no comparable source and is 78.4% one
  homogeneous public corpus, which is easier to model.
- **Tokenizer.** Nepali's fertility is *better* (1.416 tokens per word against
  1.503), so its model spends fewer prediction steps per sentence. On its own,
  that helps Model L.

The lower-resource model scoring better is therefore a statement about corpus
homogeneity, not about the two languages or the two networks. It is also a
warning about the metric: a single held-out bits-per-byte cannot separate "the
model learned more" from "the held-out text was more predictable".

### 5.2 Generation quality

Full numbers in [`results.md`](results.md) Section 4. Continuations from held-out
prefixes are in [Gujarati](gujarati/gujarati_16k/samples.md) and
[Nepali](nepali/nepali_16k/samples.md); the same prompts run by hand at each
temperature are in `model_output.md` beside them. Both models behave the same
way across the four decoding settings:

- **Greedy** collapses into loops within twenty tokens or so. Repetition-4 is
  0.703 (Gujarati) and 0.631 (Nepali) against 0.007 and 0.006 for the held-out
  reference, and Distinct-2 is about a quarter of the reference's. Sample 1 in
  each file shows it directly: one clause repeated to the token budget.
- **Temperature 0.5** is fluent and grammatical but still circles a topic,
  restating the same clause with small variations. It scores the best BLEU-4 and
  ROUGE-L of the four, because staying close to the prompt's vocabulary is what
  n-gram overlap rewards.
- **Temperature 1.0** reads best. chrF++ peaks here for both models (20.13 and
  21.02) and the diversity diagnostics land almost exactly on the reference:
  Distinct-2 0.856 against 0.824 for Gujarati, 0.882 against 0.862 for Nepali.
  Sentences are well formed, agreement and case endings are usually correct, and
  named entities are plausible for the register, but claims drift - dates,
  numbers and places are invented, and consecutive sentences need not be about
  the same event.
- **Temperature 1.5** breaks down. Distinct-2 reaches 0.982 for both models,
  above the reference, which is the diagnostic showing the text is now more
  varied than real text. Words are mostly real words and morphology is mostly
  legal, but the output is not coherent, and Gujarati mixes registers within one
  sentence.

The pattern is what a 25M-parameter model trained for one epoch should show:
local fluency without discourse-level coherence. Language correctness is the
part that is clearly learned - script, morphology, agreement and punctuation are
right far more often than not in both languages - while factual and topical
consistency across sentences is not.

### 5.3 Attention

Per-head grids in each model's page; the numbers below are from
`json/attention.json`. The two models arrive at closely matched structure:
Gujarati mean normalised entropy 0.691 and mean attention distance 67.9 tokens,
Nepali 0.669 and 68.2.

- **Layers 0-2 are diffuse and long-range** in both models: normalised entropy
  0.85-0.95, mean distance 87-114 tokens. Early attention averages broadly over
  the prefix rather than selecting from it, which is the usual behaviour where
  representations are still close to raw embeddings.
- **The middle of the stack is where local, positional heads appear.** Across
  layers 3-6 entropy drops to 0.56-0.76 and mean distance to 39-52 tokens. Individual heads become
  strongly previous-token: Gujarati L6 H0 puts 58% of its mass on exactly `t-1`
  (entropy 0.303, distance 12.0), and Nepali L7 H3 puts 58% (entropy 0.278).
  Gujarati has 8 heads above 0.2 previous-token mass and Nepali 6, spread over
  layers 3-10 in both.
- **Late layers split into two kinds.** Some stay local - Nepali L10 H2 has a
  mean distance of 11.3 tokens - while others become long-range and
  content-driven: Gujarati L8 H1 reaches a mean distance of 145 tokens at
  entropy 0.354, which is focused attention on something far back rather than an
  average over the prefix.
- **The clearest difference between the models is the attention sink.** Gujarati
  parks 5.9% of its mass on `<s>` overall and 21% at layer 8, with L8 H1 alone
  at 41.9%. Nepali parks 2.6% overall and never exceeds 10.9% in any head. A
  sink head is one that has nothing to contribute at that position and dumps its
  mass somewhere harmless; Model H uses that escape route roughly twice as
  often, consistent with the more varied corpus it is modelling.

Read against the heatmaps, layer 0 shows the broad triangular wash the entropy
numbers predict, and layer 10 shows the mix of narrow near-diagonal bands and
isolated bright columns far to the left.

---

## 6. Vocabulary ablation: 16k vs 12k

Phase 1 selected a 16,000-piece vocabulary from a sweep whose runner-up was
12,000, using a rule - reject anything over 34% of the parameter budget, reject
anything using under 95% of its vocabulary, take the largest survivor - rather
than a measurement of downstream quality. Phase 2 tests that choice by training
both vocabularies for both languages.

The runs are matched on **parameters, not architecture**. A 12,000-piece
embedding costs 4.61M parameters against 16,000's 6.14M, and leaving the depth
at 11 blocks would have shrunk the model by 6%, confounding vocabulary with
capacity. The freed 1.54M buys most of a twelfth block instead, putting the 12k
model at 25,860,864 parameters against 25,625,856 - within 0.92%. Vocabulary is
then the variable and depth is what absorbs it. Everything else is identical:
same corpus, same splits, same optimizer, same schedule, same seed, one epoch.

| Model | Metric (test split) | 16k | 12k | Change |
|---|---|---:|---:|---:|
| Gujarati | Perplexity - *not comparable* | 51.62 | 44.64 | -13.5% |
| Gujarati | **Bits per byte - comparable** | **0.5764** | **0.5759** | **-0.08%** |
| Nepali | Perplexity - *not comparable* | 38.93 | 33.47 | -14.0% |
| Nepali | **Bits per byte - comparable** | **0.4247** | **0.4236** | **-0.26%** |

Perplexity is per token and the two runs do not share a tokenizer: choosing the
next piece out of 12,000 is an easier question than out of 16,000, so a model
that had learned nothing extra would still post a lower perplexity. Bits-per-byte
divides the same cross-entropy by the bytes of text it explains, and bytes are a
property of the corpus.

**Conclusion.** The Phase 1 choice stands. A 12,000-piece vocabulary buys
0.08-0.26% of bits-per-byte and costs 8.9-13.0% more wall-clock time per epoch,
for two compounding reasons: the same corpus becomes 3.7-4.0% more tokens to
read, and every one of those tokens passes through 12 blocks instead of 11.
chrF++ at the best decoding setting moves -2.9% (Gujarati) and -0.7% (Nepali),
that is, slightly against the 12k run. The 16k run remains the reported Phase 2
model and the checkpoint Phase 3 finetunes from.

The methodological point is the part worth keeping: a 13-14% perplexity drop
that survives no correction for tokenization is easy to claim and easy to
believe, and bits-per-byte is what separates it from a real one.

Full tables and figures: [`vocab_comparison.md`](vocab_comparison.md).

---

## 6b. Bonus: no positional embeddings

The brief's optional ablation, done on Gujarati 16k: the same model retrained
with `positional: none`. RoPE is computed rather than learned, so the ablated
model has the same 25,625,856 parameters, saw the same 518,586,368 tokens over
the same 15,826 steps, and differs from the reported model in exactly one config
line. Removing position costs 8.6% test perplexity and 2.1% bits-per-byte, and
its clearest practical symptom is more looping under greedy decoding.

Everything the ablation needs is in [`report/bonus/`](../bonus/README.md); the checkpoint
is `gujarati/train/runs/gujarati-25m-16k-nopos/`.

## 7. Reproducing

```bash
cd gujarati                                 # or nepali

python3 model/transformer.py --self-test    # RoPE, attention, masking, init
python3 model/param_budget.py               # parameter count, formula vs model
python3 eval/text_metrics.py --self-test    # BLEU / chrF vs sacrebleu
python3 train/prepare_data.py               # splits -> uint16 token streams
python3 train/pretrain.py                   # pretrain (resumable; --max-hours)
python3 eval/run_eval.py                    # the whole evaluation suite
python3 eval/infer.py --prompt "..."        # try the model by hand
```

Then, once both languages have finished:

```bash
python3 report/phase2/make_report_assets.py
```

The configs ship set to the reported 16k run. Reproducing the 12k control means
switching `tokenizer_model`, `run_name`, `data_dir`, `reports_dir` and
`logs_subdir` in `configs/pretrain.yaml` to their `_12k` values, setting
`n_layers: 12` in `configs/model.yaml`, and repeating the commands above; then:

```bash
python3 report/phase2/make_vocab_comparison.py
```

On a hosted GPU the same commands run unchanged, with `--data-dir`, `--runs-dir`
and `--logs-dir` pointed at writable storage.

---

## 8. What is committed

Checkpoints (`*.pt`, ~295 MB per run) are published at
<https://huggingface.co/hardik300/LMA_mini_project/tree/main>, one folder per run;
the link is also in the top-level README. Everything small enough to be evidence
is committed, in two places:

    <language>/logs/pretrain/pretrain_<vocab>/
        train.log, metrics.jsonl, config.json, COMPLETED

    report/phase2/<language>/<language>_<vocab>/
        JSON results, and for the reported run its figures, generated samples
        and model_output.md - the prompts in eval/input.txt answered at each
        temperature, which is where the qualitative reading in Section 5.2 comes from

`<vocab>` is `16k` or `12k`. Each parent directory keeps its plain name and holds
one subdirectory per run, because the two runs write identical filenames.

The 16k run is the reported model and carries the full evidence. The 12k run is
a control for Section 6, so it keeps only the evaluation JSON that comparison reads:
its figures and samples are not committed, because the ablation is answered by
the numbers and the 16k figures already show what attention looks like at this
scale.

The run logs sit under the language directory rather than inside the report,
beside the Phase 1 logs (`logs/collection`, `logs/tokenizer`), because that is
where a log for this pipeline belongs. The report reads `metrics.jsonl` from
there to draw the loss curves, so the figures and the log cannot disagree.
