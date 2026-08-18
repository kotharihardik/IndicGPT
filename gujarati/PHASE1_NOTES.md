# Phase 1 — Gujarati (Model H): my working notes

Not the submission README. This is the "what did I actually do and why" file, in
plain language, so I can defend every choice in the evaluation.

**Status: Phase 1 complete.** Final number that matters:

> **580,463,746 training tokens, of which 183,415,487 (31.60%) are manually
> collected. Requirement is >=20%. PASS.**

Final run: 16 Aug 2026, vocabulary 16,000. Six news sources were withdrawn
before this run (see §2), and the vocabulary was changed from 32,000 to 16,000
on the sweep evidence (see §6), which is why these numbers differ from earlier
drafts of this file.

---

## 1. What Phase 1 asked for

- Build a Gujarati corpus of ~500M tokens **after tokenization**
- **At least 20% of those tokens must be collected by me**, not downloaded
  from an existing dataset
- Clean it, remove duplicates, split into train/validation/test
- Train a tokenizer from scratch (no pretrained ones)
- Report statistics

The 20% is counted in *tokens*, not characters or files. That matters, because
during collection the tokenizer doesn't exist yet, so every progress number was
an estimate until the very end.

---

## 2. Data collection

### The problem I started with

My first collector was a normal web crawler: fetch a page, pull the text out,
find its links, sleep 2 seconds, repeat.

It ran for **3 hours 16 minutes** and produced **10.3M characters**. That is
**0.9 K characters/second**. At that speed the 20% floor needs about **five
days** of nonstop crawling. It also stopped early — at 3,777 pages out of a
planned 30,000 — because link-following from a news homepage only reaches
recent articles and then runs out.

Three things were wrong:

1. A 2-second sleep after every page. Politeness was implemented as *going
   slowly*, which caps throughput by design.
2. **Every page was downloaded twice** — once for its text, once again to read
   its links. Half of all requests were wasted.
3. Link-following can't reach a news archive, so the page budget was
   unreachable no matter how long it ran.

Meanwhile the book/OCR path had collected **zero**. archive.org served PDFs at
10–300 KB/s (the log's own ETA said **341 hours**), and Tesseract on this
machine has no Gujarati model, so no scan could have been read anyway.

### The fix: WordPress REST API

I probed **63 Gujarati domains**. The useful discovery: five Gujarati
publishers run WordPress, which exposes a public read API.

```
GET /wp-json/wp/v2/posts?per_page=100
```

returns **100 complete articles as JSON** in one request — no navigation, no
ads, no scripts. It's WordPress's own documented endpoint, on by default, and
`robots.txt` allows it on all five sites.

Measured difference:

| Method | Speed |
|---|---|
| WordPress REST API | **~570 K chars/s** |
| sitemap + fetch each page | ~7 K chars/s |
| my original crawler | 0.9 K chars/s |

**~600× faster**, and this is not aggression — it's the opposite. One request
returns 100 documents, so I make *fewer* requests than a browser would. All
550M characters took about **2,700 requests total** and **28 minutes**.

The single clearest example: `gujarativishwakosh` gave me **20,743 documents
from 220 requests** = 94 documents per request.

### Sources I actually used

13 manual sources plus one public dataset. Raw characters collected, and the
tokens each contributed to the final train split:

| Source | Method | What it is | Raw chars | Train tokens |
|---|---|---|---:|---:|
| `ai4bharat_sangraha` | public download | The public dataset (the ~68% I did not collect) | 1.60B | 397.0M |
| `abtakmedia` | wp_rest | Rajkot news | 160.0M | 36.2M |
| `ia_books_fulltext` | ia_fulltext | **archive.org books, all eras** | 152.8M | 45.2M |
| `westerntimesnews` | wp_rest | Ahmedabad news + lifestyle | 95.0M | 21.0M |
| `gujarativishwakosh` | wp_rest | **Gujarati encyclopedia** | 85.0M | 21.0M |
| `gujarattoday` | wp_rest | General news | 73.2M | 15.7M |
| `opinionmagazine` | sitemap | Opinion and essays | 70.0M | 17.0M |
| `gujaratabroad` | wp_rest | Diaspora news | 19.8M | 4.6M |
| `aksharnaad` | sitemap | Literature | 16.2M | 4.1M |
| `divya_bhaskar` | sitemap | National daily | 10.3M | 2.3M |
| `layastaro` | sitemap | Poetry | 8.6M | 1.7M |
| `webdunia_gujarati` | sitemap | Mixed portal | 4.1M | 0.9M |
| `ia_books_pdf` | pdf text layer | Internet Archive books, PDF text layer | 3.7M | 1.1M |
| `sandesh` | sitemap | Second newspaper group | 0 | 0 |

**Six sources I withdrew (16 Aug).** `mantavyanews`, `gujaratmirror`,
`loksansar`, `sunvillasamachar`, `eaglenews`, `gstv` — together 248.9M
characters, all WordPress news. The corpus was already dominated by news
registers and comfortably over target, so removing them improved the mix
without risking the floor. Their files were **moved** to `data/archive/manual/`,
not deleted, so the run stays reproducible. The pipeline reads `data/raw/`, and
`data/archive/` sits outside it, so they simply stop being seen.

**Why the encyclopedia and the books matter more than their size.** Nearly
everything else is news. If the tokenizer only ever sees news, it fits news
vocabulary and handles academic writing badly. The evidence is in how well each
source compresses (lower = more varied vocabulary, so the tokenizer works
harder):

```
ia_books_fulltext   2.411 chars/token   <- books, by far the most varied
layastaro           3.054               <- poetry
gujarativishwakosh  3.425               <- encyclopedia
ai4bharat_sangraha  3.715               <- public web text
abtakmedia          3.962               <- news
gujarattoday        4.085               <- news, most predictable
```

Books compress at 2.411 against news at 4.085. That gap is the whole argument
for collecting them: they are genuinely different language, not more of the
same.

### Sources I rejected, and why

Keep these — "I checked and it doesn't work" is part of the deliverable.

| Source | Why rejected |
|---|---|
| `loksatta.com` | API works, **992,573 posts** — but it's **Marathi**. Gujarati ratio 0.00 |
| `deshgujarat.com` | 78,394 posts — but written in **English**. Ratio 0.01 |
| `matrubharti.com` | 296,146 sitemap URLs but JavaScript-rendered — 0/15 pages gave any text |
| `gujaratsamachar.com` | Also JS-rendered, 1/15 pages worked |
| `readgujarati.com`, `oneindia.com` | **robots.txt says no.** Not crawled |
| `gu.wikisource.org` | Wikipedia text is already inside Sangraha — would just be deduplicated away |

**Lesson worth remembering:** the two biggest reservoirs I found were the wrong
language. A working API is not a source. Always check script ratio first.

### Politeness (be ready to defend this)

- `robots.txt` is checked **at every run**, not once. Reason: `sandesh.com`
  said DISALLOW on one probe and ALLOW on a later one. Things change, so the
  code re-checks instead of trusting a comment.
- Concurrency is capped per host (5–10 in flight). Speed comes from *batching
  100 articles per request*, not from hammering servers.
- Real contact email in the User-Agent.
- The code **refuses to run** if the User-Agent still has a placeholder address.

---

## 3. Cleaning (preprocessing)

Everything here is Gujarati-specific and standard practice (the filters come
from the Gopher / C4 / RefinedWeb playbook — these are what real corpora use).

**Normalisation:**
- **NFC** Unicode form — so `જ઼` is always one code point, never two spellings
- **Remove ZWJ (U+200D), keep ZWNJ (U+200C)** — Gujarati never needs the joiner,
  so stray ZWJ just creates duplicate spellings of the same word. ZWNJ *is*
  meaningful (it blocks a conjunct), so it stays
- **Gujarati digits ૦-૯ → 0-9** — rare in modern prose, and they'd otherwise
  waste vocabulary slots
- Strip URLs and emails, fold fancy quotes/dashes

**Filters (rejects the document):**
- ≥85% of letters must be **Gujarati Unicode block** (U+0A80–U+0AFF)
- <5% Devanagari (catches Hindi/Marathi pages leaking in)
- <15% Latin
- Repetition checks: repeated lines, and the most common 2/3/4-word phrase
  can't dominate the text
- Must contain a few common Gujarati words (menus and name-lists fail this)
- Boilerplate lines appearing in ≥50 documents get stripped

**Why Unicode blocks instead of a language-ID model?** The spec bans pretrained
models. Gujarati and Devanagari are separate Unicode blocks, so this is *exact*,
not probabilistic. Nice answer if asked.

**Result: 864,122 -> 822,113 documents (95.1% kept), 2.16B characters.**

Top rejections: `not_gujarati` 28,867 · `devanagari_leakage` 5,725 ·
`repetitive_4gram` 2,537 · `too_short` 1,880. The top two are both script
purity, which is exactly what you would expect for an Indic corpus, and
together they are 82% of everything dropped.

Worth comparing with Nepali, which keeps only 90.6%. Gujarati can be identified
by its Unicode block alone, so `not_gujarati` is an exact test. Nepali shares
Devanagari with Hindi and has no such test — its equivalent bucket is 102,027
documents that are genuinely Devanagari but not Nepali, catchable only by a
word-level test. Same pipeline, very different amounts of work landing on the
language filter.

---

## 4. Deduplication

Two passes, both standard:

1. **Exact** — SHA1 of whitespace-collapsed lowercased text. Cheap, catches the bulk.
2. **Near-duplicate** — MinHash + LSH, 5-word shingles, Jaccard threshold 0.8.
   Catches syndicated news reprinted with a changed headline.

**Runs BEFORE splitting.** Critical: if a near-copy of a test document sits in
train, every Phase 2 perplexity number is inflated and meaningless.

Manual documents are streamed **first**, so if a manual and a public document
collide, the manual copy survives and the 20% quota doesn't erode.

**Result: 822,113 -> 809,290 (1.6% removed): 2,841 exact, 9,982 near-duplicate.**

Near-duplicates outnumber exact ones 3.5:1. That ratio is the justification
for running MinHash at all: those 9,982 documents are syndicated stories with
changed headlines, which a hash comparison structurally cannot catch.

### "Why did so little get removed? Professor said ~20%"

This is the question I expect to be asked. I audited it instead of guessing.

I re-ran duplicate detection on **60,000 random documents from the final
corpus**, using **4× stronger settings** than the pipeline (256 permutations
vs 64):

```
Exact duplicates surviving : 0      (0.000%)
Near  duplicates surviving : 10     (0.017%)
```

So dedup caught essentially everything. The low number is **real**, and here's why:

1. **Sangraha is already deduplicated** by AI4Bharat before release — that's
   75% of my corpus arriving pre-cleaned.
2. **My collection is URL-unique by construction.** The WordPress API returns
   each article exactly once. A normal crawler hits the same article via 5
   different URLs (AMP, print view, tracking parameters) — *that's* where crawl
   duplication comes from.
3. **I'm not using raw Common Crawl.** The 20–50% figure is from raw crawl
   dumps (C4 removed ~30%, RefinedWeb ~50%). Correct for that setting, doesn't
   apply to mine.

The 10 survivors were weekly horoscope columns republished with new dates —
same template, genuinely different content. Arguably correct to keep.

**One weakness I found and fixed:** the pipeline used `num_perm=64`. MinHash's
error is about `1/√num_perm` = 12.5% at 64, which is noisy enough to miss real
duplicates. Standard is 128+. I raised the default to 128. The audit says
re-running would recover only ~0.017%, so I didn't burn 7.5 hours redoing it —
but the code is now defensible.

---

## 5. Splits

| Split | Docs | Chars | Tokens | Manual % |
|---|---:|---:|---:|---:|
| train | 803,151 | 2,112,674,786 | 580,463,746 | 31.60% |
| validation | 3,156 | 8,000,377 | 2,180,775 | 27.60% |
| test | 2,983 | 8,001,679 | 2,203,661 | 36.15% |

- **Document-level**, never sentence-level (half an article in train and half
  in test would be leakage)
- Deterministic — split decided by hashing the document id with a fixed seed,
  so it's reproducible
- Validation/test **capped at 8M characters** so Phase 2 doesn't waste GPU time
  evaluating on a huge set

**"Held out"** just means kept out of training. The tokenizer is fitted on
train only; all tokenizer statistics are measured on validation. Otherwise I'd
be testing memory, not ability. Don't touch `test` until the very end.

---

## 6. Tokenizer

**SentencePiece Unigram, 16,000 vocab, trained from scratch on train only.**

### Why Unigram, not BPE

Gujarati glues suffixes onto stems: `ઘર → ઘરમાં → ઘરોમાં → ઘરોમાંથી`.

BPE merges greedily by frequency, so the same stem gets cut differently in
different words (`ઘર|ોમાં` here, `ઘરો|માં` there) — splitting one stem's
evidence across two embedding rows. Unigram picks the most-likely segmentation,
which lands on real morpheme boundaries.

Verified on my own tokenizer — stem spelled identically every time:
```
▁ઘર | ▁ઘરમાં | ▁ઘરોમાં | ▁ઘરોમાં|થી
```

### Settings that matter

| Setting | Why |
|---|---|
| `normalization_rule_name="identity"` | **Critical.** My preprocessing already normalised. SentencePiece's default would normalise *again, differently*, so the tokenizer wouldn't match the corpus |
| `byte_fallback=True` | Nothing is unrepresentable, so UNK ≈ 0 **by construction** — which is why I report **byte-fallback rate (2.1%)** as the real coverage number |
| `required_chars` = all **91** Gujarati code points | Indic tokenizers break badly if a rare matra or virama is missing — every word containing it degrades to bytes |
| `split_by_unicode_script=True` | No piece straddles Gujarati and Latin |
| `split_digits=True` | Digits individually, not memorised numbers like "2024" |

### Why 16,000 and not 32k/48k

This is the decision I am most likely to be quizzed on, so here it is in full.

Vocabulary costs parameters. Input and output embeddings are **tied** (they
share one matrix), so the cost is `vocab x d_model`. At `d_model=384` one
Transformer block costs `12 x 384^2 = 1.77M`. Against a ~25M budget, measured
on 3,000 held-out documents:

| Vocab | chars/token | fertility | vocab used | Embedding | % of budget | Layers left |
|---:|---:|---:|---:|---:|---:|---:|
| 8,000 | 3.346 | 1.749 | 97.7% | 3.07M | 12% | ~12.4 |
| **16,000** | **3.668** | **1.595** | **98.6%** | **6.14M** | **25%** | **~10.7** |
| 24,000 | 3.830 | 1.528 | 98.2% | 9.22M | 37% | ~8.9 |
| 32,000 | 3.933 | 1.488 | 96.8% | 12.29M | 49% | ~7.2 |
| 48,000 | 4.067 | 1.439 | 92.4% | 18.43M | 74% | ~3.7 |

**The trap: fertility improves at every single step.** If I picked on fertility
alone I would pick 48,000 every time, because bigger is always "better" by that
one number. That is the wrong answer here, and the marginal-cost table shows why:

| Step | Compression gained | Parameters spent per 1% gained |
|---|---:|---:|
| 8,000 -> 16,000 | 9.6% | **319 K** |
| 16,000 -> 24,000 | 4.4% | 696 K |
| 24,000 -> 32,000 | 2.7% | 1.14 M |
| 32,000 -> 48,000 | 3.4% | 1.80 M |

The step *into* 16,000 is more than twice as efficient as any step past it.
That is the knee of the curve.

Three reasons bigger is worse **at this model size**:

1. **Embedding parameters are lazy.** A token looks up *one* row; the other
   15,999 do nothing for it. A Transformer parameter is used by *every* token.
   Same cost, very different value.
2. **Depth is what Phase 3 needs.** The finetuning task is transitive reasoning
   (A>B, B>C therefore A>C). Chaining needs layers. Four layers cannot compose.
3. **Rare rows barely train.** At 48,000, 7.6% of pieces never appear at all in
   held-out text — roughly 3,600 embedding rows that would stay near their
   random initialisation.

**A bigger vocabulary does NOT mean more diverse output.** Any Gujarati word can
be built from pieces either way; bigger just means fewer pieces per word. The
vocabulary is a set of building blocks, not a word list.

**And the argument that settles it:** Phase 2 reports **bits-per-byte (BPB)**,
which the spec says exists "to make the two languages comparable when tokenizers
differ". BPB is tokenizer-invariant. A bigger vocabulary makes *perplexity* look
better and does nothing for BPB. So tuning vocabulary for fertility is chasing
an artifact of the measurement.

The selection rule I actually used, applied by `tokenizer_stats.py`:

> Reject anything whose embedding exceeds 34% of the parameter budget. Reject
> anything using under 95% of its vocabulary on held-out text. Take the largest
> survivor.

For Gujarati that rejects 24,000 and 32,000 on budget, 48,000 on utilisation,
leaving **16,000**. Nepali's sweep, run completely separately, landed on 16,000
too.

### A mistake worth recording: 32,000 was in the config for weeks

During collection the config said `vocab_size: 32000`, and that was deliberate
but for a completely different reason. A bigger vocabulary compresses better, so
it produces **fewer** tokens from the same text. Of all the candidates, 32,000
therefore gave the smallest token count and was the hardest 500M target to hit.
Sizing the corpus against it meant every smaller vocabulary would clear 500M
automatically.

That job finished the moment the corpus hit 540.8M tokens at 32,000. But nothing
in the code goes back and changes the number afterwards: **stage 5 prints a
recommendation and stage 6 just reads the config.** They can silently disagree,
and they did, for weeks. I only caught it when I re-read the pipeline output.

Re-running the last four stages at 16,000 raised the count from 540.8M to
**580.5M tokens** from exactly the same 2.11B characters, and the manual share
did not move at all (31.60% -> 31.60%).

### Tokens at every vocabulary size (measured, not estimated)

I tokenized the whole train split once per candidate model and counted manual
and public separately, rather than dividing characters by an average rate:

| Vocab | Manual tokens | Public tokens | Total | Manual % | vs 500M |
|---:|---:|---:|---:|---:|---:|
| 8,000 | 200,955,045 | 435,344,760 | 636,299,805 | 31.58% | 127% |
| **16,000** | **183,376,003** | **397,057,005** | **580,433,008** | **31.59%** | **116%** |
| 24,000 | 175,572,066 | 380,193,823 | 555,765,889 | 31.59% | 111% |
| 32,000 | 170,797,331 | 370,047,723 | 540,845,054 | 31.58% | 108% |
| 48,000 | 164,917,458 | 357,887,054 | 522,804,512 | 31.54% | 105% |

Two things to take from this:

1. **Every vocabulary size clears 500M.** That was the point of sizing the
   corpus against the worst case.
2. **The manual share barely moves** — 31.54% to 31.59% across a 6x range of
   vocabulary. Vocabulary changes the token count by 114M but not the
   provenance ratio, so the 20% requirement is safe whatever I pick. Manual and
   public text do compress slightly differently, which is why I measured this
   instead of assuming.

(These are measured with the *sweep* models. The final 16,000 model gives
580,463,746 rather than 580,433,008 — a 0.005% difference, because it is a
separate training run of the same size. Immaterial, but that is why two numbers
for 16,000 appear in my files.)

### The metrics, what they mean, and what they told me

`project_doc.md` §1.3 asks for vocabulary size, token-frequency statistics,
average characters per token, tokenization examples, and unknown-token
statistics, and says to choose using fertility / unknown-token rate. All of that
is computed. Here is what each number actually means, in plain terms:

| Metric | What it means | Gujarati (16k) | What I read from it |
|---|---|---:|---|
| **chars/token** | Average characters each token covers. Higher = better compression | 3.668 | Sensible for an Indic script. Rises with vocabulary, which is why it cannot be the only criterion |
| **fertility** | Tokens per whitespace word. Lower = words stay whole | 1.595 | A Gujarati word costs ~1.6 tokens. Below 1.5 only happens at vocabularies too big to afford |
| **UNK rate** | Share of characters the tokenizer cannot represent | **0.000000** | **Useless here, and I need to say why.** With `byte_fallback=True` an unknown character becomes `<0xNN>` byte pieces instead of `<unk>`, so zero is true *by construction*, not an achievement |
| **byte-fallback rate** | Share of tokens that had to drop to raw bytes | **2.34%** | The metric that actually measures coverage. Mine is above the 1% bar I set and the script warns about it — see below |
| **vocabulary utilisation** | Share of pieces that appear at least once in held-out text | 98.64% | Almost every piece earns its place. At 48,000 this falls to 92.4%, meaning ~3,600 dead rows |
| **hapax pieces** | Pieces seen exactly once in held-out text | 45 | Very low. Confirms the vocabulary is not padded with junk |
| **embedding parameters** | vocab x d_model, the Phase 2 cost | 6.14M (25%) | The number that actually drives the decision |
| **chars/token by source** | Compression per source | 2.411 to 4.085 | Proof the corpus has real register diversity, not one house style repeated |

**The one result I am not happy with: byte-fallback is 2.34%.**

It sits at 2.1-2.6% at *every* vocabulary size, so it is a property of the
corpus, not of the tokenizer. Nepali's is 0.58-0.69%, roughly four times better.

The cause shows up in the held-out script share: my validation text is **78.7%
Gujarati script**, against Nepali's 83.0% Devanagari. Every Gujarati code point
is pinned into the vocabulary by `required_chars`, so Gujarati letters *cannot*
fall back to bytes. What is falling back is Latin text that survived
preprocessing — English brand names, URLs fragments, transliterations embedded
in Gujarati news copy.

So this is a **preprocessing finding, not a tokenizer fault**. The fix would be
tightening `max_latin_ratio` (currently 0.15) and re-running. I have not done it
because it changes the corpus and the corpus is already over target, but I
should be able to say all of this if asked.

### Final tokenizer, measured on held-out validation

| Metric | Value |
|---|---|
| Algorithm | SentencePiece Unigram |
| Vocabulary | 16,000 |
| Trained on | train split only, 100M character sample |
| chars/token | 3.668 |
| Fertility | 1.595 tokens/word |
| UNK rate | 0.000000 (by construction) |
| **Byte-fallback rate** | **2.34%** <- the real coverage metric |
| Vocabulary used on held-out | **98.64%** |
| Hapax pieces | 45 |
| Embedding cost | 6.14M params (25% of 25M budget) |
| Held-out set | 3,000 documents, 7,634,430 characters |

Files: `tokenizer/model/16k_unigram_guj.model` and
`tokenizer/vocab/16k_unigram_guj.vocab`. The five sweep candidates live in
`tokenizer/model/sweep/`.

## 7. Corpus statistics

Final train split:

```
Documents          : 803,151
Characters         : 2,112,674,786
Words              : 359,528,943
Tokens             : 580,463,746   <- the number the 20% is measured against
Manual tokens      : 183,415,487   (31.60%)
Public tokens      : 397,048,259   (68.40%)
```

**Words are not tokens.** This confused me more than once. Whitespace words are
359.5M, which is *below* 500M and looks alarming until you remember the target
is in tokens. Subword tokenization raises the count: 359.5M words become 580.5M
tokens, a fertility of 1.615. The stage tables report words because you can
compare them across stages before a tokenizer exists; the 500M target is only
ever about the final token count.

**Why the manual share rises from 29.5% of words to 31.6% of tokens.** Manual
text (books, encyclopedia, poetry) has richer vocabulary, so it breaks into more
pieces per word than public web text does. Same documents, different ratio,
depending on which unit you count in. Worth being able to explain, because the
requirement is specifically in tokens.

### The pipeline's shrinkage, end to end

| Stage | Docs | Chars | Words | Kept |
|---|---:|---:|---:|---:|
| Collected (raw) | 864,122 | 2,298,677,721 | 388,693,175 | — |
| After cleaning | 822,113 | 2,161,762,634 | 367,654,754 | 94.6% of words |
| After dedup | 809,290 | 2,128,676,842 | 362,250,461 | 98.5% of words |
| Train split | 803,151 | 2,112,674,786 | 359,528,943 | 99.2% of words |
| **Total lost** | **7.1% docs** | **8.1% chars** | **7.5% words** | |

Split manual vs public, because that is what the 20% audit needs:

| Stage | Manual docs | Manual chars | Public docs | Public chars |
|---|---:|---:|---:|---:|
| Collected | 214,898 | 698,677,286 | 649,224 | 1,600,000,435 |
| After cleaning | 207,805 | 656,699,641 | 614,308 | 1,505,062,993 |
| After dedup | 204,358 | 646,071,023 | 604,932 | 1,482,605,819 |
| Train split | 202,746 | 641,185,183 | 600,405 | 1,471,489,603 |

Manual holds 29.4% -> 29.5% of words all the way through. Nothing in cleaning or
dedup preferentially eats manual text, which is the thing I was worried about
and the reason `deduplicate.py` streams manual documents first.

## 8. Bugs I hit and fixed (good stories for the viva)

**1. `Document.id` wasn't unique — the dangerous one.**
The id was built from `source_url`, but `collect_public.py` gives *every*
Sangraha document the same URL (the dataset landing page). So **614,326
documents shared one id**. Dedup crashed on it (`ValueError: key already
exists`). The crash was the *lucky* part — `build_splits.py` assigns splits by
hashing the id, so without the crash **the entire public corpus would have
landed in one split as a single block**, and I'd never have noticed. Fixed: id
now hashes source + URL + text.

**2. The pipeline silently used a truncated corpus.**
My chain script piped output through `tail`, which threw away the exit code, so
after dedup crashed it happily ran `build_splits.py` on a half-written file.
The splits *looked* fine — 119,523 documents, plausible sizes — but were the
first 14% of the corpus and 100% manual. Fixed with `set -euo pipefail` and no
pipe. **Lesson: a stage that fails must stop the pipeline.**

**3. Tokenizer training ate 18 GB on an 8 GB machine.**
SentencePiece Unigram builds a suffix array; peak memory is ~35–40× the input
text. 470M characters → 18 GB → ~30 GB of swap → disk down to 8.7 GB free.
Killed it, dropped the training text to 100M chars (peak ~3.7 GB). Note: this
limits only tokenizer *fitting* — corpus statistics still cover 100% of data.

**4. Preprocessing would have run out of RAM.**
Pass A counts every distinct line across 889k documents — millions of strings,
several GB. Added periodic pruning of singletons (lossy counting). Real
boilerplate appears thousands of times so it survives pruning. Memory stayed at
**14 MB** instead of gigabytes.

**5. One dead URL cost 3.7 of 6.3 minutes.**
`gstv.in/sitemap_index.xml` doesn't exist and *hangs* instead of returning 404.
Retried 4× at a 45s timeout. Fixed: guessed URLs get one short attempt, no retries.

**6. Wrong deny pattern.** I wrote `/webstory`; the real path is `/web-stories/`.
That was 15 of 17 network failures (all HTTP 500).

**7. An 860K-char book was thrown away as "too long."**
The length filter is sized for web pages. Books are now split at paragraph
boundaries instead of rejected.

**8. Mislabelled OCR.** `ia_books_pdf` was labelled `manual_ocr`, but **no OCR
ever ran** — Tesseract has no Gujarati model, so it was PyMuPDF text-layer
extraction only. Added a separate `manual_pdf` label. Don't call it OCR.

**9. Tokenizer training text was biased.** Shards are read in filename order,
so capping the export would have trained the tokenizer on ~100% `abtakmedia`.
Fixed by subsampling whole documents by probability. It now prints its
composition, which is how I verify it:
```
ai4bharat_sangraha 74.5% · abtakmedia 7.6% · mantavyanews 6.4% ...
```
matching the corpus proportions.

**10. Example sentences weren't normalised.** They were tokenized raw, showing
byte-fallback on an em-dash that preprocessing actually removes. Fixed — took
one example from 21 tokens to 18.

**11. My own vocab recommendation was wrong at first.** It said 48,000 because
it only looked at fertility — which always favours the largest vocab. Added the
parameter-budget and vocabulary-utilisation constraints. Now says 16,000.

---

**12. The config and the sweep silently disagreed for weeks.** Stage 5 prints
"RECOMMENDED vocab_size = 16,000" and stage 6 reads `vocab_size` from
`configs/pipeline.yaml`. Nothing connects them. The config said 32,000, so the
final tokenizer was trained at 32,000 while the pipeline's own output said 16,000
on the same screen. It ran that way through a complete pipeline before I noticed.
Fixed by setting 16,000 and writing the evidence into the config comment, but the
real lesson is that a script printing advice it does not act on is a trap.

**13. Two different reports were writing to the same filename.**
`tokenizer_stats.py --compare-sweep` produces the five-way comparison;
`tokenizer_stats.py` produces the final model report. Both wrote to
`tokenizer_stats.json`, so whichever ran last destroyed the other. My pipeline
runs them in that order, so I lost the sweep comparison — the exact table that
justifies the vocabulary choice — without any error message. Fixed: the sweep
now writes `tokenizer_sweep.{json,md}`. Silent overwrites are the worst kind of
bug because nothing looks wrong.

**14. Running a script with `--help` started a real job.** `build_splits.py` has
no argparse, so it ignored the flag and began rebuilding the splits. A timeout
killed it mid-write and left `train.jsonl.gz` truncated from 468 MB to 107 MB.
Recoverable only because the split is `md5(seed:doc_id)` with a fixed seed, so
rebuilding reproduced it exactly — 803,151 / 3,156 / 2,983 documents, identical.
Determinism is not a nicety; it is what made this a 40-minute annoyance instead
of a disaster.

**15. Reading a progress number too early and drawing a conclusion.** Partway
through dedup I saw exact and near duplicates running about 1:1 and said so. The
final split was 3.5:1. Near-duplicates concentrate in the news sources, which sit
later in filename order. Do not characterise a distribution from the first
quarter of a stream.

## 9. All commands

Run everything from inside `gujarati/`. Plain `python3` works because the venv
is on the path; if it is not, activate it first with
`source ../.venv/bin/activate`.

```bash
cd gujarati
```

### Collect (I did NOT re-run this for the final pipeline)

```bash
# Public dataset (~68% of the corpus). --dry-run first to check the fields.
python3 data/collection/collect_public.py --dry-run
python3 data/collection/collect_public.py

# Manual collection. ALWAYS --check first: robots.txt + reachability, writes nothing.
python3 data/collection/collect_manual.py --check
python3 data/collection/collect_manual.py            # collect (safe to Ctrl-C, resumes)
python3 data/collection/collect_manual.py --status   # progress vs the 20% floor

# Useful variants
python3 data/collection/collect_manual.py --only aksharnaad --limit-chars 1000000
python3 data/collection/collect_manual.py --max-hours 2      # time-boxed

# Downloads the book PDFs that the `pdf` method reads (Gujarati only)
python3 data/collection/fetch_ia_pdfs.py --dry-run
python3 data/collection/fetch_ia_pdfs.py --max-items 700 --max-gb 20
```

**Why collection is not re-run.** Sites publish new articles constantly, so
running it again would *grow* the corpus rather than reproduce it. `data/raw/`
is the corpus as collected, and everything downstream is deterministic from it.

### The 9 commands of an actual pipeline run

```bash
# 1. Clean: normalise Unicode, strip boilerplate, drop bad documents  (~60 min)
python3 data/preprocessing/preprocess.py

# 2. Deduplicate: exact SHA1, then MinHash near-duplicates            (~60 min)
python3 data/preprocessing/deduplicate.py

# 3. Split: train / validation / test, deterministic                  (~24 min)
python3 data/splits/build_splits.py

# 4. Train all 5 candidate vocabularies                               (~11 min)
python3 tokenizer/train_tokenizer.py --sweep

# 5. Compare them, print the recommendation                           (~1 min)
python3 tokenizer/tokenizer_stats.py --compare-sweep

#    <- set tokenizer.vocab_size in configs/pipeline.yaml from step 5 output!
#       Nothing does this automatically. See bug 12.

# 6. Train the final tokenizer at that size                           (~3 min)
python3 tokenizer/train_tokenizer.py

# 7. Final tokenizer report + figures                                 (~1 min)
python3 tokenizer/tokenizer_stats.py

# 8. THE GRADED NUMBER: exact tokens and manual %                     (~3 min)
python3 data/splits/corpus_stats.py --tokenizer

# 9. Per-stage retention, manual vs public                            (~2 min)
python3 data/splits/stage_stats.py
```

Total: about 2 hours 45 minutes. Stages 1, 2 and 3 are the slow ones.

Useful flag: `train_tokenizer.py --reuse-input` skips re-exporting the 246 MB
training text sample, which saves a few minutes when only the vocabulary changed.

### Phase 2 planning

```bash
python3 model/param_budget.py            # exact parameter count for a config
python3 model/param_budget.py --sweep    # layers affordable per vocab size
```

---

## 10. Where things live

```
gujarati/
  configs/pipeline.yaml           every threshold and decision, heavily commented
  data/
    corpus_io.py                  Document schema (the provenance fields live here)
    sources/manual_sources.yaml   all sources, probe results, rejections
    collection/
      collect_manual.py           entry point: drives a collection run
      collect_public.py           entry point: streams the public dataset
      fetch_ia_pdfs.py            downloads the book PDFs the pdf method reads
      adapters/                   ONE MODULE PER COLLECTION METHOD
        base.py                   cleaning, quality gate, resume, Document building
        http.py                   robots.txt + retrying fetch
        progress.py               the live status line
        wordpress.py              wp_rest
        sitemap.py                sitemap + link crawl
        pdf.py                    pdf text layer
        archive_org.py            ia_fulltext
    preprocessing/                preprocess.py, deduplicate.py
    splits/                       build_splits.py, corpus_stats.py, stage_stats.py
  tokenizer/
    model/16k_unigram_guj.model         <- Phase 2 loads this
    vocab/16k_unigram_guj.vocab         <- human-readable piece list
    model/sweep/8k|16k|24k|32k|48k_unigram_guj.model   <- the candidates
    vocab/sweep/...
  model/param_budget.py           Phase 2 parameter calculator
  logs/{collection,preprocessing,tokenizer,stats}/

report/
  phase1_report.md                the submission write-up (both languages)
  gujarati/phase1/                tables, JSON summaries, figures
    corpus_stats.{md,json}        final token counts, per source and per split
    tokenizer_sweep.{md,json}     the five-way vocabulary comparison
    tokenizer_stats.{md,json}     the final tokenizer
    cleaning.json  dedup.json  splits.json
    sweep_token_split.json        manual/public tokens at each vocabulary size
    collection_notes.md           how the collection strategy was chosen
    figures/*.png
```

**Naming.** Tokenizer files are `<size>_<algorithm>_<language>`, so
`16k_unigram_guj.model` describes itself. Sweep candidates go in `sweep/` so the
final model is never lost among them.

---

## 11. Before the evaluation — checklist

**Things I must be able to explain:**
- Why the WordPress REST API is faster (100 articles per request, not aggression)
- Why only 1.6% was deduplicated (Sangraha is pre-deduplicated, and my collection
  is URL-unique by construction; the 20-50% figure is for raw Common Crawl)
- Why Unigram not BPE (morpheme boundaries, stem consistency)
- Why 16k not 48k (embedding competes with layers at 25M params; and BPB, which
  Phase 2 grades, is tokenizer-invariant so fertility tuning is chasing an artifact)
- Why UNK is 0.000000 and why that is *not* a good result (byte_fallback makes it
  true by construction; quote byte-fallback rate instead)
- Why byte-fallback is 2.34% here and 0.65% for Nepali (Latin text surviving
  preprocessing, not a tokenizer fault)
- Why dedup runs before splitting (test-set leakage would inflate Phase 2 PPL)
- Why `ia_books_fulltext` is labelled `manual_web` and not `manual_ocr`

**Honest caveats to state up front, not hide:**
- The tokenizer is fitted on a 100M-character proportional sample, because
  SentencePiece needs ~35-40x the training text in RAM and 470M characters drove
  it to 18 GB on an 8 GB machine. Corpus statistics still cover 100% of the data.
- `ia_books_pdf` is **not** OCR. No Gujarati Tesseract model is installed, so it
  reads embedded PDF text layers and is labelled `manual_pdf` / `pdf_text_layer`.
- `ia_books_fulltext` uses OCR that **archive.org** ran on its own servers. I
  downloaded and cleaned it. Labelled `manual_web` for that reason.
- Byte-fallback at 2.34% is above my own 1% bar. It is Latin text, not missing
  Gujarati coverage, and the fix is a preprocessing change I chose not to make
  because the corpus is already over target.
- **If WordPress REST output were ruled to be "structured data" rather than
  manual collection**, non-WordPress manual would be about 13% of the corpus,
  under the floor. Removing WordPress data cannot fix this — it shrinks the
  denominator as fast as the numerator. Only more collection would.

**Still to do:**
- [ ] **Commit to git** — everything is still uncommitted
- [ ] Upload corpus + tokenizer to Drive, put the links in the top-level README
- [ ] Regenerate `tokenizer_config.json` if it was written with `--reuse-input`
      (it can end up with an empty training-composition field)
