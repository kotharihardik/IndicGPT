# Nepali (Model L) — Phase 1

Corpus and tokenizer results. Setup and commands are in the [root
README](../../../README.md).

**Corpus download:** [OneDrive](https://iiithydstudents-my.sharepoint.com/:f:/g/personal/hardik_k_students_iiit_ac_in/IgDZ46BkCoOhT74gAOPIOBrCAVnN7uIm5sKRY9rQiqDukj0?e=0hlbPM)

## Why Nepali

It is on the allowed lower-resource list and has a real public web presence. It
is also harder than Gujarati in one specific way that shaped most of the
language-specific code: Nepali shares Devanagari with Hindi, Marathi and
Sanskrit and has **no exclusive characters**, so a script test cannot identify
it.

## Corpus

| | Value |
|---|---:|
| **Total tokens** | **541,960,704** |
| Against the ~500M target | **108.4%** |
| Manual tokens | 116,938,372 |
| Public tokens | 425,022,332 |
| **Manual share** | **21.58%** |
| Documents | 1,239,652 |
| Characters | 2,528,493,333 |
| Manual sources | 23 |

| Split | Documents | Characters | Tokens | Manual % |
|---|---:|---:|---:|---:|
| train | 1,116,564 | 2,276,899,633 | 488,046,826 | 21.59% |
| validation | 61,604 | 125,283,173 | 26,814,777 | 21.18% |
| test | 61,484 | 126,310,527 | 27,099,101 | 21.68% |
| **corpus** | **1,239,652** | **2,528,493,333** | **541,960,704** | **21.58%** |

The corpus exceeds 500M tokens and clears the 20% manual floor.

**On the training split.** Read strictly as the training split alone, Nepali
reaches 488.0M — 11.95M short. The corpus itself clears the target; the
difference is that 10% is held out. That ratio was chosen so Phase 2 gets a
validation set spanning all 23 sources (61,604 documents rather than the ~3,900
a 0.35% split gives), and §1.2 of the brief allows the lower-resource language
to report the maximum feasible amount with a justification. A 96/2/2 split would
return the training split to roughly 520M if the stricter reading is required.

## Sources

One public dataset (AI4Bharat Sangraha) plus 23 collected by hand — far more
sources than Gujarati needed, for less text.

| Source | Category | Documents | Tokens | % |
|---|---|---:|---:|---:|
| `ai4bharat_sangraha` | public dataset, mixed | 978,095 | 425,022,332 | 78.42% |
| `onlinekhabar` | news | 59,094 | 29,989,965 | 5.53% |
| `sahityasangraha_html` | **literature** | 12,877 | 19,794,023 | 3.65% |
| `nepalpress_html` | news | 39,934 | 14,018,135 | 2.59% |
| `reportersnepal_html` | news | 39,336 | 13,516,434 | 2.49% |
| `khabarhub_html` | news | 39,842 | 13,183,948 | 2.43% |
| `clickmandu_html` | business | 18,185 | 7,074,374 | 1.31% |
| *16 smaller sources* | culture, finance, health, legal, … | 52,289 | 19,361,493 | 3.58% |
| **collected directly** | 23 sources | **261,557** | **116,938,372** | **21.58%** |

Collection uses interchangeable adapters chosen per source by what the site
offers: a JSON article endpoint where one exists, sitemap-driven crawling and
link discovery, and text-layer extraction from downloaded PDFs. Every source is
handled the same way regardless — checked for language and terms of use,
`robots.txt` re-verified on every run, markup and site boilerplate stripped,
script ratio and the lexical Nepali test applied — before anything is written.

**Nepali is harder to collect than Gujarati**, and that is a property of the web
rather than a choice. Some Gujarati publishers expose a JSON endpoint returning
complete article bodies; most Nepali news sites run custom CMSes with no
equivalent, so 20 of 23 sources had to be fetched a page at a time and parsed
from HTML. That is why Nepali needed nearly twice as many sources for less
text.

The source list is broad but thin outside news: 11 non-news categories, yet
health, education, agriculture, legal and magazine together are under 1.5% of
the corpus. The largest non-news source is `sahityasangraha_html` at 19.8M
tokens of genuine literary Nepali. `court_pdf` shows the limit at 22 usable
documents — roughly 88% of the Supreme Court PDFs are scanned images with no
text layer, and no Nepali OCR model was available.

## Language identification

Gujarati owns its Unicode block, so "≥85% of letters are Gujarati" is exact.
**Nepali owns nothing** — a Hindi page passes any Devanagari check perfectly.
Since pretrained models are not allowed, identification is **lexical**:

1. the text must be ≥80% Devanagari, ruling out other scripts;
2. it must contain Nepali function words — छ, छन्, हो, भएको, गरेको, लाई;
3. Nepali markers must outnumber Hindi markers (है, हैं, था, किया) by 1.5:1.

On real pages Nepali news scores 11–14 Nepali markers per 1000 characters and
effectively zero Hindi markers; Hindi pages invert that exactly. This is the
most important piece of Nepali-specific code and the most expensive: it rejected
**102,137 documents** that are genuinely Devanagari and still not Nepali.
Gujarati's equivalent buckets total 11,788 — a ninth as many — and are caught by
the script test for free.

## Cleaning

**Normalisation:** NFC; ZWJ stripped but ZWNJ **kept**, since ZWNJ blocks a
conjunct and changes the word; danda (। ॥) spacing regularised; URLs and emails
stripped.

**Two rules follow a ruling from the teaching staff.** Every Latin-script word
is removed, proper nouns included. Digits stay but in Devanagari — `2024`
becomes `२०२४`, reversing an earlier choice of folding native digits to ASCII.

Removal is at word level, not document level, which is why `not_devanagari`
almost disappears from the rejection table: a page with some English now
survives with the English removed rather than being discarded whole.

**Boilerplate removal has three layers:** `trafilatura` for page structure, a
rule for "सम्बन्धित समाचार" link lists that themes inject inside the article
body, and a corpus-wide pass blocking any line appearing in ≥50 documents.

**Result: 1,378,885 → 1,255,276 documents (91.0% kept).**

| Reason | Count |
|---|---:|
| `not_nepali` | **102,137** |
| `repetitive_4gram` | 10,024 |
| `too_short` | 4,810 |
| `repetitive_3gram` | 4,664 |
| everything else | 1,974 |

Gujarati keeps 97.6% at this stage against Nepali's 91.0%, and the whole gap is
the language filter.

## Deduplication

Runs **before** splitting, so no near-copy of a test document can sit in train.
Exact hashing first, then MinHash + LSH (5-word shingles, Jaccard 0.8,
`num_perm=128`). Manual documents stream first so a collision never eats manual
text.

**Result: 1,255,276 → 1,239,652 (1.2% removed): 5,045 exact, 10,579 near.**

The low rate is real for the same reasons as Gujarati — Sangraha ships
deduplicated and sitemap crawling yields each article once.

**A serious bug surfaced here.** `Document.id` was not unique: 614,326 documents
shared a single id. It crashed dedup, but the silent version would have been far
worse — it would have put the entire public corpus into one split. Fixed by
deriving the id from a hash of source + URL + text.

## Splits

90/5/5, assigned per document by `md5(seed:doc_id)` with a fixed seed —
document-level and deterministic, so the pipeline reproduces exactly.

## Tokenizer

SentencePiece **Unigram**, vocabulary **16,000**, trained from scratch on the
training split only. Nothing is shared with Gujarati.

**Why Unigram over BPE.** Nepali marks case and number with chains of suffixes.
Unigram scores a whole segmentation probabilistically, so it cuts where the
morphology cuts and keeps a stem stable across inflections; BPE merges greedily
and can split the same stem two ways. Practically, `required_chars` — which pins
every Devanagari code point into the vocabulary — is incompatible with the
SentencePiece BPE trainer.

Measured on the **complete** validation split (61,604 documents, 125.3M chars):

| Vocab | chars/token | fertility | byte fallback | UNK | vocab used | % of 25M budget |
|---:|---:|---:|---:|---:|---:|---:|
| 8,000 | 4.222 | 1.573 | 0.320% | 0.0 | 98.2% | 12% |
| 12,000 | 4.508 | 1.473 | 0.342% | 0.0 | 98.8% | 18% |
| **16,000** | **4.689** | **1.416** | **0.356%** | **0.0** | **99.1%** | **25%** |
| 24,000 | 4.925 | 1.348 | 0.373% | 0.0 | 99.4% | 37% |
| 32,000 | 5.068 | 1.310 | 0.384% | 0.0 | 99.5% | 49% |

**Why 16,000.** Same rule as Gujarati, applied to an independently run sweep:
reject anything whose embedding exceeds 34% of the 25M parameter budget, reject
anything using under 95% of its vocabulary, take the largest survivor. That
rejects 24,000 and 32,000 and leaves 16,000. Two separate sweeps landing on the
same size is convergence, not copying — the vocabularies share nothing.

**UNK is 0.0 at every size and means nothing** — with `byte_fallback=True` an
unknown character becomes `<0xNN>` byte pieces rather than `<unk>`. The real
coverage metric is the **byte-fallback rate, 0.356%**, measured line by line
because the tokenizer is fitted on one line per paragraph and has no piece for a
newline.

The tokenizer is fitted on 248.4M characters (10.9% of train), sampled
proportionally across sources. **This caps only vocabulary fitting** — all token
counts and metrics above are computed over 100% of the relevant split.

### Examples

**News prose with digits** — 17 tokens, 6.06 chars/token, 1.06 tokens/word:

> नेपाल सरकारले २०८१ सालको बजेटमा शिक्षा र स्वास्थ्य क्षेत्रका लागि छुट्याइएको रकम बढाउने निर्णय गरेको छ।

`▁नेपाल | ▁सरकारले | ▁२०८१ | ▁सालको | ▁बजेटमा | ▁शिक्षा | ▁र | ▁स्वास्थ्य | ▁क्षेत्रका | ▁लागि | ▁छुट्याइएको | ▁रकम | ▁बढाउने | ▁निर्णय | ▁गरेको | ▁छ | ।`

Every word is one token, including the long derived form `छुट्याइएको`. At 1.06
tokens per word this is near the practical floor.

**Academic prose** — 22 tokens, 5.59 chars/token, 1.69 tokens/word:

> यस अनुसन्धानपत्रमा लेखकले भाषाशास्त्रीय दृष्टिकोणबाट नेपाली क्रियापदका रूपान्तरणको प्रक्रियाको विश्लेषण प्रस्तुत गरेका छन्।

`▁यस | ▁अनुसन्धान | पत्रमा | ▁लेखकले | ▁भाषा | शास्त्र | ीय | ▁दृष्टिकोण | बाट | ▁नेपाली | ▁क्रिया | पद | का | ▁रूपान्तरण | को | ▁प्रक्रिया | को | ▁विश्लेषण | ▁प्रस्तुत | ▁गरेका | ▁छन् | ।`

`भाषा` + `शास्त्र` + `ीय` matches the Gujarati segmentation of the same word
exactly — convergence on shared Sanskritic morphology, from two tokenizers
trained on disjoint corpora.

**Inflection paradigm** — 34 tokens, 3.47 chars/token:

> घर, घरमा, घरहरूमा, घरहरूबाट, घरवालाहरू - एउटै मूल शब्दबाट बनेका यी पाँच रूपहरू हेर्दा मूल शब्द अक्षुण्ण रहेको देखिन्छ।

`▁घर | , | ▁घरमा | , | ▁घर | हरूमा | , | ▁घर | हरूबाट | , | ▁घर | वाला | हरू | ▁- | ▁एउटै | ▁मूल | ▁शब्द | बाट | ...`

`घर` is an independent token in four of five forms, with the plural marker
`हरू` and case endings `मा` / `बाट` as separate pieces.

## Provenance

Every document carries `manual_or_public` from collection onward, so the manual
share is measured rather than estimated. `court_pdf` is labelled `manual_pdf`
not `manual_ocr` — it reads an embedded text layer and no OCR runs.

**One caveat.** The teaching staff indicated that data arriving in a structured
format might count toward the public 80%. On the strictest reading of that the
manual share would fall to **15.53%**. The counter-argument is that this project
selected each source, verified its language and terms of use, fetched and
paginated it, stripped markup, removed boilerplate, filtered on script ratio and
deduplicated — none of which a packaged corpus release requires, which is what
`public_hf` denotes.

## Files here

| Path | Contents |
|---|---|
| `json/` | machine-written stage summaries — the source of every number above |
| `figures/` | vocabulary sweep, piece-frequency curve, piece-length distribution |

Cross-language tables and figures are in [`../tables/`](../tables/) and
[`../figures/`](../figures/).
