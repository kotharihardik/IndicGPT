# Gujarati (Model H) — Phase 1

Corpus and tokenizer results. Setup and commands are in the [root
README](../../../README.md).

**Corpus download:** [OneDrive](https://iiithydstudents-my.sharepoint.com/:f:/g/personal/hardik_k_students_iiit_ac_in/IgDZ46BkCoOhT74gAOPIOBrCAVnN7uIm5sKRY9rQiqDukj0?e=0hlbPM)

## Why Gujarati

It has an **exclusive Unicode block** (U+0A80–U+0AFF) that no other language
uses, so language identification is an exact character test rather than a
guess — which matters because pretrained models are not allowed. Public text is
also plentiful, leaving room to collect a large manual share on top.

## Corpus

| | Value |
|---|---:|
| **Total tokens** | **573,667,143** |
| Against the ~500M target | **114.7%** |
| Manual tokens | 180,350,586 |
| Public tokens | 393,316,557 |
| **Manual share** | **31.44%** |
| Documents | 830,410 |
| Characters | 2,165,416,159 |
| Manual sources | 13 |

| Split | Documents | Characters | Tokens | Manual % |
|---|---:|---:|---:|---:|
| train | 747,765 | 1,951,290,045 | 517,108,008 | 31.51% |
| validation | 41,171 | 107,259,317 | 28,416,795 | 31.64% |
| test | 41,474 | 106,866,797 | 28,142,340 | 29.95% |
| **corpus** | **830,410** | **2,165,416,159** | **573,667,143** | **31.44%** |

Both requirements are met: the corpus exceeds 500M tokens and manual collection
is well above the 20% floor. The manual share varies by only 1.6 points across
splits, which is what a hash-based split should give.

## Sources

One public dataset (AI4Bharat Sangraha) plus 13 collected by hand.

| Source | Register | Documents | Tokens | % |
|---|---|---:|---:|---:|
| `ai4bharat_sangraha` | public dataset, mixed web | 622,831 | 393,316,557 | 68.56% |
| `ia_books_fulltext` | books, all eras | 868 | 48,123,533 | 8.39% |
| `abtakmedia` | Saurashtra news | 65,810 | 38,108,858 | 6.64% |
| `gujarativishwakosh` | encyclopedia | 20,623 | 22,568,398 | 3.93% |
| `westerntimesnews` | city news | 46,022 | 22,143,962 | 3.86% |
| `opinionmagazine` | opinion, essays | 11,149 | 18,015,441 | 3.14% |
| `gujarattoday` | general news | 44,868 | 16,365,808 | 2.85% |
| *7 smaller sources* | literature, poetry, diaspora, PDFs | 18,239 | 15,024,586 | 2.62% |
| **collected directly** | 13 sources | **207,579** | **180,350,586** | **31.44%** |

Collection uses four interchangeable adapters, chosen per source by what the
site offers: a JSON article endpoint where one exists, sitemap-driven crawling
and link discovery, archive.org full-text retrieval, and text-layer extraction
from downloaded PDFs. Every source is handled the same way regardless — checked
for language and terms of use, `robots.txt` re-verified on every run, markup and
site boilerplate stripped, script ratio filtered — before anything is written.

**Why the small sources are kept.** They are far slower to collect per character
and contribute a few percent each, but a corpus of only news sites is a corpus
of house styles. The evidence is in compression: book text runs at 2.97
chars/token against 4.31 for news, so it is genuinely different language.

Six news sources were withdrawn on 16 Aug (248.9 M characters) because
the corpus already over-represented that register. Their files were moved to
`data/archive/` rather than deleted, so the run stays reproducible.

## Cleaning

**Normalisation:** NFC (not NFKC, which would destroy wanted distinctions); ZWJ
removed but ZWNJ kept, since ZWNJ blocks a conjunct and changes the word; URLs
and emails stripped; typographic quotes and dashes folded.

**Two rules follow a ruling from the teaching staff.** Every Latin-script word
is removed, proper nouns included, so the corpus is strictly monolingual. Digits
stay but are written in Gujarati script — `2024` becomes `૨૦૨૪`. The digit rule
reverses an earlier choice of folding native digits to ASCII.

Latin removal is at **word** level, not document level. On 5,057 raw documents,
41.9% contain at least one Latin word but Latin is only **0.99% of characters** —
dropping those documents would discard 41.9% of the corpus to remove 1% of the
text. The cost is that an inflected loanword leaves a stranded case marker;
badly degraded documents are caught by the minimum-word and repetition filters.

**Filters:** ≥85% Gujarati letters, <5% Devanagari, <15% Latin, minimum length,
repeated-line and repeated-n-gram checks, and a common-word check that catches
menus and name lists.

**Boilerplate removal has three layers:** `trafilatura` for page structure; a
rule for "also read" link lists, which themes inject *inside* the article body;
and a corpus-wide pass blocking any line appearing in ≥50 documents. Only the
third finds boilerplate it was never told about.

**Result: 864,122 → 843,660 documents (97.6% kept).** Top rejections:
`devanagari_leakage` 6,208, `not_gujarati` 5,580, `repetitive_4gram` 2,715.

## Deduplication

Runs **before** splitting — a near-copy of a test document sitting in train
would inflate every Phase 2 perplexity number. Exact duplicates go first by SHA1
of whitespace-collapsed text, then MinHash + LSH over 5-word shingles at Jaccard
0.8 catches syndicated news under changed headlines. Manual documents stream
first, so a collision never eats manual text.

**Result: 843,660 → 830,410 (1.6% removed): 3,365 exact, 9,885 near.**

1.6% is low enough to check: re-running a 60,000-document sample at
`num_perm=256` left 0.017% surviving duplicates, so the rate is real. Sangraha
ships pre-deduplicated and API/sitemap collection is URL-unique by construction;
the usual 20–50% figures describe raw Common Crawl.

## Splits

90/5/5, assigned per document by `md5(seed:doc_id)` with a fixed seed —
document-level so no article is divided, and deterministic so the pipeline
reproduces exactly.

## Tokenizer

SentencePiece **Unigram**, vocabulary **16,000**, trained from scratch on the
training split only. Nothing is shared with Nepali.

**Why Unigram over BPE.** Gujarati marks case with chains of suffixes. Unigram
scores a whole segmentation probabilistically, so it cuts where the morphology
cuts and keeps a stem spelled the same way across inflections; BPE merges
greedily and can split the same stem two ways in two words, which would divide
its evidence across two embedding rows. Practically, `required_chars` — which
pins every Gujarati code point into the vocabulary so no word can fall back to
bytes — is incompatible with the SentencePiece BPE trainer.

Measured on the **complete** validation split (41,171 documents, 107.3M chars):

| Vocab | chars/token | fertility | byte fallback | UNK | vocab used | % of 25M budget |
|---:|---:|---:|---:|---:|---:|---:|
| 8,000 | 3.512 | 1.654 | 0.408% | 0.0 | 98.1% | 12% |
| 12,000 | 3.723 | 1.560 | 0.432% | 0.0 | 98.7% | 18% |
| **16,000** | **3.865** | **1.503** | **0.449%** | **0.0** | **99.0%** | **25%** |
| 24,000 | 4.046 | 1.436 | 0.470% | 0.0 | 99.3% | 37% |
| 32,000 | 4.170 | 1.393 | 0.484% | 0.0 | 99.4% | 49% |

**Why 16,000.** Compression improves at every step, so a fertility-only rule
always picks the largest — which is wrong at 25M parameters. The embedding is
`vocab × d_model` and comes out of the same budget as the Transformer blocks: at
32,000 it is 12.29M parameters, 49% of the budget, before a single attention
head exists. The rule applied is: reject anything whose embedding exceeds 34% of
budget, reject anything using under 95% of its vocabulary, take the largest
survivor. That rejects 24,000 and 32,000 and leaves 16,000.

**UNK is 0.0 at every size and means nothing** — with `byte_fallback=True` an
unknown character becomes `<0xNN>` byte pieces rather than `<unk>`, so any
tokenizer configured this way scores zero. The real coverage metric is the
**byte-fallback rate, 0.449%**, and what remains is symbols like `°` and `₹`.
It must be measured line by line: the tokenizer is fitted on one line per
paragraph and has no piece for a newline, so encoding whole documents charges
every paragraph break to byte fallback (2.778% document-wise vs 0.394%
line-wise — 87% of it newlines).

The tokenizer is fitted on 210.5M characters (10.8% of train), sampled
proportionally across sources. The limit is memory; beyond ~10% the machine
pages. **This caps only vocabulary fitting** — all token counts and metrics
above are computed over 100% of the relevant split.

### Examples

**News prose with digits** — 19 tokens, 6.32 chars/token, 1.19 tokens/word:

> ગુજરાત સરકારે ૨૦૨૪ના શૈક્ષણિક વર્ષથી રાજ્યની તમામ પ્રાથમિક શાળાઓમાં માતૃભાષામાં શિક્ષણ ફરજિયાત બનાવવાનો નિર્ણય લીધો હતો.

`▁ગુજરાત | ▁સરકારે | ▁૨૦૨૪ | ના | ▁શૈક્ષણિક | ▁વર્ષથી | ▁રાજ્યની | ▁તમામ | ▁પ્રાથમિક | ▁શાળાઓમાં | ▁માતૃભાષા | માં | ▁શિક્ષણ | ▁ફરજિયાત | ▁બનાવવાનો | ▁નિર્ણય | ▁લીધો | ▁હતો | .`

Almost every word is one token, and the year is a single piece.

**Academic prose** — 24 tokens, 5.00 chars/token, 1.85 tokens/word:

> આ સંશોધનપત્રમાં લેખકે ભાષાશાસ્ત્રીય દૃષ્ટિકોણથી ગુજરાતી ક્રિયાપદોના રૂપાંતરણની પ્રક્રિયાનું વિશ્લેષણ પ્રસ્તુત કર્યું છે.

`▁આ | ▁સંશોધન | પત્ર | માં | ▁લેખક | ે | ▁ભાષા | શાસ્ત્ર | ીય | ▁દૃષ્ટિકોણ | થી | ▁ગુજરાતી | ▁ક્રિયાપદ | ોના | ▁રૂપાંતર | ણ | ની | ▁પ્રક્રિયા | નું | ▁વિશ્લેષણ | ▁પ્રસ્તુત | ▁કર્યું | ▁છે | .`

Compression drops because Sanskritic compounds split into real morphemes —
`સંશોધન` + `પત્ર` + `માં` — so the model gets reusable parts, not rare tokens.

**Inflection paradigm** — 34 tokens, 3.50 chars/token:

> ઘર, ઘરમાં, ઘરોમાં, ઘરોમાંથી, ઘરવાળાં - એક જ મૂળ શબ્દમાંથી બનેલાં આ પાંચ રૂપો તપાસવાથી જણાય છે કે મૂળ શબ્દ અકબંધ રહે છે.

`▁ઘર | , | ▁ઘરમાં | , | ▁ઘરોમાં | , | ▁ઘરોમાં | થી | , | ▁ઘર | વાળાં | ▁- | ▁એક | ▁જ | ▁મૂળ | ▁શબ્દ | માંથી | ...`

The Unigram argument made visible: the stem `ઘર` is recoverable in all five
forms, and `ઘરોમાંથી` is cut as `ઘરોમાં` + `થી`.

## Provenance

Every document carries `manual_or_public` from collection onward, surviving
cleaning, dedup and splitting, so the manual share is measured rather than
estimated. Two labels are deliberately weak: `ia_books_pdf` is `manual_pdf` not
`manual_ocr` (it reads an embedded text layer; no OCR runs), and
`ia_books_fulltext` is `manual_web` because archive.org ran that OCR on its own
servers. Both count either way.

**One caveat.** The teaching staff indicated that data arriving in a structured
format might count toward the public 80%. On the strictest reading of that the
manual share would fall to **13.34%**. The counter-argument is that this project
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
