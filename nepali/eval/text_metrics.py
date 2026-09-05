"""Generation-quality metrics, implemented here rather than imported.

BLEU, chrF++ and ROUGE-L are all counting exercises over n-grams, so they are
written out below. That keeps the evaluation free of extra dependencies on a
hosted GPU worker, and it means the exact variant used is visible rather than
hidden behind a library default.

Everything is word-level or character-level on the *text*, never on tokenizer
pieces. That matters for this project specifically: Model H and Model L have
different vocabularies, so a piece-level score would measure the tokenizers as
much as the models. Words and characters are the same units in both languages.

    BLEU-4     corpus level, modified n-gram precision with a brevity penalty
    chrF++     character 1-6 grams plus word 1-2 grams, beta = 2
    ROUGE-L    longest common subsequence, F1, averaged over pairs
    Distinct-n unique n-grams / all n-grams, over the whole set of generations
    Repetition fraction of n-grams in a generation that are not the first
               occurrence, averaged over generations
"""

from __future__ import annotations

import argparse
import math
import unicodedata
from collections import Counter
from functools import lru_cache
from typing import Dict, List, Sequence, Tuple

#: Unicode categories that belong *inside* a word in these scripts even though
#: they are not letters or digits:
#:   Mn/Mc/Me  combining marks - Gujarati and Devanagari vowel signs, anusvara,
#:             virama. A regex `\w` class does NOT match these, and splitting on
#:             them shatters every word into consonant fragments.
#:   Cf        format characters - the corpus deliberately keeps ZWNJ, which
#:             blocks a conjunct and is part of the spelling.
_WORD_CATEGORIES = frozenset(("Mn", "Mc", "Me", "Cf"))


@lru_cache(maxsize=4096)
def _is_word_char(ch: str) -> bool:
    """True if this character continues a word rather than ending one."""
    return (ch.isalnum() or ch == "_"
            or unicodedata.category(ch) in _WORD_CATEGORIES)


def tokenize_words(text: str) -> List[str]:
    """Split text into word tokens, with punctuation as separate tokens.

    The standard `13a` BLEU tokenizer's behaviour extended to Unicode:
    punctuation is detached from the words it touches, but a combining mark
    stays with its consonant. It goes beyond `13a` in detaching the danda,
    which `13a` does not know about and which is sentence punctuation in both
    of these scripts exactly as a full stop is.
    """
    tokens: List[str] = []
    buffer: List[str] = []
    for ch in text:
        if ch.isspace():
            if buffer:
                tokens.append("".join(buffer))
                buffer = []
        elif _is_word_char(ch):
            buffer.append(ch)
        else:
            if buffer:
                tokens.append("".join(buffer))
                buffer = []
            tokens.append(ch)
    if buffer:
        tokens.append("".join(buffer))
    return tokens


def ngrams(tokens: Sequence[str], n: int) -> Counter:
    """Count every n-gram in a token sequence."""
    return Counter(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))


# BLEU


def corpus_bleu(hypotheses: Sequence[str], references: Sequence[str],
                max_n: int = 4) -> Dict[str, float]:
    """Corpus-level BLEU-4 over one reference per hypothesis.

    Precisions are pooled across the corpus before the geometric mean, which is
    what "corpus BLEU" means and why it is not the average of sentence BLEUs.

    A zero higher-order precision is floored at `1 / (2 * total_n)` instead of
    collapsing the score to zero. Open-ended continuations frequently share no
    4-gram with the reference, and a hard zero would report the same 0.0 for
    every decoding setting, hiding exactly the differences being measured.
    """
    clipped = [0] * (max_n + 1)
    totals = [0] * (max_n + 1)
    hyp_len = ref_len = 0

    for hyp, ref in zip(hypotheses, references):
        h_tokens, r_tokens = tokenize_words(hyp), tokenize_words(ref)
        hyp_len += len(h_tokens)
        ref_len += len(r_tokens)
        for n in range(1, max_n + 1):
            h_counts, r_counts = ngrams(h_tokens, n), ngrams(r_tokens, n)
            clipped[n] += sum(min(c, r_counts[g]) for g, c in h_counts.items())
            totals[n] += max(sum(h_counts.values()), 0)

    precisions, smoothed = [], False
    for n in range(1, max_n + 1):
        if totals[n] == 0:
            precisions.append(0.0)
        elif clipped[n] == 0:
            precisions.append(1.0 / (2 * totals[n]))
            smoothed = True
        else:
            precisions.append(clipped[n] / totals[n])

    if min(precisions) <= 0 or hyp_len == 0:
        score = 0.0
        brevity = 0.0
    else:
        # Precision alone rewards short output, so a hypothesis shorter than
        # its reference is penalised in proportion to how much shorter it is.
        brevity = 1.0 if hyp_len > ref_len else math.exp(1 - ref_len / hyp_len)
        score = brevity * math.exp(sum(math.log(p) for p in precisions) / max_n)

    return {"bleu": 100 * score,
            "precisions": [100 * p for p in precisions],
            "brevity_penalty": brevity,
            "hypothesis_length": hyp_len,
            "reference_length": ref_len,
            "smoothed": smoothed}


# chrF++


def _fscore(matches: float, hyp_total: float, ref_total: float,
            beta: float) -> float:
    """F-score from pooled match counts, 0 when either side is empty."""
    if hyp_total == 0 or ref_total == 0 or matches == 0:
        return 0.0
    precision = matches / hyp_total
    recall = matches / ref_total
    return ((1 + beta ** 2) * precision * recall
            / (beta ** 2 * precision + recall))


def corpus_chrf(hypotheses: Sequence[str], references: Sequence[str],
                char_order: int = 6, word_order: int = 2,
                beta: float = 2.0) -> Dict[str, float]:
    """Corpus-level chrF++ : character n-grams plus short word n-grams.

    chrF suits both languages here: they are heavily inflected, so a correct
    stem with the wrong case ending scores zero on word-level BLEU but keeps
    most of its character n-grams. `word_order = 0` gives plain chrF.

    Following Popovic (2015), precision and recall are averaged over the n-gram
    orders first and one F-score is computed from those averages; an F-score per
    order, averaged, would be a different and non-standard number.

    Whitespace is removed before taking character n-grams, so spacing does not
    count as content. Beta = 2 weights recall twice as heavily as precision,
    the published default.
    """
    orders: List[Tuple[str, int]] = ([("char", n) for n in range(1, char_order + 1)]
                                     + [("word", n) for n in range(1, word_order + 1)])
    stats = {key: [0, 0, 0] for key in orders}      # matches, hyp total, ref total

    for hyp, ref in zip(hypotheses, references):
        h_chars = "".join(hyp.split())
        r_chars = "".join(ref.split())
        h_words, r_words = tokenize_words(hyp), tokenize_words(ref)
        for kind, n in orders:
            h_units = h_chars if kind == "char" else h_words
            r_units = r_chars if kind == "char" else r_words
            h_counts, r_counts = ngrams(h_units, n), ngrams(r_units, n)
            stats[(kind, n)][0] += sum(min(c, r_counts[g]) for g, c in h_counts.items())
            stats[(kind, n)][1] += sum(h_counts.values())
            stats[(kind, n)][2] += sum(r_counts.values())

    # An order with nothing on either side carries no information and is
    # dropped, rather than being counted as a zero that drags the mean down.
    precisions, recalls = [], []
    for key in orders:
        matches, hyp_total, ref_total = stats[key]
        if hyp_total > 0 and ref_total > 0:
            precisions.append(matches / hyp_total)
            recalls.append(matches / ref_total)

    overall = 0.0
    if precisions:
        avg_p = sum(precisions) / len(precisions)
        avg_r = sum(recalls) / len(recalls)
        if avg_p + avg_r > 0:
            overall = ((1 + beta ** 2) * avg_p * avg_r
                       / (beta ** 2 * avg_p + avg_r))
    return {"chrf": 100 * overall, "precision": 100 * (sum(precisions) / len(precisions)
                                                       if precisions else 0.0),
            "recall": 100 * (sum(recalls) / len(recalls) if recalls else 0.0),
            "char_order": char_order, "word_order": word_order, "beta": beta}


# ROUGE-L


def lcs_length(a: Sequence[str], b: Sequence[str]) -> int:
    """Length of the longest common subsequence, with a rolling table."""
    if not a or not b:
        return 0
    previous = [0] * (len(b) + 1)
    for token in a:
        current = [0]
        for j, other in enumerate(b):
            current.append(previous[j] + 1 if token == other
                           else max(current[j], previous[j + 1]))
        previous = current
    return previous[-1]


def corpus_rouge_l(hypotheses: Sequence[str], references: Sequence[str],
                   beta: float = 1.0) -> Dict[str, float]:
    """ROUGE-L: LCS-based F-score, averaged over hypothesis/reference pairs.

    LCS rewards words appearing in the right order without demanding they be
    adjacent, so it is more forgiving than BLEU about the insertions a language
    model makes. Beta = 1 gives the plain F1 that most papers report.
    """
    scores = []
    for hyp, ref in zip(hypotheses, references):
        h_tokens, r_tokens = tokenize_words(hyp), tokenize_words(ref)
        lcs = lcs_length(h_tokens, r_tokens)
        scores.append(_fscore(lcs, len(h_tokens), len(r_tokens), beta))
    mean = sum(scores) / len(scores) if scores else 0.0
    return {"rouge_l": 100 * mean, "beta": beta, "pairs": len(scores)}


# Diversity and repetition


def distinct_n(texts: Sequence[str], n: int) -> float:
    """Unique n-grams over all n-grams, pooled across the whole set.

    Low Distinct-2 is the signature of a model that has collapsed onto a few
    stock phrases; it is a property of the set of generations, not of one.
    """
    seen, total = set(), 0
    for text in texts:
        counts = ngrams(tokenize_words(text), n)
        seen.update(counts.keys())
        total += sum(counts.values())
    return len(seen) / total if total else 0.0


def repetition_rate(texts: Sequence[str], n: int) -> float:
    """Mean fraction of n-grams inside one generation that are repeats.

    Measured per generation and then averaged, because the failure being looked
    for - a model looping on one phrase - happens within a sample.
    """
    rates = []
    for text in texts:
        counts = ngrams(tokenize_words(text), n)
        total = sum(counts.values())
        if total:
            rates.append(1 - len(counts) / total)
    return sum(rates) / len(rates) if rates else 0.0


def diversity_report(texts: Sequence[str]) -> Dict[str, float]:
    """Distinct-1/2 and repetition at n = 1..4 for one set of generations."""
    report = {f"distinct_{n}": distinct_n(texts, n) for n in (1, 2)}
    report.update({f"repetition_{n}": repetition_rate(texts, n)
                   for n in (1, 2, 3, 4)})
    lengths = [len(tokenize_words(t)) for t in texts]
    report["mean_words"] = sum(lengths) / len(lengths) if lengths else 0.0
    return report


def all_metrics(hypotheses: Sequence[str], references: Sequence[str]
                ) -> Dict[str, object]:
    """Every generation metric for one decoding setting."""
    return {"bleu": corpus_bleu(hypotheses, references),
            "chrf": corpus_chrf(hypotheses, references),
            "rouge_l": corpus_rouge_l(hypotheses, references),
            "diversity": diversity_report(hypotheses),
            "reference_diversity": diversity_report(references)}


# Verification


#: Reference values produced by sacrebleu 2.6.0 on the fixtures below, recorded
#: so the implementations can be checked anywhere without installing it:
#:
#:   sacrebleu.corpus_bleu(hyps, [refs], tokenize="13a").score
#:   sacrebleu.corpus_chrf(hyps, [refs], char_order=6, word_order=W, beta=2).score
#:
#: BLEU agrees exactly. chrF agrees to ~1e-3, the difference being sacrebleu's
#: epsilon floor on empty n-gram orders. The fixtures carry no punctuation on
#: purpose: this tokenizer detaches the danda and `13a` does not, so punctuation
#: would compare two different tokenizations rather than two metrics.
_FIXTURES = {
    "ascii": {
        "hyp": ["the quick brown fox jumps over the lazy dog again and again",
                "a second sentence with punctuation",
                "short one"],
        "ref": ["the quick brown fox jumps over the lazy dog",
                "a second sentence with different punctuation",
                "short one indeed"],
        "bleu": 72.628661, "chrf": 78.426400, "chrf_pp": 79.694764,
    },
    "nepali": {
        "hyp": ["नेपाल सरकारले २०८१ सालको बजेटमा शिक्षा र स्वास्थ्य क्षेत्रका लागि रकम बढाउने निर्णय गरेको छ",
                "काठमाडौंमा आज दिनभर पानी पर्‍यो र मानिसहरू घरमै बस्नुपर्‍यो",
                "यस अनुसन्धानपत्रमा लेखकले भाषाशास्त्रीय दृष्टिकोणबाट विश्लेषण प्रस्तुत गरेका छन्"],
        "ref": ["नेपाल सरकारले २०८१ सालको बजेटमा शिक्षा र स्वास्थ्य क्षेत्रका लागि छुट्याइएको रकम बढाउने निर्णय गरेको छ",
                "काठमाडौंमा हिजो दिनभर भारी पानी पर्‍यो र मानिसहरूलाई घरमै बस्न सुझाव दिइयो",
                "यस अनुसन्धानपत्रमा लेखकले भाषाशास्त्रीय दृष्टिकोणबाट नेपाली क्रियापदका रूपान्तरणको विश्लेषण प्रस्तुत गरेका छन्"],
        "bleu": 54.747955, "chrf": 74.127192, "chrf_pp": 73.036617,
    },
}


def self_test(tolerance: float = 1e-3) -> bool:
    """Check BLEU and chrF against recorded sacrebleu values.

    Also checks the two properties that make the word tokenizer correct for
    these scripts: a combining mark never starts a new token, and punctuation
    always does.
    """
    ok = True
    for name, case in _FIXTURES.items():
        got = {"bleu": corpus_bleu(case["hyp"], case["ref"])["bleu"],
               "chrf": corpus_chrf(case["hyp"], case["ref"], word_order=0)["chrf"],
               "chrf_pp": corpus_chrf(case["hyp"], case["ref"], word_order=2)["chrf"]}
        for key, value in got.items():
            delta = abs(value - case[key])
            mark = "ok" if delta <= tolerance else "FAILED"
            ok &= delta <= tolerance
            print(f"  {name:<9}{key:<9}{value:>12.6f}  vs sacrebleu "
                  f"{case[key]:>12.6f}  delta {delta:.2e}  {mark}")

    checks = [
        (tokenize_words("गरेको छ।"), ["गरेको", "छ", "।"]),
        (tokenize_words("मानिसहरू, घरमै"), ["मानिसहरू", ",", "घरमै"]),
        (tokenize_words("a, b"), ["a", ",", "b"]),
    ]
    for got_tokens, want in checks:
        mark = "ok" if got_tokens == want else "FAILED"
        ok &= got_tokens == want
        print(f"  tokenizer{'':<9}{str(got_tokens):<40} {mark}")

    # ROUGE-L of a sequence against itself is 100 by construction.
    identical = corpus_rouge_l(["a b c d"], ["a b c d"])["rouge_l"]
    ok &= abs(identical - 100.0) < 1e-9
    print(f"  rouge_l  identity{identical:>12.6f}  "
          f"{'ok' if abs(identical - 100) < 1e-9 else 'FAILED'}")
    return bool(ok)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Generation metrics: self-check.")
    ap.add_argument("--self-test", action="store_true",
                    help="compare against recorded sacrebleu values")
    args = ap.parse_args()
    if args.self_test:
        raise SystemExit(0 if self_test() else 1)
    ap.print_help()
