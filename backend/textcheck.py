"""Text analysis: plagiarism overlap and AI-generation indicators.

Two very different problems, deliberately kept apart because their evidence is
of completely different quality:

* **Plagiarism** is *measurable*. Given a reference text, the shared-n-gram
  overlap between two documents is an exact quantity, and the matching passages
  can be pointed at. The percentage means something precise.

* **AI-generated text** is probabilistic. A trained TF-IDF classifier supplies
  the primary signal, while sentence-length variance, vocabulary diversity,
  repetition, and model-typical phrasing explain supporting style evidence.
  Generator and domain shift still cause errors, so the result is never proof.

The honest framing matters here: published AI-text detectors routinely
misclassify human writing, and the consequences (accusing a student of
cheating) are serious. This module reports the individual signals alongside the
score so a reader can judge them, and it never returns a bare verdict.
"""

from __future__ import annotations

import math
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------- tuning
MIN_WORDS = 40           # below this, every statistic is noise
NGRAM = 5                # shingle size for overlap; long enough that a match
                         # is unlikely to be coincidental

# Phrases current large language models produce far more often than people do.
# Presence is weak evidence on its own, which is why it is one signal of six.
LLM_PHRASES = (
    "delve into", "it is important to note", "it's important to note",
    "in conclusion", "furthermore", "moreover", "in today's world",
    "navigate the complexities", "rich tapestry", "a testament to",
    "plays a crucial role", "it is worth noting", "in the realm of",
    "underscores the importance", "multifaceted", "ever-evolving",
    "shed light on", "at the end of the day", "when it comes to",
    "the landscape of", "paves the way", "cannot be overstated",
)

_WORD = re.compile(r"[a-z0-9']+")
_SENTENCE = re.compile(r"[^.!?]+[.!?]*")
_MODEL_TOKEN = re.compile(r"(?u)\b\w\w+\b")

_TEXT_MODEL_DIR = Path(__file__).resolve().parent / "text_models"
_TEXT_MODEL: "_TextClassifier | None" = None


class _TextClassifier:
    """Dependency-free inference for the locally trained TF-IDF classifier."""

    def __init__(self, directory: Path):
        vocabulary = json.loads((directory / "vocabulary.json").read_text("utf-8"))
        self.vocabulary = {term: index for index, term in enumerate(vocabulary)}
        with np.load(directory / "classifier.npz") as data:
            self.idf = data["idf"].astype(np.float64)
            self.coefficients = data["coefficients"].astype(np.float64)
            self.intercept = float(data["intercept"][0])
            self.calibration_coefficient = float(data["calibration_coefficient"][0])
            self.calibration_intercept = float(data["calibration_intercept"][0])
        self.metadata = json.loads((directory / "metadata.json").read_text("utf-8"))

    def predict(self, text: str) -> float:
        tokens = _MODEL_TOKEN.findall(text.lower())
        terms = tokens + [f"{a} {b}" for a, b in zip(tokens, tokens[1:])]
        counts = Counter(
            index for term in terms
            if (index := self.vocabulary.get(term)) is not None
        )
        if not counts:
            return 0.5

        norm_squared = 0.0
        numerator = 0.0
        for index, count in counts.items():
            value = count * self.idf[index]
            norm_squared += value * value
            numerator += self.coefficients[index] * value
        raw = numerator / math.sqrt(norm_squared) + self.intercept
        calibrated = (
            self.calibration_coefficient * raw + self.calibration_intercept
        )
        if calibrated >= 0:
            return 1.0 / (1.0 + math.exp(-calibrated))
        exp_value = math.exp(calibrated)
        return exp_value / (1.0 + exp_value)


def _text_classifier() -> _TextClassifier | None:
    global _TEXT_MODEL
    if _TEXT_MODEL is None:
        try:
            _TEXT_MODEL = _TextClassifier(_TEXT_MODEL_DIR)
        except (FileNotFoundError, OSError, ValueError, KeyError, json.JSONDecodeError):
            return None
    return _TEXT_MODEL


def _words(text: str) -> list[str]:
    return _WORD.findall(text.lower())


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE.findall(text) if s.strip()]


def _sentence_spans(text: str) -> list[dict]:
    """Sentences with character offsets into the original text.

    Offsets rather than substrings, so the UI can mark a passage in place with
    its original capitalisation and punctuation intact.
    """
    spans = []
    for m in _SENTENCE.finditer(text):
        raw = m.group()
        stripped = raw.strip()
        if not stripped:
            continue
        lead = len(raw) - len(raw.lstrip())
        start = m.start() + lead
        spans.append({"start": start, "end": start + len(stripped), "text": stripped})
    return spans


def _shingles(words: list[str], n: int = NGRAM) -> list[tuple[str, ...]]:
    return [tuple(words[i:i + n]) for i in range(max(0, len(words) - n + 1))]


# ----------------------------------------------------------------- plagiarism
def check_plagiarism(text: str, reference: str) -> dict:
    """Exact n-gram overlap between a document and a reference.

    Uses word 5-grams: long enough that a shared run is very unlikely to be
    coincidental, short enough to survive light paraphrasing of the
    surrounding sentence.

    The percentage is the share of the submitted text's 5-grams that also
    appear in the reference - a real measurement, not an estimate.
    """
    words = _words(text)
    ref_words = _words(reference or "")

    if len(words) < NGRAM:
        return {
            "available": False,
            "reason": f"Need at least {NGRAM} words to compare.",
        }
    if len(ref_words) < NGRAM:
        return {
            "available": False,
            "reason": "Paste a reference text to compare against. Without one there is "
                      "nothing to measure overlap with - this compares two documents, it "
                      "does not search the web.",
        }

    doc = _shingles(words)
    ref = set(_shingles(ref_words))

    matched_flags = [s in ref for s in doc]
    matched = sum(matched_flags)
    overlap = matched / len(doc) if doc else 0.0

    # Which word positions fall inside a matching run. A shingle at index i
    # covers words i..i+NGRAM-1, so a hit marks all of them.
    flagged = [False] * len(words)
    for i, hit in enumerate(matched_flags):
        if hit:
            for j in range(i, min(i + NGRAM, len(words))):
                flagged[j] = True

    # Character offsets into the ORIGINAL text, so the UI can highlight the
    # passage in place rather than showing a reconstructed lowercase copy.
    # Walking the original with the same tokenizer keeps the two in step.
    spans: list[dict] = []
    positions = [(m.start(), m.end()) for m in _WORD.finditer(text.lower())]

    run_start = None
    for i in range(len(words) + 1):
        inside = i < len(words) and flagged[i]
        if inside and run_start is None:
            run_start = i
        elif not inside and run_start is not None:
            start_char = positions[run_start][0]
            end_char = positions[i - 1][1]
            spans.append({
                "start": start_char,
                "end": end_char,
                "words": i - run_start,
                "text": text[start_char:end_char],
            })
            run_start = None

    passages = sorted((s["text"] for s in spans), key=len, reverse=True)

    return {
        "available": True,
        "overlap_percent": round(overlap * 100, 1),
        "matched_ngrams": matched,
        "total_ngrams": len(doc),
        "ngram_size": NGRAM,
        "matched_passages": passages[:12],
        # Character ranges, in document order, for inline highlighting.
        "matched_spans": sorted(spans, key=lambda s: s["start"]),
        "flagged_words": sum(flagged),
        "total_words": len(words),
        "reference_words": len(ref_words),
        "word_coverage_percent": round(sum(flagged) / len(words) * 100, 1),
        "unique_matched_passages": len(spans),
        "longest_match_words": max((s["words"] for s in spans), default=0),
        "verdict": (
            "HIGH" if overlap >= 0.25 else
            "MODERATE" if overlap >= 0.10 else
            "LOW" if overlap >= 0.03 else
            "MINIMAL"
        ),
        "note": (
            "Share of this text's 5-word sequences that also appear in the reference. "
            "This is an exact measurement against the text you provided; it does not "
            "search the internet."
        ),
    }


# ------------------------------------------------------------- AI indicators
def _burstiness(sentences: list[str]) -> float:
    """Coefficient of variation of sentence length.

    Human prose mixes long and short sentences; generated prose tends toward a
    uniform rhythm. Higher means more human-like variation.
    """
    lengths = [len(_words(s)) for s in sentences if _words(s)]
    if len(lengths) < 3:
        return 0.0
    mean = sum(lengths) / len(lengths)
    if mean == 0:
        return 0.0
    var = sum((x - mean) ** 2 for x in lengths) / len(lengths)
    return math.sqrt(var) / mean


def check_ai_text(text: str) -> dict:
    """Trained human-vs-AI classification with supporting style indicators."""
    words = _words(text)
    sentences = _sentences(text)

    if len(words) < MIN_WORDS:
        return {
            "available": False,
            "reason": f"Need at least {MIN_WORDS} words; got {len(words)}. "
                      f"Short samples cannot support any of these statistics.",
        }

    lowered = text.lower()

    # 1. Burstiness - low variation leans machine.
    burst = _burstiness(sentences)
    burst_signal = max(0.0, min(1.0, (0.55 - burst) / 0.45))

    # 2. Vocabulary diversity, length-normalised so long texts are not punished.
    unique = len(set(words))
    ttr = unique / len(words)
    expected_ttr = max(0.28, 0.85 - 0.06 * math.log(max(len(words), 2)))
    diversity_signal = max(0.0, min(1.0, (expected_ttr - ttr) / max(expected_ttr, 1e-6)))

    # 3. Repeated 5-grams within the text itself.
    shingles = _shingles(words)
    repeats = sum(c - 1 for c in Counter(shingles).values() if c > 1)
    repeat_ratio = repeats / len(shingles) if shingles else 0.0
    repeat_signal = min(1.0, repeat_ratio * 12)

    # 4. Phrases current models overuse.
    found_phrases = [p for p in LLM_PHRASES if p in lowered]
    phrase_signal = min(1.0, len(found_phrases) / 4)

    # 5. Sentence-length uniformity in absolute terms.
    lengths = [len(_words(s)) for s in sentences if _words(s)]
    avg_len = sum(lengths) / len(lengths) if lengths else 0
    uniform_signal = max(0.0, min(1.0, (avg_len - 14) / 16)) if avg_len else 0.0

    # 6. Punctuation variety - generated prose leans on commas and full stops.
    marks = Counter(c for c in text if c in ",;:-()\"'?!")
    variety = len(marks) / 10
    punct_signal = max(0.0, min(1.0, 1 - variety))

    heuristic_signals = [
        ("Sentence-length variation", burst_signal, 0.28,
         f"coefficient of variation {burst:.2f}",
         "Human writing mixes long and short sentences more than generated text."),
        ("Vocabulary diversity", diversity_signal, 0.20,
         f"{unique} unique of {len(words)} words ({ttr:.2f})",
         "Lower-than-expected diversity for this length."),
        ("Internal repetition", repeat_signal, 0.16,
         f"{repeats} repeated 5-grams",
         "Repeated phrasing within the same passage."),
        ("Model-typical phrasing", phrase_signal, 0.16,
         ", ".join(found_phrases[:5]) if found_phrases else "none found",
         "Phrases current language models overuse."),
        ("Sentence length", uniform_signal, 0.10,
         f"{avg_len:.1f} words average",
         "Consistently long sentences are more common in generated prose."),
        ("Punctuation variety", punct_signal, 0.10,
         f"{len(marks)} distinct marks",
         "Narrow punctuation range."),
    ]

    heuristic_score = sum(
        value * weight for _, value, weight, _, _ in heuristic_signals
    )

    # The trained classifier is the primary evidence.  The older style rules
    # remain useful as an explanation layer, but they fail badly on dialogue,
    # scripts, lists and markdown because those formats naturally have varied
    # sentence lengths and punctuation.  That was the cause of obviously
    # generated screenplay text receiving a ~4% score.
    classifier = _text_classifier()
    model_probability = classifier.predict(text) if classifier else None
    if model_probability is not None:
        heuristic_weights = (0.07, 0.05, 0.04, 0.04, 0.03, 0.02)
        signals = [
            (
                "Trained text classifier",
                model_probability,
                0.75,
                f"{model_probability * 100:.1f}% model probability",
                "A TF-IDF logistic classifier trained on 40,559 labelled human and AI passages.",
            ),
            *[
                (name, value, weight, detail, meaning)
                for (name, value, _, detail, meaning), weight
                in zip(heuristic_signals, heuristic_weights)
            ],
        ]
        score = 0.75 * model_probability + 0.25 * heuristic_score
        confidence = "medium"
        score_kind = "trained_classifier_with_style_support"
    else:
        signals = heuristic_signals
        score = heuristic_score
        confidence = "low"
        score_kind = "style_indicators_only"

    # ---------------------------------------------------------------- regions
    # Which sentences look most machine-like, so the result can point at
    # passages rather than only scoring the document as a whole.
    #
    # Only signals that are meaningful for a single sentence are used here.
    # Burstiness and length-normalised vocabulary diversity are properties of a
    # document and say nothing about one sentence, so including them would
    # invent precision that is not there.
    shingle_counts = Counter(shingles)
    repeated = {s for s, c in shingle_counts.items() if c > 1}

    flagged_sentences = []
    for span in _sentence_spans(text):
        s_words = _words(span["text"])
        if len(s_words) < 6:
            continue

        s_lower = span["text"].lower()
        reasons = []
        s_score = 0.0

        hits = [p for p in LLM_PHRASES if p in s_lower]
        if hits:
            s_score += min(1.0, len(hits) / 2) * 0.55
            reasons.append("model-typical phrasing: " + ", ".join(hits[:3]))

        # Long sentences relative to this document, not an absolute threshold.
        if avg_len and len(s_words) > avg_len * 1.35 and len(s_words) > 20:
            s_score += 0.2
            reasons.append(f"{len(s_words)} words, well above this document's average")

        s_shingles = _shingles(s_words)
        if s_shingles:
            rep_here = sum(1 for sh in s_shingles if sh in repeated)
            if rep_here:
                s_score += min(1.0, rep_here / len(s_shingles)) * 0.25
                reasons.append("phrasing repeated elsewhere in the text")

        s_unique = len(set(s_words))
        if len(s_words) >= 12 and s_unique / len(s_words) < 0.6:
            s_score += 0.15
            reasons.append("low word variety within the sentence")

        if s_score >= 0.3 and reasons:
            flagged_sentences.append({
                "start": span["start"],
                "end": span["end"],
                "strength": round(min(1.0, s_score) * 100, 1),
                "reasons": reasons,
                "words": len(s_words),
            })

    flagged_words = sum(f["words"] for f in flagged_sentences)

    return {
        "available": True,
        "ai_likelihood_percent": round(score * 100, 1),
        # Character ranges for inline marking, in document order.
        "flagged_sentences": flagged_sentences,
        "flagged_word_count": flagged_words,
        "total_word_count": len(words),
        "confidence": confidence,
        "score_kind": score_kind,
        "model_probability_percent": (
            round(model_probability * 100, 1)
            if model_probability is not None else None
        ),
        "model": (
            classifier.metadata["model"] if classifier else None
        ),
        "model_test_accuracy_percent": (
            round(classifier.metadata["metrics"]["accuracy"] * 100, 1)
            if classifier else None
        ),
        "verdict": (
            "STRONG INDICATORS" if score >= 0.65 else
            "SOME INDICATORS" if score >= 0.45 else
            "FEW INDICATORS" if score >= 0.25 else
            "MINIMAL INDICATORS"
        ),
        "word_count": len(words),
        "sentence_count": len(sentences),
        "signals": [
            {
                "name": name,
                "strength": round(value * 100, 1),
                "weight": round(weight * 100),
                "detail": detail,
                "meaning": meaning,
            }
            for name, value, weight, detail, meaning in signals
        ],
        "note": (
            "The trained classifier is supported by transparent style indicators, "
            "but AI-text detection is still probabilistic. Generator, language, "
            "editing and sample length can change the result; never use this score "
            "alone as proof of authorship."
            if classifier else
            "The trained classifier is unavailable, so this score uses style "
            "indicators only. Treat it as a reason to look closer, never as proof."
        ),
    }


# -------------------------------------------------------- news credibility
# What this is, and firmly what it is not.
#
# It does NOT determine whether a claim is true. Nothing in this file can:
# establishing that a statement about the world is false requires knowing
# about the world, and this module only has the string. A piece can be
# impeccably written and entirely false, or badly written and perfectly
# accurate, and no amount of counting adverbs separates those two.
#
# What IS measurable is *presentation* - the writing habits that distinguish
# reporting from agitation. Sourced claims, hedged conclusions and calm
# punctuation are conventions of journalism; unsourced absolutes, shouting and
# manufactured outrage are conventions of something else. Those conventions are
# countable, and departures from them are a reason to check a piece more
# carefully.
#
# So the output is a "concern" score about *style*, never a truth verdict, and
# the wording throughout says so. Calling it a fake-news detector would be a
# lie about what the numbers can carry.

MIN_NEWS_WORDS = 25       # headlines are short; below this even style is noise

SENSATIONAL = (
    "you won't believe", "you wont believe", "shocking", "shocked the world",
    "doctors hate", "the truth about", "what they don't want you to know",
    "what they dont want you to know", "wake up", "mainstream media",
    "bombshell", "exposed", "destroyed", "slams", "obliterates", "eviscerates",
    "miracle cure", "one weird trick", "gone wrong", "will blow your mind",
    "this is why", "here's what happened", "heres what happened",
    "no one is talking about", "they don't want you", "they dont want you",
    "secret that", "banned", "censored", "silenced",
)

# Markers of attributed reporting. Their ABSENCE is the signal - a piece making
# factual claims while naming no one who made them is the single most reliable
# stylistic difference between reporting and assertion.
ATTRIBUTION = (
    "according to", "said", "says", "told", "reported", "reports",
    "confirmed", "announced", "stated", "spokesperson", "spokesman",
    "spokeswoman", "researchers", "study", "studies", "survey", "data from",
    "quoted", "cited", "in a statement", "press release", "court filing",
    "peer-reviewed", "journal",
)

# Claims phrased so they cannot be checked or contradicted.
ABSOLUTES = (
    "everyone knows", "nobody knows", "no one knows", "everybody knows",
    "always", "never", "undeniable", "undeniably", "proven fact",
    "irrefutable", "100% proven", "completely false", "totally false",
    "beyond any doubt", "without question", "the fact is", "make no mistake",
    "it is obvious", "obviously", "clearly the", "any reasonable person",
)

LOADED = (
    "outrageous", "disgusting", "horrifying", "sickening", "evil", "corrupt",
    "scam", "hoax", "lies", "lying", "betrayal", "treason", "tyranny",
    "sheeple", "brainwashed", "agenda", "puppet", "shill", "propaganda",
    "elites", "globalist", "witch hunt", "hysteria",
)

_CAPS_WORD = re.compile(r"\b[A-Z]{3,}\b")
_PUNCT_RUN = re.compile(r"[!?]{2,}")


def check_news_credibility(text: str) -> dict:
    """Stylistic credibility signals for a news-like passage.

    Reports how a piece is *written*, not whether it is true. See the comment
    block above for why that distinction is not hedging but the actual limit of
    what the text can support.
    """
    words = _words(text)
    sentences = _sentences(text)

    if len(words) < MIN_NEWS_WORDS:
        return {
            "available": False,
            "reason": f"Need at least {MIN_NEWS_WORDS} words; got {len(words)}. "
                      f"Style cannot be measured on a fragment.",
        }

    lowered = text.lower()

    # 1. Sensationalist and clickbait phrasing.
    sensational_hits = [p for p in SENSATIONAL if p in lowered]
    sensational_signal = min(1.0, len(sensational_hits) / 3)

    # 2. Shouting. Measured as a share of words so a long piece is not
    #    penalised for one acronym, and acronyms of 3+ letters are common
    #    enough in real reporting that the threshold sits above a trace level.
    caps = _CAPS_WORD.findall(text)
    caps_ratio = len(caps) / max(len(words), 1)
    caps_signal = max(0.0, min(1.0, (caps_ratio - 0.01) / 0.06))

    # 3. Punctuation intensity - "!!!" and "?!" are not house style anywhere
    #    that employs a subeditor.
    runs = _PUNCT_RUN.findall(text)
    punct_signal = min(1.0, len(runs) / 3)

    # 4. Absence of attribution, scaled by how much is being claimed. A short
    #    opinion column needs fewer sources than a long factual account, so the
    #    expectation grows with length rather than being a flat requirement.
    attribution_hits = [a for a in ATTRIBUTION if a in lowered]
    expected_sources = max(1, len(sentences) // 4)
    attribution_signal = max(0.0, min(
        1.0, (expected_sources - len(attribution_hits)) / expected_sources))

    # 5. Unfalsifiable absolutes.
    absolute_hits = [a for a in ABSOLUTES if a in lowered]
    absolute_signal = min(1.0, len(absolute_hits) / 3)

    # 6. Loaded and emotive vocabulary.
    loaded_hits = [w for w in LOADED if w in lowered]
    loaded_signal = min(1.0, len(loaded_hits) / 3)

    signals = [
        ("Missing attribution", attribution_signal, 0.26,
         f"{len(attribution_hits)} source markers for {len(sentences)} sentences",
         "Reporting names who said what. Claims with no source behind them are "
         "the clearest stylistic departure from it."),
        ("Sensationalist phrasing", sensational_signal, 0.20,
         ", ".join(sensational_hits[:4]) if sensational_hits else "none found",
         "Clickbait and outrage constructions written to be shared rather than read."),
        ("Loaded language", loaded_signal, 0.18,
         ", ".join(loaded_hits[:4]) if loaded_hits else "none found",
         "Emotive vocabulary that characterises rather than describes."),
        ("Unfalsifiable claims", absolute_signal, 0.16,
         ", ".join(absolute_hits[:4]) if absolute_hits else "none found",
         "Absolutes phrased so they cannot be checked or contradicted."),
        ("Shouting", caps_signal, 0.12,
         f"{len(caps)} all-caps words of {len(words)}",
         "Capitalisation used for emphasis rather than for acronyms."),
        ("Punctuation intensity", punct_signal, 0.08,
         f"{len(runs)} runs of !! or ?!",
         "Repeated exclamation and question marks."),
    ]

    score = sum(value * weight for _, value, weight, _, _ in signals)

    # Sentence-level marking, same approach as the AI check: only signals that
    # mean something for a single sentence are used. Attribution is a property
    # of the piece, not of any one line, so it is deliberately excluded here.
    flagged_sentences = []
    for span in _sentence_spans(text):
        s_words = _words(span["text"])
        if len(s_words) < 5:
            continue

        s_lower = span["text"].lower()
        reasons = []
        s_score = 0.0

        for label, table, weight in (
            ("sensationalist phrasing", SENSATIONAL, 0.5),
            ("loaded language", LOADED, 0.4),
            ("unfalsifiable claim", ABSOLUTES, 0.35),
        ):
            hits = [p for p in table if p in s_lower]
            if hits:
                s_score += min(1.0, len(hits) / 2) * weight
                reasons.append(f"{label}: " + ", ".join(hits[:3]))

        s_caps = _CAPS_WORD.findall(span["text"])
        if len(s_caps) >= 2:
            s_score += 0.25
            reasons.append(f"{len(s_caps)} all-caps words")

        if _PUNCT_RUN.search(span["text"]):
            s_score += 0.2
            reasons.append("repeated ! or ?")

        if s_score >= 0.3 and reasons:
            flagged_sentences.append({
                "start": span["start"],
                "end": span["end"],
                "strength": round(min(1.0, s_score) * 100, 1),
                "reasons": reasons,
                "words": len(s_words),
            })

    return {
        "available": True,
        "concern_percent": round(score * 100, 1),
        "flagged_sentences": flagged_sentences,
        "flagged_word_count": sum(f["words"] for f in flagged_sentences),
        "total_word_count": len(words),
        "attribution_found": attribution_hits[:8],
        "confidence": "low",
        "verdict": (
            "STRONG CONCERNS" if score >= 0.60 else
            "SOME CONCERNS" if score >= 0.40 else
            "FEW CONCERNS" if score >= 0.22 else
            "MINIMAL CONCERNS"
        ),
        "word_count": len(words),
        "sentence_count": len(sentences),
        "signals": [
            {
                "name": name,
                "strength": round(value * 100, 1),
                "weight": round(weight * 100),
                "detail": detail,
                "meaning": meaning,
            }
            for name, value, weight, detail, meaning in signals
        ],
        "note": (
            "This measures how the passage is WRITTEN, not whether it is true. "
            "It cannot verify a single fact - it has no access to the world, only "
            "to this text. Well-written falsehoods score low here and clumsily "
            "written truths score high. Use it to decide what deserves a closer "
            "look, and check the claims themselves against primary sources."
        ),
    }


def analyze(text: str, reference: str = "", want_plagiarism: bool = True,
            want_ai: bool = True, want_news: bool = False) -> dict:
    """Run whichever checks were requested."""
    text = (text or "").strip()
    if not text:
        raise ValueError("No text provided.")

    result: dict = {
        "word_count": len(_words(text)),
        "character_count": len(text),
        "sentence_count": len(_sentences(text)),
        "checks_run": [],
    }

    if want_plagiarism:
        result["plagiarism"] = check_plagiarism(text, reference)
        result["checks_run"].append("plagiarism")
    if want_ai:
        result["ai_text"] = check_ai_text(text)
        result["checks_run"].append("ai_text")
    if want_news:
        result["news_credibility"] = check_news_credibility(text)
        result["checks_run"].append("news_credibility")

    if not result["checks_run"]:
        raise ValueError("Select at least one check.")

    return result
