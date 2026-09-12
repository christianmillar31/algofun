"""Loughran-McDonald finance sentiment lexicon.

Loughran & McDonald (Journal of Finance, 2011) showed that general-purpose
word lists misclassify most finance text: "liability", "tax" and "cost" are
not bad news in a 10-K. Their dictionary was built from filings and is the
standard baseline for finance text. It is also leak-free: a word list built
in 2011 cannot know how a 2018 headline turned out, which is more than can
be said for any language model scoring history it was trained on.

The master dictionary is published by the University of Notre Dame's
Software Repository for Accounting and Finance (sraf.nd.edu). It is free for
personal and academic use; commercial use needs their licence. We read the
copy bundled in the MIT-licensed `pysentiment2` package rather than
vendoring it, and cache it next to the bars.
"""
from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)

LM_URL = "https://raw.githubusercontent.com/nickderobertis/pysentiment/master/pysentiment2/static/LM.csv"
LM_FILE = "LM.csv"

# Loughran & McDonald's negation rule: a positive word within three tokens after one of
# these is counted as negative ("not profitable", "no longer growing").
NEGATORS = frozenset({"NO", "NOT", "NONE", "NEITHER", "NEVER", "NOBODY", "NOR", "CANNOT", "WITHOUT"})
NEGATION_WINDOW = 3

_TOKEN = re.compile(r"[A-Za-z][A-Za-z'\-]*")


@dataclass(frozen=True)
class Lexicon:
    positive: frozenset[str]
    negative: frozenset[str]
    uncertainty: frozenset[str] = frozenset()
    litigious: frozenset[str] = frozenset()
    name: str = "lm"

    def __len__(self) -> int:
        return len(self.positive) + len(self.negative)


@dataclass(frozen=True)
class TextScore:
    positive: int
    negative: int
    uncertainty: int
    n_tokens: int

    @property
    def tone(self) -> float:
        """(pos - neg) / (pos + neg) in [-1, 1]; 0 when the text has no sentiment words."""
        s = self.positive + self.negative
        return (self.positive - self.negative) / s if s else 0.0

    @property
    def negativity(self) -> float:
        """Share of tokens that are negative words. LM's headline result: this is the
        part that predicts anything; positive words are mostly noise."""
        return self.negative / self.n_tokens if self.n_tokens else 0.0


def tokenize(text: str) -> list[str]:
    """Upper-case alphabetic tokens; the dictionary is upper case."""
    if not text:
        return []
    return [t.upper().strip("'-") for t in _TOKEN.findall(text) if len(t) > 1]


def score_text(text: str, lexicon: Lexicon) -> TextScore:
    toks = tokenize(text)
    pos = neg = unc = 0
    for i, t in enumerate(toks):
        if t in lexicon.positive:
            lo = max(0, i - NEGATION_WINDOW)
            if any(w in NEGATORS for w in toks[lo:i]):
                neg += 1
            else:
                pos += 1
        elif t in lexicon.negative:
            neg += 1
        if t in lexicon.uncertainty:
            unc += 1
    return TextScore(pos, neg, unc, len(toks))


def parse_lm_csv(path: str | os.PathLike) -> Lexicon:
    """Parse the master dictionary CSV (Word, Negative, Positive, Uncertainty, Litigious, ...).

    A category column holds the year the word was added (or removed, negative), so
    "> 0" means the word is currently in that list."""
    df = pd.read_csv(path, usecols=lambda c: c in {"Word", "Negative", "Positive", "Uncertainty", "Litigious"})
    if "Word" not in df.columns or "Negative" not in df.columns or "Positive" not in df.columns:
        raise ValueError(f"{path} does not look like the Loughran-McDonald master dictionary")
    words = df["Word"].astype(str).str.upper()

    def members(col: str) -> frozenset[str]:
        if col not in df.columns:
            return frozenset()
        return frozenset(words[pd.to_numeric(df[col], errors="coerce").fillna(0) > 0])

    return Lexicon(positive=members("Positive"), negative=members("Negative"),
                   uncertainty=members("Uncertainty"), litigious=members("Litigious"), name="lm")


def load_lm_lexicon(cache_dir: str | os.PathLike = "data/cache/text", refresh: bool = False,
                    path: str | os.PathLike | None = None) -> Lexicon:
    """Load the dictionary from `path`, else from the cache, else download it."""
    if path:
        return parse_lm_csv(path)
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    f = cache / LM_FILE
    if refresh or not f.exists():
        import requests
        log.info("downloading Loughran-McDonald dictionary from %s", LM_URL)
        r = requests.get(LM_URL, timeout=60)
        r.raise_for_status()
        f.write_bytes(r.content)
    return parse_lm_csv(f)
