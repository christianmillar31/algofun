import pandas as pd

from algofun.text import Lexicon, load_lm_lexicon, score_text, tokenize
from algofun.text.lexicon import parse_lm_csv

LEX = Lexicon(positive=frozenset({"PROFITABLE", "GROWTH", "STRONG"}),
              negative=frozenset({"LOSS", "DECLINE", "LAWSUIT", "WEAK"}),
              uncertainty=frozenset({"MAY", "UNCERTAIN"}))


def test_tokenize_upper_alpha_only():
    assert tokenize("Q3 profit up 12%, 'strong' growth; loss-making unit sold.") == \
        ["PROFIT", "UP", "STRONG", "GROWTH", "LOSS-MAKING", "UNIT", "SOLD"]
    assert tokenize("") == []


def test_score_counts_and_tone():
    s = score_text("Strong growth and profitable quarter despite a lawsuit", LEX)
    assert (s.positive, s.negative) == (3, 1)
    assert s.tone == 0.5
    assert s.negativity == 1 / s.n_tokens
    assert score_text("nothing to see here", LEX).tone == 0.0


def test_negation_flips_positive_within_three_words():
    s = score_text("The company is not profitable this year", LEX)
    assert (s.positive, s.negative) == (0, 1)
    far = score_text("no one at the firm expected such strong results", LEX)   # 'no' is 7 tokens away
    assert (far.positive, far.negative) == (1, 0)


def test_parse_lm_csv_uses_year_columns(tmp_path):
    df = pd.DataFrame({
        "Word": ["GOOD", "BAD", "REMOVED", "MAYBE", "SUE"],
        "Sequence Number": [1, 2, 3, 4, 5],
        "Negative": [0, 2009, -2018, 0, 0],
        "Positive": [2009, 0, 2009, 0, 0],
        "Uncertainty": [0, 0, 0, 2009, 0],
        "Litigious": [0, 0, 0, 0, 2009],
        "Word Count": [1, 1, 1, 1, 1],
    })
    p = tmp_path / "LM.csv"
    df.to_csv(p, index=False)
    lex = parse_lm_csv(p)
    assert lex.positive == {"GOOD", "REMOVED"}
    assert lex.negative == {"BAD"}          # a negative year means "removed from the list"
    assert lex.uncertainty == {"MAYBE"} and lex.litigious == {"SUE"}
    assert load_lm_lexicon(path=p).positive == lex.positive
