"""Stimulus text preparation: spelling checks and the Adv/Ele rephrasing carry-over.

Lifted from `experiment_builder/text_adjustments.ipynb` in stage F.

`adv_texts_with_ids.csv` and `ele_texts_with_ids.csv` hold the same paragraphs in advanced
and elementary form. The elementary file is **deliberately unused** downstream -- it is kept
for a future condition, not a missing one.
"""

import re
from pathlib import Path
import pandas as pd
from spellchecker import SpellChecker


WORD = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)*")

ACCEPTED = {"underserved", "behaviour", "carpathians"}

spell = SpellChecker()

CORRECTIONS = {"iduring": "during"}


def spell_words(text):
    # normalise the typographic apostrophe so "don't"/"don’t" tokenise the same way
    return WORD.findall(str(text).replace("’", "'"))


def spell_issues(text):
    """{word: suggested correction} for words in `text` no dictionary form recognises."""
    issues = {}
    for w in spell_words(text):
        forms = {w.lower(), w.lower().removesuffix("'s")}  # "Greece's" -> "greece"
        if forms & ACCEPTED or len(spell.unknown(forms)) < len(forms):
            continue
        issues[w] = spell.correction(w.lower().removesuffix("'s"))
    return issues


def apply_corrections(series):
    out = series
    for wrong, right in CORRECTIONS.items():
        out = out.str.replace(rf"\b{re.escape(wrong)}\b", right, regex=True)
    return out
