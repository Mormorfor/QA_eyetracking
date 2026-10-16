"""Spelling check for the stimulus texts.

`spell_issues` lists the words of a text that the dictionary does not know, each with a
suggested correction, for review by hand; `apply_corrections` then applies the reviewed
fixes (`CORRECTIONS`). They were run on the rephrased correct answers
(`rephrased_answer_A`), in `notebooks/experiment/text_adjustments.ipynb`.

That notebook first added every word of the source texts (paragraphs, questions and
answers) to the dictionary, so that the names and terms the passages use count as known.
This module does not, so on its own `spell_issues` also flags those, e.g. "Greece".

The stimulus texts are in `data/stimuli/`: `adv_texts_with_ids.csv`, the paragraphs the
lists use, and `ele_texts_with_ids.csv`, the same paragraphs in simpler English. The lists
use only the first; the second is kept in step with it for a possible later condition.
"""

import re
from pathlib import Path
import pandas as pd
from spellchecker import SpellChecker


# A word: letters, with apostrophes inside it (don't, Greece's).
WORD = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)*")

# Checked by hand: correct words the dictionary does not know, such as British
# spellings and proper-noun plurals.
ACCEPTED = {"underserved", "behaviour", "carpathians"}

spell = SpellChecker()

# Checked by hand: {misspelling: correction}.
CORRECTIONS = {"iduring": "during"}


def spell_words(text):
    """The words of `text`. The typographic apostrophe (’) counts as a plain one, so
    "don’t" and "don't" are the same word."""
    return WORD.findall(str(text).replace("’", "'"))


def spell_issues(text):
    """{word: suggested correction} for the words of `text` the dictionary does not know.

    A word passes if one of its forms -- in lower case, or also without a final 's -- is
    in the dictionary or in `ACCEPTED`.
    """
    issues = {}
    for w in spell_words(text):
        forms = {w.lower(), w.lower().removesuffix("'s")}
        if forms & ACCEPTED or len(spell.unknown(forms)) < len(forms):
            continue
        issues[w] = spell.correction(w.lower().removesuffix("'s"))
    return issues


def apply_corrections(series):
    """`series` with each misspelling in `CORRECTIONS` replaced by its correction, as a
    whole word only."""
    out = series
    for wrong, right in CORRECTIONS.items():
        out = out.str.replace(rf"\b{re.escape(wrong)}\b", right, regex=True)
    return out
