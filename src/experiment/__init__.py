"""How Study 2's materials were made: the texts and the counterbalanced trial lists.

Unlike the rest of `src/`, this package makes what participants saw instead of analysing
what they looked at. `ingest/` reads the recordings that came back; this is what went out.

    lists.py   the 54 ordered trial lists: which texts, questions and regime order each
               participant gets
    texts.py   the spelling check of the rephrased correct answers

The notebooks that built the materials are in `notebooks/experiment/`. They keep their own
copies of these functions, and nothing else in `src/` imports this package: it is here so
that a new experiment can start from functions rather than from notebook cells. The
materials themselves, and the lists that went to the presentation software (which is not
in this repo), are in `data/stimuli/`.

Both random draws in `lists.py` are seeded (42), but the same lists come out again only
from the same input rows in the same order; `lists.py` explains why.
"""
