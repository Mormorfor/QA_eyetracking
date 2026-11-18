"""The guessing data: one row per guess, one row per item, and the histogram of items.

Two datasets, kept apart because the question and answer texts changed between them:

* `"guess"` -- `data_raw/guess.tsv`. Each submission answers two OneStop and two RACE
  questions. About three guesses per item.
* `"pilot"` -- the pilot batches. OneStop only, an earlier version of the questions. Each
  submission works through one article, a question per paragraph, answering each question
  first before the paragraph is shown (`initial_response`, the guess used here) and again
  after reading it (`final_response`). About seventeen guesses per item.

**An item is a question at one difficulty level**, as both files define it. OneStop gives
the same question and answers at both levels (Ele, Adv), so the two levels of a OneStop
question are two items here that show the guesser identical text.

**The answer coding.** The answers are stored correct one first (`options`, separated by
`|`), and a response is the index of the chosen answer in that order, so `0` is right. The
screen order was shuffled (`new_options`) and plays no part here. Both loaders check the
coding against the data, since a wrong reading of it would still give plausible numbers.
"""

from __future__ import annotations

from fractions import Fraction
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from scipy.stats import binom

from src.config.datasets import GUESS_PATH, GUESS_PILOT_DIR, GUESS_PILOT_FILES

DATASETS = {
    "guess": "Prolific guessing data (guess.tsv)",
    "pilot": "Pilot batches, answered before reading the paragraph",
}
# What identifies one item in each file. The pilot has no item id, so its key is the text.
ITEM_KEY: Dict[str, List[str]] = {
    "guess": ["item_id"],
    "pilot": ["article", "difficulty", "question"],
}
ANSWER_LABELS = ["A", "B", "C", "D"]  # in stored order: A is the correct answer
CHANCE = 0.25
PRACTICE = "practice_reading_comprehension"
SEP = "|"


def _drop_practice(raw: pd.DataFrame, name: str, verbose: bool) -> pd.DataFrame:
    """The practice questions left out: they are the same two questions for everyone, and
    not part of the item set."""
    practice = raw["trial_type"].eq(PRACTICE)
    assert set(raw.loc[~practice, "trial_type"]) == {"reading_comprehension"}, (
        f"{name}: unexpected trial types {set(raw['trial_type'])}")
    if verbose:
        print(f"{name}: {int(practice.sum())} practice rows left out, {int((~practice).sum())} guesses kept")
    return raw[~practice].copy()


def _add_answers(g: pd.DataFrame, response_col: str) -> pd.DataFrame:
    """`is_correct`, the chosen answer's label, and the correct answer's text."""
    assert g[response_col].isin(range(4)).all(), f"{response_col} outside 0-3"
    options = g["options"].str.split(SEP, regex=False)
    assert options.str.len().eq(4).all(), "an options string does not split into exactly four answers"
    g["chosen"] = g[response_col].astype(int)
    g["chosen_label"] = g["chosen"].map(dict(enumerate(ANSWER_LABELS)))
    g["is_correct"] = g["chosen"].eq(0).astype(int)
    g["correct_answer"] = options.str[0]
    return g


def load_guess(verbose: bool = True) -> pd.DataFrame:
    """`data_raw/guess.tsv`, one row per guess, practice questions left out.

    **The coding check.** The file records the running score shown before each question
    (`cumulative_score`). Within a submission it must rise by one after exactly the
    answers coded `0`; anything else stops the load."""
    g = _drop_practice(pd.read_csv(GUESS_PATH, sep="\t"), "guess", verbose)
    g = _add_answers(g, "answer_response")
    g = g.sort_values(["submission_ind", "trial_number"])
    assert not g.duplicated(["submission_ind", "trial_number"]).any(), "a submission repeats a trial number"
    gain = -g.groupby("submission_ind")["cumulative_score"].diff(-1)  # next score minus this one
    has_next = gain.notna()
    mismatched = int((gain[has_next] != g.loc[has_next, "is_correct"]).sum())
    assert mismatched == 0, f"guess: the running score disagrees with answer 0 being correct on {mismatched} trials"
    g["dataset"] = "guess"
    g["participant"] = g["submission_ind"].astype(str)
    return g.reset_index(drop=True)


def load_pilot(verbose: bool = True) -> pd.DataFrame:
    """The pilot batches, one row per guess (`initial_response`), practice questions left
    out. `position` is the question's place in the participant's session (1 = first).

    **The coding check.** The pilot has no running score. Instead: after reading the
    paragraph, the answer stored first must be the one chosen more than half the time.
    Were `0` not the correct answer, it would get about one choice in four."""
    raw = pd.concat(
        [pd.read_csv(GUESS_PILOT_DIR / f, low_memory=False).assign(batch=f.split(".")[0])
         for f in GUESS_PILOT_FILES],
        ignore_index=True,
    )
    g = _drop_practice(raw, "pilot", verbose)
    g = _add_answers(g, "initial_response")
    after_reading = g["final_response"].eq(0).mean()
    assert after_reading > 0.5, (
        f"pilot: after reading, the answer stored first was chosen on only {after_reading:.0%} of "
        "trials, so 0 may not be the correct answer")
    assert not g.duplicated(["participant_id", "article", "difficulty", "question"]).any(), (
        "pilot: a participant answered the same item twice")
    g["source"] = "Onestop"
    g["dataset"] = "pilot"
    g["participant"] = g["participant_id"].astype(str)
    g["position"] = g.groupby(["batch", "submission_id"])["trial_number"].rank(method="first").astype(int)
    if verbose:
        print(f"pilot: after reading the paragraph, {after_reading:.0%} right (the guess: {g['is_correct'].mean():.0%})")
    return g


def load(dataset: str, verbose: bool = True) -> pd.DataFrame:
    """One dataset's guesses, by its key in `DATASETS`."""
    return {"guess": load_guess, "pilot": load_pilot}[dataset](verbose=verbose)


def item_table(guesses: pd.DataFrame) -> pd.DataFrame:
    """One row per item: how many guessed it, how many were right, the share right, how many
    chose each answer, the most any one wrong answer got, and the question and correct
    answer. Sorted from most to least often guessed right."""
    dataset = guesses["dataset"].iloc[0]
    key = ITEM_KEY[dataset]
    text = guesses.groupby(key)[["source", "difficulty", "question", "options"]].nunique()
    assert (text == 1).all().all(), f"{dataset}: an item has more than one question, answer set, source or level"

    grouped = guesses.groupby(key)
    items = grouped[["source", "difficulty", "question", "correct_answer"]].first()
    items = items.loc[:, ~items.columns.isin(key)]
    items["n_guesses"] = grouped.size()
    items["n_correct"] = grouped["is_correct"].sum()
    items["share_correct"] = items["n_correct"] / items["n_guesses"]
    chose = pd.crosstab([guesses[c] for c in key], guesses["chosen_label"])
    chose = chose.reindex(columns=ANSWER_LABELS, fill_value=0)  # an answer nobody chose: a real zero
    items = items.join(chose.add_prefix("chose_"))
    items["max_wrong_n"] = items[[f"chose_{c}" for c in ANSWER_LABELS[1:]]].max(axis=1)
    assert len(items) == len(chose) and items["n_guesses"].sum() == len(guesses)
    return (items.reset_index()
                 .sort_values(["share_correct", "n_guesses"], ascending=[False, False], ignore_index=True))


def histogram_table(items: pd.DataFrame, bin_width: Optional[float] = None) -> pd.DataFrame:
    """How many items were guessed right by each share of their guessers, per source,
    against two expectations:

    * `expected_same_rate` -- if every item of the source were guessed right at the
      source's overall rate. More items than this at the two ends means items really
      differ in how guessable they are, beyond the luck of a few guesses.
    * `expected_chance` -- if every guess were one in four.

    Each item contributes its own binomial, with its own number of guesses. With
    `bin_width`, shares are pooled into bins of that width, the last one closed; without
    it, each share that occurs is its own bar (for items with only a few guesses)."""
    rows = []
    for source, it in items.groupby("source"):
        rate = it["n_correct"].sum() / it["n_guesses"].sum()
        for _, item in it.iterrows():
            n = int(item["n_guesses"])
            k = np.arange(n + 1)
            rows.append(pd.DataFrame({
                "source": source, "rate": rate, "n": n, "k": k,
                "observed": (k == item["n_correct"]).astype(float),
                "expected_same_rate": binom.pmf(k, n, rate),
                "expected_chance": binom.pmf(k, n, CHANCE),
            }))
    long = pd.concat(rows, ignore_index=True)
    long["share"] = long["k"] / long["n"]
    if bin_width is None:
        long["bin"] = [str(Fraction(int(k), int(n))) for k, n in zip(long["k"], long["n"])]
        long["bin_low"] = long["bin_high"] = long["share"]
    else:
        edges = np.round(np.arange(0, 1 + bin_width / 2, bin_width), 10)
        idx = np.clip(np.digitize(long["share"], edges[1:-1], right=False), 0, len(edges) - 2)
        long["bin_low"], long["bin_high"] = edges[idx], edges[idx + 1]
        long["bin"] = [f"{lo:.2f}-{hi:.2f}" for lo, hi in zip(long["bin_low"], long["bin_high"])]
    out = (long.groupby(["source", "bin_low", "bin_high", "bin"], as_index=False)
               .agg(rate=("rate", "first"), items_observed=("observed", "sum"),
                    items_expected_same_rate=("expected_same_rate", "sum"),
                    items_expected_chance=("expected_chance", "sum")))
    n_items = items.groupby("source").size()
    assert (out.groupby("source")["items_observed"].sum() == n_items).all(), "an item fell outside the bins"
    return out


def summary_table(guesses: pd.DataFrame) -> pd.DataFrame:
    """Guesses, items, participants and the share right, by source and level."""
    key = ITEM_KEY[guesses["dataset"].iloc[0]]
    by = ["source", "difficulty"]
    out = guesses.groupby(by).agg(guesses=("is_correct", "size"), participants=("participant", "nunique"),
                                  share_correct=("is_correct", "mean"))
    out["items"] = guesses.drop_duplicates(key).groupby(by).size()
    return out.reset_index()


def position_table(guesses: pd.DataFrame) -> pd.DataFrame:
    """Pilot only: the share of guesses right by the question's place in the session.
    Every session is one article, a paragraph per question, so from the second question on
    the participant has read earlier paragraphs of the same article before guessing."""
    return (guesses.groupby("position")
                   .agg(guesses=("is_correct", "size"), share_correct=("is_correct", "mean"))
                   .reset_index())


def extremes(items: pd.DataFrame, n: int = 15) -> Dict[str, pd.DataFrame]:
    """The `n` items most often and least often guessed right. Ties at the top are broken by
    more guesses; at the bottom, by more guesses, then by how strongly the guessers agreed on
    one wrong answer."""
    most = items.sort_values(["share_correct", "n_guesses"], ascending=[False, False]).head(n)
    least = (items.assign(_wrong=items["max_wrong_n"] / items["n_guesses"])
                  .sort_values(["share_correct", "n_guesses", "_wrong"], ascending=[True, False, False])
                  .drop(columns="_wrong").head(n))
    return {"most": most.reset_index(drop=True), "least": least.reset_index(drop=True)}
