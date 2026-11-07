"""Deliberation features built around the answer the participant selected.

Six per trial, none of which needs the answer key.

From the merged fixation sequence (`simpl_fix_by_label`, one entry per visit to an area):

    selected_switches        moves straight between the chosen answer and another answer,
                             either way
    selected_alt_run         the longest stretch going back and forth between the chosen
                             answer and one other answer; the question breaks it. The
                             sequence statistic `longest_alt_answer_run`, limited to runs
                             that include the chosen answer
    others_before_selected   how many other answers were visited before the chosen one was
                             first reached (0-3)
    others_after_selected    how many other answers were visited after it was first
                             reached (0-3)

From the press times (`button_clicks_data.csv`) and the fixation report, which share one
clock (`features/last_visited.py` reads them together the same way):

    first_look_to_select_ms  from the start of the first fixation on the chosen answer to
                             the final selection press; negative if it was selected before
                             it was ever looked at
    select_to_confirm_ms     from the final selection press to the confirming press

The final selection press is the latest one, and the confirming press the latest confirm,
as in `features/last_visited.py`. Study 2 logs most confirms twice, 1 ms apart; the latest
is the one `CONFIRM_FINAL_ANSWER_RT` measures.

**Left missing, not filled:** on a trial where the chosen answer was never looked at,
`others_before_selected`, `others_after_selected` and `first_look_to_select_ms` have no
value, and on a trial with no recorded selection press neither time has one. How many is
printed when the features are built. The model fills them with 0, with a warning, as it
fills every missing value (`conventions.md`, accepted deviations).
"""

from __future__ import annotations

import ast
from typing import Dict, List

import numpy as np
import pandas as pd

from src.checks import assert_full_coverage
from src.config import columns as Con
from src.config.columns import TRIAL_ID_COLS
from src.config.datasets import FIX_ANSWERS_PATH, KNOW_QA_FIX_ANSWERS_PATH, dataset
from src.features.fixations import load_fixations, place_fixations

# Each dataset's answer-screen fixation report: the files its build reads.
FIXATION_REPORTS = {"l1": FIX_ANSWERS_PATH, "knowqa": KNOW_QA_FIX_ANSWERS_PATH}

SEQUENCE_FEATURES: List[str] = [
    "selected_switches",
    "selected_alt_run",
    "others_before_selected",
    "others_after_selected",
]
TIMING_FEATURES: List[str] = ["first_look_to_select_ms", "select_to_confirm_ms"]
SELECTED_FEATURES: List[str] = SEQUENCE_FEATURES + TIMING_FEATURES

def _last_press(times) -> float:
    """The latest of a trial's presses. The press table holds them as a text list, e.g.
    "[45120, 47380]"; a trial with no press of that kind has no list."""
    return max(ast.literal_eval(times), default=np.nan) if isinstance(times, str) else np.nan


def sequence_features(seq, selected: str) -> Dict[str, float]:
    """The four sequence features of one trial, from its merged sequence of areas and the
    selected answer's area (`answer_A` .. `answer_D`)."""
    seq = list(seq)
    is_other = lambda a: a in Con.ANSWER_AREAS and a != selected

    switches = sum(
        1 for a, b in zip(seq, seq[1:]) if selected in (a, b) and (is_other(a) or is_other(b))
    )

    # A back-and-forth between the chosen answer and one other answer is a stretch in
    # which every move is between the same two of them.
    best = 1 if selected in seq else 0
    pair, length = None, 1
    for a, b in zip(seq, seq[1:]):
        this = frozenset((a, b)) if selected in (a, b) and (is_other(a) or is_other(b)) else None
        if this is None:
            pair, length = None, 1
            continue
        length = length + 1 if this == pair else 2
        pair = this
        best = max(best, length)

    if selected not in seq:
        before = after = np.nan
    else:
        first = seq.index(selected)
        before = len({a for a in seq[:first] if is_other(a)})
        after = len({a for a in seq[first + 1:] if is_other(a)})

    return {
        "selected_switches": switches,
        "selected_alt_run": best,
        "others_before_selected": before,
        "others_after_selected": after,
    }


def build_selected_features(dataset_key: str, verbose: bool = True) -> pd.DataFrame:
    """The six features for every trial of `dataset_key` ("l1" or "knowqa"): one row per
    trial, the trial ids and `SELECTED_FEATURES`.

    Reads the trial table, the press table, the word table and the answer screen's
    fixation report (about 1.8 GB on Study 1, so this takes a minute or two). Stops unless
    the press times and the fixation report agree with the tracker's own answering time:
    the confirming press minus the trial's first fixation must match
    `CONFIRM_FINAL_ANSWER_RT` to within 100 ms on the typical trial, or the two clocks
    are not the same and every time here would be wrong.
    """
    ds = dataset(dataset_key)
    keys = list(TRIAL_ID_COLS)
    trials = pd.read_csv(
        ds.trials,
        usecols=[*keys, Con.SIMPLIFIED_FIX_SEQ_BY_LABEL, Con.SELECTED_ANSWER_LABEL_COLUMN,
                 Con.CONFIRM_FINAL_ANSWER_RT],
        dtype={Con.PARTICIPANT_ID: str},
    )
    selected = Con.ANSWER_PREFIX + trials[Con.SELECTED_ANSWER_LABEL_COLUMN]

    # --- From the sequence -------------------------------------------------------
    seqs = trials[Con.SIMPLIFIED_FIX_SEQ_BY_LABEL].map(ast.literal_eval)
    out = pd.concat(
        [trials[keys], pd.DataFrame([sequence_features(s, sel) for s, sel in zip(seqs, selected)])],
        axis=1,
    )

    # --- From the presses ----------------------------------------------------------
    presses = pd.read_csv(
        ds.button_clicks,
        usecols=[*keys, Con.SELECT_ANS_TIMESTAMPS, Con.CONFIRM_TIMESTAMPS],
        dtype={Con.PARTICIPANT_ID: str},
    )
    presses[Con.TRIAL_ID] = presses[Con.TRIAL_ID].astype(trials[Con.TRIAL_ID].dtype)
    presses = presses.assign(
        _select=presses[Con.SELECT_ANS_TIMESTAMPS].map(_last_press),
        _confirm=presses[Con.CONFIRM_TIMESTAMPS].map(_last_press),
    )[[*keys, "_select", "_confirm"]]
    # The press table also holds practice and excluded trials; every analysed trial
    # must have exactly one row.
    assert_full_coverage(trials, presses, TRIAL_ID_COLS, "press times")

    # --- From the fixation report ------------------------------------------------------
    words = pd.read_csv(
        ds.interest_areas,
        usecols=[*keys, Con.INTEREST_AREA_ID, Con.AREA_LABEL_COLUMN],
        dtype={Con.PARTICIPANT_ID: str},
    )
    fixations = load_fixations(
        FIXATION_REPORTS[dataset_key],
        [Con.CURRENT_FIX_START, Con.CURRENT_FIX_INTEREST_AREA_ID, Con.NEAREST_IA],
        verbose=verbose,
    )
    placed = place_fixations(fixations, words, [Con.AREA_LABEL_COLUMN], label="selected", verbose=verbose)
    first_fix = placed.groupby(keys)[Con.CURRENT_FIX_START].min().rename("_first_fix")
    on_selected = placed.merge(
        pd.DataFrame({**{k: trials[k] for k in keys}, Con.AREA_LABEL_COLUMN: selected}),
        on=[*keys, Con.AREA_LABEL_COLUMN],
    )
    first_on_selected = on_selected.groupby(keys)[Con.CURRENT_FIX_START].min().rename("_first_on_selected")

    times = (
        trials[[*keys, Con.CONFIRM_FINAL_ANSWER_RT]]
        .merge(presses, on=keys, how="left")
        .merge(first_fix.reset_index(), on=keys, how="left")
        .merge(first_on_selected.reset_index(), on=keys, how="left")
    )
    gap = (times["_confirm"] - times["_first_fix"] - times[Con.CONFIRM_FINAL_ANSWER_RT]).abs()
    assert gap.median() < 100, (
        f"{dataset_key}: the confirming press minus the first fixation differs from "
        f"CONFIRM_FINAL_ANSWER_RT by {gap.median():.0f} ms on the typical trial; the press "
        f"times and the fixation report are not on the same clock."
    )
    times["first_look_to_select_ms"] = times["_select"] - times["_first_on_selected"]
    times["select_to_confirm_ms"] = times["_confirm"] - times["_select"]
    out = out.merge(times[[*keys, *TIMING_FEATURES]], on=keys, how="left", validate="one_to_one")

    if verbose:
        never = int(out["others_before_selected"].isna().sum())
        no_press = int(times["_select"].isna().sum())
        negative = int((out["first_look_to_select_ms"] < 0).sum())
        print(
            f"{dataset_key}: {len(out)} trials | chosen answer never looked at: {never} | "
            f"no selection press: {no_press} | selected before first looking at it: {negative}"
        )
    return out[[*keys, *SELECTED_FEATURES]]


def add_selected_features(hidden_table: pd.DataFrame, dataset_key: str, verbose: bool = True) -> pd.DataFrame:
    """`hidden_table` (`table.build_answer_hidden_table`) with `SELECTED_FEATURES` joined on,
    one row per trial; stops if any trial has no row."""
    feats = build_selected_features(dataset_key, verbose=verbose)
    assert_full_coverage(hidden_table, feats, TRIAL_ID_COLS, "selected-answer features")
    return hidden_table.merge(feats, on=list(TRIAL_ID_COLS), how="left")
