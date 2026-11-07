"""The trial table with nothing in it that says which answer is correct.

The paper's trial table names the four answers by meaning: `answer_A` is always the
correct one, and the comparisons (`__correct`, `__wrong_mean`, `__contrast`, ...) are
built around it. A model given any of those columns knows which answer is correct. This
module rebuilds the table so that it does not. Every measure taken on the answers is given
three other ways, none of which needs the answer key:

* **by position** -- the measure at each screen position: `skip_rate__answer_A` becomes
  `skip_rate__answer_0(top)` etc. The positions are shuffled on every trial, and the
  correct answer is in each one on about a quarter of the trials in both studies, so
  position says nothing about correctness.
* **by selection** -- the paper's five comparisons with the answer the participant
  selected in place of the correct one: `skip_rate__selected`, `__unselected_mean`,
  `__selected_contrast`, `__selected_distance_furthest`, `__selected_distance_closest`,
  made by the paper's own function. Which answer was selected is observed; whether it was
  right is not. **One way in for the answer key remains:** the answers differ in length,
  and the selected answer's length alone predicts correctness a little above chance (a
  balanced accuracy of about 0.53 on Study 1 and 0.55 on Study 2), so a measure that grows
  with length can carry some of that.
* **by spread** -- the largest, average and smallest value over the four answers
  (`skip_rate__answer_max`, `__answer_mean`, `__answer_min`): how attention was spread,
  with no answer singled out. Answers with no value (pupil size, first-fixation duration
  on an answer never looked at) are skipped, as the paper's comparisons skip them.

Everything that does not depend on which answer is which is kept as it is: the question
area, the sequence statistics, the number of selection presses, the total answering time.

The table is built from a list of the columns to keep (`ANSWER_FREE_COLS`, plus the
re-keyed ones), not by removing the known leaky ones, so a column added to the paper's
table later does not slip in. The last step checks that no column name refers to an answer
letter or to the correct / wrong comparisons.

**Paragraph features are left out**, though they say nothing about the answer screen: the
paragraph's parts are defined by the answer key ("critical" holds the correct answer,
"distractor" supports a wrong one), so time on them is time on the correct answer's
evidence. KnowQA has no paragraph screen in any case.

`hide_answers` turns one of the paper's feature lists into its version on this table, by
one rule per way of naming the answers, so every model here is the image of a model in the
paper.

`load_hidden_table` adds the features built around the selected answer
(`selected_features.py`) and saves the result beside the paper's `model_ready.csv`, so it
is built once.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd

from src.checks import assert_full_coverage
from src.config import columns as Con
from src.config.columns import TRIAL_ID_COLS
from src.config.datasets import dataset
from src.features.build import add_answer_correct_wrong_contrast_columns
from src.features.sequences import DERIVED_COLS
from src.explorations.answer_hidden.cache import table_digest
from src.explorations.answer_hidden.selected_features import FIXATION_REPORTS, add_selected_features

SEP = Con.DERIVED_SEP

# The four answer positions, in the order of `LOC_CHOICES`: top, left, right, bottom.
POSITION_AREAS: List[str] = list(Con.LOC_CHOICES[1:])

# Left out of a set of 0/1 columns, or of shares that add up to 1, as the reference, so the
# columns are not perfectly collinear. The paper's per-answer last-look columns leave out
# answer_D for the same reason.
REFERENCE_POSITION = POSITION_AREAS[-1]

# The two "last look" families: the area looked at just before the confirming press, and
# just before the last selection press.
LAST_LOOK_PREFIXES = ("last_before_confirm", "last_before_select")

# Columns measured on one answer, each a stem and the separator before the area name: the
# measures (`MEASURE_STEMS`), and with them the last-look 0/1 columns (`PER_ANSWER_STEMS`),
# which get positions and "was it the selected answer", but no comparisons and no spread.
MEASURE_STEMS = (
    [(m, SEP) for m in Con.AREA_METRIC_COLUMNS_MODELING]
    + [(m, "_") for m in Con.RT_TFD_METRICS]
)
PER_ANSWER_STEMS = MEASURE_STEMS + [(p, "_") for p in LAST_LOOK_PREFIXES]

# The three ways of naming the answers without the answer key (see the module docstring).
NAMINGS = ("position", "selected", "spread")

# The paper's five comparisons, and their versions with the selected answer as reference.
SELECTED_FOR = {
    Con.CORRECT_SUFFIX: "selected",
    Con.WRONG_MEAN_SUFFIX: "unselected_mean",
    Con.CONTRAST_SUFFIX: "selected_contrast",
    Con.DISTANCE_FURTHEST_SUFFIX: "selected_distance_furthest",
    Con.DISTANCE_CLOSEST_SUFFIX: "selected_distance_closest",
}
SELECTED_SUFFIXES = list(SELECTED_FOR.values())

SPREAD_SUFFIXES = ["answer_max", "answer_mean", "answer_min"]

# What a column name ends in when it is measured on an answer named by meaning, or on a
# comparison of the correct answer with the wrong ones. "wrong" is the last-look family's
# "the last look was at a wrong answer".
_BY_MEANING = (
    *Con.ANSWER_AREAS,
    Con.CORRECT_SUFFIX,
    Con.WRONG_MEAN_SUFFIX,
    Con.CONTRAST_SUFFIX,
    Con.DISTANCE_FURTHEST_SUFFIX,
    Con.DISTANCE_CLOSEST_SUFFIX,
    "wrong",
)

# Model features that do not depend on which answer is correct, taken over unchanged.
ANSWER_FREE_COLS: List[str] = (
    [f"{m}{SEP}{Con.QUESTION_AREA}" for m in Con.AREA_METRIC_COLUMNS_MODELING]
    + [f"{m}_{Con.QUESTION_AREA}" for m in Con.RT_TFD_METRICS]
    + [f"{p}_{Con.QUESTION_AREA}" for p in LAST_LOOK_PREFIXES]
    + list(DERIVED_COLS)
    + [Con.NUM_OF_SELECTS, Con.TOTAL_ANSWERING_RT, Con.TOTAL_ANSWERING_RT_NORMALIZED]
)

# Carried along but never features. Study 2 only, except the item id.
CARRIED_COLS = (
    Con.TEXT_ID_WITH_Q_COLUMN,
    Con.REGIME_COLUMN,
    Con.SESSION_ID,
    Con.CONFIDENCE_RATING_COLUMN,
)


def _split(col: str) -> Optional[tuple]:
    """`(stem, separator, area)` for a column measured on one area or comparison, else None."""
    if SEP in col:
        stem, area = col.split(SEP, 1)
        return stem, SEP, area
    for stem in (*Con.RT_TFD_METRICS, *LAST_LOOK_PREFIXES):
        if col.startswith(f"{stem}_"):
            return stem, "_", col[len(stem) + 1:]
    return None


def _names_an_answer_by_meaning(col: str) -> bool:
    parts = _split(col)
    return (parts is not None and parts[2] in _BY_MEANING) or any(
        a in col for a in Con.ANSWER_AREAS
    )


# ---------------------------------------------------------------------------
# Which letter sat in which position
# ---------------------------------------------------------------------------

def letter_positions(dataset_key: str) -> pd.DataFrame:
    """One row per trial, one column per screen position, holding the letter shown there
    (`answer_A` .. `answer_D`).

    Read from the word table, where each word carries both its area's letter and its
    position: the same words the per-area measurements were computed over. Stops unless
    every trial has its four letters in four different positions, and unless this agrees
    with `answers_order`, the experiment's own record of the order.
    """
    ds = dataset(dataset_key)
    cols = [*TRIAL_ID_COLS, Con.AREA_LABEL_COLUMN, Con.AREA_SCREEN_LOCATION]
    words = pd.read_csv(ds.interest_areas, usecols=cols, dtype={Con.PARTICIPANT_ID: str})
    pairs = words[words[Con.AREA_LABEL_COLUMN].isin(Con.ANSWER_AREAS)].drop_duplicates()

    per_trial = pairs.groupby(list(TRIAL_ID_COLS))
    one_to_one = (
        (per_trial.size() == 4)
        & (per_trial[Con.AREA_LABEL_COLUMN].nunique() == 4)
        & (per_trial[Con.AREA_SCREEN_LOCATION].nunique() == 4)
    )
    assert one_to_one.all(), (
        f"{dataset_key}: {int((~one_to_one).sum())} trial(s) do not have four answers in "
        f"four different positions, so letters cannot be re-keyed to positions."
    )

    wide = pairs.pivot(
        index=list(TRIAL_ID_COLS),
        columns=Con.AREA_SCREEN_LOCATION,
        values=Con.AREA_LABEL_COLUMN,
    )[POSITION_AREAS].reset_index()
    wide.columns.name = None

    # `answers_order` lists the letters by position, e.g. "['B', 'C', 'D', 'A']".
    trials = pd.read_csv(
        ds.trials,
        usecols=[*TRIAL_ID_COLS, Con.ANSWERS_ORDER_COLUMN],
        dtype={Con.PARTICIPANT_ID: str},
    )
    order = pd.DataFrame(
        trials[Con.ANSWERS_ORDER_COLUMN].map(ast.literal_eval).tolist(),
        columns=POSITION_AREAS,
    ).radd(Con.ANSWER_PREFIX)
    order = pd.concat([trials[list(TRIAL_ID_COLS)], order], axis=1)
    check = wide.merge(order, on=list(TRIAL_ID_COLS), how="inner", suffixes=("", "_order"))
    assert len(check) == len(wide), (
        f"{dataset_key}: {len(wide) - len(check)} trial(s) of the word table are not in trials.csv"
    )
    disagree = ~np.all(
        check[POSITION_AREAS].to_numpy() == check[[f"{p}_order" for p in POSITION_AREAS]].to_numpy(),
        axis=1,
    )
    assert not disagree.any(), (
        f"{dataset_key}: on {int(disagree.sum())} trial(s) the word table and answers_order "
        f"put the answers in different positions."
    )
    return wide


SELECTED_COL = "selected_area"


def selected_letters(dataset_key: str) -> pd.DataFrame:
    """One row per trial: the letter the participant selected, as an area name
    (`answer_A` .. `answer_D`), in `selected_area`."""
    trials = pd.read_csv(
        dataset(dataset_key).trials,
        usecols=[*TRIAL_ID_COLS, Con.SELECTED_ANSWER_LABEL_COLUMN],
        dtype={Con.PARTICIPANT_ID: str},
    )
    trials[SELECTED_COL] = Con.ANSWER_PREFIX + trials.pop(Con.SELECTED_ANSWER_LABEL_COLUMN)
    return trials


# ---------------------------------------------------------------------------
# The table
# ---------------------------------------------------------------------------

def build_answer_hidden_table(trial_df: pd.DataFrame, dataset_key: str) -> pd.DataFrame:
    """`trial_df`, the paper's trial table for `dataset_key` ("l1" or "knowqa"), with
    nothing that says which answer is correct.

    Keeps the trial ids, `is_correct`, `CARRIED_COLS` when present and `ANSWER_FREE_COLS`,
    and gives every per-answer column three ways (see the module docstring):

    * by position: `PER_ANSWER_STEMS` x `POSITION_AREAS`, the same numbers as in
      `trial_df`, moved to other columns;
    * by selection: `MEASURE_STEMS` x `SELECTED_SUFFIXES`, and whether each last look
      was at the selected answer (`last_before_confirm_selected`);
    * by spread: `MEASURE_STEMS` x `SPREAD_SUFFIXES`.

    A missing value stays missing.
    """
    layout = letter_positions(dataset_key).merge(
        selected_letters(dataset_key), on=list(TRIAL_ID_COLS), how="left"
    )
    assert_full_coverage(trial_df, layout, TRIAL_ID_COLS, "answer positions")
    merged = trial_df.merge(layout, on=list(TRIAL_ID_COLS), how="left")

    # The selected letter is the only thing here not already in the paper's table, so it
    # is checked against the outcome: correct exactly when answer_A was selected.
    agrees = (merged[SELECTED_COL] == Con.CORRECT_ANSWER_AREA) == (merged[Con.IS_CORRECT_COLUMN] == 1)
    assert agrees.all(), (
        f"{dataset_key}: on {int((~agrees).sum())} trial(s) the selected answer and "
        f"is_correct disagree."
    )

    letter_index = {a: i for i, a in enumerate(Con.ANSWER_AREAS)}
    rows = np.arange(len(merged))[:, None]
    # For each trial and position, which of the four letter columns to read.
    by_position_idx = merged[POSITION_AREAS].apply(lambda s: s.map(letter_index)).to_numpy(int)
    # For each trial: the selected letter first, then the other three.
    selected = merged[SELECTED_COL].map(letter_index).to_numpy(int)
    others = np.array([[j for j in range(4) if j != s] for s in selected])
    by_selection_idx = np.column_stack([selected, others])

    carried = [c for c in CARRIED_COLS if c in merged.columns]
    out = {c: merged[c] for c in [*TRIAL_ID_COLS, Con.IS_CORRECT_COLUMN, *carried, *ANSWER_FREE_COLS]}
    # The selected answer and the other three, as temporary columns for the paper's
    # comparison function, which takes one reference area and three others.
    ref, rest = "_selected", ["_other1", "_other2", "_other3"]
    by_selection = {}
    for stem, sep in PER_ANSWER_STEMS:
        by_letter = merged[[f"{stem}{sep}{a}" for a in Con.ANSWER_AREAS]].to_numpy(float)
        by_position = by_letter[rows, by_position_idx]
        for j, p in enumerate(POSITION_AREAS):
            out[f"{stem}{sep}{p}"] = by_position[:, j]

        if stem in LAST_LOOK_PREFIXES:
            out[f"{stem}_selected"] = by_letter[rows[:, 0], selected]
            continue
        ordered = by_letter[rows, by_selection_idx]
        for j, name in enumerate([ref, *rest]):
            by_selection[f"{stem}{sep}{name}"] = ordered[:, j]
        spread = pd.DataFrame(by_letter, index=merged.index)
        out[f"{stem}{sep}answer_max"] = spread.max(axis=1)
        out[f"{stem}{sep}answer_mean"] = spread.mean(axis=1)
        out[f"{stem}{sep}answer_min"] = spread.min(axis=1)

    by_selection = pd.DataFrame(by_selection, index=merged.index)
    for sep in (SEP, "_"):
        by_selection = add_answer_correct_wrong_contrast_columns(
            by_selection,
            metric_cols=[stem for stem, s in MEASURE_STEMS if s == sep],
            sep=sep,
            correct_label=ref,
            wrong_labels=rest,
            out_correct_suffix=SELECTED_FOR[Con.CORRECT_SUFFIX],
            out_wrong_mean_suffix=SELECTED_FOR[Con.WRONG_MEAN_SUFFIX],
            out_contrast_suffix=SELECTED_FOR[Con.CONTRAST_SUFFIX],
            out_distance_furthest_suffix=SELECTED_FOR[Con.DISTANCE_FURTHEST_SUFFIX],
            out_distance_closest_suffix=SELECTED_FOR[Con.DISTANCE_CLOSEST_SUFFIX],
        )
    for stem, sep in MEASURE_STEMS:
        for suffix in SELECTED_SUFFIXES:
            out[f"{stem}{sep}{suffix}"] = by_selection[f"{stem}{sep}{suffix}"]
    out = pd.DataFrame(out, index=merged.index)

    features = [c for c in out.columns if c not in (*TRIAL_ID_COLS, Con.IS_CORRECT_COLUMN, *carried)]
    leaky = [c for c in features if _names_an_answer_by_meaning(c)]
    assert not leaky, f"columns that name an answer by meaning: {leaky}"
    return out


# ---------------------------------------------------------------------------
# The saved table
# ---------------------------------------------------------------------------

def hidden_table_path(dataset_key: str) -> Path:
    """Where the answer-hidden table of `dataset_key` is saved: beside the paper's
    `model_ready.csv`, as `answer_hidden.csv`, with `answer_hidden.json` naming what it was
    built from."""
    return dataset(dataset_key).features / "answer_hidden.csv"


def _built_from(known: pd.DataFrame, dataset_key: str) -> dict:
    """What the saved table depends on: the paper's trial table, by a fingerprint of its
    contents, and the files the selected-answer features read, by when they last changed."""
    ds = dataset(dataset_key)
    files = [ds.trials, ds.interest_areas, ds.button_clicks, Path(FIXATION_REPORTS[dataset_key])]
    return {"known_table": table_digest(known), **{str(p.name): p.stat().st_mtime for p in files}}


def load_hidden_table(
    known: pd.DataFrame,
    dataset_key: str,
    *,
    rebuild: bool = False,
    verbose: bool = True,
) -> pd.DataFrame:
    """The answer-hidden table of `dataset_key` (`build_answer_hidden_table`) with the
    selected-answer features (`selected_features.py`), built from `known`, the paper's
    trial table.

    Read from `hidden_table_path` when it was built from this same `known` and the same
    source files; otherwise -- or with `rebuild=True` -- built (a minute or two, for the
    fixation reports) and saved there first, with a note saying why. Always returned as
    read from the file, so the table is the same whether it was just built or not.
    """
    path = hidden_table_path(dataset_key)
    meta = path.with_suffix(".json")
    built_from = _built_from(known, dataset_key)
    # pandas' default number parser can come back one unit off in the last digit;
    # "round_trip" reads every value back exactly as it was written.
    read = dict(dtype={Con.PARTICIPANT_ID: str}, float_precision="round_trip")

    if path.exists() and meta.exists() and not rebuild:
        saved = json.loads(meta.read_text(encoding="utf-8"))
        if saved == built_from:
            if verbose:
                print(f"{dataset_key}: reading the answer-hidden table from {path}")
            return pd.read_csv(path, **read)
        changed = sorted(k for k in built_from if saved.get(k) != built_from[k])
        print(f"{dataset_key}: rebuilding the answer-hidden table -- changed since it was saved: {changed}")
    elif verbose:
        print(f"{dataset_key}: building the answer-hidden table")

    table = add_selected_features(build_answer_hidden_table(known, dataset_key), dataset_key, verbose=verbose)
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False)
    meta.write_text(json.dumps(built_from, indent=1), encoding="utf-8")
    return pd.read_csv(path, **read)


# ---------------------------------------------------------------------------
# Feature sets
# ---------------------------------------------------------------------------

def _replacement(stem: str, sep: str, area: str, by: str) -> List[str]:
    """What one answer-keyed column becomes under naming `by`."""
    last_look = stem in LAST_LOOK_PREFIXES
    if by == "position":
        positions = [p for p in POSITION_AREAS if not (last_look and p == REFERENCE_POSITION)]
        return [f"{stem}{sep}{p}" for p in positions]
    if by == "selected":
        if last_look:
            return [f"{stem}_selected"]
        if area in SELECTED_FOR:
            return [f"{stem}{sep}{SELECTED_FOR[area]}"]
        return [f"{stem}{sep}{s}" for s in SELECTED_SUFFIXES]
    if by == "spread":
        return [] if last_look else [f"{stem}{sep}{s}" for s in SPREAD_SUFFIXES]
    raise ValueError(f"by must be one of {NAMINGS}, got {by!r}")


# Under each naming, the shares of looking time that others fix exactly, since the
# question's and the answers' shares add up to 1: (the shares fixed, the shares fixing
# them). One is left out when everything it follows from is in the set.
_SHARE = Con.AREA_DWELL_PROPORTION
_REDUNDANT_SHARE = {
    "position": ([REFERENCE_POSITION], [Con.QUESTION_AREA, *POSITION_AREAS[:-1]]),
    "selected": (["unselected_mean", "selected_contrast"], [Con.QUESTION_AREA, "selected"]),
    "spread": (["answer_mean"], [Con.QUESTION_AREA]),
}


def hide_answers(cols: Sequence[str], by: str = "position") -> List[str]:
    """The version of a paper feature list that does not know which answer is correct,
    with the answers named `by` "position", "selected" or "spread".

    A column measured on an answer by letter, or on a comparison of the correct answer
    with the wrong ones (`skip_rate__correct`, `num_label_visits__contrast`,
    `RT_normalized_answer_B`), becomes:

    * position: the same measure at each of the four positions;
    * selected: the matching comparison with the selected answer
      (`__correct` -> `__selected`, `__wrong_mean` -> `__unselected_mean`, ...), so the
      set keeps its size; a column on one letter becomes all five;
    * spread: the measure's largest, average and smallest value over the four answers.

    A last-look column about the answers (`last_before_confirm_correct`) becomes one 0/1
    column per position (bottom left out as the reference), whether the last look was at
    the selected answer, or nothing under spread: the only fact about it that names no
    answer, "the last look was at some answer", the question and no-look columns already
    fix.

    Several columns can map to the same ones; each is kept once. A column that does not
    depend on which answer is correct (`ANSWER_FREE_COLS`) is kept. Anything else stops
    the run, paragraph features included (see the module docstring).

    The shares of looking time add up to 1 over the question and the answers, so a share
    that others in the set fix exactly (`_REDUNDANT_SHARE`) is left out. Only shares: the
    paper's own comparisons are collinear too (`__contrast` is `__correct` minus
    `__wrong_mean`), and a set that has all three keeps all three versions here.
    """
    out: List[str] = []
    for col in cols:
        parts = _split(col)
        if parts is not None and parts[2] in _BY_MEANING:
            out += _replacement(*parts, by)
        elif col in ANSWER_FREE_COLS:
            out.append(col)
        else:
            raise ValueError(
                f"{col!r} has no version that hides the answers: it is neither measured "
                f"on an answer nor independent of which answer is correct."
            )
    out = list(dict.fromkeys(out))

    redundant, fixed_by = _REDUNDANT_SHARE[by]
    if all(f"{_SHARE}{SEP}{a}" in out for a in fixed_by):
        out = [c for c in out if c not in {f"{_SHARE}{SEP}{r}" for r in redundant}]
    return out
