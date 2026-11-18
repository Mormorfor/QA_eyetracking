"""One row per trial and answer, for either study: where the answer sat, what kind it was, how
long it was, when the scan first reached it, how much gaze it drew, and whether it was chosen.

Built on the answer-screen fixations and decision times of `explorations/drift_diffusion/
data.py`: the decision is the final selection press, and only gaze before it counts.

**The unit of randomisation.** Both studies place the four answers at the four positions at
random, but not at the same grain. Study 2 shuffles every trial on its own. Study 1 shuffles
each item once per counterbalancing list, so everyone on a list who sees an item sees the same
order. `unit` names the shuffle a trial belongs to, and every randomisation test re-draws the
shuffle per unit, so trials that share one move together.

**Trials with no decision time, or no look at an answer before it,** keep their choice and
placement, so they count wherever only those matter. What needs gaze before the decision --
`order`, `n_reached`, `first_position`, `gaze` -- is missing on them, and `excluded` gives the
reason. Every analysis that needs gaze uses `with_gaze`, and its table says how many trials it
had.
"""

from __future__ import annotations

import ast
from typing import Iterator, Sequence, Tuple

import numpy as np
import pandas as pd

from src.checks import assert_full_coverage
from src.config import columns as Con
from src.config.datasets import dataset
from src.explorations.drift_diffusion.data import CONDITION, CONDITION_ORDER, GAZE_COLS, KEYS, LABELS

PID = Con.PARTICIPANT_ID
ITEM = Con.TEXT_ID_WITH_Q_COLUMN
POSITIONS = ["top", "left", "right", "bottom"]  # numbered 0-3, as `answers_order` counts them
LOC_TO_POSITION = dict(zip(Con.LOC_CHOICES[1:], POSITIONS))
ORDER_NAMES = {0: "not reached", 1: "1st", 2: "2nd", 3: "3rd", 4: "4th"}
KIND_NAMES = {
    "A": "A (correct)",
    "B": "B (misreads the right sentence)",
    "C": "C (from another sentence)",
    "D": "D (not in the text)",
}

# The trials that share one shuffle of the answers.
UNIT_COLS = {
    "knowqa": [PID, Con.TRIAL_ID],
    "l1": [ITEM, Con.LIST_COLUMN],
}

GUESSING = "no knowledge"
ALL_PARTICIPANTS = "all participants"
# The groups each table is computed for. Study 1's two groups also go in together: they see the
# same items, in the same lists.
SUBSETS = {
    "knowqa": list(CONDITION_ORDER["knowqa"]),
    "l1": [ALL_PARTICIPANTS, *CONDITION_ORDER["l1"]],
}
STUDY_NAMES = {"knowqa": "Study 2", "l1": "Study 1"}


def subsets(answers: pd.DataFrame, dataset_key: str) -> Iterator[Tuple[str, pd.DataFrame]]:
    """Each group of `SUBSETS` with its rows."""
    for name in SUBSETS[dataset_key]:
        yield name, answers if name == ALL_PARTICIPANTS else answers[answers[CONDITION] == name]


def with_gaze(answers: pd.DataFrame) -> pd.DataFrame:
    """The rows of trials with a decision time and a look at an answer before it."""
    return answers[answers["excluded"] == ""]


def participant_mean(df: pd.DataFrame, by: Sequence[str], value: str) -> pd.DataFrame:
    """Each participant's mean of `value` within `by`, then the mean over participants, with its
    standard error and the number of participants and rows behind it."""
    by = list(by)
    per = df.groupby([*by, PID])[value].agg(["mean", "size"]).reset_index()
    out = per.groupby(by).agg(mean=("mean", "mean"), sd=("mean", "std"), n_participants=("mean", "size"),
                              n_rows=("size", "sum"))
    out["se"] = out["sd"] / np.sqrt(out["n_participants"])
    return out.drop(columns="sd").reset_index()


def _screen(dataset_key: str) -> pd.DataFrame:
    """Where each answer sat on each trial and how many words it has, from the word table,
    checked against the design's own record of the shuffle (`answers_order`)."""
    ds = dataset(dataset_key)
    words = pd.read_csv(ds.interest_areas, usecols=[*KEYS, Con.AREA_LABEL_COLUMN, Con.AREA_SCREEN_LOCATION],
                        dtype={PID: str})
    words = words[words[Con.AREA_LABEL_COLUMN].isin(Con.ANSWER_AREAS)]
    screen = words.groupby([*KEYS, Con.AREA_LABEL_COLUMN]).agg(
        n_places=(Con.AREA_SCREEN_LOCATION, "nunique"),
        place=(Con.AREA_SCREEN_LOCATION, "first"),
        n_words=(Con.AREA_SCREEN_LOCATION, "size"),
    ).reset_index()
    assert (screen["n_places"] == 1).all(), "an answer's words are spread over more than one screen position"
    screen["label"] = screen[Con.AREA_LABEL_COLUMN].str.removeprefix(Con.ANSWER_PREFIX)
    screen["position"] = screen["place"].map(LOC_TO_POSITION)

    # The tests rest on the shuffle, so the positions analysed must be the shuffle's positions.
    orders = pd.read_csv(ds.trials, usecols=[*KEYS, Con.ANSWERS_ORDER_COLUMN], dtype={PID: str})
    orders = orders[[*KEYS, Con.ANSWERS_ORDER_COLUMN]]  # read_csv keeps the file's column order
    shuffle = pd.DataFrame(
        [(pid, trial, label, POSITIONS[p])
         for pid, trial, order in orders.itertuples(index=False)
         for p, label in enumerate(ast.literal_eval(order))],
        columns=[*KEYS, "label", "shuffled_to"],
    )
    both = screen.merge(shuffle, on=[*KEYS, "label"], how="outer", indicator=True)
    assert (both["_merge"] == "both").all() and (both["position"] == both["shuffled_to"]).all(), (
        "the word table and answers_order disagree on where an answer sat"
    )
    return screen[[*KEYS, "label", "position", "n_words"]]


def _attractiveness(answers: pd.DataFrame, dataset_key: str) -> pd.Series:
    """How often other people chose this answer: the share choosing it among the trials of the
    same item outside this trial's unit -- so outside this placement. Study 2 counts the same
    regime only, since knowing the answer changes what attracts; Study 1 counts both groups.
    Missing when nobody outside the unit saw the item."""
    pool = [ITEM, "label"] + ([CONDITION] if dataset_key == "knowqa" else [])
    item = answers.groupby(pool)["is_chosen"]
    unit = answers.groupby([*pool, "unit"])["is_chosen"]
    others = item.transform("size") - unit.transform("size")
    chosen = item.transform("sum") - unit.transform("sum")
    return (chosen / others).where(others > 0)


def answers_table(dataset_key: str, fix: pd.DataFrame, trials: pd.DataFrame) -> pd.DataFrame:
    """One row per trial and answer, every condition.

    Per answer: `position`, `label` (its kind, A-D), `n_words`, `is_chosen`, `attractiveness`
    (`_attractiveness`), `gaze` (its share of the answer gaze before the decision) and `order`
    (when it was first reached before the decision: 1-4, or 0 if not reached by then). Per
    trial: `unit`, `n_reached`, `first_position` (where the first look at an answer landed),
    the condition, `decision_ms`, `excluded`, and Study 2's confidence rating.

    `fix` and `trials` are `drift_diffusion.data.load_fixations` and `load_trials`.
    """
    ds = dataset(dataset_key)
    extra = [Con.CONFIDENCE_RATING_COLUMN] if dataset_key == "knowqa" else []
    cols = list(dict.fromkeys([*KEYS, *UNIT_COLS[dataset_key], ITEM, Con.ANSWERS_ORDER_COLUMN, *extra]))
    raw = pd.read_csv(ds.trials, usecols=cols, dtype={PID: str})
    raw["unit"] = pd.factorize(raw[UNIT_COLS[dataset_key]].astype(str).agg("|".join, axis=1))[0]
    shared = raw.groupby("unit")[Con.ANSWERS_ORDER_COLUMN].nunique()
    assert (shared == 1).all(), "trials of one unit do not share one order of the answers"
    assert_full_coverage(trials, raw, KEYS, "trial table")

    out = _screen(dataset_key).merge(
        trials[[*KEYS, CONDITION, "chosen", Con.IS_CORRECT_COLUMN, "decision_ms", "excluded", *GAZE_COLS]],
        on=KEYS, how="inner")
    out = out.merge(raw[[*KEYS, "unit", ITEM, *extra]], on=KEYS, how="left")
    assert out[KEYS].drop_duplicates().shape[0] == len(trials), "the word table does not cover every trial"

    before = fix[fix["before_decision"] & fix["area"].isin(Con.ANSWER_AREAS)]
    reached = before.groupby([*KEYS, "area"])["start"].min().reset_index()
    reached["order"] = reached.groupby(KEYS)["start"].rank(method="first")
    reached["label"] = reached["area"].str.removeprefix(Con.ANSWER_PREFIX)
    out = out.merge(reached[[*KEYS, "label", "order"]], on=[*KEYS, "label"], how="left")
    # 0 is a category of its own -- not reached before the decision -- never a number.
    out["order"] = out["order"].fillna(0)

    g = out[GAZE_COLS].to_numpy()
    out["gaze"] = g[np.arange(len(out)), out["label"].map({l: i for i, l in enumerate(LABELS)}).to_numpy()]
    out = out.drop(columns=GAZE_COLS)
    out["is_chosen"] = (out["label"] == out["chosen"]).astype(int)
    out["n_reached"] = (out["order"] > 0).groupby([out[k] for k in KEYS]).transform("sum")
    first = out[out["order"] == 1][[*KEYS, "position"]].rename(columns={"position": "first_position"})
    out = out.merge(first, on=KEYS, how="left")
    no_gaze = out["excluded"] != ""
    out.loc[no_gaze, ["order", "n_reached", "gaze", "first_position"]] = np.nan
    assert out.loc[~no_gaze, "first_position"].notna().all(), "a trial with gaze before the decision reached no answer"

    out = out.sort_values([*KEYS, "label"]).reset_index(drop=True)
    per = out.groupby(KEYS).agg(n=("label", "size"), chosen=("is_chosen", "sum"))
    assert (per["n"] == 4).all() and (per["chosen"] == 1).all(), "not four answers with one chosen on every trial"
    out["attractiveness"] = _attractiveness(out, dataset_key)
    return out


def random_placements(n_units: int, n_draws: int, seed: int) -> Iterator[np.ndarray]:
    """`n_draws` random placements: each an array of the position (0-3) of each kind (0-3) in
    each unit, drawn afresh per unit, as the design drew them."""
    rng = np.random.default_rng(seed)
    base = np.tile(np.arange(4), (n_units, 1))
    for _ in range(n_draws):
        yield rng.permuted(base, axis=1)
