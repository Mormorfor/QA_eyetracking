"""The order in which areas were looked at, and statistics of it.

For each trial this builds:

    fix_by_label / fix_by_loc         one entry per run of fixations on one
                                      word: which area (by meaning / by place)
    simpl_fix_by_label / _by_loc      the same with repeats merged:
                                      A A B A -> A B A
    num_label_visits / num_loc_visits how many times each area was entered

and five statistics of the merged sequence (`DERIVED_COLS`):

    seq_len                 how many visits: the merged sequence's length
    has_xyx                 1 if the sequence ENDS in x-y-x (e.g. A B A)
    has_xyxy                1 if it ENDS in x-y-x-y (e.g. A B A B)
    longest_alt_answer_run  the longest stretch anywhere that alternates
                            between two answers
    trial_mean_dwell        average looking time per word on the screen

Careful with `has_xyx` and `has_xyxy`: they look only at the LAST three or four
entries, not for a back-and-forth anywhere. Any area counts, including the
question, so "answer_A, question, answer_A" at the end counts as x-y-x.
"""

from __future__ import annotations

from collections import Counter
from typing import Sequence

import ast

import numpy as np
import pandas as pd

from pathlib import Path

from src.config import columns as Con
from src.config.datasets import FIX_ANSWERS_PATH
from src.features.fixations import load_fixations, place_fixations

# The fixation-report columns a sequence is built from.
SEQUENCE_FIXATION_COLUMNS = [
    Con.CURRENT_FIX_START,
    Con.CURRENT_FIX_INTEREST_AREA_ID,
    Con.NEAREST_IA,
]


def sequence_len_literal_eval(x) -> int:
    """The length of a stored sequence (text like "['a', 'b']", or a list).

    Anything that cannot be read as a list -- missing, or malformed text --
    counts as length 0, so it looks the same as an empty sequence.
    """
    if x is None:
        return 0
    if isinstance(x, float) and np.isnan(x):
        return 0
    if isinstance(x, str):
        try:
            x = ast.literal_eval(x)
        except Exception:
            return 0
    return len(x) if isinstance(x, (list, tuple)) else 0


def parse_seq(x):
    """Read a stored sequence into a list or tuple.

    Accepts the text form (after a CSV round trip) or the list/tuple itself.
    Anything that cannot be read gives None.
    """
    if x is None:
        return None
    if isinstance(x, float) and np.isnan(x):
        return None
    if isinstance(x, str):
        try:
            x = ast.literal_eval(x)
        except Exception:
            return None
    return x if isinstance(x, (list, tuple)) else None


def has_back_and_forth_xyx(seq) -> bool:
    """True if the last three entries are x, y, x (with x and y different).

    Only the end of the sequence is checked.
    """
    if seq is None or len(seq) < 3:
        return False
    a, b, c = seq[-1], seq[-2], seq[-3]
    return (a == c) and (a != b)


def has_back_and_forth_xyxy(seq) -> bool:
    """True if the last four entries are x, y, x, y (with x and y different).

    Only the end of the sequence is checked. For a version that looks at the
    whole sequence, see ``longest_alternating_answer_run``.
    """
    if seq is None or len(seq) < 4:
        return False
    a, b, c, d = seq[-4], seq[-3], seq[-2], seq[-1]
    return (a == c) and (b == d) and (a != b)


def longest_alternating_answer_run(seq, exclude_label: str = "question") -> int:
    """The longest stretch, anywhere in the sequence, that goes back and forth
    between two answers (x y x y x ...).

    The question (``exclude_label``) breaks a stretch, as does a repeat
    (``x x``). Moving on to a third answer starts a new stretch from the last
    two entries.

    Examples (answers a, b, c; q = question):
        [a, b, a, b, a]      -> 5
        [a, b, c]            -> 2   (a,b and b,c are each 2-symbol runs)
        [a, b, q, a, b, a]   -> 3   (the question breaks the run)
        [a, a, a]            -> 1   (an answer, but no alternation)
        [q, q] / [] / None   -> 0   (no answers)
    """
    if seq is None:
        return 0

    best = 0
    run = 0
    for i, cur in enumerate(seq):
        if cur == exclude_label:
            run = 0
            continue
        if run == 0:
            run = 1
        else:
            prev = seq[i - 1]
            if cur == prev:
                run = 1
            elif run >= 2 and cur != seq[i - 2]:
                # alternation switches to a new pair (prev, cur)
                run = 2
            else:
                run += 1
        if run > best:
            best = run
    return best


def compute_trial_mean_dwell_per_word(
    df: pd.DataFrame,
    dwell_col: str = Con.IA_DWELL_TIME,
) -> pd.Series:
    """Average looking time per word on the trial, repeated on every word row.

    The result has one value per word row, not per trial, so collapse it to one
    row per trial before running any test on it.
    """
    total_dwell = df.groupby([Con.TRIAL_ID, Con.PARTICIPANT_ID])[dwell_col].transform(
        "sum"
    )
    n_words = df.groupby([Con.TRIAL_ID, Con.PARTICIPANT_ID])[dwell_col].transform(
        "count"
    )
    return total_dwell / n_words

# ---------------------------------------------------------------------------
# Building the sequences
# ---------------------------------------------------------------------------


def build_area_sequences(
    df: pd.DataFrame,
    area_cols: Sequence[str],
    *,
    fixations: pd.DataFrame,
    drop_leading_question: bool = False,
    verbose: bool = True,
) -> pd.DataFrame:
    """For each trial, the areas the eyes passed through, in time order.

    Both screens use this. For each trial it:

    1. places every fixation on a word with `fixations.place_fixations` -- the
       word it landed on, else the word it landed nearest -- and puts the
       fixations in time order;
    2. merges fixations in a row on the same word into one entry, so there is
       one entry per run of fixations on one word;
    3. turns each entry into its area;
    4. optionally (answer screen) drops a single opening entry on the question.

    `area_cols` names the area column(s) to use. The answer screen passes two
    -- area by meaning and by place -- and gets a list column for each; the
    paragraph screen passes its one column. `df` is the screen's word table and
    `fixations` its fixation table (`fixations.load_fixations`).

    A fixation that cannot be placed at all (no word and no nearest word) is
    left out, and the number left out is printed.
    """
    from src.features import area_metrics as am

    keys = [Con.TRIAL_ID, Con.PARTICIPANT_ID]
    area_cols = list(area_cols)
    trials = df[keys].drop_duplicates()

    placed = place_fixations(
        fixations[[Con.PARTICIPANT_ID, Con.TRIAL_ID] + SEQUENCE_FIXATION_COLUMNS],
        df,
        area_cols,
        label="area sequences",
        verbose=verbose,
    )
    # The fixations of `df`'s trials, each trial in time order.
    placed = placed.merge(trials, on=keys, how="inner")
    placed = placed.sort_values(keys + [Con.CURRENT_FIX_START], kind="mergesort")

    # A new entry starts wherever the word changes. A fixation with no word
    # never matches its neighbour, so it is an entry of its own, and is then
    # left out.
    word = placed[Con.INTEREST_AREA_ID]
    same_trial = placed[Con.TRIAL_ID].eq(placed[Con.TRIAL_ID].shift()) & placed[
        Con.PARTICIPANT_ID
    ].eq(placed[Con.PARTICIPANT_ID].shift())
    new_entry = ~(same_trial & word.eq(word.shift()))
    unplaced = int(word.isna().sum())
    entries = placed[new_entry.to_numpy() & word.notna().to_numpy()]
    lists = entries.groupby(keys, sort=False)[area_cols].agg(list)

    rows = []
    for key in trials.itertuples(index=False, name=None):
        seqs = [list(lists.at[key, col]) if key in lists.index else [] for col in area_cols]
        if drop_leading_question and seqs:
            seqs = list(am.drop_leading_question_fixation(seqs[0], *seqs[1:]))
        row = dict(zip(keys, key))
        row.update({col: seq for col, seq in zip(area_cols, seqs)})
        rows.append(row)

    if verbose and unplaced:
        print(
            f"  area sequences: {unplaced:,} fixation(s) could not be placed on "
            f"any word and were left out"
        )
    return pd.DataFrame(rows, columns=keys + area_cols)


def create_fixation_sequence_tags(df, fix_path: Path = FIX_ANSWERS_PATH):
    """The answer screen's fixation sequences: `fix_by_label` (which area) and
    `fix_by_loc` (where on screen), one entry per run of fixations on one word.

    A single opening entry on the question is dropped, as left over from the
    previous screen. `fix_path` is the fixation report: either the table the
    pipeline has already read, or a path to read it from.
    """
    fixations = (
        fix_path
        if isinstance(fix_path, pd.DataFrame)
        else load_fixations(fix_path, SEQUENCE_FIXATION_COLUMNS)
    )
    out = build_area_sequences(
        df,
        [Con.AREA_LABEL_COLUMN, Con.AREA_SCREEN_LOCATION],
        fixations=fixations,
        drop_leading_question=True,
    )
    return out.rename(
        columns={
            Con.AREA_LABEL_COLUMN: Con.FIX_SEQUENCE_BY_LABEL,
            Con.AREA_SCREEN_LOCATION: Con.FIX_SEQUENCE_BY_LOCATION,
        }
    )


def create_simplified_fixation_tags(df: pd.DataFrame) -> pd.DataFrame:
    """Merge repeats in each trial's fixation sequences, so each entry is a visit.

    Reads `fix_by_label` and `fix_by_loc` (from `create_fixation_sequence_tags`)
    and writes `simpl_fix_by_label` and `simpl_fix_by_loc`. For example:

        ['question', 'question', 'answer_A', 'answer_A', 'answer_B']
     -> ['question', 'answer_A', 'answer_B']

    Repeats are merged by area; the place sequence is merged at the same points.
    """
    rows = []
    for (trial_id, participant_id), g in df.groupby(
        [Con.TRIAL_ID, Con.PARTICIPANT_ID], sort=False
    ):
        r = g.iloc[0]

        labels = list(r[Con.FIX_SEQUENCE_BY_LABEL] or [])
        locs = list(r[Con.FIX_SEQUENCE_BY_LOCATION] or [])

        n = len(labels)

        simpl_labels = []
        simpl_locs = []

        prev_label = None
        for lab, loc in zip(labels, locs):
            if lab != prev_label:
                simpl_labels.append(lab)
                simpl_locs.append(loc)
                prev_label = lab

        rows.append(
            {
                Con.TRIAL_ID: trial_id,
                Con.PARTICIPANT_ID: participant_id,
                Con.SIMPLIFIED_FIX_SEQ_BY_LABEL: tuple(simpl_labels),
                Con.SIMPLIFIED_FIX_SEQ_BY_LOCATION: tuple(simpl_locs),
            }
        )

    return pd.DataFrame(rows)


def create_simplified_visit_counts(df: pd.DataFrame) -> pd.DataFrame:
    """How many times each area was entered, one row per trial and area.

    `num_label_visits` counts the area (by meaning) in the merged sequence;
    `num_loc_visits` counts its screen position in the merged place sequence.
    An area never entered gets 0.
    """
    df_local = df.copy()

    counts_rows = []
    for (trial_id, participant_id), g in df_local.groupby(
        [Con.TRIAL_ID, Con.PARTICIPANT_ID]
    ):
        labels_seq = g[Con.SIMPLIFIED_FIX_SEQ_BY_LABEL].iloc[0]
        locs_seq = g[Con.SIMPLIFIED_FIX_SEQ_BY_LOCATION].iloc[0]

        label_counts = Counter(labels_seq)
        loc_counts = Counter(locs_seq)

        counts_rows.append(
            {
                Con.TRIAL_ID: trial_id,
                Con.PARTICIPANT_ID: participant_id,
                "_label_counts": label_counts,
                "_loc_counts": loc_counts,
            }
        )

    counts_df = pd.DataFrame(counts_rows)
    df_local = df_local.merge(counts_df, on=[Con.TRIAL_ID, Con.PARTICIPANT_ID], how="left")

    df_local[Con.NUM_LABEL_VISITS] = df_local.apply(
        lambda r: (
            int(r["_label_counts"].get(r[Con.AREA_LABEL_COLUMN], 0))
            if isinstance(r["_label_counts"], Counter)
            else 0
        ),
        axis=1,
    )

    df_local[Con.NUM_LOC_VISITS] = df_local.apply(
        lambda r: (
            int(r["_loc_counts"].get(r[Con.AREA_SCREEN_LOCATION], 0))
            if isinstance(r["_loc_counts"], Counter)
            else 0
        ),
        axis=1,
    )

    out = df_local.groupby(
        [Con.TRIAL_ID, Con.PARTICIPANT_ID, Con.AREA_LABEL_COLUMN], as_index=False
    ).agg(
        **{
            Con.NUM_LABEL_VISITS: (Con.NUM_LABEL_VISITS, "max"),
            Con.NUM_LOC_VISITS: (Con.NUM_LOC_VISITS, "max"),
        }
    )

    return out


# ---------------------------------------------------------------------------
# The five sequence statistics
#
# Each is its own registry entry (see ingest/registry.py), one value per trial.
# ---------------------------------------------------------------------------


#: The five sequence statistics, in reporting order. `modeling/feature_sets.py`
#: imports this list, so the model's feature names always match what is built.
DERIVED_COLS = [
    "seq_len",
    "has_xyx",
    "has_xyxy",
    "longest_alt_answer_run",
    "trial_mean_dwell",
]


def _trial_sequences(df: pd.DataFrame, seq_col: str) -> pd.DataFrame:
    """One parsed sequence per trial.

    The sequence is the same on every word row of a trial, so taking the first
    row per trial loses nothing.
    """
    out = (
        df.groupby(list(Con.TRIAL_ID_COLS), as_index=False)[seq_col]
        .first()
    )
    # `parse_seq` rather than `ast.literal_eval`: the sequence is text after it
    # has been saved and read back, but still a tuple during the pipeline run,
    # and `literal_eval` fails on a tuple.
    out["_seq"] = out[seq_col].apply(parse_seq)
    return out


def create_seq_len(
    df: pd.DataFrame, seq_col: str = Con.SIMPLIFIED_FIX_SEQ_BY_LABEL
) -> pd.DataFrame:
    """`seq_len`: how many visits the trial had, i.e. the merged sequence's length.

    A A B A -> A B A -> 3. Visits to the question count. 0 if the sequence is
    missing.
    """
    out = _trial_sequences(df, seq_col)
    out["seq_len"] = out["_seq"].apply(
        lambda s: len(s) if isinstance(s, (list, tuple)) else 0
    )
    return out[list(Con.TRIAL_ID_COLS) + ["seq_len"]]


def create_has_xyx(
    df: pd.DataFrame, seq_col: str = Con.SIMPLIFIED_FIX_SEQ_BY_LABEL
) -> pd.DataFrame:
    """`has_xyx`: 1 if the merged sequence ENDS in x, y, x (e.g. A B A), else 0.

    Only the last three entries are checked. Any area counts, including the
    question.
    """
    out = _trial_sequences(df, seq_col)
    out["has_xyx"] = out["_seq"].apply(
        lambda s: bool(has_back_and_forth_xyx(s)) if s is not None else False
    ).astype(int)
    return out[list(Con.TRIAL_ID_COLS) + ["has_xyx"]]


def create_has_xyxy(
    df: pd.DataFrame, seq_col: str = Con.SIMPLIFIED_FIX_SEQ_BY_LABEL
) -> pd.DataFrame:
    """`has_xyxy`: 1 if the merged sequence ENDS in x, y, x, y (e.g. A B A B), else 0.

    Only the last four entries are checked. Any area counts, including the
    question.
    """
    out = _trial_sequences(df, seq_col)
    out["has_xyxy"] = out["_seq"].apply(
        lambda s: bool(has_back_and_forth_xyxy(s)) if s is not None else False
    ).astype(int)
    return out[list(Con.TRIAL_ID_COLS) + ["has_xyxy"]]


def create_longest_alt_answer_run(
    df: pd.DataFrame, seq_col: str = Con.SIMPLIFIED_FIX_SEQ_BY_LABEL
) -> pd.DataFrame:
    """`longest_alt_answer_run`: the longest back-and-forth between two answers,
    anywhere in the merged sequence. See `longest_alternating_answer_run`."""
    out = _trial_sequences(df, seq_col)
    out["longest_alt_answer_run"] = out["_seq"].apply(
        lambda s: longest_alternating_answer_run(s) if s is not None else 0
    ).astype(int)
    return out[list(Con.TRIAL_ID_COLS) + ["longest_alt_answer_run"]]


def create_trial_mean_dwell(
    df: pd.DataFrame, dwell_col: str = Con.MEAN_DWELL_TIME
) -> pd.DataFrame:
    """`trial_mean_dwell`: average looking time per word over the whole screen.

    That is, the trial's total looking time divided by its number of words.

    It is computed by averaging the area-level `mean_dwell_time` over the word
    rows. Each area's value is repeated once per word in that area, so this
    gives exactly total looking time / number of words (checked on all Study 1
    trials). It needs the word-level frame: averaging the five area values
    directly would weight every area equally and give a different number.
    """
    g = df.groupby(list(Con.TRIAL_ID_COLS), as_index=False)[dwell_col].agg(
        _sum="sum", _n="count"
    )
    g["trial_mean_dwell"] = g["_sum"] / g["_n"]
    return g[list(Con.TRIAL_ID_COLS) + ["trial_mean_dwell"]]
