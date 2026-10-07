"""Fixation-sequence features: parsing, hesitation patterns, trial mean dwell.

Split out of derived/correctness_measures.py (2026-10-06). These six are the
half that `features/build.py` consumes as FEATURES; the
correctness summaries that used to sit beside them are analysis and stayed
behind until stage E gives them a home.

Note what `has_back_and_forth_xyx` actually checks: the LAST three tokens of
the collapsed sequence, not any back-and-forth anywhere in it. The names read
more broadly than the behaviour.
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


def sequence_len_literal_eval(x) -> int:
    """Sequences come as string representations of lists/tuples. Invalid -> 0."""
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
    """Parse sequence column via ast.literal_eval (expects list/tuple)."""
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
    """Does the sequence END in an x-y-x return? Only the last 3 tokens are read."""
    if seq is None or len(seq) < 3:
        return False
    a, b, c = seq[-1], seq[-2], seq[-3]
    return (a == c) and (a != b)


def has_back_and_forth_xyxy(seq) -> bool:
    """Does the sequence END in an x-y-x-y alternation? Only the last 4 tokens are read.

    For the graded version over the whole sequence, see
    ``longest_alternating_answer_run``.
    """
    if seq is None or len(seq) < 4:
        return False
    a, b, c, d = seq[-4], seq[-3], seq[-2], seq[-1]
    return (a == c) and (b == d) and (a != b)


def longest_alternating_answer_run(seq, exclude_label: str = "question") -> int:
    """
    Length of the longest contiguous run that strictly alternates between two
    distinct answer labels (an "x y x y x ..." pattern), where x and y are any
    two distinct labels other than ``exclude_label`` (the question).

    The ``exclude_label`` breaks a run (it is not part of the alternation), as
    does any repeat (``x x``) or a switch to a third symbol.

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
    """Mean dwell per word on the trial, broadcast back onto every IA row.

    ``transform`` keeps the IA grain, so the result is still one value per word
    -- collapse before testing on it.
    """
    total_dwell = df.groupby([Con.TRIAL_ID, Con.PARTICIPANT_ID])[dwell_col].transform(
        "sum"
    )
    n_words = df.groupby([Con.TRIAL_ID, Con.PARTICIPANT_ID])[dwell_col].transform(
        "count"
    )
    return total_dwell / n_words

# ---------------------------------------------------------------------------
# Moved here from data_prep/data_csv_generation.py in stage C (2026-10-06).
# These are the registry's `create_*` / `add_*` entries -- the functions the
# prep pipeline calls to put columns on the IA frame. They mutate the frame
# they are handed and depend on registry order; that is kept as-is by decision
# (Diana, 2026-10-06), with the default runner executing them in that order.
# ---------------------------------------------------------------------------



def build_area_sequences(
    df: pd.DataFrame,
    area_cols: Sequence[str],
    *,
    fixations: pd.DataFrame | None = None,
    nearest_by_trial: dict | None = None,
    drop_leading_question: bool = False,
    verbose: bool = True,
) -> pd.DataFrame:
    """Per trial, the ordered sequence of areas the eyes passed through.

    **One implementation for both screens** (stage D step 2c). The five steps --
    parse the serialized id sequence, resolve off-area fixations against the
    nearest-interest-area queue, map ids to areas, optionally drop an opening
    question fixation, and return -- were written twice: once inline here for
    the answer screen, and once in `features/paragraph/spans.py` for the
    paragraph screen. The paragraph copy already called the shared primitives in
    `area_metrics`; the answer copy predated them and carried its own versions.

    `area_cols` is the vocabulary (or vocabularies) to map onto. The answer
    screen passes two -- area label and screen location -- and gets one list
    column per vocabulary back; the paragraph screen passes its span column
    alone. That is the only structural difference between the screens here.

    The nearest queue comes either from `fixations` (built here in **one
    groupby**) or prebuilt as `nearest_by_trial`, which is keyed
    `(str(participant_id), str(trial_id))` either way -- stringified because
    KnowQA's trial index is a composite string and L1's is an integer. The paragraph screen builds
    its own during the single streaming pass it already makes over a multi-GB
    report, so it passes the dict.

    A fixation that can be placed neither directly nor from the queue is
    **dropped and counted**, never fatal (Diana, 2026-09-23) -- see
    `area_metrics.resolve_fixation_sequence`.
    """
    from src.features import area_metrics as am

    keys = [Con.TRIAL_ID, Con.PARTICIPANT_ID]
    area_cols = list(area_cols)

    if nearest_by_trial is None:
        if fixations is None:
            raise ValueError(
                "build_area_sequences needs either `fixations` (to build the "
                "nearest-interest-area queue from) or a prebuilt `nearest_by_trial`."
            )
        # One pass, not one scan per trial. The previous answer-screen version
        # filtered the whole fixation frame inside the per-trial loop -- 19,436
        # boolean scans of a 718k-row frame on L1, which is what the registry
        # comment meant by "slow for now".
        nearest_by_trial = {
            (str(pid), str(tid)): am.nearest_ia_queue(g)
            for (tid, pid), g in fixations.groupby(keys, sort=False)
        }

    rows, dropped, unplaced_trials, no_queue = [], 0, 0, 0
    for key, group in df.groupby(keys, sort=False):
        trial_id, participant_id = key
        queue = nearest_by_trial.get((str(participant_id), str(trial_id)), [])
        known = set(group[Con.INTEREST_AREA_ID].unique())
        maps = [
            dict(zip(group[Con.INTEREST_AREA_ID], group[col])) for col in area_cols
        ]
        sequence = am.parse_ia_sequence(
            group[Con.INTEREST_AREA_FIXATION_SEQUENCE].iloc[0]
        )
        off_area = [i for i in sequence if i not in known]
        if off_area and not queue:
            no_queue += 1
        resolved, n_dropped = am.resolve_fixation_sequence(sequence, known, queue)
        dropped += n_dropped
        unplaced_trials += bool(n_dropped)

        seqs = [[m[i] for i in resolved if i in m] for m in maps]
        if drop_leading_question and seqs:
            seqs = list(am.drop_leading_question_fixation(seqs[0], *seqs[1:]))

        row = dict(zip(keys, key))
        row.update({col: seq for col, seq in zip(area_cols, seqs)})
        rows.append(row)

    if verbose and dropped:
        print(
            f"  area sequences: {dropped:,} fixation(s) on {unplaced_trials:,} trial(s) "
            f"could not be placed in any interest area and were dropped"
        )
    # A queue keyed differently from this frame would look exactly like "no
    # off-area fixations anywhere" -- every one silently dropped. Counted so a
    # key or dtype mismatch is visible instead of being absorbed.
    if no_queue:
        print(
            f"  area sequences: {no_queue:,} trial(s) have off-area fixations but NO "
            f"nearest-interest-area queue entry. If this is most of the trials, the "
            f"queue is keyed differently from the frame."
        )
    return pd.DataFrame(rows, columns=keys + area_cols)


def create_fixation_sequence_tags(df, fix_path: Path = FIX_ANSWERS_PATH):
    """Per-trial fixation sequences for the answer screen, by label and by location.

    A thin call since stage D step 2c. The parse / resolve / map / trim logic it
    used to contain inline is now `build_area_sequences`, shared with the
    paragraph screen, which calls the same `area_metrics` primitives the
    paragraph side already used.

    Two vocabularies, so two list columns come back: `fix_by_label` (which area)
    and `fix_by_loc` (where on the screen). `drop_leading_question=True` is the
    answer screen's one extra step -- a lone opening fixation on the question is
    spillover from the question screen, and the paragraph screen has no question.
    """
    fixations = fix_path if isinstance(fix_path, pd.DataFrame) else pd.read_csv(fix_path)
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
    """
    Create simplified fixation sequences by collapsing consecutive fixations
    on the same area into a single step.

    For each (TRIAL_ID, PARTICIPANT_ID), this function:
    1. Reads the fixation sequence from INTEREST_AREA_FIXATION_SEQUENCE
       (a serialized list of IA_IDs, e.g. "[1, 2, 2, 3, 3, 3, 2]").
    2. Maps each IA_ID to:
       - AREA_LABEL_COLUMN       (e.g. 'question', 'answer_A', ...)
       - AREA_SCREEN_LOCATION    (e.g. 'top', 'left', ...)
    3. Filters out IA_IDs that are not present in the group's IA set.
    4. Collapses consecutive fixations on the same label into a single entry
       (run-length compression).

    Example
    -------
    Raw label sequence:
        ['question', 'question', 'answer_A', 'answer_A', 'answer_B']
    Simplified label sequence:
        ['question', 'answer_A', 'answer_B']

    The location sequence is compressed in parallel, taking the location
    of the first fixation in each run.

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
    """

    For each (trial, participant):
      - Count occurrences of each label in SIMPLIFIED_FIX_SEQ_BY_LABEL
      - Count occurrences of each location in SIMPLIFIED_FIX_SEQ_BY_LOCATION

    Then for each (trial, participant, area_label):
      - num_label_visits = count of that area_label in label sequence
      - num_loc_visits   = count of that area's screen location in location sequence
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
