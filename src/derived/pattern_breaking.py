from __future__ import annotations

import ast
from collections import Counter
from typing import Literal, Sequence

import pandas as pd

from src import constants as C
from src.constants import TRIAL_ID_COLS


SeqKind = Literal["location", "label"]

DEFAULT_WINDOW_LEN = 4


def _parse_seq(x) -> tuple:
    """Normalise a stored simplified-fixation sequence into a tuple.

    The sequence columns hold tuples in-memory (as produced by the
    data_csv_generation pipeline) but come back as serialized strings once the
    frame has been round-tripped through a CSV. Anything unparseable / missing
    becomes an empty tuple.
    """
    if isinstance(x, tuple):
        return x
    if isinstance(x, list):
        return tuple(x)
    if isinstance(x, str):
        try:
            parsed = ast.literal_eval(x)
        except (ValueError, SyntaxError):
            return ()
        return tuple(parsed) if isinstance(parsed, (list, tuple)) else ()
    return ()


def _starting_window(
    seq, window_len: int = DEFAULT_WINDOW_LEN, drop_question: bool = False
) -> tuple:
    """First ``window_len`` tokens of a simplified sequence.

    ``drop_question`` removes ``"question"`` tokens *before* taking the window.
    The data generator only trims the single leading question fixation (see
    ``data_csv_generation``), so residual question tokens can survive; dropping
    them here keeps the strategy to pure answer-area scanning order.
    """
    tokens = list(_parse_seq(seq))
    if drop_question:
        tokens = [tok for tok in tokens if tok != "question"]
    return tuple(tokens[:window_len])


def levenshtein_sequence_distance(
    seq_a: Sequence,
    seq_b: Sequence,
    normalize: bool = False,
) -> float:
    """Token-level Levenshtein (edit) distance between two sequences.

    The minimum number of single-token insertions, deletions, or substitutions
    to turn ``seq_a`` into ``seq_b``. Operates on whole tokens, not characters,
    so each location/label (e.g. ``"answer_0(top)"``) is one edit unit.

    A graded generalisation of the binary "breaks pattern" flag: 0 when the two
    sequences are identical, larger the more they differ.

    If ``normalize``, the distance is divided by ``max(len(seq_a), len(seq_b))``
    (0 when both are empty), giving a value in [0, 1].
    """
    a = list(seq_a) if seq_a is not None else []
    b = list(seq_b) if seq_b is not None else []
    n, m = len(a), len(b)

    if n == 0 or m == 0:
        dist = float(max(n, m))
    else:
        # Wagner-Fischer with a single rolling row.
        prev = list(range(m + 1))
        for i in range(1, n + 1):
            curr = [i] + [0] * m
            ai = a[i - 1]
            for j in range(1, m + 1):
                cost = 0 if ai == b[j - 1] else 1
                curr[j] = min(
                    prev[j] + 1,          # deletion
                    curr[j - 1] + 1,      # insertion
                    prev[j - 1] + cost,   # substitution / match
                )
            prev = curr
        dist = float(prev[m])

    if normalize:
        denom = max(n, m)
        return dist / denom if denom else 0.0
    return dist


def build_starting_strategies(
    df: pd.DataFrame,
    kind: SeqKind = "location",
    window_len: int = DEFAULT_WINDOW_LEN,
    drop_question: bool = True,
    out_col: str = C.STARTING_STRATEGY_COL,
) -> pd.DataFrame:
    """Per-trial starting strategy from the simplified fixation sequence.

    One row per (PARTICIPANT_ID, TRIAL_ID). The starting strategy is the first
    ``window_len`` entries of ``simpl_fix_by_loc`` (or ``simpl_fix_by_label``
    when ``kind="label"``), stored as a tuple in ``out_col``.

    Required columns:
      C.PARTICIPANT_ID
      C.TRIAL_ID
      the relevant simplified-sequence column (see ``kind``)
    """
    if kind == "location":
        seq_col = C.SIMPLIFIED_FIX_SEQ_BY_LOCATION
    elif kind == "label":
        seq_col = C.SIMPLIFIED_FIX_SEQ_BY_LABEL
    else:
        raise ValueError("kind must be 'location' or 'label'")

    required = [C.PARTICIPANT_ID, C.TRIAL_ID, seq_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    out = df[required].drop_duplicates(subset=[C.PARTICIPANT_ID, C.TRIAL_ID]).copy()
    out[out_col] = out[seq_col].apply(
        lambda s: _starting_window(s, window_len=window_len, drop_question=drop_question)
    )

    return out[[C.PARTICIPANT_ID, C.TRIAL_ID, out_col]].reset_index(drop=True)


def compute_dominant_starting_strategy(
    df: pd.DataFrame,
    kind: SeqKind = "location",
    window_len: int = DEFAULT_WINDOW_LEN,
    drop_question: bool = True,
    id_col: str = C.PARTICIPANT_ID,
) -> pd.DataFrame:
    """Participant-level dominant starting strategy and its dominance score.

    For each participant, this:
      1. computes the per-trial starting strategy (first ``window_len`` tokens
         of ``simpl_fix_by_loc``, see :func:`build_starting_strategies`),
      2. finds ``dominant_starting_strategy`` -- the most common such strategy
         across that participant's trials,
      3. computes ``dominance_score`` -- the proportion of trials (in [0, 1])
         on which the dominant strategy is used.

    Ties for the most common strategy are broken deterministically (highest
    count, then lexicographic order of the strategy tuple).

    Returns one row per participant with columns:
      [id_col, C.DOMINANT_STARTING_STRATEGY, C.DOMINANCE_SCORE,
       C.N_STRATEGY_TRIALS]

    Required columns:
      id_col
      C.TRIAL_ID
      the relevant simplified-sequence column (see ``kind``)
    """
    per_trial = build_starting_strategies(
        df.rename(columns={id_col: C.PARTICIPANT_ID})
        if id_col != C.PARTICIPANT_ID
        else df,
        kind=kind,
        window_len=window_len,
        drop_question=drop_question,
    )
    if id_col != C.PARTICIPANT_ID:
        per_trial = per_trial.rename(columns={C.PARTICIPANT_ID: id_col})

    return _dominant_from_strategies(per_trial, id_col=id_col)


def _dominant_from_strategies(
    per_trial: pd.DataFrame,
    id_col: str = C.PARTICIPANT_ID,
    strat_col: str = C.STARTING_STRATEGY_COL,
) -> pd.DataFrame:
    """Collapse a per-trial starting-strategy frame to one row per participant.

    Picks the most common strategy per participant (ties broken deterministically
    by highest count, then lexicographic order of the strategy tuple) and its
    dominance score (proportion of trials using it).
    """
    rows = []
    for pid, g in per_trial.groupby(id_col, sort=False):
        counts = Counter(g[strat_col])
        n_trials = int(sum(counts.values()))

        # deterministic: most frequent first, ties broken lexicographically
        dominant, top_count = min(counts.items(), key=lambda kv: (-kv[1], kv[0]))

        rows.append(
            {
                id_col: pid,
                C.DOMINANT_STARTING_STRATEGY: dominant,
                C.DOMINANCE_SCORE: top_count / n_trials if n_trials else float("nan"),
                C.N_STRATEGY_TRIALS: n_trials,
            }
        )

    return pd.DataFrame(
        rows,
        columns=[
            id_col,
            C.DOMINANT_STARTING_STRATEGY,
            C.DOMINANCE_SCORE,
            C.N_STRATEGY_TRIALS,
        ],
    )


def build_trial_level_pattern_features(
    df: pd.DataFrame,
    kind: SeqKind = "location",
    window_len: int = DEFAULT_WINDOW_LEN,
    add_interaction: bool = True,
    add_distance: bool = True,
) -> pd.DataFrame:
    """Per-trial pattern-breaking features for the model.

    One row per (PARTICIPANT_ID, TRIAL_ID) with, for each of the two variants
    (question tokens kept / dropped before forming the starting strategy):

      - ``breaks_pattern_{with,no}_q`` -- 1 if this trial's starting strategy
        differs from the participant's dominant starting strategy, else 0.
      - ``dominance_score_{with,no}_q`` -- the participant's dominance score
        (proportion of trials on the dominant strategy), broadcast to each trial.

    When ``add_interaction`` (default), also adds
    ``breaks_x_dominance_{with,no}_q`` = ``breaks_pattern`` * ``dominance_score``
    per variant -- the dominance score on pattern-breaking trials, 0 otherwise.

    When ``add_distance`` (default), also adds ``strategy_distance_{with,no}_q``
    -- the token-level Levenshtein distance between the trial's starting strategy
    and the participant's dominant one (a graded ``breaks_pattern``; 0 when they
    match).

    The dominant strategy and dominance score are computed over the trials
    present in ``df``, so pass the full (unfiltered) trial set.

    Required columns:
      C.PARTICIPANT_ID
      C.TRIAL_ID
      the relevant simplified-sequence column (see ``kind``)
    """
    variants = [
        (False, C.BREAKS_PATTERN_WITH_Q, C.DOMINANCE_SCORE_WITH_Q,
         C.BREAKS_X_DOMINANCE_WITH_Q, C.STRATEGY_DISTANCE_WITH_Q),
        (True, C.BREAKS_PATTERN_NO_Q, C.DOMINANCE_SCORE_NO_Q,
         C.BREAKS_X_DOMINANCE_NO_Q, C.STRATEGY_DISTANCE_NO_Q),
    ]

    out: pd.DataFrame | None = None
    for drop_question, breaks_col, score_col, inter_col, dist_col in variants:
        per_trial = build_starting_strategies(
            df, kind=kind, window_len=window_len, drop_question=drop_question
        )
        dominant = _dominant_from_strategies(per_trial, id_col=C.PARTICIPANT_ID)

        merged = per_trial.merge(dominant, on=C.PARTICIPANT_ID, how="left")
        merged[breaks_col] = (
            merged[C.STARTING_STRATEGY_COL] != merged[C.DOMINANT_STARTING_STRATEGY]
        ).astype(int)

        cols = [breaks_col]
        if add_distance:
            merged[dist_col] = [
                levenshtein_sequence_distance(trial_strat, dominant_strat)
                for trial_strat, dominant_strat in zip(
                    merged[C.STARTING_STRATEGY_COL],
                    merged[C.DOMINANT_STARTING_STRATEGY],
                )
            ]
            cols.append(dist_col)

        merged = merged.rename(columns={C.DOMINANCE_SCORE: score_col})
        cols.append(score_col)
        if add_interaction:
            merged[inter_col] = merged[breaks_col] * merged[score_col]
            cols.append(inter_col)

        piece = merged[list(TRIAL_ID_COLS) + cols]
        out = piece if out is None else out.merge(piece, on=list(TRIAL_ID_COLS))

    return out.reset_index(drop=True)
