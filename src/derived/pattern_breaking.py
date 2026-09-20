from __future__ import annotations

import ast
from collections import Counter, defaultdict
from typing import Literal, Optional, Sequence

import numpy as np
import pandas as pd

from src import constants as C
from src.constants import TRIAL_ID_COLS


SeqKind = Literal["location", "label"]

DEFAULT_WINDOW_LEN = 4

# A participant "has a dominant strategy" when their modal opening scan covers
# AT LEAST this share of their trials. The paper's wording is "at least in half
# of the trials", so the comparison is `>=`, not `>` -- see has_dominant_strategy.
DEFAULT_DOMINANCE_THRESHOLD = 0.5


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

    return dominant_strategy_by_participant(per_trial, id_col=id_col)


def dominant_strategy_by_participant(
    per_trial: pd.DataFrame,
    id_col: str = C.PARTICIPANT_ID,
    strat_col: str = C.STARTING_STRATEGY_COL,
) -> pd.DataFrame:
    """Collapse a per-trial starting-strategy frame to one row per participant.

    Picks the most common strategy per participant (ties broken deterministically
    by highest count, then lexicographic order of the strategy tuple) and its
    dominance score (proportion of trials using it).

    This is the single implementation of "who is this participant's dominant
    strategy, and what share of their trials does it cover". The descriptive
    figures in ``viz/visualisations_strategies.py`` and
    ``viz/visualisations_dominant_eye.py`` call it too, rather than
    reimplementing the modal pick -- four near-copies of it, with three
    different tie-breaks, were merged into this one on 2026-09-20.

    Note the SCOPE caveat that applies to every caller: the dominance score is
    computed over whatever trials are in ``per_trial``. See the module note on
    ``build_trial_level_pattern_features`` and ``todo.md`` T3.21.
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


def has_dominant_strategy(
    dominance_score,
    threshold: float = DEFAULT_DOMINANCE_THRESHOLD,
):
    """Does this dominance score count as "having a dominant strategy"?

    ``>=``, not ``>``. The paper states the criterion as "at least in half of
    the trials", and on the L1 data the difference is not cosmetic: 5 hunters
    and 4 gatherers have a dominance score of exactly 0.50, which is 2.8 and
    2.2 percentage points of their groups.

    Accepts a scalar or a Series; returns the same shape. Every threshold
    comparison in the project goes through here, so the operator is written
    down once (this was `todo.md` T1.1 -- three call sites previously
    disagreed, and the same data was reported as both 46.1% and 48.9%).
    """
    return dominance_score >= threshold


# ---------------------------------------------------------------------------
# Descriptive dominance quantities
#
# Everything below is computation only -- no plotting. It backs the paper's
# First-scan behavior subsection (findings.md 1.1-1.6). It lives here rather
# than in viz/ because a number that is computed inside a plotting function is
# a number that only exists as pixels: the plot returns a frame, the notebook
# drops it, and nothing reaches reports/report_data/. Moved out of
# viz/visualisations_strategies.py and viz/visualisations_dominant_eye.py on
# 2026-09-20; behaviour deliberately unchanged in the move.
# ---------------------------------------------------------------------------


def build_prefix_completion_map(
    series: pd.Series,
    full_len: int = DEFAULT_WINDOW_LEN,
) -> dict:
    """Learn how interrupted scans tend to be completed.

    From the strategies observed at full length, map each proper prefix to the
    full sequence that most often extends it. This is the "interrupted-scan
    completion" the paper describes, used to repair scan passes shorter than
    ``full_len``.

    SCOPE: the map is learned population-wide over whatever ``series`` is given,
    so hunters and gatherers currently learn different maps. That is row 5 of
    `todo.md` T3.21 and is left as-is until that item is taken up.

    Returns ``{prefix_tuple: full_tuple}``.
    """
    full_counts = series[series.map(len).eq(full_len)].value_counts()
    by_prefix = defaultdict(Counter)
    for full_seq, c in full_counts.items():
        for k in range(1, full_len):
            pref = full_seq[:k]
            by_prefix[pref][full_seq] += c
    return {
        pref: max(counter.items(), key=lambda kv: (kv[1], kv[0]))[0]
        for pref, counter in by_prefix.items()
    }


def add_completed_strategy_column(
    df: pd.DataFrame,
    strat_col: str = C.STRATEGY_COL,
    full_len: int = DEFAULT_WINDOW_LEN,
    col_suffix: str = "_completed",
    prefix2full: Optional[dict] = None,
):
    """Fill strategies shorter than ``full_len`` using a prefix-completion map.

    Adds two columns:
      ``<strat_col><col_suffix>``   -- the completed sequence
      ``<strat_col>_was_completed`` -- whether completion was attempted, i.e.
                                       the original was shorter than full_len

    If ``prefix2full`` is None the map is learned from ``df[strat_col]`` itself
    -- see the scope note on :func:`build_prefix_completion_map`.

    Returns ``(df, prefix2full)``.
    """
    df = df.copy()
    series = df[strat_col]

    if prefix2full is None:
        prefix2full = build_prefix_completion_map(series, full_len=full_len)

    df[f"{strat_col}_was_completed"] = series.map(lambda t: len(t) < full_len)

    def _complete(t):
        if len(t) >= full_len:
            return t[:full_len]
        return prefix2full.get(t, t)

    df[f"{strat_col}{col_suffix}"] = series.map(_complete)

    return df, prefix2full


def dominance_gap_by_participant(
    per_trial: pd.DataFrame,
    id_col: str = C.PARTICIPANT_ID,
    strat_col: str = C.STARTING_STRATEGY_COL,
) -> pd.DataFrame:
    """How far clear of the runner-up is each participant's dominant strategy?

    ``p1`` is the dominant strategy's share (the same quantity as
    ``dominance_score``), ``p2`` the second-most-common strategy's share, and
    ``gap = p1 - p2``. A participant using only one strategy has ``p2 = 0``.

    This is the one dominance quantity that genuinely needs the full
    strategy-by-participant matrix rather than just the modal pick.

    Returns one row per participant, indexed by ``id_col``, columns
    ``[p1, p2, gap]``.
    """
    counts = per_trial.groupby([id_col, strat_col]).size().unstack(fill_value=0)
    props = counts.div(counts.sum(axis=1), axis=0)

    p1 = props.max(axis=1)

    def _second_largest(row):
        vals = row[row > 0].nlargest(2)
        return vals.iloc[-1] if len(vals) > 1 else 0

    p2 = props.apply(_second_largest, axis=1)

    return pd.DataFrame({"p1": p1, "p2": p2, "gap": p1 - p2})


def strategy_variety_by_participant(
    per_trial: pd.DataFrame,
    id_col: str = C.PARTICIPANT_ID,
    strat_col: str = C.STARTING_STRATEGY_COL,
) -> pd.Series:
    """How many distinct opening scans does each participant produce?

    The counterweight to the dominance score: "dominant" means MODAL, not
    consistent. A participant whose dominant strategy covers half their trials
    still produced a range of others across the rest.
    """
    return per_trial.groupby(id_col)[strat_col].nunique()


def dominant_strategy_counts(
    per_trial: pd.DataFrame,
    id_col: str = C.PARTICIPANT_ID,
    strat_col: str = C.STARTING_STRATEGY_COL,
    threshold: float = DEFAULT_DOMINANCE_THRESHOLD,
) -> pd.Series:
    """Which strategies dominate, and for how many participants?

    Restricted to participants who have a dominant strategy at ``threshold``
    (see :func:`has_dominant_strategy`). Descending by count.
    """
    dominant = dominant_strategy_by_participant(
        per_trial, id_col=id_col, strat_col=strat_col
    )
    mask = has_dominant_strategy(dominant[C.DOMINANCE_SCORE], threshold=threshold)
    return (
        dominant.loc[mask, C.DOMINANT_STARTING_STRATEGY]
        .value_counts()
        .sort_values(ascending=False)
    )


def summarize_completion_effect(
    per_trial: pd.DataFrame,
    id_col: str = C.PARTICIPANT_ID,
    raw_col: str = C.STRATEGY_COL,
    comp_col: Optional[str] = None,
    threshold: float = DEFAULT_DOMINANCE_THRESHOLD,
    full_len: int = DEFAULT_WINDOW_LEN,
):
    """What does interrupted-scan completion change?

    Compares each participant's dominance before and after completion, and
    reports how many participants' dominant label the repair actually flipped.
    The point is robustness: if completion barely moves the result, the
    first-scan finding is not an artifact of how short scans were handled.

    Returns ``(summary, both)``:
      ``summary`` -- dict of aggregates (means, the two prevalence figures at
                     ``threshold``, and how much changed)
      ``both``    -- one row per participant: raw/comp scores, delta, the two
                     dominant labels, whether the label changed, and per-trial
                     sequence-change counts
    """
    if comp_col is None:
        comp_col = f"{raw_col}_completed"

    dominant_raw = dominant_strategy_by_participant(
        per_trial, id_col=id_col, strat_col=raw_col
    ).set_index(id_col)
    dominant_comp = dominant_strategy_by_participant(
        per_trial, id_col=id_col, strat_col=comp_col
    ).set_index(id_col)

    both = pd.DataFrame(
        {
            "raw": dominant_raw[C.DOMINANCE_SCORE],
            "comp": dominant_comp[C.DOMINANCE_SCORE],
        }
    ).dropna()
    both["delta"] = both["comp"] - both["raw"]
    both["raw_label"] = dominant_raw[C.DOMINANT_STARTING_STRATEGY].reindex(both.index)
    both["comp_label"] = dominant_comp[C.DOMINANT_STARTING_STRATEGY].reindex(both.index)
    both["changed_label"] = both["raw_label"] != both["comp_label"]

    comp_series = per_trial[comp_col]
    raw_series = per_trial[raw_col]
    mask_valid = comp_series.notna() & raw_series.notna()

    def _norm(t):
        if t is None:
            return None
        t = tuple(t)
        return t[:full_len] if len(t) > full_len else t

    changed_rows = (
        comp_series[mask_valid].map(_norm) != raw_series[mask_valid].map(_norm)
    )

    per_part_changed = (
        pd.DataFrame(
            {
                "changed": changed_rows,
                "total": True,
                id_col: per_trial.loc[mask_valid, id_col].values,
            }
        )
        .groupby(id_col)
        .agg(
            seq_pct_changed=(
                "changed",
                lambda s: float(s.mean()) if len(s) else np.nan,
            ),
            seq_changed_n=("changed", "sum"),
            seq_total_n=("total", "sum"),
        )
    )

    both = both.join(per_part_changed, how="left")

    changed_n = int(both["changed_label"].sum())
    changed_pct = float(changed_n / len(both) * 100) if len(both) else np.nan
    has_seq_pct = both["seq_pct_changed"].notna().any()

    summary = {
        "participants": int(len(both)),
        "mean_raw": float(both["raw"].mean()) if len(both) else np.nan,
        "mean_completed": float(both["comp"].mean()) if len(both) else np.nan,
        "mean_delta": float(both["delta"].mean()) if len(both) else np.nan,
        f"raw_≥{int(threshold*100)}%": float(
            has_dominant_strategy(both["raw"], threshold=threshold).mean() * 100
        )
        if len(both)
        else np.nan,
        f"comp_≥{int(threshold*100)}%": float(
            has_dominant_strategy(both["comp"], threshold=threshold).mean() * 100
        )
        if len(both)
        else np.nan,
        "changed_label_n": changed_n,
        "changed_label_pct": changed_pct,
        "mean_seq_pct_changed": float(both["seq_pct_changed"].mean() * 100)
        if has_seq_pct
        else np.nan,
        "median_seq_pct_changed": float(both["seq_pct_changed"].median() * 100)
        if has_seq_pct
        else np.nan,
    }

    return summary, both


def attach_dominant_eye(
    dominant: pd.DataFrame,
    df: pd.DataFrame,
    id_col: str = C.PARTICIPANT_ID,
    eye_col: str = C.DOMINANT_EYE_COLUMN,
) -> pd.DataFrame:
    """Attach each participant's tracked eye to a per-participant frame.

    The eye is a property of the recording, constant within a participant, so
    it is joined on rather than being part of any grouping key.
    """
    if eye_col not in df.columns:
        raise KeyError(f"Column '{eye_col}' not found in df.")

    eye_map = (
        df[[id_col, eye_col]]
        .dropna(subset=[id_col])
        .groupby(id_col)[eye_col]
        .agg(lambda s: s.dropna().iloc[0] if s.dropna().size > 0 else np.nan)
    )

    out = dominant.copy()
    out[eye_col] = out[id_col].map(eye_map)
    return out


def dominant_strategy_by_eye_crosstab(
    dominant_with_eye: pd.DataFrame,
    eye_col: str = C.DOMINANT_EYE_COLUMN,
    strat_col: str = C.DOMINANT_STARTING_STRATEGY,
    min_count: int = 1,
) -> pd.DataFrame:
    """Dominant strategy against tracked eye, as a crosstab.

    Strategies used by fewer than ``min_count`` participants are pooled into
    ``"OTHER"``. Strategies are rendered as ``a -> b -> c`` strings, which is
    what the figures label them with.

    Descriptive only -- no significance test is run on this anywhere, and the
    association is not currently claimed in the paper (`findings.md` 1.6).
    """
    df = dominant_with_eye.copy()
    df["strategy_str"] = df[strat_col].apply(
        lambda s: " → ".join(map(str, s)) if isinstance(s, (list, tuple)) else str(s)
    )

    strat_counts_all = df["strategy_str"].value_counts()
    rare = strat_counts_all[strat_counts_all < min_count].index
    df.loc[df["strategy_str"].isin(rare), "strategy_str"] = "OTHER"

    return pd.crosstab(df["strategy_str"], df[eye_col])


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
        dominant = dominant_strategy_by_participant(per_trial, id_col=C.PARTICIPANT_ID)

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
