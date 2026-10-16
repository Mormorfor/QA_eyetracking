"""The opening scan ("starting strategy"): the first four areas a participant
visits on the answer screen.

A trial's starting strategy is the first four entries of its merged location
sequence, repeats included: e.g. (top, right, bottom, left), but also
(top, right, top, left). Four visits are enough for a full circuit of the four
answers, if the participant makes one; nothing requires four different answers.
Usually visits to the question are removed first, so the scan is of the answers
alone (a question visit between two visits to the same answer leaves both); the
model features come in both versions, with the question kept (`_with_q`) and
removed (`_no_q`). A scan with fewer than four visits stays shorter -- it is not
padded.

From the per-trial strategies:

- each participant's **dominant strategy** is their most common one, and their
  **dominance score** is the share of their trials that use it;
- a participant **has a dominant strategy** if that share is at least 0.5;
- **prefix completion** repairs short scans: a scan that reached only two
  answers is completed the way most full scans with that start continue. The
  paper's descriptive figures report both raw and completed scans; the model
  features use raw scans only.

Everything here only computes; the figures are in `analyses/scan_strategies/`.

**Which trials a dominant strategy is worked out from matters.** It summarises
a person over the trials it is given, so the answer depends on that choice.
See `scope_df` and `scope_by` on `build_trial_level_pattern_features`.
"""

from __future__ import annotations

import ast
from collections import Counter, defaultdict
from typing import Literal, Optional, Sequence

import numpy as np
import pandas as pd

from src.config import columns as C
from src.config.columns import TRIAL_ID_COLS


SeqKind = Literal["location", "label"]

DEFAULT_WINDOW_LEN = 4

# A participant "has a dominant strategy" when their most common opening scan
# covers AT LEAST this share of their trials. The paper says "at least in half
# of the trials", so the test is `>=`, not `>` -- see has_dominant_strategy.
DEFAULT_DOMINANCE_THRESHOLD = 0.5


def _parse_seq(x) -> tuple:
    """Read a stored merged sequence into a tuple.

    The sequence is a tuple during a pipeline run, and text once it has been
    saved and read back; both are handled. Anything that cannot be read --
    missing, or malformed text -- becomes an empty tuple.
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
    """The first ``window_len`` entries of a merged sequence.

    ``drop_question`` removes the question's entries *before* taking them, so
    the result is the order of the answers alone. (Earlier steps only remove a
    single opening fixation on the question; later question visits are still
    in the sequence.) No padding: a shorter sequence gives a shorter tuple.
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
    """How many single-step edits turn one sequence into the other.

    An edit is inserting, deleting or swapping one whole entry (e.g. one
    "answer_0(top)"), not one character. 0 means the two are identical. This is
    the graded version of "breaks pattern".

    With ``normalize`` the count is divided by the longer sequence's length,
    giving a value from 0 to 1 (0 when both are empty).
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
    keep_cols: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Each trial's starting strategy, one row per trial.

    The first ``window_len`` entries of ``simpl_fix_by_loc`` (or of
    ``simpl_fix_by_label`` with ``kind="label"``), as a tuple in ``out_col``.

    ``keep_cols`` carries other columns along, e.g. ``regime`` or
    ``session_id``, so the strategies can later be grouped by them. They must
    have one value per trial; the run stops if one does not.
    """
    if kind == "location":
        seq_col = C.SIMPLIFIED_FIX_SEQ_BY_LOCATION
    elif kind == "label":
        seq_col = C.SIMPLIFIED_FIX_SEQ_BY_LABEL
    else:
        raise ValueError("kind must be 'location' or 'label'")

    keep_cols = [c for c in (keep_cols or []) if c not in (C.PARTICIPANT_ID, C.TRIAL_ID)]
    required = [C.PARTICIPANT_ID, C.TRIAL_ID, seq_col] + keep_cols
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    out = df[required].drop_duplicates(subset=[C.PARTICIPANT_ID, C.TRIAL_ID]).copy()

    if keep_cols:
        # Only the first row of each trial is kept above. If a keep_col had
        # several values in one trial, one would be picked arbitrarily.
        n_trials = len(df.drop_duplicates(subset=[C.PARTICIPANT_ID, C.TRIAL_ID]))
        n_distinct = len(df[[C.PARTICIPANT_ID, C.TRIAL_ID] + keep_cols].drop_duplicates())
        assert n_distinct == n_trials, (
            f"keep_cols {keep_cols} are not constant within a trial: "
            f"{n_distinct} distinct (trial, {keep_cols}) combinations for "
            f"{n_trials} trials"
        )

    out[out_col] = out[seq_col].apply(
        lambda s: _starting_window(s, window_len=window_len, drop_question=drop_question)
    )

    return out[[C.PARTICIPANT_ID, C.TRIAL_ID] + keep_cols + [out_col]].reset_index(
        drop=True
    )


def compute_dominant_starting_strategy(
    df: pd.DataFrame,
    kind: SeqKind = "location",
    window_len: int = DEFAULT_WINDOW_LEN,
    drop_question: bool = True,
    id_col: str = C.PARTICIPANT_ID,
) -> pd.DataFrame:
    """Each participant's dominant strategy and dominance score, from a
    word- or trial-level frame.

    Builds the per-trial strategies (`build_starting_strategies`), then picks
    each participant's most common one (`dominant_strategy_by_participant`).
    Returns one row per participant: the dominant strategy, its share of their
    trials, and the number of trials.
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
    by: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Each participant's most common strategy, and the share of trials using it.

    `per_trial` has one row per trial with a strategy in `strat_col`. Returns
    one row per participant: `dominant_starting_strategy`, `dominance_score`
    (its share of their trials) and `n_strategy_trials` (how many trials that
    share is out of).

    A tie for most common goes to the strategy that sorts first.

    The result describes only the trials in `per_trial` -- choosing those is
    the caller's job. `by` splits each participant further, e.g. `by=["regime"]`
    gives one dominant strategy per participant and regime. By default all of a
    participant's trials are pooled, across regimes and sessions.

    Every figure and feature in the project gets its dominant strategy from this
    one function.
    """
    by = list(by or [])
    group_cols = [id_col] + by
    missing = [c for c in group_cols if c not in per_trial.columns]
    if missing:
        raise KeyError(f"Missing grouping columns: {missing}")

    rows = []
    for key, g in per_trial.groupby(group_cols, sort=False):
        key = key if isinstance(key, tuple) else (key,)
        counts = Counter(g[strat_col])
        n_trials = int(sum(counts.values()))

        # most frequent first; a tie goes to the strategy that sorts first
        dominant, top_count = min(counts.items(), key=lambda kv: (-kv[1], kv[0]))

        rows.append(
            {
                **dict(zip(group_cols, key)),
                C.DOMINANT_STARTING_STRATEGY: dominant,
                C.DOMINANCE_SCORE: top_count / n_trials if n_trials else float("nan"),
                C.N_STRATEGY_TRIALS: n_trials,
            }
        )

    return pd.DataFrame(
        rows,
        columns=group_cols
        + [
            C.DOMINANT_STARTING_STRATEGY,
            C.DOMINANCE_SCORE,
            C.N_STRATEGY_TRIALS,
        ],
    )


def has_dominant_strategy(
    dominance_score,
    threshold: float = DEFAULT_DOMINANCE_THRESHOLD,
):
    """Whether a dominance score counts as "having a dominant strategy".

    Uses ``>=``, not ``>``, because the paper says "at least in half of the
    trials". The difference matters: on Study 1, 5 hunters and 4 gatherers have
    a score of exactly 0.50 -- 2.8 and 2.2 percentage points of their groups.

    Takes a single score or a Series. Every threshold test in the project goes
    through here.
    """
    return dominance_score >= threshold


# ---------------------------------------------------------------------------
# Numbers for the paper's description of the opening scan
#
# Computation only; the figures that show these are in analyses/scan_strategies/.
# ---------------------------------------------------------------------------


def build_prefix_completion_map(
    series: pd.Series,
    full_len: int = DEFAULT_WINDOW_LEN,
) -> dict:
    """Learn how short scans are usually completed: the "interrupted-scan
    completion" the paper describes.

    Looks only at the full-length (``full_len``) strategies in ``series``. For
    every start of one -- its first 1, 2, ... entries -- records which full
    strategy most often begins that way. A tie goes to the strategy that sorts
    last.

    Learned from whatever ``series`` is given. The scan-strategy analysis passes
    the whole dataset, so hunters and gatherers share one map.

    Returns ``{start: full strategy}``.
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
    """Complete the strategies shorter than ``full_len`` using a completion map.

    Adds two columns:
      ``<strat_col><col_suffix>``   the completed strategy (a short one with no
                                    entry in the map stays as it is)
      ``<strat_col>_was_completed`` True if the original was shorter than
                                    ``full_len``

    If ``prefix2full`` is not given, the map is learned from this frame alone.

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
    """How far ahead of the runner-up each participant's dominant strategy is.

    ``p1`` is the dominant strategy's share (the same as ``dominance_score``),
    ``p2`` the second most common one's, and ``gap = p1 - p2``. Someone who used
    only one strategy has ``p2 = 0``.

    Returns one row per participant (indexed by ``id_col``).
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
    """How many different opening scans each participant used.

    The counterweight to the dominance score: "dominant" means most common, not
    consistent. Someone whose dominant strategy covers half their trials still
    used other openings on the rest.
    """
    return per_trial.groupby(id_col)[strat_col].nunique()


def dominant_strategy_counts(
    per_trial: pd.DataFrame,
    id_col: str = C.PARTICIPANT_ID,
    strat_col: str = C.STARTING_STRATEGY_COL,
    threshold: float = DEFAULT_DOMINANCE_THRESHOLD,
) -> pd.Series:
    """Which strategies are dominant, and for how many participants each.

    Counts only participants who have a dominant strategy at ``threshold``
    (see :func:`has_dominant_strategy`). Largest count first.
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
    """How much completing short scans changes the result.

    Compares each participant's dominance score and dominant strategy before
    and after completion. If completion barely changes anything, the opening-
    scan finding does not depend on how short scans were handled.

    Returns ``(summary, both)``:
      ``summary`` a dict: average scores, the share of participants with a
                  dominant strategy before and after, and how many
                  participants' dominant strategy changed
      ``both``    one row per participant: both scores, the difference, both
                  dominant strategies, whether it changed, and how many of
                  their trials' strategies changed
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
    """Add each participant's dominant eye (left/right) to a per-participant frame.

    The first non-missing value per participant is used; the eye does not
    change within a participant.
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
    """A count table: dominant strategy against dominant eye.

    Strategies held by fewer than ``min_count`` participants are grouped as
    "OTHER". Strategies are written as "a → b → c", as in the figures.

    Description only: no significance test is run on this table.
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
    *,
    scope_df: Optional[pd.DataFrame] = None,
    scope_by: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """The opening-scan features the model uses, one row per trial.

    For each of two versions -- question visits kept (`_with_q`) or removed
    (`_no_q`) before taking the strategy:

      breaks_pattern       1 if this trial's strategy differs from the
                           participant's dominant one, else 0
      dominance_score      the participant's dominance score, on every trial
      breaks_x_dominance   breaks_pattern x dominance_score (if add_interaction)
      strategy_distance    edits between this trial's strategy and the dominant
                           one; 0 when they match (if add_distance)
      n_strategy_trials    how many trials the dominant strategy was worked
                           out from

    **Which trials the dominant strategy is worked out from.** Two settings:

    ``scope_df`` -- which trials. By default, `df` itself and nothing else.
      This matters for the model. A score worked out over all of a person's
      trials includes the held-out ones, so the model would see a summary of
      its own test data. `modeling/crossval.py` avoids that by rebuilding these
      features inside each fold -- but only when it is not handed a ready-made
      table (`trial_df`); with one, the scores do include held-out trials. For
      a descriptive figure, where nothing is held out, pass the participant's
      full set of trials instead.

    ``scope_by`` -- how to split each participant's trials. By default not at
      all: one dominant strategy per participant, pooling Study 2's regimes and
      sessions. ``["regime"]`` gives one per participant and regime -- the only
      way to ask whether someone's scan changes between knowledge regimes. The
      columns must have one value per trial.

    Study 1's hunters/gatherers split needs neither: each participant is in one
    group only, so their trials are never split by it.

    `n_strategy_trials` is kept on every row because it is the only sign of how
    many trials a score came from: 0.5 out of 12 trials is far less certain than
    0.5 out of 50.

    This is deliberately not a registry entry (ingest/registry.py). Registry
    entries run once, over the whole dataset, so a dominance score built there
    would always include the held-out trials.
    """
    scope_by = list(scope_by or [])

    variants = [
        (False, C.BREAKS_PATTERN_WITH_Q, C.DOMINANCE_SCORE_WITH_Q,
         C.BREAKS_X_DOMINANCE_WITH_Q, C.STRATEGY_DISTANCE_WITH_Q,
         C.N_STRATEGY_TRIALS_WITH_Q),
        (True, C.BREAKS_PATTERN_NO_Q, C.DOMINANCE_SCORE_NO_Q,
         C.BREAKS_X_DOMINANCE_NO_Q, C.STRATEGY_DISTANCE_NO_Q,
         C.N_STRATEGY_TRIALS_NO_Q),
    ]

    out: pd.DataFrame | None = None
    for drop_question, breaks_col, score_col, inter_col, dist_col, n_col in variants:
        per_trial = build_starting_strategies(
            df,
            kind=kind,
            window_len=window_len,
            drop_question=drop_question,
            keep_cols=scope_by,
        )

        # The dominant strategy comes from scope_df if given, else from df. Both
        # go through the same function, so a strategy means the same thing.
        if scope_df is None:
            scope_per_trial = per_trial
        else:
            scope_per_trial = build_starting_strategies(
                scope_df,
                kind=kind,
                window_len=window_len,
                drop_question=drop_question,
                keep_cols=scope_by,
            )

        join_keys = [C.PARTICIPANT_ID] + scope_by
        dominant = dominant_strategy_by_participant(
            scope_per_trial, id_col=C.PARTICIPANT_ID, by=scope_by
        )

        merged = per_trial.merge(dominant, on=join_keys, how="left")
        # A trial whose participant is missing from scope_df would get no
        # dominant strategy, and would then quietly count as breaking the
        # pattern. So the run stops instead.
        unmatched = int(merged[C.DOMINANT_STARTING_STRATEGY].isna().sum())
        assert unmatched == 0, (
            f"{unmatched} of {len(merged)} trials have no dominant strategy: their "
            f"{join_keys} do not appear in the estimating frame. Pass a scope_df "
            "that covers every trial in df."
        )

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

        merged = merged.rename(
            columns={C.DOMINANCE_SCORE: score_col, C.N_STRATEGY_TRIALS: n_col}
        )
        cols.append(score_col)
        if add_interaction:
            merged[inter_col] = merged[breaks_col] * merged[score_col]
            cols.append(inter_col)
        cols.append(n_col)

        piece = merged[list(TRIAL_ID_COLS) + cols]
        out = piece if out is None else out.merge(piece, on=list(TRIAL_ID_COLS))

    return out.reset_index(drop=True)
