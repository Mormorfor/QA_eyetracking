# src/features/area_metrics.py
#
# The eight per-area eye-movement metrics, in ONE implementation parameterized by
# the grouping column.
#
# The project measures the same eight things on two different screens:
#
#     answer screen     grouped by `area_label`           question / answer_A..D
#     paragraph screen  grouped by `auxiliary_span_type`  critical / distractor / outside
#
# Until 2026-09-23 those were two codebases (`data_prep/data_csv_generation.py`
# and `predictive_modeling/answer_RTs/features.py`), which is how they came to
# disagree on `mean_first_fixation_duration` (T3.6) without anyone noticing.
# `features.py:47-49` asserted the two "line up 1:1"; nothing made that true.
# Now one set of functions serves both, and identical behaviour is structural
# rather than aspirational. (`todo.md` T1.7.)
#
# WHAT IS DELIBERATELY *NOT* SHARED
# ---------------------------------
# Two things differ between the screens for real reasons, and are parameters
# rather than divergence:
#
#   * the dwell-proportion denominator is the trial's total over *the areas of
#     that screen* -- five answer areas, or three paragraph spans. Same formula,
#     different scope, and the scope follows from `area_col`.
#   * the isolated leading fixation on the question is dropped as spillover from
#     the question screen. There is no question area on the paragraph screen, so
#     `drop_leading_question` is a QA-only step.
#
# Everything else -- the "." handling, the missing-value conventions, the
# nearest-interest-area fallback, the first-encounter ordering -- is identical on
# both screens by construction.
#
# MISSING-VALUE CONVENTIONS (docs/pitfalls.md section 2; do not "unify" these)
# ---------------------------------------------------------------------------
#   dwell time, fixation count      unread words count as 0 -- zero attention is
#                                   a real amount of attention
#   first fixation duration, pupil  unread words are EXCLUDED -- these are
#                                   properties *of a fixation* and undefined
#                                   when there was none

from __future__ import annotations

import ast
from collections import Counter
from typing import Iterable, Sequence

import pandas as pd

from src.config import columns as C
from src.config.columns import TRIAL_ID_COLS

# The two grouping columns in use. Passing anything else is fine -- the functions
# only ever use `area_col` as a groupby key -- but these are the two that exist.
ANSWER_AREA_COL = C.AREA_LABEL_COLUMN
PARAGRAPH_AREA_COL = C.AUXILIARY_SPAN_TYPE_COLUMN


def _group_cols(area_col: str) -> list[str]:
    return list(TRIAL_ID_COLS) + [area_col]


# ---------------------------------------------------------------------------
# Coercion -- done ONCE, up front
# ---------------------------------------------------------------------------

def coerce_ia_columns(
    df: pd.DataFrame,
    *,
    pupil: bool = False,
    inplace: bool = False,
) -> pd.DataFrame:
    """Coerce the raw IA measure columns from report text to numbers.

    The reports write `"."` for "no fixation landed here". This resolves that
    once, for every metric, instead of each metric coercing its own source
    column on the way past -- which is what created the undocumented ordering
    dependency between `create_mean_first_fix_duration` and
    `create_first_encounter_pupil_size` (`todo.md` T3.11).

    The conventions differ by column and the difference is intentional:

    * `IA_DWELL_TIME` / `IA_FIXATION_COUNT` -> `"."` becomes 0. In practice these
      columns carry no `"."` at all (the report writes a real 0), so this is a
      no-op that documents the convention rather than a substitution.
    * `IA_FIRST_FIXATION_DURATION` -> `"."` becomes NaN. A fixation of length
      zero does not exist, so the sentinel cannot be read as a measurement
      (T3.6).
    * pupil columns -> `"."` becomes NaN, same reasoning. Only coerced when
      `pupil=True`, because the answer pipeline scales these to mm and z-scores
      them in an earlier step and must not undo that.

    `inplace=True` mutates the caller's frame. The answer pipeline needs that:
    its coerced columns are expected in the saved IA-level table, and changing
    it would alter `all_participants.csv`'s schema.
    """
    out = df if inplace else df.copy()

    for col in (C.IA_DWELL_TIME, C.IA_FIXATIONS_COUNT):
        if col in out.columns:
            out[col] = pd.to_numeric(
                out[col].replace(".", 0), errors="coerce"
            ).astype("int64")

    if C.IA_FIRST_FIXATION_DURATION in out.columns:
        out[C.IA_FIRST_FIXATION_DURATION] = pd.to_numeric(
            out[C.IA_FIRST_FIXATION_DURATION], errors="coerce"
        )

    if pupil:
        for col in (
            C.IA_MAX_FIX_PUPIL_SIZE,
            C.IA_MIN_FIX_PUPIL_SIZE,
            C.IA_AVERAGE_FIX_PUPIL_SIZE,
        ):
            if col in out.columns:
                out[col] = pd.to_numeric(out[col], errors="coerce")

    return out


# ---------------------------------------------------------------------------
# The metrics
# ---------------------------------------------------------------------------

def _require_numeric(df: pd.DataFrame, cols: Sequence[str], fn: str) -> None:
    """Fail loudly if a measure column is still report text.

    `todo.md` T3.11. These columns arrive from the report carrying `"."` for
    "no fixation landed here", and every metric below assumes they have already
    been resolved to numbers. That used to be arranged by each metric coercing
    the caller's frame in place on the way past, which made the result depend
    on the order the metrics happened to run in -- and running one on its own
    failed with an opaque dtype error deep inside a groupby.

    The coercion is now a single explicit step (`coerce_ia_columns`), done once
    before any metric: `generate_new_row_features` for the answer pipeline,
    `paragraph_prep` for the paragraph one. This says so when it is skipped,
    instead of letting a comparison between `str` and `int` decide.
    """
    bad = [
        c for c in cols
        if c in df.columns and not pd.api.types.is_numeric_dtype(df[c])
    ]
    if bad:
        raise TypeError(
            f"{fn}: {bad} arrived as text rather than numbers. Run "
            f"coerce_ia_columns() on the frame once before computing any "
            f"metric -- it resolves the report's '.' sentinel, and doing it per "
            f"metric is what T3.11 removed."
        )


def mean_dwell_time(df: pd.DataFrame, area_col: str) -> pd.DataFrame:
    """Mean per-word dwell time in the area. Unread words count as 0."""
    _require_numeric(df, [C.IA_DWELL_TIME], "mean_dwell_time")
    return df.groupby(_group_cols(area_col), as_index=False).agg(
        **{C.MEAN_DWELL_TIME: (C.IA_DWELL_TIME, "mean")}
    )


def mean_fixations_count(df: pd.DataFrame, area_col: str) -> pd.DataFrame:
    """Mean fixations per word in the area. Unread words count as 0."""
    _require_numeric(df, [C.IA_FIXATIONS_COUNT], "mean_fixations_count")
    return df.groupby(_group_cols(area_col), as_index=False).agg(
        **{C.MEAN_FIXATIONS_COUNT: (C.IA_FIXATIONS_COUNT, "mean")}
    )


def mean_first_fix_duration(df: pd.DataFrame, area_col: str) -> pd.DataFrame:
    """Mean first-fixation duration over the words actually fixated (T3.6).

    Unread words arrive as NaN from `coerce_ia_columns` and `.mean()` skips
    them, so an area in which nothing was fixated yields NaN rather than 0.
    """
    _require_numeric(df, [C.IA_FIRST_FIXATION_DURATION], "mean_first_fix_duration")
    return df.groupby(_group_cols(area_col), as_index=False).agg(
        **{C.MEAN_FIRST_FIXATION_DURATION: (C.IA_FIRST_FIXATION_DURATION, "mean")}
    )


def skip_rate(
    df: pd.DataFrame,
    area_col: str,
    *,
    write_indicator: bool = False,
) -> pd.DataFrame:
    """Proportion of the area's words that were never fixated.

    `write_indicator=True` writes the per-word `area_skipped` flag back onto the
    caller's frame. The answer pipeline relies on that -- the column reaches the
    saved IA-level table -- so it is preserved rather than quietly dropped.
    """
    _require_numeric(df, [C.IA_DWELL_TIME], "skip_rate")
    if write_indicator:
        df[C.AREA_SKIPPED] = (df[C.IA_DWELL_TIME] == 0).astype(int)
        d = df
    else:
        d = df[_group_cols(area_col) + [C.IA_DWELL_TIME]].copy()
        d[C.AREA_SKIPPED] = (d[C.IA_DWELL_TIME] == 0).astype(int)

    return d.groupby(_group_cols(area_col), as_index=False).agg(
        **{C.SKIP_RATE: (C.AREA_SKIPPED, "mean")}
    )


def dwell_proportion(
    df: pd.DataFrame,
    area_col: str,
    *,
    keep_totals: bool = False,
) -> pd.DataFrame:
    """Share of the trial's total dwell time spent in each area.

    The denominator is the trial's total over the areas of THIS screen, so it
    sums to 1 across the five answer areas or across the three paragraph spans.
    That difference in scope is real and follows from `area_col`.

    A trial with zero total dwell would divide by zero; the resulting NaN is
    filled with 0, so on such a trial the values sum to 0 rather than 1. That is
    the one documented exception to "sums to 1".

    `keep_totals=True` also returns `total_area_dwell_time` and
    `total_dwell_time`. The answer pipeline merges those into the saved IA-level
    table, so dropping them would change `all_participants.csv`'s schema.
    """
    group = _group_cols(area_col)
    agg = (
        df.groupby(group, as_index=False)
        .agg({C.IA_DWELL_TIME: "sum"})
        .rename(columns={C.IA_DWELL_TIME: C.TOTAL_IA_DWELL_TIME})
    )
    agg[C.TOTAL_TRIAL_DWELL_TIME] = agg.groupby(list(TRIAL_ID_COLS))[
        C.TOTAL_IA_DWELL_TIME
    ].transform("sum")
    agg[C.AREA_DWELL_PROPORTION] = (
        agg[C.TOTAL_IA_DWELL_TIME] / agg[C.TOTAL_TRIAL_DWELL_TIME]
    ).fillna(0)

    if keep_totals:
        return agg
    return agg[group + [C.AREA_DWELL_PROPORTION]]


def mean_pupil_size(
    df: pd.DataFrame,
    area_col: str,
    *,
    include_raw: bool = True,
    include_z: bool = True,
) -> pd.DataFrame:
    """Per-area means of the pupil-size columns.

    Unfixated words carry NaN and `.mean()` skips them, so these are means over
    the words actually fixated -- the exclude convention, as for first-fixation
    duration.

    The answer screen reports both raw (mm) and z-scored means; the paragraph
    screen reports only the z-scored ones, because nothing consumes raw
    paragraph pupil sizes.
    """
    spec: dict = {}
    if include_raw:
        spec[C.MEAN_MAX_FIX_PUPIL_SIZE] = (C.IA_MAX_FIX_PUPIL_SIZE, "mean")
        spec[C.MEAN_MIN_FIX_PUPIL_SIZE] = (C.IA_MIN_FIX_PUPIL_SIZE, "mean")
        spec[C.MEAN_AVG_FIX_PUPIL_SIZE] = (C.IA_AVERAGE_FIX_PUPIL_SIZE, "mean")
    if include_z:
        spec[C.MEAN_MAX_FIX_PUPIL_SIZE_Z] = (f"{C.IA_MAX_FIX_PUPIL_SIZE}_z", "mean")
        spec[C.MEAN_MIN_FIX_PUPIL_SIZE_Z] = (f"{C.IA_MIN_FIX_PUPIL_SIZE}_z", "mean")
        spec[C.MEAN_AVG_FIX_PUPIL_SIZE_Z] = (f"{C.IA_AVERAGE_FIX_PUPIL_SIZE}_z", "mean")

    return df.groupby(_group_cols(area_col), as_index=False).agg(**spec)


def first_encounter_pupil_size(
    df: pd.DataFrame,
    area_col: str,
    *,
    include_raw: bool = True,
    include_z: bool = True,
) -> pd.DataFrame:
    """Pupil size at the first word of the area that was actually fixated.

    "First" means first in READING ORDER, so the frame is sorted explicitly by
    `IA_ID` before taking the head of each group. The answer pipeline used to
    sort only by the group keys and rely on the frame already arriving in IA
    order -- true in practice, but an implicit dependency on row order that a
    re-sort anywhere upstream would have silently changed.

    Unfixated words are excluded by `IA_FIRST_FIXATION_DURATION > 0`. Since T3.6
    that column is NaN rather than 0 for an unread word, and `NaN > 0` is False,
    so the filter behaves exactly as before.
    """
    group = _group_cols(area_col)
    mm_col = C.IA_AVERAGE_FIX_PUPIL_SIZE
    z_col = f"{mm_col}_z"

    d = df[df[C.IA_FIRST_FIXATION_DURATION] > 0]
    d = d.sort_values(list(TRIAL_ID_COLS) + [C.INTEREST_AREA_ID])
    first = d.groupby(group, as_index=False).head(1)

    out_cols = list(group)
    rename: dict = {}
    if include_raw:
        out_cols.append(mm_col)
        rename[mm_col] = C.FIRST_ENCOUNTER_AVG_PUPIL_SIZE
    if include_z:
        out_cols.append(z_col)
        rename[z_col] = C.FIRST_ENCOUNTER_AVG_PUPIL_SIZE_Z

    return first[out_cols].rename(columns=rename)


# ---------------------------------------------------------------------------
# Fixation sequences and visit counts
# ---------------------------------------------------------------------------

def parse_ia_sequence(raw) -> list:
    """Parse the serialized `INTEREST_AREA_FIXATION_SEQUENCE` cell.

    DataViewer writes `"."` (its missing marker) for a trial with no
    interest-area fixations at all, which is a real state and becomes `[]`.
    """
    if isinstance(raw, (list, tuple)):
        return list(raw)
    if isinstance(raw, str):
        raw = raw.strip()
        return ast.literal_eval(raw) if raw.startswith("[") else []
    if raw is None or pd.isna(raw):
        return []
    return []


def _parse_interest_area_list(x) -> list:
    if isinstance(x, list):
        return x
    if isinstance(x, str):
        x = x.strip()
        return ast.literal_eval(x) if x else []
    if pd.isna(x):
        return []
    return []


def resolve_fixation_sequence(
    ia_ids: Sequence,
    known_ids: set,
    nearest_ids: Sequence,
) -> tuple[list, int]:
    """Map a raw fixation sequence onto interest areas, filling off-area hits.

    A fixation that landed outside every interest area appears in the sequence
    as an id the trial does not have. Rather than drop it -- which silently
    discards a real fixation -- it is resolved to the nearest interest area,
    taken in order from `nearest_ids` (the fixation report's
    `CURRENT_FIX_NEAREST_INTEREST_AREA` for the rows whose
    `CURRENT_FIX_INTEREST_AREAS` is empty).

    This used to happen on the answer screen only. The paragraph screen dropped
    those fixations, which is why the two sides' `num_label_visits` disagreed on
    77% of trials while sharing a column name. Shared here so the mechanics match
    (Diana, 2026-09-23, `todo.md` T1.7).

    **A fixation that cannot be placed is DROPPED, never fatal** (Diana,
    2026-09-23). Not knowing which area a fixation fell into is a normal
    limitation of the recording, not a reason to lose the trial or stop the run.
    The count is returned rather than printed per occurrence, so the caller can
    report one number instead of a line per fixation -- dropping stays visible
    without becoming noise.

    Returns (resolved ids, number of fixations dropped).
    """
    pointer = 0
    resolved = []
    dropped = 0
    for ia_id in ia_ids:
        if ia_id in known_ids:
            resolved.append(ia_id)
        elif pointer < len(nearest_ids):
            resolved.append(nearest_ids[pointer])
            pointer += 1
        else:
            dropped += 1
    return resolved, dropped


def nearest_ia_queue(fix_group: pd.DataFrame) -> list:
    """The trial's off-area fixations' nearest interest areas, in order."""
    areas = fix_group[C.CURRENT_FIX_INTEREST_AREAS].apply(_parse_interest_area_list)
    return fix_group.loc[areas.apply(len) == 0, C.NEAREST_IA].astype(int).tolist()


def drop_leading_question_fixation(labels: list, *others: list) -> tuple[list, ...]:
    """Drop an isolated opening fixation on the question.

    A single fixation on the question immediately followed by a move elsewhere
    is spillover from the question screen, not part of the answer scan. ANSWER
    SCREEN ONLY -- the paragraph screen has no question area.
    """
    if len(labels) >= 2 and labels[0] == "question" and labels[1] != "question":
        return (labels[1:], *[o[1:] for o in others])
    return (labels, *others)


def collapse_runs(seq: Iterable) -> list:
    """Collapse consecutive repeats, turning a fixation sequence into transitions."""
    out = []
    for i, x in enumerate(seq):
        if i == 0 or x != seq[i - 1]:
            out.append(x)
    return out


def visit_counts_from_sequence(
    collapsed: Iterable,
    areas: Sequence[str],
    out_col: str = C.NUM_LABEL_VISITS,
) -> dict:
    """Count entries into each area from an already-collapsed sequence."""
    counts = Counter(collapsed)
    return {area: int(counts.get(area, 0)) for area in areas}

# ---------------------------------------------------------------------------
# Moved here from data_prep/data_csv_generation.py in stage C (2026-10-06).
# These are the registry's `create_*` / `add_*` entries -- the functions the
# prep pipeline calls to put columns on the IA frame. They mutate the frame
# they are handed and depend on registry order; that is kept as-is by decision
# (Diana, 2026-10-06), with the default runner executing them in that order.
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# One per-area metric block, for either screen
# ---------------------------------------------------------------------------

def build_area_metrics(
    df: pd.DataFrame,
    scr,
    visit_counts: pd.DataFrame | None = None,
    metric_cols: Sequence[str] | None = None,
) -> pd.DataFrame:
    """One row per (participant, trial, area) carrying every per-area metric.

    **The merge loop that both screens used to keep their own copy of**
    (`docs/decisions/2026-10-06-stage-d-proposal.md` section 3.8). `scr` is a
    `config.screens.Screen`; substituting it is the whole difference between
    "answer screen" and "paragraph screen" here.

    `visit_counts` is passed in rather than computed, because the two screens
    count visits from different sources: the answer screen from the simplified
    label sequence, the paragraph screen by resolving the raw IA sequence
    against its nearest-interest-area queue. Everything else is shared.
    """
    group = list(TRIAL_ID_COLS) + [scr.area_col]
    parts = [
        mean_dwell_time(df, scr.area_col),
        mean_fixations_count(df, scr.area_col),
        mean_first_fix_duration(df, scr.area_col),
        skip_rate(df, scr.area_col, write_indicator=scr.write_skip_indicator),
        dwell_proportion(df, scr.area_col, keep_totals=scr.keep_dwell_totals),
        mean_pupil_size(df, scr.area_col, include_raw=scr.include_raw_pupil, include_z=True),
        first_encounter_pupil_size(
            df, scr.area_col, include_raw=scr.include_raw_pupil, include_z=True
        ),
    ]
    if visit_counts is not None:
        parts.append(visit_counts)

    out = parts[0]
    for part in parts[1:]:
        out = out.merge(part, on=group, how="outer")
    if metric_cols is None:
        return out
    return out[group + list(metric_cols)]

def create_mean_area_dwell_time(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the mean dwell time per (trial, participant, area_label) group.

    This function aggregates interest-area-level data into area-level summaries
    by computing the mean dwell time for each unique combination of:
    - TRIAL_ID
    - PARTICIPANT_ID
    - AREA_LABEL_COLUMN (e.g., 'question', 'answer_A', ...)

    """
    return mean_dwell_time(df, C.AREA_LABEL_COLUMN)


def create_mean_area_fix_count(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the mean number of fixations per (trial, participant, area_label) group.

    This function aggregates interest-area-level data by computing the mean
    number of fixations for each unique combination of:
    - TRIAL_ID
    - PARTICIPANT_ID
    - AREA_LABEL_COLUMN (e.g., 'question', 'answer_A', ...)

    """
    return mean_fixations_count(df, C.AREA_LABEL_COLUMN)


def create_mean_first_fix_duration(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the mean first-fixation duration per (trial, participant, area_label).

    This function:
    Groups by (TRIAL_ID, PARTICIPANT_ID, AREA_LABEL_COLUMN).
    Computes the mean first-fixation duration over the words that were
    ACTUALLY FIXATED within each group.

    Unread words are excluded rather than counted as zero: a fixation of length
    zero does not exist, so the "." sentinel can only ever mean "no fixation
    landed here", never a measurement. Averaging it in as 0 made this metric
    largely a restatement of skip_rate (r = -0.70 to -0.94 per area before the
    change). Coercing to NaN lets `.mean()` skip those words, so the result is
    intensity per *read* word.

    This is deliberately the OPPOSITE convention from mean_dwell_time and
    mean_fixations_count, where 0 is a real measurement -- a word nobody read
    genuinely received 0 ms and 0 fixations, and those metrics are meant to
    capture attention per *available* word. Do not unify the family for
    tidiness; see docs/pitfalls.md section 2. It matches the paragraph path
    (answer_RTs/features.py::_mean_first_fix_duration), which always coerced
    this way, so the two are now one measure. (todo.md T3.6)

    Consequence: an area in which no word was fixated yields NaN, not 0 --
    5,810 question areas and 253-829 per answer area on L1.
    """
    return mean_first_fix_duration(df, C.AREA_LABEL_COLUMN)


def create_skip_rate(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the skip rate per (trial, participant, area_label).

    A skip is defined as an interest area (IA) with *zero dwell time*.
    The skip rate is the proportion of IAs within an area (e.g., 'answer_A')
    that were skipped by the participant during the trial.

    - Create an indicator AREA_SKIPPED:
          1 if IA_DWELL_TIME == 0
          0 otherwise
    - Group by (TRIAL_ID, PARTICIPANT_ID, AREA_LABEL_COLUMN)
    - Compute the mean of AREA_SKIPPED → skip_rate

    """
    # write_indicator=True keeps `area_skipped` on the caller's frame, where the
    # saved IA-level table expects it.
    return skip_rate(df, C.AREA_LABEL_COLUMN, write_indicator=True)


def create_dwell_proportions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute dwell time proportions per area within each trial and participant.

    For each (TRIAL_ID, PARTICIPANT_ID, AREA_LABEL_COLUMN), this function:
    1. Sums IA_DWELL_TIME to obtain TOTAL_IA_DWELL_TIME per area.
    2. Sums TOTAL_IA_DWELL_TIME over all areas within a trial/participant to
       obtain TOTAL_TRIAL_DWELL_TIME.
    3. Computes AREA_DWELL_PROPORTION as:
           TOTAL_IA_DWELL_TIME / TOTAL_TRIAL_DWELL_TIME

    Any resulting NaN values (e.g., if TOTAL_TRIAL_DWELL_TIME is 0) are replaced by 0.
    """
    # keep_totals=True: total_area_dwell_time and total_dwell_time are merged into
    # the saved IA-level table, so dropping them would change its schema.
    return dwell_proportion(df, C.AREA_LABEL_COLUMN, keep_totals=True)
