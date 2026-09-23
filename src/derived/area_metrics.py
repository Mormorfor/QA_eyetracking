# src/derived/area_metrics.py
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
from typing import Iterable, Optional, Sequence

import pandas as pd

from src import constants as C
from src.constants import TRIAL_ID_COLS

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

def mean_dwell_time(df: pd.DataFrame, area_col: str) -> pd.DataFrame:
    """Mean per-word dwell time in the area. Unread words count as 0."""
    return df.groupby(_group_cols(area_col), as_index=False).agg(
        **{C.MEAN_DWELL_TIME: (C.IA_DWELL_TIME, "mean")}
    )


def mean_fixations_count(df: pd.DataFrame, area_col: str) -> pd.DataFrame:
    """Mean fixations per word in the area. Unread words count as 0."""
    return df.groupby(_group_cols(area_col), as_index=False).agg(
        **{C.MEAN_FIXATIONS_COUNT: (C.IA_FIXATIONS_COUNT, "mean")}
    )


def mean_first_fix_duration(df: pd.DataFrame, area_col: str) -> pd.DataFrame:
    """Mean first-fixation duration over the words actually fixated (T3.6).

    Unread words arrive as NaN from `coerce_ia_columns` and `.mean()` skips
    them, so an area in which nothing was fixated yields NaN rather than 0.
    """
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
