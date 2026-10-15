"""The per-area measurements: one set of functions for both screens.

Each function turns word rows into one row per trial and area. The same code
measures both screens; only the column naming the area changes:

    answer screen     area_label           question, answer_A..D
    paragraph screen  auxiliary_span_type  critical, distractor, outside

What is measured, per area:

    mean_dwell_time               average looking time per word
    mean_fixations_count          average number of fixations per word
    mean_first_fixation_duration  average first-fixation length per word
    skip_rate                     share of words never looked at
    area_dwell_proportion         share of the trial's looking time
    mean_*_pupil_size(_z)         average pupil size during fixations
    first_encounter_*_pupil_size  pupil size on the area's first fixated word
                                  (see `first_encounter_pupil_size`)

**Words nobody looked at are handled two ways, on purpose:**

    dwell time, fixation count      count as 0 -- zero attention is a real
                                    amount of attention
    first fixation duration, pupil  left out -- these describe a fixation, and
                                    there was none

Do not make these consistent; the difference is deliberate.

Two things differ between the screens, and follow from the area column:

* `area_dwell_proportion` is a share of the looking time on *that screen*, so
  it sums to 1 over the five answer-screen areas or over the three paragraph
  parts.
* An isolated first fixation on the question is dropped as left over from the
  previous screen. The paragraph screen has no question area, so this only
  applies to the answer screen.
"""

from __future__ import annotations

import ast
from collections import Counter
from typing import Iterable, Sequence

import pandas as pd

from src.config import columns as C
from src.config.columns import TRIAL_ID_COLS

# The two area columns in use, one per screen.
ANSWER_AREA_COL = C.AREA_LABEL_COLUMN
PARAGRAPH_AREA_COL = C.AUXILIARY_SPAN_TYPE_COLUMN


def _group_cols(area_col: str) -> list[str]:
    """participant_id, TRIAL_INDEX and the area column: one group per trial and area."""
    return list(TRIAL_ID_COLS) + [area_col]


# ---------------------------------------------------------------------------
# Turning the report's text columns into numbers -- done once, before anything
# ---------------------------------------------------------------------------

def coerce_ia_columns(
    df: pd.DataFrame,
    *,
    pupil: bool = False,
    inplace: bool = False,
) -> pd.DataFrame:
    """Turn the per-word measurement columns from report text into numbers.

    The report writes "." where no fixation landed. This is done once, before
    any measurement, so every measurement sees the same numeric columns.

    The "." becomes a different value per column, on purpose:

    * dwell time, fixation count -> 0. (In practice these columns have no "."
      -- the report already writes 0 -- so nothing actually changes.)
    * first fixation duration -> NaN. A fixation of length zero does not exist,
      so "." cannot be read as a measurement.
    * pupil columns -> NaN, for the same reason. Only when `pupil=True`: the
      answer-screen pipeline has already converted and z-scored them by this
      point, and must not convert them again.

    `inplace=True` changes the caller's frame instead of returning a copy; the
    answer-screen pipeline uses that so the numeric columns are what get saved.
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
# The measurements
# ---------------------------------------------------------------------------

def _require_numeric(df: pd.DataFrame, cols: Sequence[str], fn: str) -> None:
    """Stop with a clear error if a measurement column is still text.

    Every measurement below expects `coerce_ia_columns` to have run first.
    Without this check, forgetting it would fail with an unclear type error
    deep inside pandas.
    """
    bad = [
        c for c in cols
        if c in df.columns and not pd.api.types.is_numeric_dtype(df[c])
    ]
    if bad:
        raise TypeError(
            f"{fn}: {bad} arrived as text rather than numbers. Run "
            f"coerce_ia_columns() on the frame once before computing any "
            f"measurement -- it turns the report's '.' into numbers."
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
    """Average first-fixation duration over the words that were looked at.

    Words nobody looked at are NaN (from `coerce_ia_columns`) and the average
    skips them, so an area where nothing was looked at gives NaN, not 0.
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
    """Share of the area's words that were never looked at (dwell time 0).

    `write_indicator=True` also adds the per-word 0/1 flag `area_skipped` to
    the caller's frame, so it ends up in the saved word table.
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
    """Share of the trial's looking time spent in each area.

    The total is over the areas of this screen only, so the shares add up to 1
    over the five answer-screen areas, or over the three paragraph parts.

    Exception: a trial with no looking time at all would divide by zero. Its
    shares are set to 0, so they add up to 0 instead of 1.

    `keep_totals=True` also returns the looking time per area
    (`total_area_dwell_time`) and per trial (`total_dwell_time`).
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
    """Three average pupil sizes per area: the average over its words of each
    word's mean, largest and smallest pupil size during fixations.

    Words nobody looked at have no pupil size (NaN) and are left out of the
    average. `include_raw` gives the values in mm, `include_z` the z-scored
    ones; both screens use both.
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
    """Pupil size on the area's first word, by position, that was looked at.

    Takes the words of the area that were looked at, picks the one with the
    lowest word number (`IA_ID`, i.e. first in reading order), and reports its
    average pupil size over all its fixations.

    Careful, the name suggests more than this does: it is the first word **by
    position**, not the first word the participant looked at in time. On Study
    1's answer screen the two are different words in about a third of areas.

    Words nobody looked at are left out (their first-fixation duration is NaN,
    and `NaN > 0` is False).
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
    """Read the tracker's word-sequence cell (`INTEREST_AREA_FIXATION_SEQUENCE`)
    into a list of word ids.

    A trial with no fixation on any word is written as "."; that is a real
    situation, not an error, and gives an empty list.
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
    """Read a fixation row's list of interest areas; empty if there are none."""
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
    """Turn a trial's fixation sequence into a sequence of word ids, placing
    fixations that missed every word on the nearest word.

    A fixation that landed between words shows up with an id the trial does not
    have. Instead of being dropped, it is given the nearest word, taken in turn
    from `nearest_ids` (see `nearest_ia_queue`).

    A fixation with no nearest word either cannot be placed, and is dropped.
    That is a normal limit of the recording, not a reason to stop. The number
    dropped is returned, so the caller can report it.

    Both screens use this, so their visit counts are made the same way.

    Returns (word ids, number of fixations dropped).
    """
    pointer = 0
    resolved = []
    dropped = 0
    for ia_id in ia_ids:
        if ia_id in known_ids:
            resolved.append(ia_id)
        elif pointer < len(nearest_ids):
            nearest = nearest_ids[pointer]
            pointer += 1
            # `None`: on no word and with no nearest word, so it cannot be
            # placed. Dropped and counted. The pointer still moves on, because
            # the queue is matched up by position.
            if nearest is None:
                dropped += 1
            else:
                resolved.append(nearest)
        else:
            dropped += 1
    return resolved, dropped


def nearest_ia_queue(fix_group: pd.DataFrame) -> list:
    """For each fixation in the trial that missed every word, the nearest word.

    Taken from the fixation report, in order. `None` marks a fixation with no
    nearest word either (the report writes "."). It is kept in the list rather
    than removed, because the list is matched to the fixations by position:
    removing one would give every later fixation its neighbour's nearest word.
    """
    areas = fix_group[C.CURRENT_FIX_INTEREST_AREAS].apply(_parse_interest_area_list)
    ids = pd.to_numeric(
        fix_group.loc[areas.apply(len) == 0, C.NEAREST_IA], errors="coerce"
    )
    return [None if pd.isna(v) else int(v) for v in ids]


def drop_leading_question_fixation(labels: list, *others: list) -> tuple[list, ...]:
    """Drop a single opening fixation on the question.

    If the sequence starts with exactly one fixation on the question and then
    moves elsewhere, that fixation is left over from the previous screen (which
    showed the question), not part of looking at the answers. `others` are
    parallel sequences trimmed the same way. Answer screen only.
    """
    if len(labels) >= 2 and labels[0] == "question" and labels[1] != "question":
        return (labels[1:], *[o[1:] for o in others])
    return (labels, *others)


def collapse_runs(seq: Iterable) -> list:
    """Merge repeats in a row: A A B A becomes A B A, so each entry is a move."""
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
    """How many times each area was entered, from a merged sequence.

    Returns {area: count} for every area in `areas`, 0 for areas never entered.
    `out_col` is not used.
    """
    counts = Counter(collapsed)
    return {area: int(counts.get(area, 0)) for area in areas}

# ---------------------------------------------------------------------------
# Every per-area measurement at once, for either screen
# ---------------------------------------------------------------------------

def build_area_metrics(
    df: pd.DataFrame,
    scr,
    visit_counts: pd.DataFrame | None = None,
    metric_cols: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Every per-area measurement, one row per participant, trial and area.

    `scr` is a `config.screens.Screen` and says which screen this is. Includes
    the dwell-time totals, and pupil size in mm as well as z-scored. As a side
    effect, adds the per-word `area_skipped` flag to `df`.

    `visit_counts` (how many times each area was entered) is passed in rather
    than computed here, because each screen builds it from a different sequence.
    `metric_cols` keeps only those columns.
    """
    group = list(TRIAL_ID_COLS) + [scr.area_col]
    parts = [
        mean_dwell_time(df, scr.area_col),
        mean_fixations_count(df, scr.area_col),
        mean_first_fix_duration(df, scr.area_col),
        # Both screens get every column: the per-word skip flag, the dwell
        # totals, and pupil size in mm as well as z-scored.
        skip_rate(df, scr.area_col, write_indicator=True),
        dwell_proportion(df, scr.area_col, keep_totals=True),
        mean_pupil_size(df, scr.area_col, include_raw=True, include_z=True),
        first_encounter_pupil_size(df, scr.area_col, include_raw=True, include_z=True),
    ]
    if visit_counts is not None:
        parts.append(visit_counts)

    out = parts[0]
    for part in parts[1:]:
        out = out.merge(part, on=group, how="outer")
    if metric_cols is None:
        return out
    return out[group + list(metric_cols)]

# ---------------------------------------------------------------------------
# Registry entries for the answer screen
#
# Each is a thin wrapper: one measurement above, grouped by area_label. These
# are the names listed in ingest/registry.py.
# ---------------------------------------------------------------------------

def create_mean_area_dwell_time(df: pd.DataFrame) -> pd.DataFrame:
    """Average looking time per word, per answer-screen area. Unread words count as 0."""
    return mean_dwell_time(df, C.AREA_LABEL_COLUMN)


def create_mean_area_fix_count(df: pd.DataFrame) -> pd.DataFrame:
    """Average number of fixations per word, per answer-screen area. Unread words count as 0."""
    return mean_fixations_count(df, C.AREA_LABEL_COLUMN)


def create_mean_first_fix_duration(df: pd.DataFrame) -> pd.DataFrame:
    """Average first-fixation length, per answer-screen area, over the words
    that were looked at.

    Unread words are left out rather than counted as 0, because a fixation of
    length zero does not exist. (Counting them as 0 would make this measure
    mostly a copy of skip_rate.) This is the opposite of dwell time and fixation
    count, where 0 is a real value -- see the module docstring.

    An area where no word was looked at gives NaN, not 0. On Study 1 that is
    5,810 question areas and 253-829 per answer.
    """
    return mean_first_fix_duration(df, C.AREA_LABEL_COLUMN)


def create_skip_rate(df: pd.DataFrame) -> pd.DataFrame:
    """Share of each answer-screen area's words never looked at (dwell time 0).

    Also adds the per-word 0/1 flag `area_skipped` to the frame.
    """
    return skip_rate(df, C.AREA_LABEL_COLUMN, write_indicator=True)


def create_dwell_proportions(df: pd.DataFrame) -> pd.DataFrame:
    """Share of the trial's looking time spent on each answer-screen area.

    Also returns the looking time per area and per trial. A trial with no
    looking time at all gets shares of 0.
    """
    return dwell_proportion(df, C.AREA_LABEL_COLUMN, keep_totals=True)
