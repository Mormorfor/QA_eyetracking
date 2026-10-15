"""How long each area was looked at, three ways, one row per trial.

Three measures, each per area (e.g. RT_pure_answer_A):

    TFD_*              total fixation duration: the looking time on the area's
                       words, added up
    RT_*               time actually spent on the area. A visit is a run of
                       fixations in a row on the area; each visit counts from
                       the start of its first fixation to the start of the next
                       fixation anywhere. Visits are added up.
    TimeSinceOffset_*  from the start of the first look at the area to the end
                       of the last look -- so time spent elsewhere in between
                       counts too

Each comes as `_pure` (milliseconds) and `_normalized` (divided by the number of
words in the area).

An area nobody looked at gets 0, never NaN. That 0 is a real value: every area
is on screen in every trial, so 0 can only mean "not looked at".

Both screens use the same functions; `build_screen_rt_tfd` puts them together.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from typing import Sequence

import pandas as pd

from src.config import columns as C
from src.config.datasets import RT_AND_TFD_PATH
from src.config.columns import TRIAL_ID_COLS
from src.config.screens import screen
from src.features.fixations import place_fixations

KEYS = list(TRIAL_ID_COLS)


def compute_reading_times(
    data: pd.DataFrame,
    area_col: str,
    regions: Sequence[str],
) -> pd.DataFrame:
    """The two measures that come straight from the word table, one row per trial.

    For each area in `regions`:

    - `RT_pure_*` / `RT_normalized_*`: from the start of the first look at the
      area to the end of the last look (the "span"). In the normal run these
      columns are renamed to `TimeSinceOffset_*` by `build_screen_rt_tfd`.
    - `TFD_pure_*` / `TFD_normalized_*`: the area's looking time, added up.

    `_normalized` divides by the number of words in the area. An area never
    looked at gets 0.
    """
    regions = list(regions)
    first_time, last_time = C.IA_FIRST_FIXATION_TIME, C.IA_LAST_FIXATION_TIME
    last_duration, dwell = C.IA_LAST_FIXATION_DURATION, C.IA_DWELL_TIME
    rt_cols = [first_time, last_time, last_duration]
    group_keys = KEYS + [area_col]

    df = data[group_keys + rt_cols + [dwell]].copy()
    for c in rt_cols + [dwell]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    fix = df.dropna(subset=[first_time, last_time])

    agg = (
        fix.groupby(group_keys)
        .agg(
            min_first=(first_time, "min"),
            max_last=(last_time, "max"),
        )
        .reset_index()
    )

    last_idx = fix.loc[fix.groupby(group_keys)[last_time].idxmax()]
    last_dur = last_idx[group_keys + [last_duration]].rename(
        columns={last_duration: "last_fix_duration"}
    )
    agg = agg.merge(last_dur, on=group_keys)
    agg["RT"] = agg["max_last"] - agg["min_first"] + agg["last_fix_duration"]

    tfd = df.groupby(group_keys)[dwell].sum().reset_index(name="TFD")
    word_counts = data.groupby(group_keys).size().reset_index(name="n_words")

    rt_pure = agg.pivot_table(
        index=KEYS,
        columns=area_col,
        values="RT",
    ).reset_index()
    tfd_pure = tfd.pivot_table(
        index=KEYS,
        columns=area_col,
        values="TFD",
    ).reset_index()
    n_words = word_counts.pivot_table(
        index=KEYS,
        columns=area_col,
        values="n_words",
    ).reset_index()

    all_pairs = data[KEYS].drop_duplicates()
    rt_pure = all_pairs.merge(rt_pure, on=KEYS, how="left")
    tfd_pure = all_pairs.merge(tfd_pure, on=KEYS, how="left")
    n_words = all_pairs.merge(n_words, on=KEYS, how="left")

    for wide in (rt_pure, tfd_pure, n_words):
        for col in regions:
            if col not in wide.columns:
                wide[col] = 0
        wide[regions] = wide[regions].fillna(0)

    out = all_pairs.copy()
    for col in regions:
        denom = n_words[col].replace(0, pd.NA).values
        out[f"{C.RT_PURE}_{col}"] = rt_pure[col].values
        out[f"{C.RT_NORMALIZED}_{col}"] = (
            pd.Series(rt_pure[col].values / denom).fillna(0).values
        )
        out[f"{C.TFD_PURE}_{col}"] = tfd_pure[col].values
        out[f"{C.TFD_NORMALIZED}_{col}"] = (
            pd.Series(tfd_pure[col].values / denom).fillna(0).values
        )
    return out


def compute_run_based_rt_from_fixations(
    fixations: pd.DataFrame,
    ia_data: pd.DataFrame,
    area_col: str,
    regions: Sequence[str],
    verbose: bool = True,
) -> pd.DataFrame:
    """The RT_* measure: time actually spent looking at each area.

    The trial's fixations are put in time order and cut into visits -- runs of
    fixations in a row on the same area. Each visit counts

        start of the next fixation in the trial - start of the visit's first fixation

    and an area's RT is the sum over its visits. So time spent on another area
    between two visits counts for that other area, not this one. The trial's
    last visit has no next fixation, so it ends at the end of its own last
    fixation.

    (`compute_reading_times` differs: it takes one span from an area's first
    look to its last, so time spent elsewhere in between counts too.)

    Parameters
    ----------
    fixations : the screen's fixation table (`fixations.load_fixations`).
    ia_data : the word table for the same screen, used to find each word's
        area and to count the words in each area.
    area_col, regions : the column that names a word's area, and the areas to
        report (a `config.screens.Screen`'s `area_col` and `regions`).

    Returns
    -------
    One row per trial, with `RT_pure_<area>` and `RT_normalized_<area>`.
    """
    regions = list(regions)
    keys = KEYS
    pid, tid = C.PARTICIPANT_ID, C.TRIAL_ID
    start, duration = C.CURRENT_FIX_START, C.CURRENT_FIX_DURATION

    cols = keys + [start, duration, C.CURRENT_FIX_INTEREST_AREA_ID, C.NEAREST_IA]
    fix = place_fixations(
        fixations[cols],
        ia_data,
        [area_col],
        label=f"run-based RT ({area_col})",
        verbose=verbose,
    )

    # Put each trial's fixations in time order, then cut them into visits.
    fix = fix.sort_values(keys + [start], kind="mergesort").reset_index(drop=True)
    same_trial = fix[pid].eq(fix[pid].shift()) & fix[tid].eq(fix[tid].shift())
    # A fixation with no area never matches its neighbour, so it is its own
    # "visit" -- it separates visits but is not counted for any area.
    same_area = fix[area_col].eq(fix[area_col].shift()) & fix[area_col].notna()
    fix["_run"] = (~(same_trial & same_area)).cumsum()

    # When the next fixation in the same trial starts; NaN for the trial's last.
    next_start = fix[start].shift(-1)
    next_same_trial = fix[pid].eq(fix[pid].shift(-1)) & fix[tid].eq(fix[tid].shift(-1))
    fix["_next_start"] = next_start.where(next_same_trial)

    runs = fix.groupby("_run", sort=False).agg(
        **{pid: (pid, "first"), tid: (tid, "first")},
        area=(area_col, "first"),
        first_start=(start, "first"),
    )
    # Take each visit's real last row with `.tail(1)`, NOT `.agg("last")`.
    # `.agg("last")` skips NaN, so for the trial's last visit it would pick up an
    # earlier fixation's "next start" instead of the NaN meaning "nothing
    # follows" -- and the visit would lose its last fixation's duration.
    tail = fix.groupby("_run", sort=False).tail(1).set_index("_run")
    runs["last_start"] = tail[start]
    runs["last_duration"] = tail[duration]
    runs["next_start"] = tail["_next_start"]
    runs = runs[runs["area"].isin(regions)]

    end = runs["next_start"].fillna(runs["last_start"] + runs["last_duration"])
    runs["rt"] = end - runs["first_start"]

    rt_wide = (
        runs.groupby(keys + ["area"], observed=True)["rt"]
        .sum()
        .unstack("area")
    )

    n_words = (
        ia_data.groupby(keys + [area_col]).size().unstack(area_col)
    )

    all_pairs = ia_data[keys].drop_duplicates().reset_index(drop=True)
    out = all_pairs.copy()
    for r in regions:
        rt_vals = (
            all_pairs.join(rt_wide[r], on=keys)[r].values
            if r in rt_wide.columns
            else pd.Series(0.0, index=all_pairs.index).values
        )
        nw = (
            all_pairs.join(n_words[r], on=keys)[r].values
            if r in n_words.columns
            else pd.Series(0.0, index=all_pairs.index).values
        )
        rt_vals = pd.Series(rt_vals, dtype="float64").fillna(0.0)
        denom = pd.Series(nw, dtype="float64").replace(0, pd.NA)
        out[f"{C.RT_PURE}_{r}"] = rt_vals.values
        out[f"{C.RT_NORMALIZED}_{r}"] = (rt_vals / denom).fillna(0).values
    return out



def rename_rt_to_time_since_offset(df: pd.DataFrame) -> pd.DataFrame:
    """Rename the span measure's `RT_*` columns to `TimeSinceOffset_*`.

    `compute_reading_times` names its first-look-to-last-look span `RT_*`.
    That span counts time spent elsewhere in between, so it is not really
    reading time. It is kept, under the name `TimeSinceOffset_*`, and the
    `RT_*` names go to the visit-based measure.
    """
    pairs = (
        (C.RT_PURE, C.TIME_SINCE_OFFSET_PURE),
        (C.RT_NORMALIZED, C.TIME_SINCE_OFFSET_NORMALIZED),
    )
    rename = {
        c: new + c[len(old):]
        for c in df.columns
        for old, new in pairs
        if c.startswith(f"{old}_")
    }
    return df.rename(columns=rename)


def build_screen_rt_tfd(
    ia_data: pd.DataFrame,
    scr,
    fixations: pd.DataFrame | None = None,
    include_run_based_rt: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    """All three time measures for one screen, one row per trial.

    `scr` is a `config.screens.Screen`, which says which column names the areas
    and what the areas are.

      * `TFD_*` and `TimeSinceOffset_*` come from the word table (`ia_data`);
      * `RT_*` comes from `fixations`, the screen's fixation table
        (`fixations.load_fixations`).

    With `include_run_based_rt=False` no fixations are needed, and the columns
    named `RT_*` hold the span measure instead (there are then no
    `TimeSinceOffset_*` columns).
    """
    span_based = compute_reading_times(ia_data, area_col=scr.area_col, regions=scr.regions)
    if not include_run_based_rt:
        return span_based

    out = rename_rt_to_time_since_offset(span_based)

    if verbose:
        print(f"Computing {scr.label} RT (run-based)...")
    run_rt = compute_run_based_rt_from_fixations(
        fixations, ia_data, area_col=scr.area_col, regions=scr.regions, verbose=verbose
    )
    merged = out.merge(run_rt, on=list(TRIAL_ID_COLS), how="left")
    assert len(merged) == len(out), (
        f"{scr.label} run-based RT join changed the row count: "
        f"{len(out)} -> {len(merged)}"
    )
    return merged

def build_rt_and_tfd(
    all_participants: pd.DataFrame,
    fixations: pd.DataFrame,
    output_path: Path = RT_AND_TFD_PATH,
    include_run_based_rt: bool = True,
    save: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    """The three time measures for the answer screen, optionally saved.

    `all_participants` is the answer screen's word table and `fixations` its
    fixation table (`fixations.load_fixations`). `save=False` returns the table
    without writing it to `output_path`.

    Answer screen only; the paragraph screen's times are built by
    `features/paragraph/spans.py`, using the same functions.
    """
    if verbose:
        print("Computing answer-region RT/TFD...")
    answer = build_screen_rt_tfd(
        all_participants,
        screen("answers"),
        fixations=fixations,
        include_run_based_rt=include_run_based_rt,
        verbose=verbose,
    )

    rt_and_tfd = answer

    if save:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        rt_and_tfd.to_csv(output_path, index=False)
        if verbose:
            print(f"Saved {len(rt_and_tfd)} rows to {output_path}")
    return rt_and_tfd
