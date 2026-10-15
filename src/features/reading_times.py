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

import ast
from typing import Sequence

import pandas as pd

from src.config.datasets import (
    FIX_ANSWERS_PATH,
    FIX_PARAGRAPH_PATH,
    GATHERERS_PROCESSED_PATH,
    HUNTERS_PROCESSED_PATH,
    RT_AND_TFD_PATH,
)
from src.checks import assert_full_coverage
from src.config.columns import TRIAL_ID_COLS
from src.config.screens import screen


ANSWER_AREA_COL = "area_label"
ANSWER_REGIONS = ("question", "answer_A", "answer_B", "answer_C", "answer_D")

PARAGRAPH_AREA_COL = "auxiliary_span_type"
PARAGRAPH_REGIONS = ("outside", "distractor", "critical")

IA_ID_COL = "IA_ID"

# Columns of a fixation report (one row per fixation) that RT needs.
FIX_START_COL = "CURRENT_FIX_START"
FIX_DURATION_COL = "CURRENT_FIX_DURATION"
FIX_IA_ID_COL = "CURRENT_FIX_INTEREST_AREA_ID"
# The word a fixation landed NEAR, when it landed on none. When the tracker
# fills both this and the column above, they always agree (checked over every
# fixation in both Study 1 reports), so "the word it is on, else the nearest
# one" gives the same answer as "the nearest one" wherever both exist.
FIX_NEAREST_IA_COL = "CURRENT_FIX_NEAREST_INTEREST_AREA"


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
    rt_cols = [
        "IA_FIRST_FIXATION_TIME",
        "IA_LAST_FIXATION_TIME",
        "IA_LAST_FIXATION_DURATION",
    ]
    group_keys = ["participant_id", "TRIAL_INDEX", area_col]

    df = data[group_keys + rt_cols + ["IA_DWELL_TIME"]].copy()
    for c in rt_cols + ["IA_DWELL_TIME"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    fix = df.dropna(subset=["IA_FIRST_FIXATION_TIME", "IA_LAST_FIXATION_TIME"])

    agg = (
        fix.groupby(group_keys)
        .agg(
            min_first=("IA_FIRST_FIXATION_TIME", "min"),
            max_last=("IA_LAST_FIXATION_TIME", "max"),
        )
        .reset_index()
    )

    last_idx = fix.loc[fix.groupby(group_keys)["IA_LAST_FIXATION_TIME"].idxmax()]
    last_dur = last_idx[group_keys + ["IA_LAST_FIXATION_DURATION"]].rename(
        columns={"IA_LAST_FIXATION_DURATION": "last_fix_duration"}
    )
    agg = agg.merge(last_dur, on=group_keys)
    agg["RT"] = agg["max_last"] - agg["min_first"] + agg["last_fix_duration"]

    tfd = df.groupby(group_keys)["IA_DWELL_TIME"].sum().reset_index(name="TFD")
    word_counts = data.groupby(group_keys).size().reset_index(name="n_words")

    rt_pure = agg.pivot_table(
        index=["participant_id", "TRIAL_INDEX"],
        columns=area_col,
        values="RT",
    ).reset_index()
    tfd_pure = tfd.pivot_table(
        index=["participant_id", "TRIAL_INDEX"],
        columns=area_col,
        values="TFD",
    ).reset_index()
    n_words = word_counts.pivot_table(
        index=["participant_id", "TRIAL_INDEX"],
        columns=area_col,
        values="n_words",
    ).reset_index()

    all_pairs = data[["participant_id", "TRIAL_INDEX"]].drop_duplicates()
    rt_pure = all_pairs.merge(rt_pure, on=["participant_id", "TRIAL_INDEX"], how="left")
    tfd_pure = all_pairs.merge(
        tfd_pure, on=["participant_id", "TRIAL_INDEX"], how="left"
    )
    n_words = all_pairs.merge(n_words, on=["participant_id", "TRIAL_INDEX"], how="left")

    for wide in (rt_pure, tfd_pure, n_words):
        for col in regions:
            if col not in wide.columns:
                wide[col] = 0
        wide[regions] = wide[regions].fillna(0)

    out = all_pairs.copy()
    for col in regions:
        denom = n_words[col].replace(0, pd.NA).values
        out[f"RT_pure_{col}"] = rt_pure[col].values
        out[f"RT_normalized_{col}"] = (
            pd.Series(rt_pure[col].values / denom).fillna(0).values
        )
        out[f"TFD_pure_{col}"] = tfd_pure[col].values
        out[f"TFD_normalized_{col}"] = (
            pd.Series(tfd_pure[col].values / denom).fillna(0).values
        )
    return out


def load_fixations(
    fixations_path: Path,
    participants: Sequence[str] | None = None,
    chunksize: int = 2_000_000,
    trial_index_dtype: str | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """Read the six columns of a fixation report that reading times need.

    Works for either screen. The reports are several GB, so they are read in
    chunks. `participants` keeps only those participants, for a quick check.
    Rows without a fixation start time are dropped.

    `TRIAL_INDEX` is kept as text unless `trial_index_dtype` asks for a number.
    That matters: Study 2's TRIAL_INDEX is text (`b2l01t005`), and converting it
    to a number would drop every row.
    """
    numeric = [FIX_START_COL, FIX_DURATION_COL, FIX_IA_ID_COL, FIX_NEAREST_IA_COL]
    header = pd.read_csv(fixations_path, nrows=0).columns
    present = [c for c in numeric if c in header]
    usecols = ["participant_id", "TRIAL_INDEX"] + present

    parts = []
    # Read everything as text and convert afterwards: the report writes "." for
    # a missing value, so letting pandas guess the types would give different
    # types in different chunks.
    reader = pd.read_csv(
        fixations_path, usecols=usecols, chunksize=chunksize, dtype=str
    )
    for chunk in reader:
        if participants is not None:
            chunk = chunk[chunk["participant_id"].isin(participants)]
        if not len(chunk):
            continue
        for c in present:
            chunk[c] = pd.to_numeric(chunk[c], errors="coerce")
        parts.append(chunk.dropna(subset=[FIX_START_COL]))

    if not parts:
        return pd.DataFrame(columns=usecols)

    out = pd.concat(parts, ignore_index=True)
    out = out[out["TRIAL_INDEX"].notna()]
    if trial_index_dtype is not None:
        out["TRIAL_INDEX"] = pd.to_numeric(out["TRIAL_INDEX"], errors="coerce")
        out = out.dropna(subset=["TRIAL_INDEX"])
        out["TRIAL_INDEX"] = out["TRIAL_INDEX"].astype(trial_index_dtype)
    if verbose:
        print(f"  read {len(out):,} fixations from {Path(fixations_path).name}")
    return out


def load_paragraph_fixations(
    fixations_path: Path = FIX_PARAGRAPH_PATH,
    participants: Sequence[str] | None = None,
    chunksize: int = 2_000_000,
    verbose: bool = True,
) -> pd.DataFrame:
    """`load_fixations` for the paragraph report, with TRIAL_INDEX as a whole number.

    Only Study 1 has a paragraph report, and its TRIAL_INDEX is a number, so
    converting is safe here. Study 2 data must never come through this: its
    TRIAL_INDEX is text, and converting it would drop every row.
    """
    return load_fixations(
        fixations_path,
        participants=participants,
        chunksize=chunksize,
        trial_index_dtype="int64",
        verbose=verbose,
    )


def resolve_fixation_area(
    fixations: pd.DataFrame,
    ia_data: pd.DataFrame,
    area_col: str,
    *,
    ia_id_col: str = IA_ID_COL,
    fix_ia_id_col: str = FIX_IA_ID_COL,
    fix_nearest_ia_col: str = FIX_NEAREST_IA_COL,
    label: str = "",
    verbose: bool = True,
) -> pd.DataFrame:
    """Give every fixation the area it belongs to (`area_col`).

    The rule: the word the fixation landed on; if it landed on none, the word
    it landed nearest. Both screens, and every measure that needs a fixation's
    area, use this one function, so fixations are placed the same way
    everywhere.

    A fixation that cannot be placed -- no word and no nearest word -- gets no
    area. It still separates one visit from the next, but adds no time to any
    area, because we cannot say where the reader was looking. Such fixations
    are counted and printed.

    Stops with an error if a fixation in an analysed trial points at a word
    that trial does not have: that means the fixation report and the word table
    do not match, and the area would wrongly read as not looked at.

    Returns `fixations` with the `area_col` column added.
    """
    keys = ["participant_id", "TRIAL_INDEX"]
    fix = fixations.copy()

    exact = pd.to_numeric(fix[fix_ia_id_col], errors="coerce")
    nearest = (
        pd.to_numeric(fix[fix_nearest_ia_col], errors="coerce")
        if fix_nearest_ia_col in fix.columns
        else pd.Series(pd.NA, index=fix.index, dtype="float64")
    )
    resolved = exact.fillna(nearest).astype("float64")
    n_filled = int(exact.isna().sum() - resolved.isna().sum())

    labels = (
        ia_data[keys + [ia_id_col, area_col]]
        .dropna(subset=[ia_id_col, area_col])
        .drop_duplicates(subset=keys + [ia_id_col])
        .copy()
    )
    # Word ids as decimals on both sides: the fixation report has NaN for
    # fixations on no word, so its ids are decimals, and whole numbers on the
    # other side would match nothing.
    labels[ia_id_col] = pd.to_numeric(labels[ia_id_col], errors="coerce").astype("float64")

    # The trial ids must be the same type on both sides, or the join would match
    # nothing and every area would read as not looked at. So the fixation
    # report's are converted to the word table's type. (Study 2's TRIAL_INDEX
    # is text.) The check further down still stops the run if anything fails to
    # match.
    for k in keys:
        if fix[k].dtype != labels[k].dtype:
            fix[k] = fix[k].astype(labels[k].dtype)

    fix["_ia"] = resolved
    fix["_no_ia"] = resolved.isna().to_numpy()
    merged = fix.merge(
        labels.rename(columns={ia_id_col: "_ia"}), on=keys + ["_ia"], how="left"
    )
    assert len(merged) == len(fix), (
        f"fixation -> {area_col} join duplicated rows: {len(fix)} -> {len(merged)}"
    )
    # Two very different reasons a fixation can fail to get an area:
    #   - its trial is not in the word table at all. Normal: the fixation report
    #     also holds practice and excluded trials (about 4,600 on Study 1).
    #   - its trial IS analysed, but its word id is not one of that trial's
    #     words. That would mean the two tables do not match, so it stops the run.
    analysed = set(map(tuple, labels[keys].drop_duplicates().itertuples(index=False, name=None)))
    in_analysed = merged.set_index(keys).index.isin(analysed)
    unresolved = merged[area_col].isna().to_numpy()
    no_ia = merged["_no_ia"].to_numpy()
    n_outside = int((~in_analysed).sum())
    # Counted only inside analysed trials, so the two counts do not overlap.
    n_neither = int((no_ia & in_analysed).sum())
    n_unmatched = int((unresolved & in_analysed & ~no_ia).sum())

    if verbose:
        tag = f"{label} " if label else ""
        print(
            f"  {tag}fixation->area: {len(merged):,} fixations | "
            f"{n_filled:,} placed by the nearest-IA fallback | "
            f"{n_outside:,} in trials outside the analysis set | "
            f"{n_neither:,} in the analysis set with no interest area at all"
        )
    if n_unmatched:
        raise ValueError(
            f"{label or 'fixation->area'}: {n_unmatched:,} fixation(s) in trials the IA "
            f"table DOES contain carry an interest-area id that is not among that "
            f"trial's interest areas. Every such fixation would accrue no reading time "
            f"and the region would be written as a real zero. This is a key or dtype "
            f"mismatch between the fixation report and the IA table, not missing data."
        )
    return merged.drop(columns=["_ia", "_no_ia"])


def compute_run_based_rt_from_fixations(
    fixations: pd.DataFrame,
    ia_data: pd.DataFrame,
    area_col: str = PARAGRAPH_AREA_COL,
    regions: Sequence[str] = PARAGRAPH_REGIONS,
    fix_start_col: str = FIX_START_COL,
    fix_duration_col: str = FIX_DURATION_COL,
    fix_ia_id_col: str = FIX_IA_ID_COL,
    fix_nearest_ia_col: str = FIX_NEAREST_IA_COL,
    ia_id_col: str = IA_ID_COL,
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
    fixations : a fixation report for either screen, with at least
        participant_id, TRIAL_INDEX and the fixation start, duration and word
        columns.
    ia_data : the word table for the same screen, used to find each word's
        area and to count the words in each area.

    Returns
    -------
    One row per trial, with `RT_pure_<area>` and `RT_normalized_<area>`.
    """
    regions = list(regions)
    keys = ["participant_id", "TRIAL_INDEX"]

    cols = keys + [fix_start_col, fix_duration_col, fix_ia_id_col]
    if fix_nearest_ia_col in fixations.columns:
        cols = cols + [fix_nearest_ia_col]
    fix = fixations[cols].copy()
    for c in (fix_start_col, fix_duration_col):
        fix[c] = pd.to_numeric(fix[c], errors="coerce")
    fix = fix.dropna(subset=[fix_start_col])

    # Place each fixation in an area, by the same rule used everywhere.
    fix = resolve_fixation_area(
        fix,
        ia_data,
        area_col,
        ia_id_col=ia_id_col,
        fix_ia_id_col=fix_ia_id_col,
        fix_nearest_ia_col=fix_nearest_ia_col,
        label=f"run-based RT ({area_col})",
        verbose=verbose,
    )

    # Put each trial's fixations in time order, then cut them into visits.
    fix = fix.sort_values(keys + [fix_start_col], kind="mergesort").reset_index(drop=True)
    same_trial = (
        fix["participant_id"].eq(fix["participant_id"].shift())
        & fix["TRIAL_INDEX"].eq(fix["TRIAL_INDEX"].shift())
    )
    # A fixation with no area never matches its neighbour, so it is its own
    # "visit" -- it separates visits but is not counted for any area.
    same_area = fix[area_col].eq(fix[area_col].shift()) & fix[area_col].notna()
    fix["_run"] = (~(same_trial & same_area)).cumsum()

    # When the next fixation in the same trial starts; NaN for the trial's last.
    next_start = fix[fix_start_col].shift(-1)
    next_same_trial = (
        fix["participant_id"].eq(fix["participant_id"].shift(-1))
        & fix["TRIAL_INDEX"].eq(fix["TRIAL_INDEX"].shift(-1))
    )
    fix["_next_start"] = next_start.where(next_same_trial)

    runs = fix.groupby("_run", sort=False).agg(
        participant_id=("participant_id", "first"),
        TRIAL_INDEX=("TRIAL_INDEX", "first"),
        area=(area_col, "first"),
        first_start=(fix_start_col, "first"),
    )
    # Take each visit's real last row with `.tail(1)`, NOT `.agg("last")`.
    # `.agg("last")` skips NaN, so for the trial's last visit it would pick up an
    # earlier fixation's "next start" instead of the NaN meaning "nothing
    # follows" -- and the visit would lose its last fixation's duration.
    tail = fix.groupby("_run", sort=False).tail(1).set_index("_run")
    runs["last_start"] = tail[fix_start_col]
    runs["last_duration"] = tail[fix_duration_col]
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
        out[f"RT_pure_{r}"] = rt_vals.values
        out[f"RT_normalized_{r}"] = (rt_vals / denom).fillna(0).values
    return out



def rename_rt_to_time_since_offset(df: pd.DataFrame) -> pd.DataFrame:
    """Rename the span measure's `RT_*` columns to `TimeSinceOffset_*`.

    `compute_reading_times` names its first-look-to-last-look span `RT_*`.
    That span counts time spent elsewhere in between, so it is not really
    reading time. It is kept, under the name `TimeSinceOffset_*`, and the
    `RT_*` names go to the visit-based measure.
    """
    rename = {
        c: c.replace("RT_pure_", "TimeSinceOffset_pure_", 1).replace(
            "RT_normalized_", "TimeSinceOffset_normalized_", 1
        )
        for c in df.columns
        if c.startswith("RT_pure_") or c.startswith("RT_normalized_")
    }
    return df.rename(columns=rename)


def build_screen_rt_tfd(
    ia_data: pd.DataFrame,
    scr,
    fixations: pd.DataFrame | None = None,
    fixations_path=None,
    include_run_based_rt: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    """All three time measures for one screen, one row per trial.

    `scr` is a `config.screens.Screen`, which says which column names the areas
    and what the areas are.

      * `TFD_*` and `TimeSinceOffset_*` come from the word table (`ia_data`);
      * `RT_*` comes from the fixation report.

    Pass `fixations` if the report is already loaded, or `fixations_path` to
    read it here.

    With `include_run_based_rt=False` the fixation report is not used, and the
    columns named `RT_*` hold the span measure instead (there are then no
    `TimeSinceOffset_*` columns).
    """
    span_based = compute_reading_times(ia_data, area_col=scr.area_col, regions=scr.regions)
    if not include_run_based_rt:
        return span_based

    out = rename_rt_to_time_since_offset(span_based)

    if verbose:
        print(f"Computing {scr.label} RT (run-based)...")
    if fixations is None:
        fixations = load_fixations(fixations_path, verbose=verbose)
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
    all_participants: pd.DataFrame | None = None,
    hunters_path: Path = HUNTERS_PROCESSED_PATH,
    gatherers_path: Path = GATHERERS_PROCESSED_PATH,
    fixations_path: Path = FIX_ANSWERS_PATH,
    output_path: Path = RT_AND_TFD_PATH,
    include_run_based_rt: bool = True,
    save: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    """The three time measures for the answer screen, optionally saved.

    `all_participants` is the answer screen's word table. If it is not given,
    it is read from `hunters_path` and `gatherers_path` -- but those files are
    only written on request (see `config/datasets.py`), so normally pass the
    table in. `fixations_path` is the answer screen's fixation report.

    `save=False` returns the table without writing it to `output_path`.

    Answer screen only; the paragraph screen's times are built by
    `features/paragraph/spans.py`, using the same functions.
    """
    if all_participants is None:
        if verbose:
            print("Loading processed answer IA data...")
        hunters = pd.read_csv(hunters_path)
        gatherers = pd.read_csv(gatherers_path)
        all_participants = pd.concat([hunters, gatherers], ignore_index=True)

    if verbose:
        print("Computing answer-region RT/TFD...")
    answer = build_screen_rt_tfd(
        all_participants,
        screen("answers"),
        fixations_path=fixations_path,
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


if __name__ == "__main__":
    build_rt_and_tfd()
