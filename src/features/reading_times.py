# src/features/reading_times.py
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

# Columns of the paragraph fixation report needed for the run-based RT.
FIX_START_COL = "CURRENT_FIX_START"
FIX_DURATION_COL = "CURRENT_FIX_DURATION"
FIX_IA_ID_COL = "CURRENT_FIX_INTEREST_AREA_ID"
# The interest area a fixation landed NEAR, when it landed on none. EyeLink sets
# both columns, and they never disagree when both are present -- measured
# 2026-10-06 over both full reports, 0 disagreements in 718,207 answer and
# 2,400,788 paragraph fixations. So the coalesce below is a generalisation of
# "use nearest", not a different rule.
FIX_NEAREST_IA_COL = "CURRENT_FIX_NEAREST_INTEREST_AREA"


def compute_reading_times(
    data: pd.DataFrame,
    area_col: str,
    regions: Sequence[str],
) -> pd.DataFrame:
    """Per (participant_id, TRIAL_INDEX), for each region in `regions`:
    RT_pure / RT_normalized — based on first/last IA fixation timestamps.
    TFD_pure / TFD_normalized — sum of IA_DWELL_TIME, raw and divided by n_words.
    Normalization is by number of IAs (rows) in that participant-trial-area.
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


# `compute_run_based_rt` and `_parse_fixation_pairs` were deleted on 2026-10-06.
# They computed run-based RT from the (timestamp, nearest-IA) list that
# `ingest/clicks.py` copied into the button-click table. `compute_run_based_rt_from_fixations`
# below is now the only run-based RT, for both screens, reading the fixation
# report directly -- see `docs/decisions/2026-10-06-stage-d-proposal.md` section 3.6.


def load_fixations(
    fixations_path: Path,
    participants: Sequence[str] | None = None,
    chunksize: int = 2_000_000,
    trial_index_dtype: str | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """Read only the columns of a fixation report that RT and visit counts need.

    Either screen's report, which is the point: since 2026-10-06 the answer
    screen reads its own fixation report here instead of a per-trial sequence
    that had been copied into the button-click table.

    The reports are several GB wide, so this keeps six columns and streams in
    chunks. `participants` restricts to a subset, for a quick check.

    `TRIAL_INDEX` is left as text unless `trial_index_dtype` says otherwise.
    That matters: KnowQA's TRIAL_INDEX is a composite string (`b2l01t005`) and
    coercing it to int64 would drop every row (`docs/pitfalls.md` section 5).
    """
    numeric = [FIX_START_COL, FIX_DURATION_COL, FIX_IA_ID_COL, FIX_NEAREST_IA_COL]
    header = pd.read_csv(fixations_path, nrows=0).columns
    present = [c for c in numeric if c in header]
    usecols = ["participant_id", "TRIAL_INDEX"] + present

    parts = []
    # Read everything as text: the reports use "." for a missing interest area,
    # so dtype inference differs between chunks and pandas errors on the mixed
    # -type warning path. Coerce once, here.
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
    """`load_fixations` for the paragraph report, pinning TRIAL_INDEX to int64.

    Only the two Study-1 paragraph runs have a paragraph report at all, and their
    TRIAL_INDEX is an integer, so the cast is safe here and nowhere else -- see
    `docs/pitfalls.md` section 5 for why KnowQA must never reach this.
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
    """Attach `area_col` to every fixation, on both screens, by one rule.

    **The rule (Diana, 2026-10-06): take the interest area the fixation landed
    ON; if there is none, take the one it landed NEAREST.** Applied here once,
    so every consumer -- run-based RT, visit counts, either screen -- resolves
    fixations the same way. Before this the answer screen always used `nearest`
    and the paragraph screen always used the exact id, which silently dropped
    1.3% of its fixations from RT while its own visit counts kept them.

    The coalesce is not a third behaviour: the two columns never disagree when
    both are present, so this equals "nearest" wherever nearest was already used.

    A fixation with neither column, or whose id matches no interest area in
    `ia_data` for that trial, is left unmapped -- it breaks a run without
    accruing time, which is correct, since we cannot say where the reader was
    looking. **Those are counted and reported, never silently dropped.**

    Returns `fixations` with one added column, `area_col`.
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
    # Both sides of the id join cast to float: the fixation report carries NaN
    # for off-area fixations, so an int/float mismatch would join to nothing.
    labels[ia_id_col] = pd.to_numeric(labels[ia_id_col], errors="coerce").astype("float64")

    # The trial key must agree in dtype or the merge yields nothing and every
    # region comes back 0 -- a plausible-looking wrong number, so it is asserted
    # rather than trusted. KnowQA's TRIAL_INDEX is a composite string.
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
    # Two very different reasons an id fails to resolve, and conflating them
    # hides the one that matters. A fixation in a trial the IA table does not
    # contain is routine -- the report carries practice and excluded trials that
    # the analysis set drops. A fixation in a trial we DO analyse whose id is not
    # among that trial's interest areas is a key mismatch, and would mean a
    # region silently accruing no time. Measured on L1 2026-10-06: 143,123 of
    # the first kind (the 4,610 report-only trials), ZERO of the second.
    analysed = set(map(tuple, labels[keys].drop_duplicates().itertuples(index=False, name=None)))
    in_analysed = merged.set_index(keys).index.isin(analysed)
    unresolved = merged[area_col].isna().to_numpy()
    no_ia = merged["_no_ia"].to_numpy()
    n_outside = int((~in_analysed).sum())
    # Counted WITHIN the analysis set, so the two numbers do not overlap.
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
    """Run-based per-area RT from a fixation report. **The only run-based RT.**

    Same definition as the answer-region RT: the trial's fixations are ordered,
    split into runs of consecutive fixations on the same area, and each run
    contributes

        run_RT = next_fix_start_in_trial - first_fix_start_in_run

    so a region only accrues time while it is actually being looked at. Time
    spent elsewhere between two visits belongs to the other region, not to this
    one. The trial's final run has no following fixation, so it ends at
    `last_fix_start + last_fix_duration`.

    This differs from `compute_reading_times`, which takes one span from the
    region's first to its last fixation and therefore counts every excursion
    away and back as part of the region's reading time.

    The report carries each fixation's own start and duration, so the final run
    of a trial ends at `last_start + last_duration` -- no matching back into the
    IA-level table, and no case where a fixation silently contributes 0 ms.

    Parameters
    ----------
    fixations : paragraph fixation report, at least `participant_id`,
        `TRIAL_INDEX`, and the three fixation columns.
    ia_data : paragraph IA report, used to map `IA_ID` -> `area_col` and to
        count words per area.

    Returns
    -------
    DataFrame with one row per (participant_id, TRIAL_INDEX) and
    `RT_pure_{region}` / `RT_normalized_{region}` columns.
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

    # One rule for placing a fixation, shared with visit counts and both screens.
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

    # Order within trial, then cut into runs of consecutive same-area fixations.
    fix = fix.sort_values(keys + [fix_start_col], kind="mergesort").reset_index(drop=True)
    same_trial = (
        fix["participant_id"].eq(fix["participant_id"].shift())
        & fix["TRIAL_INDEX"].eq(fix["TRIAL_INDEX"].shift())
    )
    # NaN labels (fixations off any interest area) never compare equal, so each
    # one becomes its own run -- they break runs without accruing time.
    same_area = fix[area_col].eq(fix[area_col].shift()) & fix[area_col].notna()
    fix["_run"] = (~(same_trial & same_area)).cumsum()

    # The next fixation's start, within the same trial; NaN on the trial's last.
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
    # The run's genuinely last row, NOT `.agg("last")`. **`.agg("last")` skips
    # NaN**, so for the trial's final run it would resurrect the *previous*
    # fixation's `_next_start` instead of the NaN that marks "nothing follows" --
    # and the run would then end at its own last fixation's START, losing that
    # fixation's duration. Fixed 2026-10-06; before that every trial's final run
    # was truncated this way, which is where the paragraph RT family's missing
    # time was going. Measured on L1: the final run is on average 143 ms longer
    # once the trailing fixation is counted.
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
    """Rename the span-based `RT_*` columns to `TimeSinceOffset_*`.

    Both screens do this, for the same reason: `compute_reading_times` measures
    an area from its first to its last fixation, counting every excursion away
    and back. That is a real quantity and worth keeping, but it is not reading
    time, so it is renamed out of the way and the run-based RT takes the `RT_*`
    names. **One implementation since 2026-10-06** -- it was the same dict
    comprehension, character for character, in `build_rt_and_tfd` and in
    `spans.build_paragraph_rt_tfd`.
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
    """Per-area RT / TFD / TimeSinceOffset for either screen, one row per trial.

    **The assembly both screens used to keep their own copy of** (stage D step
    2b). `scr` is a `config.screens.Screen`; it supplies the grouping column and
    the regions, and nothing else differs:

      * `TimeSinceOffset_*` -- first-to-last-fixation span, from `ia_data`;
      * `TFD_*` -- summed dwell, from `ia_data`;
      * `RT_*` -- run-based, from the fixation report, so a region only accrues
        time while it is actually being looked at.

    Pass `fixations` when the report is already in memory, or `fixations_path`
    to have it read here.
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
    """Build RT/TFD features and (optionally) save to CSV.

    Answer regions (all processed IA data, by `area_label`):
      - TFD via per-area aggregation of IA dwell times.
      - RT: when `include_run_based_rt` is True, via run-based aggregation over
        the trial's fixation sequence (loaded from `fixations_path`), to
        avoid conflating non-consecutive visits — and the fixation-span RT is
        kept as `TimeSinceOffset_*`. When False (no button-click data), the
        fixation-span RT is used directly as `RT_*` and button clicks are not
        read.
    Paragraph regions are NOT built here. Since T6.1 (2026-09-23) the paragraph
    screen has its own pipeline, `features/paragraph/spans.py`, which produces the
    per-span RT / TFD / TimeSinceOffset columns using these same functions. This
    one is answer-only, so preparing the answer screen no longer opens a
    paragraph report.

    `all_participants` may be passed in-memory (the processed answer-IA
    DataFrame); if omitted, it is loaded and concatenated from
    `hunters_path` / `gatherers_path`. Set `save=False` to skip writing the CSV
    and only return the DataFrame.
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

    # The paragraph half of this function moved to
    # `features/paragraph/spans.build_paragraph_rt_tfd` on 2026-09-23 (`todo.md`
    # T6.1). It used to open the paragraph IA and fixation reports here and merge
    # per-span RT/TFD into the answer table with `how="inner"` -- so preparing the
    # answer screen depended on multi-GB paragraph reports, and a trial with
    # answer data but no paragraph data silently disappeared. Paragraph RT/TFD is
    # now built by the paragraph pipeline and joined where it is wanted.
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
