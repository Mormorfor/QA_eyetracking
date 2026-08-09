# src/derived/reading_times.py
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import ast
from typing import Sequence

import pandas as pd

from src.data_paths import (
    BUTTON_CLICKS_PATH,
    FIX_PARAGRAPH_PATH,
    GATHERERS_PROCESSED_PATH,
    HUNTERS_PROCESSED_PATH,
    IA_PARAGRAPH_PATH,
    RT_AND_TFD_PATH,
)


ANSWER_AREA_COL = "area_label"
ANSWER_REGIONS = ("question", "answer_A", "answer_B", "answer_C", "answer_D")

PARAGRAPH_AREA_COL = "auxiliary_span_type"
PARAGRAPH_REGIONS = ("outside", "distractor", "critical")

FIXATION_TIMESTAMPS_IA_COL = "FIXATION_TIMESTAMPS_IA"
IA_ID_COL = "IA_ID"

# Columns of the paragraph fixation report needed for the run-based RT.
FIX_START_COL = "CURRENT_FIX_START"
FIX_DURATION_COL = "CURRENT_FIX_DURATION"
FIX_IA_ID_COL = "CURRENT_FIX_INTEREST_AREA_ID"


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


def _parse_fixation_pairs(x):
    """Parse a FIXATION_TIMESTAMPS_IA cell into a list of (timestamp, IA_id) tuples.

    Accepts a real Python list, a stringified list, NaN, or None.
    """
    if x is None:
        return []
    if isinstance(x, list):
        return x
    if isinstance(x, float) and pd.isna(x):
        return []
    if isinstance(x, str):
        s = x.strip()
        if not s:
            return []
        try:
            parsed = ast.literal_eval(s)
        except (ValueError, SyntaxError):
            return []
        return parsed if isinstance(parsed, list) else []
    return []


def compute_run_based_rt(
    button_clicks_df: pd.DataFrame,
    ia_data: pd.DataFrame,
    area_col: str = ANSWER_AREA_COL,
    regions: Sequence[str] = ANSWER_REGIONS,
    fixation_col: str = FIXATION_TIMESTAMPS_IA_COL,
    ia_id_col: str = IA_ID_COL,
) -> pd.DataFrame:
    """Compute per-area RT_pure / RT_normalized from temporally-ordered fixations.

    For each (participant_id, TRIAL_INDEX), the trial's (timestamp, IA_id) sequence
    from `button_clicks_df[fixation_col]` is mapped to area_labels via `ia_data`,
    split into runs of consecutive same-area fixations, and per-run RT is summed
    per area:
        run_RT = next_fix_start_in_trial - first_fix_start_in_run
    For the trial's final run there is no following fixation, so we recover its
    last fixation's duration from the IA-level `IA_LAST_FIXATION_DURATION` of the
    IA the run ends on (matched by `IA_LAST_FIXATION_TIME == last_fix_start`):
        run_RT = (last_fix_start_in_run + last_fix_duration) - first_fix_start_in_run

    Normalization is by `n_words` (rows in `ia_data` for that participant-trial-area).
    """
    regions = list(regions)
    keys = ["participant_id", "TRIAL_INDEX"]

    label_lookup = (
        ia_data[keys + [ia_id_col, area_col]]
        .dropna(subset=[ia_id_col, area_col])
        .drop_duplicates(subset=keys + [ia_id_col])
        .copy()
    )

    label_lookup[ia_id_col] = pd.to_numeric(
        label_lookup[ia_id_col], errors="coerce"
    ).astype("Int64")

    lookup_by_trial: dict[tuple, dict[int, str]] = {
        (pid, tid): dict(zip(g[ia_id_col].tolist(), g[area_col].tolist()))
        for (pid, tid), g in label_lookup.groupby(keys, sort=False)
    }

    last_dur_src = ia_data[
        keys + [ia_id_col, "IA_LAST_FIXATION_TIME", "IA_LAST_FIXATION_DURATION"]
    ].copy()

    last_dur_src[ia_id_col] = pd.to_numeric(
        last_dur_src[ia_id_col], errors="coerce"
    ).astype("Int64")

    last_dur_src["IA_LAST_FIXATION_TIME"] = pd.to_numeric(
        last_dur_src["IA_LAST_FIXATION_TIME"], errors="coerce"
    )

    last_dur_src["IA_LAST_FIXATION_DURATION"] = pd.to_numeric(
        last_dur_src["IA_LAST_FIXATION_DURATION"], errors="coerce"
    )

    last_dur_src = last_dur_src.dropna(
        subset=[ia_id_col, "IA_LAST_FIXATION_TIME", "IA_LAST_FIXATION_DURATION"]
    )
    
    last_dur_by_trial: dict[tuple, dict[int, tuple[int, int]]] = {
        (pid, tid): {
            int(ia): (int(t), int(d))
            for ia, t, d in zip(
                g[ia_id_col].tolist(),
                g["IA_LAST_FIXATION_TIME"].tolist(),
                g["IA_LAST_FIXATION_DURATION"].tolist(),
            )
        }
        for (pid, tid), g in last_dur_src.groupby(keys, sort=False)
    }

    rt_rows = []
    for _, row in button_clicks_df.iterrows():
        pid, tid = row["participant_id"], row["TRIAL_INDEX"]
        pairs = _parse_fixation_pairs(row[fixation_col])
        if not pairs:
            continue
        ia_to_label = lookup_by_trial.get((pid, tid), {})
        last_dur_for_trial = last_dur_by_trial.get((pid, tid), {})

        timestamps: list[int] = []
        ias: list[int] = []
        labels: list[object] = []
        for ts, ia in pairs:
            if pd.isna(ts) or pd.isna(ia):
                continue
            try:
                ts_i = int(ts)
                ia_i = int(ia)
            except (ValueError, TypeError):
                continue
            timestamps.append(ts_i)
            ias.append(ia_i)
            labels.append(ia_to_label.get(ia_i))

        if not timestamps:
            continue

        rt_per_area = {r: 0 for r in regions}
        n = len(timestamps)
        i = 0
        while i < n:
            j = i
            while j + 1 < n and labels[j + 1] == labels[i]:
                j += 1
            label = labels[i]
            if label in rt_per_area:
                first_ts = timestamps[i]
                if j + 1 < n:
                    end_proxy = timestamps[j + 1]
                else:
                    last_entry = last_dur_for_trial.get(ias[j])
                    if last_entry is not None and last_entry[0] == timestamps[j]:
                        end_proxy = timestamps[j] + last_entry[1]
                    else:
                        end_proxy = timestamps[j]
                rt_per_area[label] += end_proxy - first_ts
            i = j + 1

        out_row = {"participant_id": pid, "TRIAL_INDEX": tid}
        out_row.update(rt_per_area)
        rt_rows.append(out_row)

    rt_df = pd.DataFrame(rt_rows, columns=keys + regions)

    n_words = (
        ia_data.groupby(keys + [area_col])
        .size()
        .reset_index(name="n_words")
        .pivot_table(index=keys, columns=area_col, values="n_words")
        .reset_index()
    )
    for r in regions:
        if r not in n_words.columns:
            n_words[r] = 0
    n_words[regions] = n_words[regions].fillna(0)

    all_pairs = ia_data[keys].drop_duplicates().reset_index(drop=True)
    n_words_for_join = n_words.rename(columns={r: f"_nw_{r}" for r in regions})
    combined = all_pairs.merge(rt_df, on=keys, how="left").merge(
        n_words_for_join, on=keys, how="left"
    )
    for r in regions:
        if r not in combined.columns:
            combined[r] = 0
    combined[regions] = combined[regions].fillna(0)
    nw_cols = [f"_nw_{r}" for r in regions]
    combined[nw_cols] = combined[nw_cols].fillna(0)

    final = all_pairs.copy()
    for r in regions:
        rt_vals = combined[r].astype(float).values
        n_vals = combined[f"_nw_{r}"].astype(float).values
        denom = pd.Series(n_vals).replace(0, pd.NA)
        final[f"RT_pure_{r}"] = rt_vals
        final[f"RT_normalized_{r}"] = pd.Series(rt_vals / denom).fillna(0).values
    return final


def load_paragraph_fixations(
    fixations_path: Path = FIX_PARAGRAPH_PATH,
    participants: Sequence[str] | None = None,
    chunksize: int = 2_000_000,
    verbose: bool = True,
) -> pd.DataFrame:
    """Read only the columns of the paragraph fixation report that RT needs.

    The full report is several GB wide; this keeps five columns and reads in
    chunks. `participants` restricts to a subset, for a quick check.
    """
    numeric = ["TRIAL_INDEX", FIX_START_COL, FIX_DURATION_COL, FIX_IA_ID_COL]
    usecols = ["participant_id"] + numeric

    parts = []
    # Read everything as text: the report uses "." for missing interest areas,
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
        for c in numeric:
            chunk[c] = pd.to_numeric(chunk[c], errors="coerce")
        parts.append(chunk.dropna(subset=["TRIAL_INDEX", FIX_START_COL]))

    if not parts:
        return pd.DataFrame(columns=usecols)

    out = pd.concat(parts, ignore_index=True)
    out["TRIAL_INDEX"] = out["TRIAL_INDEX"].astype("int64")
    if verbose:
        print(f"  read {len(out):,} paragraph fixations")
    return out


def compute_run_based_rt_from_fixations(
    fixations: pd.DataFrame,
    ia_data: pd.DataFrame,
    area_col: str = PARAGRAPH_AREA_COL,
    regions: Sequence[str] = PARAGRAPH_REGIONS,
    fix_start_col: str = FIX_START_COL,
    fix_duration_col: str = FIX_DURATION_COL,
    fix_ia_id_col: str = FIX_IA_ID_COL,
    ia_id_col: str = IA_ID_COL,
) -> pd.DataFrame:
    """Run-based per-area RT from a fixation report, as `compute_run_based_rt`.

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

    Unlike `compute_run_based_rt` (which reconstructs fixation ends from the
    IA-level report), the fixation report carries each fixation's own start and
    duration, so no matching is needed.

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

    fix = fixations[keys + [fix_start_col, fix_duration_col, fix_ia_id_col]].copy()
    for c in (fix_start_col, fix_duration_col, fix_ia_id_col):
        fix[c] = pd.to_numeric(fix[c], errors="coerce")
    fix = fix.dropna(subset=[fix_start_col])

    labels = (
        ia_data[keys + [ia_id_col, area_col]]
        .dropna(subset=[ia_id_col, area_col])
        .drop_duplicates(subset=keys + [ia_id_col])
        .copy()
    )
    # Cast both sides of the id join to float: the fixation report's ids carry
    # NaN for off-area fixations, so an int/float mismatch would join to nothing.
    labels[ia_id_col] = pd.to_numeric(labels[ia_id_col], errors="coerce").astype(
        "float64"
    )
    fix[fix_ia_id_col] = fix[fix_ia_id_col].astype("float64")

    fix = fix.merge(
        labels.rename(columns={ia_id_col: fix_ia_id_col}),
        on=keys + [fix_ia_id_col],
        how="left",
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
        last_start=(fix_start_col, "last"),
        last_duration=(fix_duration_col, "last"),
        next_start=("_next_start", "last"),
    )
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


def build_rt_and_tfd(
    all_participants: pd.DataFrame | None = None,
    hunters_path: Path = HUNTERS_PROCESSED_PATH,
    gatherers_path: Path = GATHERERS_PROCESSED_PATH,
    paragraph_ia_path: Path = IA_PARAGRAPH_PATH,
    paragraph_fixations_path: Path = FIX_PARAGRAPH_PATH,
    button_clicks_path: Path = BUTTON_CLICKS_PATH,
    output_path: Path = RT_AND_TFD_PATH,
    include_paragraph: bool = True,
    include_run_based_rt: bool = True,
    include_paragraph_run_based_rt: bool = True,
    save: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    """Build RT/TFD features and (optionally) save to CSV.

    Answer regions (all processed IA data, by `area_label`):
      - TFD via per-area aggregation of IA dwell times.
      - RT: when `include_run_based_rt` is True, via run-based aggregation over
        the trial's fixation sequence (loaded from `button_clicks_path`), to
        avoid conflating non-consecutive visits — and the fixation-span RT is
        kept as `TimeSinceOffset_*`. When False (no button-click data), the
        fixation-span RT is used directly as `RT_*` and button clicks are not
        read.
    Paragraph regions (paragraph IA, by `auxiliary_span_type`), only when
    `include_paragraph` is True — set it False for experiments without a
    paragraph-reading screen (then only answer-region RT/TFD is returned):
      - TFD via per-area aggregation of IA dwell times.
      - RT: when `include_paragraph_run_based_rt` is True, run-based over the
        paragraph fixation report (`paragraph_fixations_path`), matching the
        answer-region treatment — and the fixation-span RT is kept as
        `TimeSinceOffset_*`. When False, the fixation-span RT (first to last
        fixation on the region, excursions away included) is used as `RT_*`,
        which is what this function did before the run-based path existed.
    When both are built they are inner-merged on (participant_id, TRIAL_INDEX).

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
    answer_full = compute_reading_times(
        all_participants,
        area_col=ANSWER_AREA_COL,
        regions=ANSWER_REGIONS,
    )

    if include_run_based_rt:
        # Keep the fixation-span RT as TimeSinceOffset_* and replace RT_* with
        # the run-based RT computed from the button-click fixation sequence.
        rt_rename = {
            c: c.replace("RT_pure_", "TimeSinceOffset_pure_", 1).replace(
                "RT_normalized_", "TimeSinceOffset_normalized_", 1
            )
            for c in answer_full.columns
            if c.startswith("RT_pure_") or c.startswith("RT_normalized_")
        }
        answer = answer_full.rename(columns=rt_rename)

        if verbose:
            print("Computing answer-region RT (run-based)...")
        button_clicks = pd.read_csv(button_clicks_path)
        answer_rt = compute_run_based_rt(
            button_clicks,
            all_participants,
            area_col=ANSWER_AREA_COL,
            regions=ANSWER_REGIONS,
        )
        answer = answer.merge(
            answer_rt, on=["participant_id", "TRIAL_INDEX"], how="left"
        )
    else:
        # No button-click data: use the fixation-span RT directly as RT_*.
        answer = answer_full

    if include_paragraph:
        if verbose:
            print("Computing paragraph-region reading times...")
        paragraph_ia = pd.read_csv(paragraph_ia_path)
        paragraph_full = compute_reading_times(
            paragraph_ia,
            area_col=PARAGRAPH_AREA_COL,
            regions=PARAGRAPH_REGIONS,
        )

        if include_paragraph_run_based_rt:
            # Same treatment as the answer regions: keep the first-to-last
            # fixation span as TimeSinceOffset_* and make RT_* run-based.
            rt_rename = {
                c: c.replace("RT_pure_", "TimeSinceOffset_pure_", 1).replace(
                    "RT_normalized_", "TimeSinceOffset_normalized_", 1
                )
                for c in paragraph_full.columns
                if c.startswith("RT_pure_") or c.startswith("RT_normalized_")
            }
            paragraph = paragraph_full.rename(columns=rt_rename)

            if verbose:
                print("Computing paragraph-region RT (run-based)...")
            paragraph_fixations = load_paragraph_fixations(
                paragraph_fixations_path, verbose=verbose
            )
            paragraph_run_rt = compute_run_based_rt_from_fixations(
                paragraph_fixations,
                paragraph_ia,
                area_col=PARAGRAPH_AREA_COL,
                regions=PARAGRAPH_REGIONS,
            )
            paragraph = paragraph.merge(
                paragraph_run_rt, on=["participant_id", "TRIAL_INDEX"], how="left"
            )
        else:
            paragraph = paragraph_full

        rt_and_tfd = answer.merge(
            paragraph, on=["participant_id", "TRIAL_INDEX"], how="inner"
        )
    else:
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
