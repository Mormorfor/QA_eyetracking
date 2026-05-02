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


def build_rt_and_tfd(
    hunters_path: Path = HUNTERS_PROCESSED_PATH,
    gatherers_path: Path = GATHERERS_PROCESSED_PATH,
    paragraph_ia_path: Path = IA_PARAGRAPH_PATH,
    button_clicks_path: Path = BUTTON_CLICKS_PATH,
    output_path: Path = RT_AND_TFD_PATH,
    verbose: bool = True,
) -> pd.DataFrame:
    """Build RT/TFD features and save to CSV.

    Answer regions (hunters + gatherers, by `area_label`):
      - TFD via per-area aggregation of IA dwell times.
      - RT via run-based aggregation over the trial's fixation sequence
        (loaded from `button_clicks_path`), to avoid conflating non-consecutive
        visits to the same area.
    Paragraph regions (paragraph IA, by `auxiliary_span_type`):
      - Both RT and TFD via per-area aggregation, since paragraph reading is
        approximately linear within a region.
    The two are inner-merged on (participant_id, TRIAL_INDEX).
    """
    if verbose:
        print("Loading data...")
    hunters = pd.read_csv(hunters_path)
    gatherers = pd.read_csv(gatherers_path)
    all_participants = pd.concat([hunters, gatherers], ignore_index=True)
    paragraph_ia = pd.read_csv(paragraph_ia_path)
    button_clicks = pd.read_csv(button_clicks_path)

    if verbose:
        print("Computing answer-region TFD...")
    answer_full = compute_reading_times(
        all_participants,
        area_col=ANSWER_AREA_COL,
        regions=ANSWER_REGIONS,
    )
    rt_rename = {
        c: c.replace("RT_pure_", "TimeSinceOffset_pure_", 1).replace(
            "RT_normalized_", "TimeSinceOffset_normalized_", 1
        )
        for c in answer_full.columns
        if c.startswith("RT_pure_") or c.startswith("RT_normalized_")
    }
    answer_tfd = answer_full.rename(columns=rt_rename)

    if verbose:
        print("Computing answer-region RT (run-based)...")
    answer_rt = compute_run_based_rt(
        button_clicks,
        all_participants,
        area_col=ANSWER_AREA_COL,
        regions=ANSWER_REGIONS,
    )
    answer = answer_tfd.merge(
        answer_rt, on=["participant_id", "TRIAL_INDEX"], how="left"
    )

    if verbose:
        print("Computing paragraph-region reading times...")
    paragraph_rt = compute_reading_times(
        paragraph_ia,
        area_col=PARAGRAPH_AREA_COL,
        regions=PARAGRAPH_REGIONS,
    )

    rt_and_tfd = answer.merge(
        paragraph_rt, on=["participant_id", "TRIAL_INDEX"], how="inner"
    )
    rt_and_tfd.to_csv(output_path, index=False)
    if verbose:
        print(f"Saved {len(rt_and_tfd)} rows to {output_path}")
    return rt_and_tfd


if __name__ == "__main__":
    build_rt_and_tfd()
