# participant_scan_patterns.py

from __future__ import annotations

import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src import constants as Con


# ---------------------------------------------------------------------------
# Constants / helpers
# ---------------------------------------------------------------------------

ANSWER_AREAS = Con.ANSWER_LABEL_CHOICES[1:]


def _build_trial_area_table(
    df: pd.DataFrame,
    metric: str,
) -> pd.DataFrame:
    """
    Build a wide table with one row per (participant, trial) and
    one column per answer area (answer_A..D) for a given metric.

    Returns columns:
        participant_id
        trial_index
        selected_label
        <Con.IS_CORRECT_COLUMN>   (per trial)
        answer_A, answer_B, answer_C, answer_D
    """
    area_level = (
        df[
            [
                Con.PARTICIPANT_ID,
                Con.TRIAL_ID,
                Con.AREA_LABEL_COLUMN,
                Con.SELECTED_ANSWER_LABEL_COLUMN,
                metric,
            ]
        ]
        .drop_duplicates(
            subset=[Con.PARTICIPANT_ID, Con.TRIAL_ID, Con.AREA_LABEL_COLUMN]
        )
        .dropna(subset=[metric])
    )

    area_level[metric] = (
        area_level[metric]
        .astype(str)
        .str.strip()
        .replace({".": np.nan, "": np.nan, "nan": np.nan, "None": np.nan})
    )
    area_level[metric] = pd.to_numeric(area_level[metric], errors="coerce")

    area_level = area_level.dropna(subset=[metric])

    area_level = area_level[
        area_level[Con.AREA_LABEL_COLUMN].isin(ANSWER_AREAS)
    ].copy()

    wide = (
        area_level
        .pivot_table(
            index=[Con.PARTICIPANT_ID, Con.TRIAL_ID],
            columns=Con.AREA_LABEL_COLUMN,
            values=metric,
            aggfunc="mean",
        )
        .reindex(columns=ANSWER_AREAS)
    )

    wide.reset_index(inplace=True)

    sel = (
        area_level[
            [Con.PARTICIPANT_ID, Con.TRIAL_ID, Con.SELECTED_ANSWER_LABEL_COLUMN]
        ]
        .drop_duplicates(subset=[Con.PARTICIPANT_ID, Con.TRIAL_ID])
        .rename(columns={Con.SELECTED_ANSWER_LABEL_COLUMN: "selected_label"})
    )

    out = wide.merge(
        sel,
        on=[Con.PARTICIPANT_ID, Con.TRIAL_ID],
        how="left",
    )

    if Con.IS_CORRECT_COLUMN in df.columns:
        corr = (
            df[
                [Con.PARTICIPANT_ID, Con.TRIAL_ID, Con.IS_CORRECT_COLUMN]
            ]
            .drop_duplicates(subset=[Con.PARTICIPANT_ID, Con.TRIAL_ID])
        )

        out = out.merge(
            corr,
            on=[Con.PARTICIPANT_ID, Con.TRIAL_ID],
            how="left",
        )

    out.rename(
        columns={
            Con.PARTICIPANT_ID: "participant_id",
            Con.TRIAL_ID: "trial_index",
        },
        inplace=True,
    )
    return out


def _label_from_area(area_name: str) -> Optional[str]:
    """
    'answer_A' -> 'A', etc.
    """
    if not isinstance(area_name, str):
        return None
    if not area_name.startswith(Con.ANSWER_PREFIX):
        return None
    return area_name[len(Con.ANSWER_PREFIX):]


# ---------------------------------------------------------------------------
# Core trial-level preference stats
# ---------------------------------------------------------------------------
#furtherst from median
def compute_trial_preference_stats(
    df: pd.DataFrame,
    metric: str = Con.MEAN_DWELL_TIME,
    uniform_rel_range: float = 0.10,
) -> pd.DataFrame:
    """
    For each (participant, trial) compute:
        - metric per answer (answer_A..D)
        - rel_range: (max - min) / mean  across answers
        - preference_strength: (max - second max) / mean
        - dominant_metric_area: answer_X with highest metric
        - dominant_metric_label: corresponding A/B/C/D
        - selected_label: chosen answer (A/B/C/D)
        - is_uniform: rel_range <= uniform_rel_range
        - mismatch_flag: dominant_metric_label != selected_label
                         AND preference_strength >= min_pref_strength

    Returns one row per (participant, trial), keeping Con.IS_CORRECT_COLUMN
    if it existed in the original df.
    """
    trial_table = _build_trial_area_table(df, metric=metric)

    stats_rows = []
    for _, row in trial_table.iterrows():
        vals: List[float] = []
        cols: List[str] = []
        for a in ANSWER_AREAS:
            v = row.get(a)
            if pd.notna(v):
                vals.append(float(v))
                cols.append(a)

        if len(vals) == 0:
            rel_range = np.nan
            pref_strength = np.nan
            dom_area = None
            dom_label = None
        else:
            vals_arr = np.array(vals, dtype=float)
            mean_val = vals_arr.mean()
            max_val = vals_arr.max()
            min_val = vals_arr.min()

            if mean_val > 0:
                rel_range = (max_val - min_val) / mean_val
            else:
                rel_range = np.nan

            if len(vals_arr) >= 2:
                sorted_vals = np.sort(vals_arr)[::-1]
                top = sorted_vals[0]
                second = sorted_vals[1]
            else:
                top = second = vals_arr[0]

            if mean_val > 0:
                pref_strength = (top - second) / mean_val
            else:
                pref_strength = np.nan

            max_idx = int(np.argmax(vals_arr))
            dom_area = cols[max_idx]
            dom_label = _label_from_area(dom_area)

        stats_rows.append(
            {
                "participant_id": row["participant_id"],
                "trial_index": row["trial_index"],
                "selected_label": row["selected_label"],
                "rel_range": rel_range,
                "preference_strength": pref_strength,
                "dominant_metric_area": dom_area,
                "dominant_metric_label": dom_label,
            }
        )

    stats_df = pd.DataFrame(stats_rows)

    out = trial_table.merge(
        stats_df,
        on=["participant_id", "trial_index", "selected_label"],
        how="left",
    )

    out = assign_pref_group(out, uniform_rel_range=uniform_rel_range)

    out["uniform_rel_range"] = uniform_rel_range
    out["preference_metric"] = metric
    print(out)
    return out


def run_trial_preference_screening(
    hunters: pd.DataFrame,
    gatherers: pd.DataFrame,
    metric: str = Con.MEAN_DWELL_TIME,
    uniform_rel_range: float = 0.10,
    output_root: str = "../reports/report_data/trial_prefs",
    save: bool = True,
) -> Dict[str, pd.DataFrame]:
    """
    Convenience wrapper: compute trial-level preference stats for
    hunters and gatherers and optionally save as CSV.

    Returns:
        {
          "hunters": <DataFrame>,
          "gatherers": <DataFrame>,
        }
    """
    os.makedirs(output_root, exist_ok=True)

    results: Dict[str, pd.DataFrame] = {}
    for name, df in [("hunters", hunters), ("gatherers", gatherers)]:
        res = compute_trial_preference_stats(
            df,
            metric=metric,
            uniform_rel_range=uniform_rel_range,
        )
        results[name] = res

        if save:
            fname = (
                f"{name}_trial_prefs_{metric}"
                f"_uniform{uniform_rel_range:.2f}.csv"
            )
            res.to_csv(
                os.path.join(output_root, fname),
                index=False,
            )

    return results


# ---------------------------------------------------------------------------
# Grouping + descriptive stats
# ---------------------------------------------------------------------------

def assign_pref_group(
    out: pd.DataFrame,
    uniform_rel_range: float,
) -> pd.DataFrame:
    """
    Ensures each trial falls into exactly one of:
      - uniform
      - matching
      - mismatch
    based on:
      - rel_range
      - dominant_metric_label
      - selected_label
    """
    out = out.copy()

    out["is_uniform"] = out["rel_range"] <= uniform_rel_range

    out["pref_group"] = "matching"
    out.loc[out["is_uniform"], "pref_group"] = "uniform"
    out.loc[
        (~out["is_uniform"]) & (out["dominant_metric_label"] != out["selected_label"]),
        "pref_group",
    ] = "mismatch"

    out["mismatch_flag"] = out["pref_group"].eq("mismatch")

    allowed = {"uniform", "matching", "mismatch"}
    if not set(out["pref_group"].unique()).issubset(allowed):
        raise ValueError(f"Unexpected pref_group values: {set(out['pref_group'].unique()) - allowed}")

    if out["pref_group"].isna().any():
        raise ValueError("Some trials have pref_group == NaN, partition failed.")

    return out


def compute_global_preference_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return:
        one-row 'meta' dict-like summary of counts
        (mostly for quick logging / sanity check).
    """
    total = len(df)
    n_uniform = int(df["is_uniform"].sum())
    n_mismatch = int(df["mismatch_flag"].sum())
    n_matching = total - n_uniform - n_mismatch

    return pd.DataFrame(
        {
            "total_trials": [total],
            "n_uniform": [n_uniform],
            "n_mismatch": [n_mismatch],
            "n_matching": [n_matching],
            "prop_uniform": [n_uniform / total if total else np.nan],
            "prop_mismatch": [n_mismatch / total if total else np.nan],
            "prop_matching": [n_matching / total if total else np.nan],
        }
    )


def compute_participant_preference_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a DataFrame with one row per participant:
        participant_id
        n_trials
        n_uniform, prop_uniform
        n_mismatch, prop_mismatch
        n_matching, prop_matching
    """
    gp = df.groupby("participant_id")

    summary = pd.DataFrame({
        "n_trials": gp.size(),
        "n_uniform": gp["is_uniform"].sum(),
        "n_mismatch": gp["mismatch_flag"].sum(),
    })

    summary["n_matching"] = (
        summary["n_trials"] - summary["n_uniform"] - summary["n_mismatch"]
    )

    summary["prop_uniform"] = summary["n_uniform"] / summary["n_trials"]
    summary["prop_mismatch"] = summary["n_mismatch"] / summary["n_trials"]
    summary["prop_matching"] = summary["n_matching"] / summary["n_trials"]

    return summary.reset_index().rename(columns={"participant_id": "participant_id"})


def correctness_by_group(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a DataFrame with:
        pref_group
        correct_rate
        n_trials
        n_correct
    Requires:
        - 'pref_group'
        - Con.IS_CORRECT_COLUMN
    """
    out = (
        df.groupby("pref_group")[Con.IS_CORRECT_COLUMN]
        .agg(correct_rate="mean", n_trials="count", n_correct="sum")
        .reset_index()
    )
    out["correct_rate"] = out["correct_rate"].round(3)
    return out
