# src/statistics/preference_correctness_tests.py

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, fisher_exact
from statsmodels.stats.multitest import multipletests

from src import constants as C


# -----------------------------
# Helpers
# -----------------------------

_ALLOWED_GROUPS = ["matching", "uniform", "mismatch"]
_PAIR_ORDER = [
    ("matching", "uniform"),
    ("matching", "mismatch"),
    ("uniform", "mismatch"),
]


def _as_int(x) -> int:
    if pd.isna(x):
        return 0
    return int(x)


def _contingency_correctness(df: pd.DataFrame, g1: str, g2: str) -> Tuple[np.ndarray, dict]:
    """
    Build a 2x2 table:
        rows:    [g1, g2]
        cols:    [correct, incorrect]
    """
    sub = df[df["pref_group"].isin([g1, g2])].copy()

    if sub.empty:
        table = np.array([[0, 0], [0, 0]], dtype=int)
        meta = {
            "n_trials_g1": 0, "n_trials_g2": 0,
            "n_correct_g1": 0, "n_correct_g2": 0,
            "correct_rate_g1": np.nan, "correct_rate_g2": np.nan,
        }
        return table, meta


    sub[C.IS_CORRECT_COLUMN] = sub[C.IS_CORRECT_COLUMN].astype(float)

    # counts per group
    grp = (
        sub.groupby("pref_group")[C.IS_CORRECT_COLUMN]
        .agg(n_trials="count", n_correct="sum")
        .reindex([g1, g2])
        .fillna(0)
    )

    n1 = _as_int(grp.loc[g1, "n_trials"])
    c1 = _as_int(grp.loc[g1, "n_correct"])
    n2 = _as_int(grp.loc[g2, "n_trials"])
    c2 = _as_int(grp.loc[g2, "n_correct"])

    table = np.array(
        [
            [c1, n1 - c1],
            [c2, n2 - c2],
        ],
        dtype=int,
    )

    meta = {
        "n_trials_g1": n1,
        "n_trials_g2": n2,
        "n_correct_g1": c1,
        "n_correct_g2": c2,
        "correct_rate_g1": (c1 / n1) if n1 else np.nan,
        "correct_rate_g2": (c2 / n2) if n2 else np.nan,
    }
    return table, meta


def _choose_test(table: np.ndarray, min_expected: float = 5.0) -> str:
    """
    Choose chi-square if expected counts are all >= min_expected,
    otherwise Fisher exact (more reliable for sparse tables).
    """
    # If any row has 0 trials, fisher is safest (but will often be undefined-ish).
    if table.sum(axis=1).min() == 0:
        return "fisher"

    try:
        _, _, _, expected = chi2_contingency(table, correction=False)
        if np.all(expected >= min_expected):
            return "chi2"
        return "fisher"
    except Exception:
        return "fisher"


def _run_test(table: np.ndarray, test_used: str) -> Tuple[float, float]:
    """
    Returns:
        p_value, test_stat
    """
    if table.sum() == 0:
        return np.nan, np.nan

    if test_used == "chi2":
        stat, p, _, _ = chi2_contingency(table, correction=False)
        return float(p), float(stat)

    # Fisher exact only supports 2x2, which we have.
    # It returns oddsratio, p-value. We'll store oddsratio as "stat".
    try:
        oddsratio, p = fisher_exact(table)
        return float(p), float(oddsratio)
    except Exception:
        return np.nan, np.nan


# -----------------------------
# Public API
# -----------------------------

def pairwise_correctness_tests(
    df: pd.DataFrame,
    metric: str | None = None,
    uniform_rel_range: float | None = None,
    group_label: str = "merged",
    alpha: float = 0.05,
    fdr_method: str = "fdr_bh",
    min_expected_for_chi2: float = 5.0,
) -> pd.DataFrame:
    """
    Pairwise statistical tests of correctness differences between:
        matching vs uniform
        matching vs mismatch
        uniform  vs mismatch

    Parameters
    ----------
    df : pd.DataFrame
        Trial-level preference table produced by participant_scan_patterns.
        Must contain:
            - 'pref_group' in {"matching","uniform","mismatch"}
            - constants.IS_CORRECT_COLUMN (binary 0/1)
        May also contain:
            - 'preference_metric'
            - 'uniform_rel_range'
    metric : str | None
        Metric name to store in the result table.
        If None, will try df['preference_metric'] else 'unknown'.
    uniform_rel_range : float | None
        Threshold to store in result table.
        If None, will try df['uniform_rel_range'] else NaN.
    group_label : str
        "hunters", "gatherers", or "merged" tag.
    alpha : float
        Significance threshold after FDR correction.
    fdr_method : str
        Method passed to statsmodels.multipletests.
    min_expected_for_chi2 : float
        Minimum expected count per cell to allow chi-square; else Fisher.

    Returns
    -------
    pd.DataFrame
        One row per pairwise comparison, with:
            metric, uniform_rel_range, group
            comparison, g1, g2
            test_used, p_value, p_fdr, significant
            correct_rate_g1, correct_rate_g2, delta_correctness
            n_trials_g1, n_trials_g2, n_correct_g1, n_correct_g2
            test_stat
    """
    if "pref_group" not in df.columns:
        raise KeyError("Missing required column: pref_group")
    if C.IS_CORRECT_COLUMN not in df.columns:
        raise KeyError(f"Missing required column: {C.IS_CORRECT_COLUMN}")

    if metric is None:
        metric = (
            df["preference_metric"].iloc[0]
            if "preference_metric" in df.columns and len(df) > 0
            else "unknown"
        )
    if uniform_rel_range is None:
        uniform_rel_range = (
            float(df["uniform_rel_range"].iloc[0])
            if "uniform_rel_range" in df.columns and len(df) > 0
            else np.nan
        )

    # # Keep only allowed groups, warn/error if unexpected values exist
    # present = set(df["pref_group"].dropna().unique())
    # bad = present - set(_ALLOWED_GROUPS)
    # if bad:
    #     raise ValueError(f"Unexpected pref_group values: {bad}")

    rows = []
    for g1, g2 in _PAIR_ORDER:
        table, meta = _contingency_correctness(df, g1, g2)
        test_used = _choose_test(table, min_expected=min_expected_for_chi2)
        p_value, test_stat = _run_test(table, test_used=test_used)

        r1 = meta["correct_rate_g1"]
        r2 = meta["correct_rate_g2"]

        rows.append(
            {
                "metric": metric,
                "uniform_rel_range": uniform_rel_range,
                "group": group_label,
                "comparison": f"{g1}_vs_{g2}",
                "g1": g1,
                "g2": g2,
                "test_used": test_used,
                "test_stat": test_stat,
                "p_value": p_value,
                "correct_rate_g1": r1,
                "correct_rate_g2": r2,
                "delta_correctness": (r1 - r2) if (pd.notna(r1) and pd.notna(r2)) else np.nan,
                **meta,
            }
        )

    out = pd.DataFrame(rows)

    # FDR correction within this metric/threshold/group call (3 tests)
    valid = out["p_value"].notna().to_numpy()
    out["p_fdr"] = np.nan
    out["significant"] = False

    if valid.any():
        pvals = out.loc[valid, "p_value"].to_numpy(dtype=float)
        _, p_adj, _, _ = multipletests(pvals, alpha=alpha, method=fdr_method)
        out.loc[valid, "p_fdr"] = p_adj
        out.loc[valid, "significant"] = out.loc[valid, "p_fdr"] < alpha

    out["correct_rate_g1"] = out["correct_rate_g1"].astype(float).round(4)
    out["correct_rate_g2"] = out["correct_rate_g2"].astype(float).round(4)
    out["delta_correctness"] = out["delta_correctness"].astype(float).round(4)
    out["p_value"] = out["p_value"].astype(float)
    out["p_fdr"] = out["p_fdr"].astype(float)

    return out
