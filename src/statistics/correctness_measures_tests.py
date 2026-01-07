# src/statistics/correctness_measures_tests.py

from __future__ import annotations

from typing import Dict, Callable
import ast

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact

from src import constants as C


def correctness_by_seq_len_threshold_test(
    df: pd.DataFrame,
    threshold: int,
    seq_col: str = C.SIMPLIFIED_FIX_SEQ_BY_LOCATION,
    correct_col: str = C.IS_CORRECT_COLUMN,
) -> Dict:
    """
    Fisher exact test comparing correctness between:
      - seq_len <= threshold
      - seq_len > threshold

    Assumes seq_col values are strings that can be parsed with ast.literal_eval
    into list/tuple of strings.

    Returns dict with:
      - contingency_table (2x2)
      - odds_ratio
      - p_value
      - counts
      - accuracies
      - delta_accuracy (long - short)
    """
    sub = df[[seq_col, correct_col]].dropna().copy()
    sub[correct_col] = sub[correct_col].astype(int)

    def _len_parsed(x) -> int:
        try:
            x = ast.literal_eval(x) if isinstance(x, str) else x
        except Exception:
            return 0
        return len(x) if isinstance(x, (list, tuple)) else 0

    sub["_seq_len"] = sub[seq_col].apply(_len_parsed)
    sub["_is_long"] = sub["_seq_len"] > threshold

    a = int(((~sub["_is_long"]) & (sub[correct_col] == 1)).sum())
    b = int(((~sub["_is_long"]) & (sub[correct_col] == 0)).sum())
    c = int(((sub["_is_long"]) & (sub[correct_col] == 1)).sum())
    d = int(((sub["_is_long"]) & (sub[correct_col] == 0)).sum())

    table = np.array([[a, b],
                      [c, d]], dtype=int)

    odds_ratio, p_value = fisher_exact(table)  # two-sided default

    n_short = a + b
    n_long = c + d
    acc_short = a / n_short if n_short else np.nan
    acc_long = c / n_long if n_long else np.nan

    return {
        "contingency_table": table,
        "odds_ratio": float(odds_ratio),
        "p_value": float(p_value),
        "counts": {
            f"≤{threshold}": {"correct": a, "incorrect": b, "n": n_short},
            f">{threshold}": {"correct": c, "incorrect": d, "n": n_long},
        },
        "accuracies": {
            f"≤{threshold}": float(acc_short) if np.isfinite(acc_short) else np.nan,
            f">{threshold}": float(acc_long) if np.isfinite(acc_long) else np.nan,
        },
        "delta_accuracy": float(acc_long - acc_short)
        if np.isfinite(acc_long) and np.isfinite(acc_short)
        else np.nan,
    }




def correctness_by_sequence_pattern_test(
    df: pd.DataFrame,
    pattern_fn: Callable[[list], bool],
    seq_col: str = C.SIMPLIFIED_FIX_SEQ_BY_LOCATION,
    correct_col: str = C.IS_CORRECT_COLUMN,
) -> Dict:
    """
    Fisher exact test comparing correctness between:
      - pattern present
      - pattern absent

    seq_col is parsed with ast.literal_eval.
    """

    sub = df[[seq_col, correct_col]].dropna().copy()
    sub[correct_col] = sub[correct_col].astype(int)

    def _parse(x):
        if isinstance(x, str):
            try:
                x = ast.literal_eval(x)
            except Exception:
                return None
        return x if isinstance(x, (list, tuple)) else None

    sub["_seq"] = sub[seq_col].apply(_parse)
    sub["_has_pattern"] = sub["_seq"].apply(lambda s: bool(pattern_fn(s)) if s is not None else False)

    a = int(((sub["_has_pattern"]) & (sub[correct_col] == 1)).sum())   # pattern present correct
    b = int(((sub["_has_pattern"]) & (sub[correct_col] == 0)).sum())   # pattern present incorrect
    c = int(((~sub["_has_pattern"]) & (sub[correct_col] == 1)).sum())  # pattern absent correct
    d = int(((~sub["_has_pattern"]) & (sub[correct_col] == 0)).sum())  # pattern absent incorrect

    table = np.array([[a, b],
                      [c, d]], dtype=int)

    odds_ratio, p_value = fisher_exact(table)

    n_yes = a + b
    n_no = c + d
    acc_yes = a / n_yes if n_yes else np.nan
    acc_no = c / n_no if n_no else np.nan

    return {
        "contingency_table": table,
        "odds_ratio": float(odds_ratio),
        "p_value": float(p_value),
        "counts": {
            "pattern_present": {"correct": a, "incorrect": b, "n": n_yes},
            "pattern_absent": {"correct": c, "incorrect": d, "n": n_no},
        },
        "accuracies": {
            "pattern_present": float(acc_yes) if np.isfinite(acc_yes) else np.nan,
            "pattern_absent": float(acc_no) if np.isfinite(acc_no) else np.nan,
        },
        "delta_accuracy": float(acc_yes - acc_no)
        if np.isfinite(acc_yes) and np.isfinite(acc_no)
        else np.nan,
    }



def correctness_by_trial_mean_dwell_threshold_test(
    df: pd.DataFrame,
    threshold: float,
    dwell_col: str = C.IA_DWELL_TIME,
    correct_col: str = C.IS_CORRECT_COLUMN,
) -> Dict:
    """
    Fisher exact test comparing correctness between trials with
    low vs high mean dwell time per word across the entire trial.

    Mean dwell per word is computed as:
        sum(IA_DWELL_TIME) / number_of_words
    per (TRIAL_ID, PARTICIPANT_ID).
    """

    d = df[[C.TRIAL_ID, C.PARTICIPANT_ID, dwell_col, correct_col]].copy()
    d[correct_col] = d[correct_col].astype(int)

    # true per-word trial mean
    total_dwell = (
        d.groupby([C.TRIAL_ID, C.PARTICIPANT_ID])[dwell_col]
        .transform("sum")
    )
    n_words = (
        d.groupby([C.TRIAL_ID, C.PARTICIPANT_ID])[dwell_col]
        .transform("count")
    )
    d["_trial_mean_dwell"] = total_dwell / n_words

    low = d["_trial_mean_dwell"] <= threshold
    high = d["_trial_mean_dwell"] > threshold

    a = int(((low) & (d[correct_col] == 1)).sum())
    b = int(((low) & (d[correct_col] == 0)).sum())
    c = int(((high) & (d[correct_col] == 1)).sum())
    d_ = int(((high) & (d[correct_col] == 0)).sum())

    table = np.array([[a, b],
                      [c, d_]], dtype=int)

    odds_ratio, p_value = fisher_exact(table)

    return {
        "contingency_table": table,
        "odds_ratio": float(odds_ratio),
        "p_value": float(p_value),
        "counts": {
            "low_dwell": {"correct": a, "incorrect": b},
            "high_dwell": {"correct": c, "incorrect": d_},
        },
    }