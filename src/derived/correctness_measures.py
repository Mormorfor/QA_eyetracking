# src/derived/correctness_measures.py
from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from src import constants as Con
from src.statistics.correctness_measures_tests import (
    correctness_by_seq_len_threshold_test,
    correctness_by_sequence_pattern_test,
    correctness_by_trial_mean_dwell_threshold_test,
)


# -----------------------------
# Small reusable primitives
# -----------------------------

def sequence_len_literal_eval(x) -> int:
    """Sequences come as string representations of lists/tuples. Invalid -> 0."""
    if x is None:
        return 0
    if isinstance(x, float) and np.isnan(x):
        return 0
    if isinstance(x, str):
        try:
            x = ast.literal_eval(x)
        except Exception:
            return 0
    return len(x) if isinstance(x, (list, tuple)) else 0


def parse_seq(x):
    """Parse sequence column via ast.literal_eval (expects list/tuple)."""
    if x is None:
        return None
    if isinstance(x, float) and np.isnan(x):
        return None
    if isinstance(x, str):
        try:
            x = ast.literal_eval(x)
        except Exception:
            return None
    return x if isinstance(x, (list, tuple)) else None


def wilson_ci(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    if n <= 0:
        return (np.nan, np.nan)
    phat = k / n
    denom = 1 + (z**2) / n
    center = (phat + (z**2) / (2 * n)) / denom
    half = (z / denom) * np.sqrt((phat * (1 - phat) + (z**2) / (4 * n)) / n)
    return (max(0.0, center - half), min(1.0, center + half))


def summarize_binary_by_group(trial_df: pd.DataFrame, group_col: str, outcome_col: str) -> pd.DataFrame:
    """
    Expects one row per trial. Produces n/k/accuracy/Wilson CI per group.
    """
    rows = []
    for group_label, dd in trial_df.groupby(group_col, sort=False, observed=False):
        n = len(dd)
        k = int(dd[outcome_col].sum())
        acc = (k / n) if n else np.nan
        lo, hi = wilson_ci(k, n) if n else (np.nan, np.nan)
        rows.append(
            dict(
                group=group_label,
                n=n,
                k_correct=k,
                accuracy=acc,
                ci_low=lo,
                ci_high=hi,
            )
        )
    return pd.DataFrame(rows)


# -----------------------------
# Trial-level feature builders
# -----------------------------

def build_trial_df_for_seq_len_threshold(
    df: pd.DataFrame,
    threshold: int,
    seq_col: str = Con.SIMPLIFIED_FIX_SEQ_BY_LOCATION,
    correct_col: str = Con.IS_CORRECT_COLUMN,
) -> pd.DataFrame:
    d = df[[Con.TRIAL_ID, Con.PARTICIPANT_ID, seq_col, correct_col]].copy()
    d[correct_col] = d[correct_col].astype(int)
    d["_seq_len"] = d[seq_col].apply(sequence_len_literal_eval)

    trial_df = (
        d.groupby([Con.TRIAL_ID, Con.PARTICIPANT_ID], as_index=False)
        .agg(seq_len=("_seq_len", "first"), is_correct=(correct_col, "first"))
    )

    trial_df["group"] = np.where(trial_df["seq_len"] > threshold, f"> {threshold}", f"≤ {threshold}")
    return trial_df


def has_back_and_forth_xyx(seq) -> bool:
    if seq is None or len(seq) < 3:
        return False
    a, b, c = seq[-1], seq[-2], seq[-3]
    return (a == c) and (a != b)


def has_back_and_forth_xyxy(seq) -> bool:
    if seq is None or len(seq) < 4:
        return False
    a, b, c, d = seq[-4], seq[-3], seq[-2], seq[-1]
    return (a == c) and (b == d) and (a != b)


def build_trial_df_for_back_and_forth_pattern(
    df: pd.DataFrame,
    seq_col: str = Con.SIMPLIFIED_FIX_SEQ_BY_LOCATION,
    correct_col: str = Con.IS_CORRECT_COLUMN,
    use_xyxy: bool = False,
) -> pd.DataFrame:
    pattern_name = "XYXY" if use_xyxy else "XYX"
    pattern_fn = has_back_and_forth_xyxy if use_xyxy else has_back_and_forth_xyx

    d = df[[Con.TRIAL_ID, Con.PARTICIPANT_ID, seq_col, correct_col]].copy()
    d[correct_col] = d[correct_col].astype(int)
    d["_seq"] = d[seq_col].apply(parse_seq)
    d["_has_pattern"] = d["_seq"].apply(lambda s: bool(pattern_fn(s)) if s is not None else False)

    trial_df = (
        d.groupby([Con.TRIAL_ID, Con.PARTICIPANT_ID], as_index=False)
        .agg(has_pattern=("_has_pattern", "first"), is_correct=(correct_col, "first"))
    )

    trial_df["group"] = np.where(trial_df["has_pattern"], f"{pattern_name} present", f"{pattern_name} absent")
    return trial_df


def compute_trial_mean_dwell_per_word(
    df: pd.DataFrame,
    dwell_col: str = Con.IA_DWELL_TIME,
) -> pd.Series:
    total_dwell = df.groupby([Con.TRIAL_ID, Con.PARTICIPANT_ID])[dwell_col].transform("sum")
    n_words = df.groupby([Con.TRIAL_ID, Con.PARTICIPANT_ID])[dwell_col].transform("count")
    return total_dwell / n_words


def build_trial_df_for_mean_dwell_threshold(
    df: pd.DataFrame,
    threshold: float,
    dwell_col: str = Con.IA_DWELL_TIME,
    correct_col: str = Con.IS_CORRECT_COLUMN,
) -> pd.DataFrame:
    d = df[[Con.TRIAL_ID, Con.PARTICIPANT_ID, dwell_col, correct_col]].copy()
    d[correct_col] = d[correct_col].astype(int)
    d["_trial_mean_dwell"] = compute_trial_mean_dwell_per_word(d, dwell_col)

    trial_df = (
        d.groupby([Con.TRIAL_ID, Con.PARTICIPANT_ID], as_index=False)
        .agg(trial_mean_dwell=("_trial_mean_dwell", "first"), is_correct=(correct_col, "first"))
    )

    trial_df["group"] = np.where(trial_df["trial_mean_dwell"] <= threshold, f"≤ {threshold}", f"> {threshold}")
    return trial_df


# -----------------------------
# Public "compute summaries + tests"
# -----------------------------

def compute_seq_len_threshold_summary(
    df: pd.DataFrame,
    threshold: int,
    seq_col: str = Con.SIMPLIFIED_FIX_SEQ_BY_LOCATION,
    correct_col: str = Con.IS_CORRECT_COLUMN,
    add_significance: bool = True,
) -> Tuple[pd.DataFrame, Optional[Dict]]:
    trial_df = build_trial_df_for_seq_len_threshold(df, threshold, seq_col, correct_col)

    # enforce plotting order (stable)
    order = [f"≤ {threshold}", f"> {threshold}"]
    trial_df["group"] = pd.Categorical(trial_df["group"], categories=order, ordered=True)

    summary_df = summarize_binary_by_group(trial_df, group_col="group", outcome_col="is_correct").sort_values("group")

    test_res = None
    if add_significance:
        test_res = correctness_by_seq_len_threshold_test(df=df, threshold=threshold, seq_col=seq_col, correct_col=correct_col)

    return summary_df.reset_index(drop=True), test_res


def compute_back_and_forth_pattern_summary(
    df: pd.DataFrame,
    seq_col: str = Con.SIMPLIFIED_FIX_SEQ_BY_LOCATION,
    correct_col: str = Con.IS_CORRECT_COLUMN,
    use_xyxy: bool = False,
    add_significance: bool = True,
) -> Tuple[pd.DataFrame, Optional[Dict], str]:
    pattern_name = "XYXY" if use_xyxy else "XYX"
    pattern_fn = has_back_and_forth_xyxy if use_xyxy else has_back_and_forth_xyx

    trial_df = build_trial_df_for_back_and_forth_pattern(df, seq_col, correct_col, use_xyxy=use_xyxy)

    order = [f"{pattern_name} absent", f"{pattern_name} present"]
    trial_df["group"] = pd.Categorical(trial_df["group"], categories=order, ordered=True)

    summary_df = summarize_binary_by_group(trial_df, "group", "is_correct").sort_values("group")

    test_res = None
    if add_significance:
        test_res = correctness_by_sequence_pattern_test(df=df, pattern_fn=pattern_fn, seq_col=seq_col, correct_col=correct_col)

    return summary_df.reset_index(drop=True), test_res, pattern_name


def compute_trial_mean_dwell_threshold_summary(
    df: pd.DataFrame,
    threshold: float,
    dwell_col: str = Con.IA_DWELL_TIME,
    correct_col: str = Con.IS_CORRECT_COLUMN,
    add_significance: bool = True,
) -> Tuple[pd.DataFrame, Optional[Dict]]:
    trial_df = build_trial_df_for_mean_dwell_threshold(df, threshold, dwell_col, correct_col)

    order = [f"≤ {threshold}", f"> {threshold}"]
    trial_df["group"] = pd.Categorical(trial_df["group"], categories=order, ordered=True)

    summary_df = summarize_binary_by_group(trial_df, "group", "is_correct").sort_values("group")

    test_res = None
    if add_significance:
        test_res = correctness_by_trial_mean_dwell_threshold_test(df=df, threshold=threshold, dwell_col=dwell_col, correct_col=correct_col)

    return summary_df.reset_index(drop=True), test_res
