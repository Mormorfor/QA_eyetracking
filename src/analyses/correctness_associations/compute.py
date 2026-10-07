# src/derived/correctness_measures.py
from __future__ import annotations

import ast
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from src.config import columns as Con
from src.lib.stats.proportions import wilson_ci  # moved to lib (T1.5)
# The feature half of this module moved to features/sequences.py (stage C).
# What is left is the analysis half; it still builds on those features.
from src.features.sequences import (
    compute_trial_mean_dwell_per_word,
    has_back_and_forth_xyx,
    has_back_and_forth_xyxy,
    parse_seq,
    sequence_len_literal_eval,
)
from src.analyses.correctness_associations.stats import (
    correctness_by_seq_len_threshold_test,
    correctness_by_sequence_pattern_test,
    correctness_by_trial_mean_dwell_threshold_test,
)


# -----------------------------
# Small reusable primitives
# -----------------------------


def summarize_binary_by_group(
    trial_df: pd.DataFrame, group_col: str, outcome_col: str
) -> pd.DataFrame:
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
    seq_col: str = Con.SIMPLIFIED_FIX_SEQ_BY_LABEL,
    correct_col: str = Con.IS_CORRECT_COLUMN,
) -> pd.DataFrame:
    """Collapse an IA-level frame to one row per trial, split by scan length.

    Takes an IA-level frame and returns a TRIAL-level one -- the grain any test
    on these groups must run at.
    """
    d = df[[Con.TRIAL_ID, Con.PARTICIPANT_ID, seq_col, correct_col]].copy()
    d[correct_col] = d[correct_col].astype(int)
    d["_seq_len"] = d[seq_col].apply(sequence_len_literal_eval)

    trial_df = d.groupby([Con.TRIAL_ID, Con.PARTICIPANT_ID], as_index=False).agg(
        seq_len=("_seq_len", "first"), is_correct=(correct_col, "first")
    )

    trial_df["group"] = np.where(
        trial_df["seq_len"] > threshold, f"> {threshold}", f"≤ {threshold}"
    )
    return trial_df


# Label sequence, not location, throughout this module and its callers: XYX is
# about which *answers* a reader flips between, so the answer letter is the
# meaningful token. Switched from SIMPLIFIED_FIX_SEQ_BY_LOCATION 2026-10-06
# (Diana). Verified free before changing: label and location are a per-trial
# bijection, so across all 19,436 L1 trials the two give identical has_xyx,
# has_xyxy, longest_alternating_answer_run and sequence length -- 0 trials
# differ. The strategy features stay on LOCATION, because the clockwise /
# counter-clockwise finding is about screen geometry, not about which answer.
def build_trial_df_for_back_and_forth_pattern(
    df: pd.DataFrame,
    seq_col: str = Con.SIMPLIFIED_FIX_SEQ_BY_LABEL,
    correct_col: str = Con.IS_CORRECT_COLUMN,
    use_xyxy: bool = False,
) -> pd.DataFrame:
    """Collapse an IA-level frame to one row per trial, split by XYX / XYXY presence.

    Returns a TRIAL-level frame. Note a trial whose sequence will not parse is
    grouped with "pattern absent" rather than flagged.
    """
    pattern_name = "XYXY" if use_xyxy else "XYX"
    pattern_fn = has_back_and_forth_xyxy if use_xyxy else has_back_and_forth_xyx

    d = df[[Con.TRIAL_ID, Con.PARTICIPANT_ID, seq_col, correct_col]].copy()
    d[correct_col] = d[correct_col].astype(int)
    d["_seq"] = d[seq_col].apply(parse_seq)
    d["_has_pattern"] = d["_seq"].apply(
        lambda s: bool(pattern_fn(s)) if s is not None else False
    )

    trial_df = d.groupby([Con.TRIAL_ID, Con.PARTICIPANT_ID], as_index=False).agg(
        has_pattern=("_has_pattern", "first"), is_correct=(correct_col, "first")
    )

    trial_df["group"] = np.where(
        trial_df["has_pattern"], f"{pattern_name} present", f"{pattern_name} absent"
    )
    return trial_df


def build_trial_df_for_mean_dwell_threshold(
    df: pd.DataFrame,
    threshold: float,
    dwell_col: str = Con.IA_DWELL_TIME,
    correct_col: str = Con.IS_CORRECT_COLUMN,
) -> pd.DataFrame:
    """Collapse an IA-level frame to one row per trial, split by mean dwell per word.

    Returns a TRIAL-level frame.
    """
    d = df[[Con.TRIAL_ID, Con.PARTICIPANT_ID, dwell_col, correct_col]].copy()
    d[correct_col] = d[correct_col].astype(int)
    d["_trial_mean_dwell"] = compute_trial_mean_dwell_per_word(d, dwell_col)

    trial_df = d.groupby([Con.TRIAL_ID, Con.PARTICIPANT_ID], as_index=False).agg(
        trial_mean_dwell=("_trial_mean_dwell", "first"),
        is_correct=(correct_col, "first"),
    )

    trial_df["group"] = np.where(
        trial_df["trial_mean_dwell"] <= threshold, f"≤ {threshold}", f"> {threshold}"
    )
    return trial_df


# -----------------------------
# Public "compute summaries + tests"
# -----------------------------


def compute_seq_len_threshold_summary(
    df: pd.DataFrame,
    threshold: int,
    seq_col: str = Con.SIMPLIFIED_FIX_SEQ_BY_LABEL,
    correct_col: str = Con.IS_CORRECT_COLUMN,
    add_significance: bool = True,
) -> Tuple[pd.DataFrame, Optional[Dict]]:
    """Accuracy above vs. below a scan-length threshold, with Wilson CIs and a Fisher test.

    Returns (summary_df, fisher_result). Both are computed on the trial-level
    frame this builds internally.
    """
    trial_df = build_trial_df_for_seq_len_threshold(df, threshold, seq_col, correct_col)

    # enforce plotting order (stable)
    order = [f"≤ {threshold}", f"> {threshold}"]
    trial_df["group"] = pd.Categorical(
        trial_df["group"], categories=order, ordered=True
    )

    summary_df = summarize_binary_by_group(
        trial_df, group_col="group", outcome_col="is_correct"
    ).sort_values("group")

    test_res = None
    if add_significance:
        test_res = correctness_by_seq_len_threshold_test(
            trial_df=trial_df, threshold=threshold
        )

    return summary_df.reset_index(drop=True), test_res


def compute_back_and_forth_pattern_summary(
    df: pd.DataFrame,
    seq_col: str = Con.SIMPLIFIED_FIX_SEQ_BY_LABEL,
    correct_col: str = Con.IS_CORRECT_COLUMN,
    use_xyxy: bool = False,
    add_significance: bool = True,
) -> Tuple[pd.DataFrame, Optional[Dict], str]:
    """Accuracy with vs. without a back-and-forth pattern, with Wilson CIs and a Fisher test.

    Returns (summary_df, fisher_result, pattern_name).
    """
    pattern_name = "XYXY" if use_xyxy else "XYX"

    trial_df = build_trial_df_for_back_and_forth_pattern(
        df, seq_col, correct_col, use_xyxy=use_xyxy
    )

    order = [f"{pattern_name} absent", f"{pattern_name} present"]
    trial_df["group"] = pd.Categorical(
        trial_df["group"], categories=order, ordered=True
    )

    summary_df = summarize_binary_by_group(trial_df, "group", "is_correct").sort_values(
        "group"
    )

    test_res = None
    if add_significance:
        test_res = correctness_by_sequence_pattern_test(trial_df=trial_df)

    return summary_df.reset_index(drop=True), test_res, pattern_name


def compute_trial_mean_dwell_threshold_summary(
    df: pd.DataFrame,
    threshold: float,
    dwell_col: str = Con.IA_DWELL_TIME,
    correct_col: str = Con.IS_CORRECT_COLUMN,
    add_significance: bool = True,
) -> Tuple[pd.DataFrame, Optional[Dict]]:
    """Accuracy above vs. below a mean-dwell threshold, with Wilson CIs and a Fisher test.

    Returns (summary_df, fisher_result).
    """
    trial_df = build_trial_df_for_mean_dwell_threshold(
        df, threshold, dwell_col, correct_col
    )

    order = [f"≤ {threshold}", f"> {threshold}"]
    trial_df["group"] = pd.Categorical(
        trial_df["group"], categories=order, ordered=True
    )

    summary_df = summarize_binary_by_group(trial_df, "group", "is_correct").sort_values(
        "group"
    )

    test_res = None
    if add_significance:
        test_res = correctness_by_trial_mean_dwell_threshold_test(
            trial_df=trial_df, threshold=threshold
        )

    return summary_df.reset_index(drop=True), test_res
