"""Fisher exact tests behind the correctness-association bar charts.

Each test splits trials into two groups and asks whether accuracy differs:

    correctness_by_seq_len_threshold_test           at most N visits vs more
    correctness_by_sequence_pattern_test            ends in a back-and-forth vs not
    correctness_by_trial_mean_dwell_threshold_test  mean dwell per word: at most N ms vs more
    correctness_by_pref_group_test                  chose the gaze-preferred answer vs not

Each returns a dict with the 2x2 table of counts, the odds ratio (the odds of a
correct answer in the first group over the odds in the second) and the two-sided
p-value. Trials missing either value are left out of the table.

The tests need one row per trial (`pitfalls.md` §1). The first three check this
and stop otherwise; the preference test does not check.

The p-values are too small. Fisher's test treats every trial as an independent
draw, but trials come in clusters: by participant (360 on Study 1, 54 trials
each) and by item (972, about 20 trials each). The intraclass correlation of
correctness -- how alike two trials from the same cluster are -- is 0.041 by
participant and 0.126 by item, so the effective sample size is nearer 6,000 than
the 19,436 trials.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact

from src.config import columns as C


def _check_trial_frame(trial_df: pd.DataFrame, name: str) -> None:
    """Stop unless `trial_df` has at least one row, and one row per trial."""
    if trial_df.empty:
        raise ValueError(
            f"{name} got zero trials. Fisher on an all-zero table returns "
            "p = 1.0 rather than failing, so an empty group would be reported "
            "as a result."
        )
    # The 2x2 counts rows, so the number of rows is the sample size. The word-level
    # table would count each trial once per word on screen (about 39 times): the
    # p-value collapses, and since trials have different numbers of words the odds
    # ratio becomes word-weighted, while the table still looks plausible.
    n_dup = int(trial_df.duplicated(subset=[C.PARTICIPANT_ID, C.TRIAL_ID]).sum())
    if n_dup:
        raise ValueError(
            f"{name} expects one row per (participant, trial); found {n_dup} "
            f"duplicated keys across {len(trial_df)} rows. Pass the trial-level "
            "frame built by correctness_associations.compute.build_trial_df_for_*, "
            "not the IA-level frame."
        )


def correctness_by_seq_len_threshold_test(
    trial_df: pd.DataFrame,
    threshold: int,
    seq_len_col: str = "seq_len",
    correct_col: str = C.IS_CORRECT_COLUMN,
) -> Dict:
    """Fisher test: accuracy of trials with at most `threshold` visits vs more.

    Takes the one-row-per-trial frame from
    `compute.build_trial_df_for_seq_len_threshold` and splits it on the same
    `seq_len` column the bars are drawn from, so the test and the figure use the
    same two groups.

    Returns the 2x2 table (rows: short, long; columns: correct, incorrect), the
    odds ratio (short over long), the p-value, counts and accuracy per group, and
    `delta_accuracy` (long minus short).
    """
    _check_trial_frame(trial_df, "correctness_by_seq_len_threshold_test")

    sub = trial_df[[seq_len_col, correct_col]].dropna().copy()
    sub[correct_col] = sub[correct_col].astype(int)
    sub["_is_long"] = sub[seq_len_col] > threshold

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
    trial_df: pd.DataFrame,
    has_pattern_col: str = "has_pattern",
    correct_col: str = C.IS_CORRECT_COLUMN,
) -> Dict:
    """Fisher test: accuracy of trials whose sequence ends in a back-and-forth vs not.

    Takes the one-row-per-trial frame from
    `compute.build_trial_df_for_back_and_forth_pattern`, which has already decided
    XYX or XYXY, and reads the same `has_pattern` column the bars are drawn from,
    so the test and the figure cannot disagree about the pattern.

    Returns the 2x2 table (rows: present, absent), the odds ratio (present over
    absent), the p-value, counts and accuracy per group, and `delta_accuracy`
    (present minus absent).
    """
    _check_trial_frame(trial_df, "correctness_by_sequence_pattern_test")

    sub = trial_df[[has_pattern_col, correct_col]].dropna().copy()
    sub[correct_col] = sub[correct_col].astype(int)
    # A real boolean column, so that `~` below means "not". On an object column it
    # would invert integers instead (~True is -2).
    sub["_has_pattern"] = sub[has_pattern_col].astype(bool)

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
    trial_df: pd.DataFrame,
    threshold: float,
    mean_dwell_col: str = "trial_mean_dwell",
    correct_col: str = C.IS_CORRECT_COLUMN,
) -> Dict:
    """Fisher test: accuracy of trials with mean dwell per word at most `threshold`
    vs more.

    Takes the one-row-per-trial frame from
    `compute.build_trial_df_for_mean_dwell_threshold`, where `trial_mean_dwell` is
    the trial's total dwell time divided by its number of words on screen.

    Returns the 2x2 table (rows: low, high), the odds ratio (low over high), the
    p-value and the counts per group; no accuracies.
    """
    _check_trial_frame(trial_df, "correctness_by_trial_mean_dwell_threshold_test")

    d = trial_df[[mean_dwell_col, correct_col]].dropna().copy()
    d[correct_col] = d[correct_col].astype(int)

    low = d[mean_dwell_col] <= threshold
    high = d[mean_dwell_col] > threshold

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



# ---------------------------------------------------------------------------
# Preference matching
# ---------------------------------------------------------------------------

def correctness_by_pref_group_test(
    df: pd.DataFrame,
    pref_col: str = "pref_group",
    correct_col: str = C.IS_CORRECT_COLUMN,
) -> Dict:
    """Fisher test: accuracy of trials where the chosen answer was the gaze-preferred
    one ("matching") vs not ("not_matching").

    Expects one row per trial, as `features.preference.compute_trial_matching`
    returns it; unlike the other tests here, it does not check. Rows with any other
    value in `pref_col` are not counted.

    Returns the 2x2 table (rows: matching, not matching), the odds ratio (matching
    over not matching), the p-value and the counts per group.
    """
    sub = df[[pref_col, correct_col]].dropna().copy()
    sub[correct_col] = sub[correct_col].astype(int)

    a = int(((sub[pref_col] == "matching") & (sub[correct_col] == 1)).sum())
    b = int(((sub[pref_col] == "matching") & (sub[correct_col] == 0)).sum())
    c = int(((sub[pref_col] == "not_matching") & (sub[correct_col] == 1)).sum())
    d = int(((sub[pref_col] == "not_matching") & (sub[correct_col] == 0)).sum())

    table = np.array([[a, b],
                      [c, d]], dtype=int)

    odds_ratio, p_value = fisher_exact(table)

    return {
        "contingency_table": table,
        "odds_ratio": float(odds_ratio),
        "p_value": float(p_value),
        "counts": {
            "matching": {"correct": a, "incorrect": b},
            "not_matching": {"correct": c, "incorrect": d},
        },
    }
