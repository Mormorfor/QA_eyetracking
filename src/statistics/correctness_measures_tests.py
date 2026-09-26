# src/statistics/correctness_measures_tests.py

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact

from src import constants as C


# One assumption remains after the grain fix, and it is a known, accepted one:
# Fisher treats the rows of its 2x2 as independent draws, and trials are not --
# they are nested in 360 participants (54 each) and 972 items (20 each). Measured
# 2026-09-26: ICC 0.041 by participant, 0.126 by item, so the effective n is
# nearer 6,000 than 19,436 and every p below is too small. A participant-clustered
# bootstrap moves no conclusion (todo.md T3.22 carries the numbers). Deliberately
# PUSHED past the restructure, which gives clustered inference a single home in
# modeling/inference.py rather than four copies.


def _check_trial_frame(trial_df: pd.DataFrame, name: str) -> None:
    """Refuse anything that is not one row per trial.

    These tests build a 2x2 of counts, so the grain *is* the sample size. Handed
    an IA-level frame, each trial is counted once per word on screen (~39x on
    L1), which collapses the p-value and -- because word counts differ by trial
    -- silently makes the odds ratio word-weighted. The table still looks
    entirely plausible, which is the failure mode conventions.md asks to assert
    against rather than absorb.
    """
    if trial_df.empty:
        raise ValueError(
            f"{name} got zero trials. Fisher on an all-zero table returns "
            "p = 1.0 rather than failing, so an empty group would be reported "
            "as a result."
        )
    n_dup = int(trial_df.duplicated(subset=[C.PARTICIPANT_ID, C.TRIAL_ID]).sum())
    if n_dup:
        raise ValueError(
            f"{name} expects one row per (participant, trial); found {n_dup} "
            f"duplicated keys across {len(trial_df)} rows. Pass the trial-level "
            "frame built by derived.correctness_measures.build_trial_df_for_*, "
            "not the IA-level frame."
        )


def correctness_by_seq_len_threshold_test(
    trial_df: pd.DataFrame,
    threshold: int,
    seq_len_col: str = "seq_len",
    correct_col: str = C.IS_CORRECT_COLUMN,
) -> Dict:
    """
    Fisher exact test comparing correctness between:
      - seq_len <= threshold
      - seq_len > threshold

    Takes the TRIAL-level frame from
    ``derived.correctness_measures.build_trial_df_for_seq_len_threshold``, which
    has already parsed the sequence and collapsed to one row per trial. The
    split is read off the same ``seq_len`` column the bars are drawn from, so
    the test and the figure cannot describe different partitions.

    Returns dict with:
      - contingency_table (2x2)
      - odds_ratio
      - p_value
      - counts
      - accuracies
      - delta_accuracy (long - short)
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
    """
    Fisher exact test comparing correctness between:
      - pattern present
      - pattern absent

    Takes the TRIAL-level frame from
    ``derived.correctness_measures.build_trial_df_for_back_and_forth_pattern``,
    which has already applied the XYX / XYXY predicate. ``pattern_fn`` is
    therefore no longer a parameter -- the test reads the same ``has_pattern``
    column the bars are drawn from, so the two cannot disagree about which
    pattern was tested.
    """
    _check_trial_frame(trial_df, "correctness_by_sequence_pattern_test")

    sub = trial_df[[has_pattern_col, correct_col]].dropna().copy()
    sub[correct_col] = sub[correct_col].astype(int)
    # astype(bool) is load-bearing under pandas 3: the builder's .apply() over an
    # object column infers the result dtype, and `&` against a boolean mask then
    # raises unless this is a real boolean column.
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
    """
    Fisher exact test comparing correctness between trials with
    low vs high mean dwell time per word across the entire trial.

    Takes the TRIAL-level frame from
    ``derived.correctness_measures.build_trial_df_for_mean_dwell_threshold``,
    where ``trial_mean_dwell`` is already sum(IA_DWELL_TIME) / n_words per
    trial. Recomputing it here is what used to keep this test IA-level:
    ``transform`` broadcasts the per-trial mean back onto every word, so the
    frame read as collapsed while still carrying ~39 rows per trial.
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