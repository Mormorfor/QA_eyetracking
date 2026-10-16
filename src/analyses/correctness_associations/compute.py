"""Accuracy in two groups of trials, for three ways of splitting them.

The numbers behind three of the bar charts in `plots.py`:

    by number of visits   at most N visits vs more (visits to any area, the
                          question included)
    by back-and-forth     whether the sequence of visits ends in x-y-x (or x-y-x-y)
    by mean dwell         whole-trial mean dwell time per word, at most N ms vs more

Each `build_trial_df_for_*` takes the word-level table and collapses it to one
row per trial, the grain any test on these groups needs (`pitfalls.md` §1). Each
`compute_*_summary` then gives accuracy per group with a 95% Wilson interval, and
the Fisher test from `stats.py`.

The sequence used is the one by meaning (`simpl_fix_by_label`), because a
back-and-forth is about which answers a reader moves between. Within a trial,
meaning and screen position correspond one to one, so on Study 1's 19,436 trials
the sequence by position gives the same lengths and the same XYX and XYXY
results. The opening-scan features use the sequence by position instead, because
the clockwise finding is about screen positions.
"""

from __future__ import annotations

import ast
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from src.config import columns as Con
from src.lib.stats.proportions import wilson_ci
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
# Accuracy per group
# -----------------------------


def summarize_binary_by_group(
    trial_df: pd.DataFrame, group_col: str, outcome_col: str
) -> pd.DataFrame:
    """Accuracy per group, with a 95% Wilson interval.

    Expects one row per trial; it does not check. Returns one row per group, with
    `group`, `n`, `k_correct`, `accuracy`, `ci_low` and `ci_high`. Trials with no
    group are left out; with a categorical `group_col`, an empty category still
    gets a row, with n = 0.
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
# One row per trial, in two groups
# -----------------------------


def build_trial_df_for_seq_len_threshold(
    df: pd.DataFrame,
    threshold: int,
    seq_col: str = Con.SIMPLIFIED_FIX_SEQ_BY_LABEL,
    correct_col: str = Con.IS_CORRECT_COLUMN,
) -> pd.DataFrame:
    """Collapse the word-level table to one row per trial, with its number of visits
    (`seq_len`) and its group: at most `threshold` ("≤ N") or more ("> N").

    A trial whose sequence is missing or will not parse counts as 0 visits, so it
    lands in the short group.
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


def build_trial_df_for_back_and_forth_pattern(
    df: pd.DataFrame,
    seq_col: str = Con.SIMPLIFIED_FIX_SEQ_BY_LABEL,
    correct_col: str = Con.IS_CORRECT_COLUMN,
    use_xyxy: bool = False,
) -> pd.DataFrame:
    """Collapse the word-level table to one row per trial, with whether its sequence
    ends in a back-and-forth (`has_pattern`) and its group ("XYX present" or
    "XYX absent"; XYXY with `use_xyxy`).

    XYX means the last three visits are x, y, x; XYXY that the last four are
    x, y, x, y. Only the end of the sequence is checked, and any area can be x or
    y, the question included. A trial whose sequence is missing or will not parse
    is counted as "absent", without a flag.
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
    """Collapse the word-level table to one row per trial, with its mean dwell time
    per word (`trial_mean_dwell`) and its group: at most `threshold` ("≤ N") or
    more ("> N").

    The mean is over every word on the answer screen, the question included, and
    words nobody looked at count as 0 ms.
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
# Summaries and tests, as the plots use them
# -----------------------------


def compute_seq_len_threshold_summary(
    df: pd.DataFrame,
    threshold: int,
    seq_col: str = Con.SIMPLIFIED_FIX_SEQ_BY_LABEL,
    correct_col: str = Con.IS_CORRECT_COLUMN,
    add_significance: bool = True,
) -> Tuple[pd.DataFrame, Optional[Dict]]:
    """Accuracy of trials with at most `threshold` visits vs more, with 95% Wilson
    intervals and, if `add_significance`, the Fisher test.

    Returns (summary, Fisher result or None), both from the one-row-per-trial
    frame built here. The summary's rows are in bar order: "≤ N", then "> N".
    """
    trial_df = build_trial_df_for_seq_len_threshold(df, threshold, seq_col, correct_col)

    # Bar order: short, then long.
    order = [f"≤ {threshold}", f"> {threshold}"]
    trial_df["group"] = pd.Categorical(
        trial_df["group"], categories=order, ordered=True
    )

    # Sort into bar order explicitly: as text, "> N" would come before "≤ N", and
    # the error bars and n labels are placed by row.
    summary_df = summarize_binary_by_group(
        trial_df, group_col="group", outcome_col="is_correct"
    ).sort_values("group", key=lambda s: s.map({g: i for i, g in enumerate(order)}))

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
    """Accuracy of trials whose sequence ends in a back-and-forth vs not, with 95%
    Wilson intervals and, if `add_significance`, the Fisher test.

    Returns (summary, Fisher result or None, "XYX" or "XYXY"). The summary's rows
    are in bar order: absent, then present.
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
        "group", key=lambda s: s.map({g: i for i, g in enumerate(order)})
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
    """Accuracy of trials with mean dwell per word at most `threshold` vs more, with
    95% Wilson intervals and, if `add_significance`, the Fisher test.

    Returns (summary, Fisher result or None). The summary's rows are in bar order:
    "≤ N", then "> N".
    """
    trial_df = build_trial_df_for_mean_dwell_threshold(
        df, threshold, dwell_col, correct_col
    )

    order = [f"≤ {threshold}", f"> {threshold}"]
    trial_df["group"] = pd.Categorical(
        trial_df["group"], categories=order, ordered=True
    )

    summary_df = summarize_binary_by_group(trial_df, "group", "is_correct").sort_values(
        "group", key=lambda s: s.map({g: i for i, g in enumerate(order)})
    )

    test_res = None
    if add_significance:
        test_res = correctness_by_trial_mean_dwell_threshold_test(
            trial_df=trial_df, threshold=threshold
        )

    return summary_df.reset_index(drop=True), test_res
