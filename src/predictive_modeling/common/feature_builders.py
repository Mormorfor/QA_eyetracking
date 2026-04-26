# feature_builders.py

from typing import Sequence, Tuple, List, Optional
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from src import constants as Con
from src.constants import TRIAL_ID_COLS

#--------------------------------
# Helpers
#--------------------------------

def select_feature_columns(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
) -> pd.DataFrame:

    df_cols = set(df.columns)
    present_cols = [c for c in feature_cols if c in df_cols]
    return df[present_cols].copy()



def build_area_metric_pivot(
    df: pd.DataFrame,
    area_col: str,
    metric_cols: Sequence[str],
) -> pd.DataFrame:
    """
    Collapse already-aggregated area-level metrics into one row per group,
    pivoting areas into feature columns.

    Input: word-level df with columns:
        group_cols + [area_col] + metric_cols
    Output: one row per group_cols, columns:
        <metric>__<area_label>
    """
    cols_needed = list(TRIAL_ID_COLS) + [area_col] + list(metric_cols)
    metrics_df = (
        df[cols_needed]
        .dropna(subset=[area_col])
        .groupby(list(TRIAL_ID_COLS) + [area_col], as_index=False)
        .agg({m: "first" for m in metric_cols})
    )

    metrics_pivot = metrics_df.pivot_table(
        index=list(TRIAL_ID_COLS),
        columns=area_col,
        values=metric_cols,
        aggfunc="first",
    )

    metrics_pivot.columns = [
        f"{metric}__{area_label}"
        for metric, area_label in metrics_pivot.columns
    ]
    metrics_pivot = metrics_pivot.reset_index()
    return metrics_pivot



def add_answer_correct_wrong_contrast_columns(
    df_pivot: pd.DataFrame,
    metric_cols: Sequence[str],
    sep: str = "__",
    correct_label: str = "answer_A",
    wrong_labels: Sequence[str] = ("answer_B", "answer_C", "answer_D"),
    out_correct_suffix: str = Con.CORRECT_SUFFIX,
    out_wrong_mean_suffix: str = Con.WRONG_MEAN_SUFFIX,
    out_contrast_suffix: str = Con.CONTRAST_SUFFIX,
    out_distance_furthest_suffix: str = Con.DISTANCE_FURTHEST_SUFFIX,
    out_distance_closest_suffix: str = Con.DISTANCE_CLOSEST_SUFFIX,
) -> pd.DataFrame:
    """
    Given a pivoted trial-level dataframe that already contains columns like:
        <metric>__answer_A, <metric>__answer_B, <metric>__answer_C, <metric>__answer_D

    add:
        <metric>__correct
        <metric>__wrong_mean
        <metric>__contrast
        <metric>__distance_furthest
        <metric>__distance_closest

    where:
        correct             = value of the correct answer
        wrong_mean          = mean value across wrong answers
        contrast            = correct - wrong_mean
        distance_furthest   = max absolute distance between correct and any wrong answer
        distance_closest    = min absolute distance between correct and any wrong answer

    Does not drop any columns.
    Assumes columns exist.
    """
    base = df_pivot.copy()
    new_cols = {}

    for metric in metric_cols:
        a = f"{metric}{sep}{correct_label}"
        bs = [f"{metric}{sep}{lbl}" for lbl in wrong_labels]

        out_correct = f"{metric}{sep}{out_correct_suffix}"
        out_wrong_mean = f"{metric}{sep}{out_wrong_mean_suffix}"
        out_contrast = f"{metric}{sep}{out_contrast_suffix}"
        out_distance_furthest = f"{metric}{sep}{out_distance_furthest_suffix}"
        out_distance_closest = f"{metric}{sep}{out_distance_closest_suffix}"

        correct_vals = pd.to_numeric(base[a], errors="coerce")
        wrong_vals = base[bs].apply(pd.to_numeric, errors="coerce")

        wrong_mean = wrong_vals.mean(axis=1)
        contrast = correct_vals - wrong_mean
        abs_diffs = wrong_vals.sub(correct_vals, axis=0).abs()

        new_cols[out_correct] = correct_vals
        new_cols[out_wrong_mean] = wrong_mean
        new_cols[out_contrast] = contrast
        new_cols[out_distance_furthest] = abs_diffs.max(axis=1)
        new_cols[out_distance_closest] = abs_diffs.min(axis=1)

    derived_df = pd.DataFrame(new_cols, index=base.index)
    out = pd.concat([base, derived_df], axis=1)

    return out

def build_trial_level_constant_numeric_features(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
) -> pd.DataFrame:
    """
    Collapse numeric columns that are constant within a trial-participant pair
    into one row per trial.

    Returns:
        one row per trial with columns:
            TRIAL_ID_COLS + feature_cols
    """
    feature_cols = list(feature_cols)
    cols_needed = list(TRIAL_ID_COLS) + feature_cols

    out = (
        df[cols_needed]
        .groupby(list(TRIAL_ID_COLS), as_index=False)
        .agg({c: "first" for c in feature_cols})
    )

    for c in feature_cols:
        out[c] = pd.to_numeric(out[c], errors="coerce")

    return out


def build_trial_level_categorical_feature(
    df: pd.DataFrame,
    feature_col: str,
    prefix: Optional[str] = None,
    drop_first: bool = False,
    dummy_na: bool = False,
) -> pd.DataFrame:
    """
    Build one-hot encoded trial-level features from a categorical column
    that should be constant within each trial.

    Returns:
        one row per trial with columns:
            group_cols + dummy columns
    """
    prefix = prefix or feature_col

    cols_needed = list(TRIAL_ID_COLS) + [feature_col]
    d = df[cols_needed].copy()

    trial_cat = (
        d.groupby(list(TRIAL_ID_COLS), as_index=False)[feature_col]
        .first()
    )

    dummies = pd.get_dummies(
        trial_cat[feature_col],
        prefix=prefix,
        drop_first=drop_first,
        dummy_na=dummy_na,
    ).astype(int)

    out = pd.concat([trial_cat[list(TRIAL_ID_COLS)], dummies], axis=1)
    return out
