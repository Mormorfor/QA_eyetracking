# data_utils.py

from typing import Sequence, Tuple
import numpy as np
import pandas as pd

from src import constants as Con


def simple_train_test_split(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
    group_col: str = Con.PARTICIPANT_ID,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Group-wise train/test split (e.g. by participant_id).
    """
    df = df.copy()
    groups = df[group_col].dropna().unique()
    rng = np.random.default_rng(random_state)
    rng.shuffle(groups)

    n_groups = len(groups)
    n_test = max(1, int(round(test_size * n_groups)))

    test_groups = set(groups[:n_test])
    train_groups = set(groups[n_test:])

    train_mask = df[group_col].isin(train_groups)
    test_mask = df[group_col].isin(test_groups)

    return df[train_mask].copy(), df[test_mask].copy()



def select_feature_columns(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    allow_missing: bool = False,
) -> pd.DataFrame:
    df_cols = set(df.columns)
    missing = [c for c in feature_cols if c not in df_cols]

    if missing and not allow_missing:
        raise KeyError(f"Missing feature columns in DataFrame: {missing}")

    present_cols = [c for c in feature_cols if c in df_cols]
    return df[present_cols].copy()



def build_area_metric_pivot(
    df: pd.DataFrame,
    group_cols: Sequence[str],
    area_col: str,
    metric_cols: Sequence[str],
) -> pd.DataFrame:
    """
    Generic helper to aggregate area-level metrics per trial:

    Input: word-level df with columns:
        group_cols + [area_col] + metric_cols
    Output: one row per group_cols, columns:
        <metric>__<area_label>
    """
    cols_needed = list(group_cols) + [area_col] + list(metric_cols)
    missing = [c for c in cols_needed if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns for area metrics: {missing}")

    metrics_df = (
        df[cols_needed]
        .dropna(subset=[area_col])
        .groupby(list(group_cols) + [area_col], as_index=False)
        .agg({m: "mean" for m in metric_cols})
    )

    metrics_pivot = metrics_df.pivot_table(
        index=list(group_cols),
        columns=area_col,
        values=metric_cols,
        aggfunc="mean",
    )

    metrics_pivot.columns = [
        f"{metric}__{area_label}"
        for metric, area_label in metrics_pivot.columns.to_list()
    ]
    metrics_pivot = metrics_pivot.reset_index()
    return metrics_pivot
