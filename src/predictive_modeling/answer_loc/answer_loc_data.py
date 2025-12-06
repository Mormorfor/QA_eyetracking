# answer_loc_data.py
import numpy as np
import pandas as pd
from typing import Sequence, Tuple
from src import constants as Con


def build_trial_level_location_table(
    df: pd.DataFrame,
    group_cols: Sequence[str] = (Con.PARTICIPANT_ID, Con.TRIAL_ID),
    last_loc_col: str = Con.LAST_VISITED_LOCATION,
    target_col: str = Con.SELECTED_ANSWER_POSITION_COLUMN,
) -> pd.DataFrame:
    cols = list(group_cols) + [last_loc_col, target_col]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns for trial table: {missing}")

    trial_df = (
        df[cols]
        .drop_duplicates()
        .dropna(subset=[last_loc_col, target_col])
        .reset_index(drop=True)
    )
    return trial_df


from src import constants as Con


def build_trial_level_with_area_metrics(
    df: pd.DataFrame,
    group_cols: Sequence[str] = (Con.PARTICIPANT_ID, Con.TRIAL_ID),
    area_col: str = Con.AREA_LABEL_COLUMN,
    metric_cols: Sequence[str] = Con.AREA_METRIC_COLUMNS,
    last_loc_col: str = Con.LAST_VISITED_LOCATION,
    target_col: str = Con.SELECTED_ANSWER_POSITION_COLUMN,
) -> pd.DataFrame:
    """
    Build a trial-level table with:
      - group_cols (e.g. participant_id, TRIAL_INDEX)
      - last_loc_col
      - target_col
      - one column per (metric, area_label):
            <metric>__<area_label>
        where area_label in AREA_LABEL_CHOICES

    Assumes df is word-level, with metrics duplicated per area.
    We aggregate by (group_cols + area_col) and use mean().
    """
    trial_core = build_trial_level_location_table(
        df=df,
        group_cols=group_cols,
        last_loc_col=last_loc_col,
        target_col=target_col,
    )

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

    trial_df = trial_core.merge(metrics_pivot, on=list(group_cols), how="left")
    return trial_df





def simple_train_test_split(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
    group_col: str = Con.PARTICIPANT_ID,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
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
