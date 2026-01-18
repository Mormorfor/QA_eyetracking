# answer_loc_data.py

from typing import Sequence, Tuple
import numpy as np
import pandas as pd

from src import constants as Con
from src.predictive_modeling.common.data_utils import (
    group_vise_train_test_split,
    select_feature_columns,
    build_area_metric_pivot,
)


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


def build_trial_level_with_area_metrics(
    df: pd.DataFrame,
    group_cols: Sequence[str] = (Con.PARTICIPANT_ID, Con.TRIAL_ID),
    area_col: str = Con.AREA_LABEL_COLUMN,
    metric_cols: Sequence[str] = Con.AREA_METRIC_COLUMNS,
    last_loc_col: str = Con.LAST_VISITED_LOCATION,
    target_col: str = Con.SELECTED_ANSWER_POSITION_COLUMN,
) -> pd.DataFrame:
    """
    Trial-level table + area metrics, for location prediction.
    """
    trial_core = build_trial_level_location_table(
        df=df,
        group_cols=group_cols,
        last_loc_col=last_loc_col,
        target_col=target_col,
    )

    metrics_pivot = build_area_metric_pivot(
        df=df,
        group_cols=group_cols,
        area_col=area_col,
        metric_cols=metric_cols,
    )

    trial_df = trial_core.merge(metrics_pivot, on=list(group_cols), how="left")
    return trial_df
