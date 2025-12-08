# answer_correctness_data.py

from typing import Sequence
import pandas as pd

from src import constants as Con
from src.predictive_modeling.common.data_utils import build_area_metric_pivot


def build_trial_level_with_area_metrics_for_correctness(
    df: pd.DataFrame,
    group_cols: Sequence[str] = (Con.PARTICIPANT_ID, Con.TRIAL_ID),
    area_col: str = Con.AREA_LABEL_COLUMN,
    metric_cols: Sequence[str] = Con.AREA_METRIC_COLUMNS,
    last_loc_col: str = Con.LAST_VISITED_LOCATION,
) -> pd.DataFrame:
    core_cols = list(group_cols) + [
        last_loc_col,
        Con.SELECTED_ANSWER_POSITION_COLUMN,
        Con.CORRECT_ANSWER_POSITION_COLUMN,
    ]
    missing_core = [c for c in core_cols if c not in df.columns]
    if missing_core:
        raise KeyError(f"Missing required columns for trial core: {missing_core}")

    trial_core = (
        df[core_cols]
        .drop_duplicates()
        .dropna(
            subset=[
                last_loc_col,
                Con.SELECTED_ANSWER_POSITION_COLUMN,
                Con.CORRECT_ANSWER_POSITION_COLUMN,
            ]
        )
        .reset_index(drop=True)
    )

    trial_core[Con.IS_CORRECT_COLUMN] = (
        trial_core[Con.SELECTED_ANSWER_POSITION_COLUMN]
        == trial_core[Con.CORRECT_ANSWER_POSITION_COLUMN]
    ).astype(int)

    metrics_pivot = build_area_metric_pivot(
        df=df,
        group_cols=group_cols,
        area_col=area_col,
        metric_cols=metric_cols,
    )

    trial_df = trial_core.merge(metrics_pivot, on=list(group_cols), how="left")
    return trial_df
