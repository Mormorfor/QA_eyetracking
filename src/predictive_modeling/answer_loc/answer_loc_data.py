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
