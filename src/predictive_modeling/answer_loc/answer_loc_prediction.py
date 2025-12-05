import numpy as np
import pandas as pd
from typing import Sequence, Tuple, Dict, Protocol
from dataclasses import dataclass

from src import constants as Con


# ---------------------------------------------------------------------------
# Dataset preparation
# ---------------------------------------------------------------------------

def build_trial_level_location_table(
    df: pd.DataFrame,
    group_cols: Sequence[str] = (Con.PARTICIPANT_ID, Con.TRIAL_ID),
    last_loc_col: str = Con.LAST_VISITED_LOCATION,
    target_col: str = Con.SELECTED_ANSWER_POSITION_COLUMN,
) -> pd.DataFrame:
    """
    One row per trial + target + simple baseline feature(s).

    Columns:
      - group_cols (e.g. participant_id, TRIAL_INDEX)
      - last_loc_col (e.g. last_area_visited_loc)
      - target_col  (e.g. selected_a_screen_loc)
    """
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


# ---------------------------------------------------------------------------
# Train–test splitting (dummy 80/20, subject-wise)
# ---------------------------------------------------------------------------

def simple_train_test_split(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
    group_col: str = Con.PARTICIPANT_ID,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Dummy train-test split (group-wise, e.g. by participant).
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

    train_df = df[train_mask].copy()
    test_df = df[test_mask].copy()
    return train_df, test_df



# ---------------------------------------------------------------------------
# Feature selection helper
# ---------------------------------------------------------------------------

def select_feature_columns(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    allow_missing: bool = False,
) -> pd.DataFrame:
    """
    Very simple feature selector
    """
    df_cols = set(df.columns)
    missing = [c for c in feature_cols if c not in df_cols]

    if missing and not allow_missing:
        raise KeyError(f"Missing feature columns in DataFrame: {missing}")

    present_cols = [c for c in feature_cols if c in df_cols]
    return df[present_cols].copy()





# ---------------------------------------------------------------------------
# Model interface
# ---------------------------------------------------------------------------

class AnswerLocationModel(Protocol):
    """
    Minimal interface that all answer-location models should implement.
    """
    name: str
    def fit(self, train_df: pd.DataFrame, target_col: str) -> None:
        """
        Fit the model on trial-level data.

        train_df: trial-level table (one row per trial).
        target_col: column containing the true labels.
        """
        ...

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        Predict locations for the given trial-level DataFrame.
        Returns a 1D array of predictions aligned with df.index.
        """
        ...


# ---------------------------------------------------------------------------
# Baseline model: last visited location
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Helper: map location labels to numeric answer positions
# ---------------------------------------------------------------------------

def map_last_location_to_position(
    series: pd.Series,
    choices: Sequence[str] = Con.AREA_LABEL_CHOICES
) -> pd.Series:
    """
    Map last_area_visited_loc values to numeric positions 0–3.

    AREA_LABEL_CHOICES defines the order:
        index 1 → answer position 0
        index 2 → answer position 1
        index 3 → answer position 2
        index 4 → answer position 3

    Any value not matching the answer_* labels becomes NaN.
    """
    mapping = {}
    for idx, label in enumerate(choices):
        if idx == 0:
            continue
        mapping[label] = idx - 1
    return series.map(mapping)



@dataclass
class LastLocationBaseline:
    name: str = "last_location"
    last_loc_col: str = Con.LAST_VISITED_LOCATION

    def fit(self, train_df: pd.DataFrame, target_col: str) -> None:
        return

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        if self.last_loc_col not in df.columns:
            raise KeyError(f"Column '{self.last_loc_col}' not found in df.")

        loc_series = df[self.last_loc_col]
        pos_series = map_last_location_to_position(loc_series)

        return pos_series.fillna(-1).astype(int).to_numpy()





# ---------------------------------------------------------------------------
# Experiment runner
# ---------------------------------------------------------------------------

def evaluate_models_on_answer_location(
    df: pd.DataFrame,
    models: Sequence[AnswerLocationModel],
    group_cols: Sequence[str] = (Con.PARTICIPANT_ID, Con.TRIAL_ID),
    split_group_col: str = Con.PARTICIPANT_ID,
    last_loc_col: str = Con.LAST_VISITED_LOCATION,
    target_col: str = Con.SELECTED_ANSWER_POSITION_COLUMN,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Dict[str, Dict[str, object]]:
    """
    High-level evaluation pipeline for answer-location prediction.

    Steps:
      1. Build trial-level table (one row per trial).
      2. Train-test split (group-wise, by split_group_col).
      3. For each model:
           - fit on train trials
           - predict on test trials
           - compute accuracy

    Returns
    -------
    results : dict
        results[model.name] = {
            "train_df": train_df,
            "test_df": test_df,
            "y_true":  y_true,
            "y_pred":  y_pred,
            "accuracy": float,
            "n_test": int,
        }
    """
    trial_df = build_trial_level_location_table(
        df,
        group_cols=group_cols,
        last_loc_col=last_loc_col,
        target_col=target_col,
    )

    if trial_df.empty:
        raise ValueError("Trial-level table is empty; cannot evaluate models.")

    train_df, test_df = simple_train_test_split(
        trial_df,
        test_size=test_size,
        random_state=random_state,
        group_col=split_group_col,
    )

    y_true = test_df[target_col].to_numpy()

    results: Dict[str, Dict[str, object]] = {}

    for model in models:
        model.fit(train_df, target_col=target_col)

        y_pred = model.predict(test_df)
        if len(y_pred) != len(test_df):
            raise ValueError(
                f"Model '{model.name}' returned {len(y_pred)} predictions "
                f"for {len(test_df)} test trials."
            )

        acc = float((y_true == y_pred).mean())

        results[model.name] = {
            "train_df": train_df,
            "test_df": test_df,
            "y_true": y_true,
            "y_pred": y_pred,
            "accuracy": acc,
            "n_test": int(len(test_df)),
        }

    return results



