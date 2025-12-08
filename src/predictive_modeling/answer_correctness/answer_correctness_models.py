# answer_correctness_models.py

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Protocol, Sequence, List

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from src import constants as Con
from src.predictive_modeling.common.features import map_last_location_to_position


class AnswerCorrectnessModel(Protocol):
    """
    Minimal interface that all answer-correctness models should implement.
    """
    name: str

    def fit(self, train_df: pd.DataFrame, target_col: str) -> None:
        ...

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        ...


# ---------------------------------------------------------------------
# Baseline: majority class
# ---------------------------------------------------------------------

@dataclass
class MajorityBaselineCorrectness:
    """
    Baseline model predicting the majority class (0 or 1) from training data.
    """
    name: str = "majority_baseline"
    majority_label_: int = field(default=1, init=False)

    def fit(self, train_df: pd.DataFrame, target_col: str) -> None:
        if target_col not in train_df.columns:
            raise KeyError(f"Target column '{target_col}' not found in train_df.")

        y = train_df[target_col].astype(int)
        counts = y.value_counts()

        if counts.empty:
            raise ValueError("Training data for MajorityBaselineCorrectness is empty.")

        # Prefer the more frequent class; if tie, this still picks a consistent label
        self.majority_label_ = int(counts.idxmax())

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return np.full(len(df), self.majority_label_, dtype=int)


# ---------------------------------------------------------------------
# Area-metrics Logistic Regression
# ---------------------------------------------------------------------

@dataclass
class AreaMetricsCorrectnessLogRegModel:
    """
    Binary logistic regression for correctness (is_correct = 0/1),
    using:
      - AREA_METRIC_COLUMNS per area in AREA_LABEL_CHOICES:
            <metric>__<area_label>
      - optionally last_loc_numeric (0–3, or -1 if no answer).
    """
    name: str = "area_metrics_correctness_log_reg"
    include_last_location: bool = False

    model: LogisticRegression = field(default=None, init=False)
    feature_cols_: List[str] = field(default_factory=list, init=False)

    def _build_feature_columns(self, df: pd.DataFrame) -> List[str]:
        cols: List[str] = []

        for metric in Con.AREA_METRIC_COLUMNS:
            for area in Con.AREA_LABEL_CHOICES:
                col = f"{metric}__{area}"
                if col in df.columns:
                    cols.append(col)

        if self.include_last_location and Con.LAST_VISITED_LOCATION in df.columns:
            cols.append("last_loc_numeric")

        return cols

    def _prepare_design_matrix(self, df: pd.DataFrame, fit: bool = False) -> pd.DataFrame:
        temp = df.copy()

        if self.include_last_location and Con.LAST_VISITED_LOCATION in temp.columns:
            temp["last_loc_numeric"] = map_last_location_to_position(
                temp[Con.LAST_VISITED_LOCATION]
            ).fillna(-1).astype(int)

        if fit or not self.feature_cols_:
            self.feature_cols_ = self._build_feature_columns(temp)

        if not self.feature_cols_:
            raise ValueError(
                "No feature columns found for AreaMetricsCorrectnessLogRegModel. "
                "Check that AREA_METRIC_COLUMNS and AREA_LABEL_CHOICES "
                "match your trial-level DataFrame."
            )

        X = temp[self.feature_cols_].fillna(0.0)
        return X

    def fit(self, train_df: pd.DataFrame, target_col: str) -> None:
        if target_col not in train_df.columns:
            raise KeyError(f"Target column '{target_col}' not found in train_df.")

        X = self._prepare_design_matrix(train_df, fit=True)
        y = train_df[target_col].astype(int)

        self.model = LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
        )
        self.model.fit(X, y)

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model has not been fitted yet.")

        X = self._prepare_design_matrix(df, fit=False)
        return self.model.predict(X).astype(int)
