# answer_correctness_models.py

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Protocol, Sequence, List

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


from src import constants as Con
from src.predictive_modeling.common.data_utils import get_coef_summary as summary

class AnswerCorrectnessModel(Protocol):
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
    majority_label_: int = None

    def fit(self, train_df: pd.DataFrame, target_col: str) -> None:
        y = train_df[target_col].astype(int)
        counts = y.value_counts()
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
      - AREA_METRIC_COLUMNS per area in ANSWER_LABEL_CHOICES:
            <metric>__<area_label>
    """
    name: str = "area_metrics_correctness_log_reg"
    model: LogisticRegression = field(default=None, init=False)
    scaler_: StandardScaler = field(default=None, init=False)
    feature_cols_: List[str] = field(default_factory=list, init=False)

    def _build_feature_columns(self, df: pd.DataFrame) -> List[str]:
        cols: List[str] = []
        for metric in Con.AREA_METRIC_COLUMNS_MODELING:
            for area in Con.LABEL_CHOICES:
                col = f"{metric}__{area}"
                if col in df.columns:
                    cols.append(col)

        return cols


    def _prepare_X(self, df: pd.DataFrame, fit: bool = False) -> pd.DataFrame:
        temp = df.copy()
        if fit or not self.feature_cols_:
            self.feature_cols_ = self._build_feature_columns(temp)

        X = df[self.feature_cols_].copy()
        for c in X.columns:
            X[c] = pd.to_numeric(X[c], errors="coerce")
        X = X.fillna(0.0)

        if fit:
            self.scaler_ = StandardScaler()
            X_scaled = self.scaler_.fit_transform(X)
        else:
            if self.scaler_ is None:
                raise RuntimeError("Scaler has not been fitted.")
            X_scaled = self.scaler_.transform(X)

        return pd.DataFrame(X_scaled, columns=self.feature_cols_, index=df.index)


    def get_coef_summary(
            self,
            train_df: pd.DataFrame,
            top_k: int = None,
    ) -> pd.DataFrame:
        """
        Return a coefficient summary table for the fitted LogisticRegression model.
        """
        if self.model is None:
            raise RuntimeError("Model has not been fitted yet.")

        X = self._prepare_X(train_df, fit=False)
        out = summary(self.model, self.feature_cols_, top_k)
        return out


    def fit(self, train_df: pd.DataFrame, target_col: str) -> None:
        X = self._prepare_X(train_df, fit=True)
        y = train_df[target_col].astype(int)

        self.model = LogisticRegression(
            max_iter=100000,
            class_weight="balanced",
        )
        self.model.fit(X, y)


    def predict(self, df: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model has not been fitted yet.")

        X = self._prepare_X(df, fit=False)
        return self.model.predict(X).astype(int)



# ---------------------------------------------------------------------
# Derived-metrics Logistic Regression
# ---------------------------------------------------------------------
@dataclass
class DerivedFeaturesCorrectnessLogRegModel:
    """
    Logistic regression on derived trial-level features:
      - seq_len
      - has_xyx
      - has_xyxy
      - trial_mean_dwell
      - plus ALL columns starting with 'pref_matching__'
    """
    name: str = "derived_features_correctness_log_reg"
    model: LogisticRegression = field(default=None, init=False)
    scaler_: StandardScaler = field(default=None, init=False)
    base_feature_cols_: List[str] = field(default_factory=lambda: [
        "seq_len", "has_xyx", "has_xyxy", "trial_mean_dwell"], init=False)
    feature_cols_: List[str] = field(default_factory=list, init=False)


    def _build_feature_cols(self, df: pd.DataFrame) -> List[str]:
        pref_cols = [c for c in df.columns if c.startswith("pref_matching__")]
        return self.base_feature_cols_ + sorted(pref_cols)

    def _prepare_X(self, df: pd.DataFrame, fit: bool = False) -> pd.DataFrame:
        if fit or not self.feature_cols_:
            self.feature_cols_ = self._build_feature_cols(df)

        X = df[self.feature_cols_].copy()
        for c in X.columns:
            X[c] = pd.to_numeric(X[c], errors="coerce")
        X = X.fillna(0.0)

        if fit:
            self.scaler_ = StandardScaler()
            X_scaled = self.scaler_.fit_transform(X)
        else:
            if self.scaler_ is None:
                raise RuntimeError("Scaler has not been fitted.")
            X_scaled = self.scaler_.transform(X)

        return pd.DataFrame(X_scaled, columns=self.feature_cols_, index=df.index)


    def get_coef_summary(
            self,
            train_df: pd.DataFrame,
            top_k: int = None,
    ) -> pd.DataFrame:
        """
        Return a coefficient summary table for the fitted LogisticRegression model.
        """
        if self.model is None:
            raise RuntimeError("Model has not been fitted yet.")

        X = self._prepare_X(train_df, fit=False)
        out = summary(self.model, self.feature_cols_, top_k)
        return out


    def fit(self, train_df: pd.DataFrame, target_col: str) -> None:
        X = self._prepare_X(train_df, fit=True)
        y = train_df[target_col].astype(int)

        self.model = LogisticRegression(
            max_iter=100000,
            class_weight="balanced",
        )
        self.model.fit(X, y)


    def predict(self, df: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model has not been fitted yet.")

        X = self._prepare_X(df, fit=False)
        return self.model.predict(X).astype(int)



# ---------------------------------------------------------------------
# Derived-metrics + Area-metrics Logistic Regression
# ---------------------------------------------------------------------
@dataclass
class FullFeaturesCorrectnessLogRegModel:
    """
    Logistic regression using:
      - area-metric features (<metric>__<area>)
      - derived features (seq_len, has_xyx, has_xyxy, trial_mean_dwell)
      - all pref_matching__* features
    """
    name: str = "full_features_correctness_log_reg"
    model: LogisticRegression = field(default=None, init=False)
    scaler_: StandardScaler = field(default=None, init=False)
    feature_cols_: List[str] = field(default_factory=list, init=False)

    def _build_feature_cols(self, df: pd.DataFrame) -> List[str]:
        cols: List[str] = []

        for metric in Con.AREA_METRIC_COLUMNS_MODELING:
            for area in Con.LABEL_CHOICES:
                col = f"{metric}__{area}"
                if col in df.columns:
                    cols.append(col)

        derived_base = [
            "seq_len",
            "has_xyx",
            "has_xyxy",
            "trial_mean_dwell",
        ]
        cols.extend([c for c in derived_base if c in df.columns])
        cols.extend(sorted(c for c in df.columns if c.startswith("pref_matching__")))

        return cols


    def _prepare_X(self, df: pd.DataFrame, fit: bool = False) -> pd.DataFrame:
        if fit or not self.feature_cols_:
            self.feature_cols_ = self._build_feature_cols(df)

        X = df[self.feature_cols_].copy()
        for c in X.columns:
            X[c] = pd.to_numeric(X[c], errors="coerce")
        X = X.fillna(0.0)

        if fit:
            self.scaler_ = StandardScaler()
            X_scaled = self.scaler_.fit_transform(X)
        else:
            if self.scaler_ is None:
                raise RuntimeError("Scaler has not been fitted.")
            X_scaled = self.scaler_.transform(X)

        return pd.DataFrame(X_scaled, columns=self.feature_cols_, index=df.index)


    def get_coef_summary(
            self,
            train_df: pd.DataFrame,
            top_k: int = None,
    ) -> pd.DataFrame:
        """
        Return a coefficient summary table for the fitted LogisticRegression model.
        """
        if self.model is None:
            raise RuntimeError("Model has not been fitted yet.")
        X = self._prepare_X(train_df, fit=False)
        out = summary(self.model, self.feature_cols_, top_k)
        return out


    def fit(self, train_df: pd.DataFrame, target_col: str) -> None:
        X = self._prepare_X(train_df, fit=True)
        y = train_df[target_col].astype(int)

        self.model = LogisticRegression(
            max_iter=100000,
            class_weight="balanced",
        )
        self.model.fit(X, y)


    def predict(self, df: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model has not been fitted yet.")

        X = self._prepare_X(df, fit=False)
        return self.model.predict(X).astype(int)

