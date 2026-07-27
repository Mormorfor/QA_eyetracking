# linreg_model.py
#
# Trial-level linear-regression model for answer reading times, mirroring the
# structure of answer_correctness/models/logreg_model.py (numeric coercion +
# NaN fill + standardization, then a scikit-learn linear estimator).

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler


@dataclass
class TrialLevelLinRegModel:
    """Standardized linear / ridge regression over trial-level features."""

    name: str = "trial_level_lin_reg"
    model_kind: str = "ridge"  # "ridge" or "linear"
    alpha: float = 1.0  # ridge regularization strength (ignored for "linear")
    fill_value: float = 0.0

    model: object = field(default=None, init=False)
    scaler_: StandardScaler = field(default=None, init=False)
    feature_cols_: list[str] = field(default_factory=list, init=False)

    def _new_estimator(self):
        if self.model_kind == "linear":
            return LinearRegression()
        if self.model_kind == "ridge":
            return Ridge(alpha=self.alpha)
        raise ValueError(f"Unsupported model_kind: {self.model_kind}")

    def _validate_feature_cols(
        self,
        df: pd.DataFrame,
        feature_cols: Sequence[str],
    ) -> list[str]:
        cols = list(feature_cols)
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise KeyError(f"Missing feature columns: {missing}")
        return cols

    def _prepare_X(
        self,
        df: pd.DataFrame,
        *,
        fit: bool,
        feature_cols: Optional[Sequence[str]] = None,
    ) -> pd.DataFrame:
        if feature_cols is None:
            if not self.feature_cols_:
                raise ValueError("feature_cols must be provided on first fit.")
            cols = list(self.feature_cols_)
        else:
            cols = self._validate_feature_cols(df, feature_cols)

        X = df[cols].copy()
        for c in X.columns:
            X[c] = pd.to_numeric(X[c], errors="coerce")
        X = X.fillna(self.fill_value)

        if fit:
            self.feature_cols_ = list(cols)
            self.scaler_ = StandardScaler()
            X_scaled = self.scaler_.fit_transform(X)
        else:
            if self.scaler_ is None:
                raise RuntimeError("Scaler has not been fitted.")
            X_scaled = self.scaler_.transform(X)

        return pd.DataFrame(X_scaled, columns=cols, index=df.index)

    def fit(
        self,
        train_df: pd.DataFrame,
        target_col: str,
        feature_cols: Sequence[str],
    ) -> None:
        X = self._prepare_X(train_df, fit=True, feature_cols=feature_cols)
        y = pd.to_numeric(train_df[target_col], errors="coerce")

        self.model = self._new_estimator()
        self.model.fit(X, y)

    def predict(
        self,
        df: pd.DataFrame,
        feature_cols: Optional[Sequence[str]] = None,
    ) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model has not been fitted yet.")
        X = self._prepare_X(df, fit=False, feature_cols=feature_cols)
        return self.model.predict(X)

    def get_coef_summary(self, top_k: Optional[int] = None) -> pd.DataFrame:
        """Standardized coefficients (features are z-scored), sorted by |coef|."""
        if self.model is None:
            raise RuntimeError("Model has not been fitted yet.")

        out = pd.DataFrame(
            {
                "feature": self.feature_cols_,
                "coef": np.asarray(self.model.coef_).ravel(),
            }
        )
        out["abs_coef"] = out["coef"].abs()
        out = out.sort_values("abs_coef", ascending=False).reset_index(drop=True)
        if top_k is not None:
            out = out.head(int(top_k))
        return out
