# linreg_model.py
#
# Trial-level linear-regression model for answer reading times, mirroring the
# structure of answer_correctness/models/logreg_model.py (numeric coercion +
# NaN fill + standardization, then a scikit-learn linear estimator).
#
# Regularization strength is either fixed (`model_kind="ridge"`, `alpha=...`) or
# chosen by cross-validation on the training split (`"ridge_cv"`, `"lasso_cv"`,
# `"elasticnet_cv"`). The CV variants fold by participant when `fit` is given
# `groups`, matching the participant-grouped train/test split in run.py -- a
# plain KFold would put the same reader on both sides of the inner split and
# pick an alpha tuned to leakage.
#
# The default is fixed rather than searched: `alpha=None` resolves to the value
# that search already settled on for a predictor set of that width (see the
# pre-tuned penalties below), so an ordinary run does not pay for the search.
# Use `"ridge_cv"` when the predictor set or target has changed enough that the
# recorded value should not be trusted.

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import (
    ElasticNetCV,
    LassoCV,
    LinearRegression,
    Ridge,
    RidgeCV,
)
from sklearn.model_selection import GroupKFold, KFold
from sklearn.preprocessing import StandardScaler

# Ridge alphas swept by "ridge_cv" when none are supplied. Wide and log-spaced:
# with ~50 standardized predictors the useful range is small, but the external
# EyeBench set (~520 columns) needs heavier shrinkage.
DEFAULT_RIDGE_ALPHAS: tuple[float, ...] = (
    0.01, 0.1, 1.0, 10.0, 100.0, 1_000.0, 10_000.0, 100_000.0,
)

# Number of alphas on sklearn's data-scaled path for the L1 kinds.
N_PATH_ALPHAS = 100

CV_MODEL_KINDS = ("ridge_cv", "lasso_cv", "elasticnet_cv")

# ---------------------------------------------------------------------------
# Pre-tuned penalties
#
# What "ridge_cv" (participant-grouped 5-fold) settled on for the answer-RT
# task, recorded so ordinary runs can skip the search -- it costs 5 extra fits
# per answer and lands in the same place. Reproduce with
# `run.sweep_answer_rt_alphas`, and re-tune if the predictor set changes.
#
# The two predictor sets want penalties an order of magnitude apart, so the
# default is resolved from the width of the feature matrix rather than fixed:
#   narrow (~49 cols, our span metrics + text): mean held-out R2 is flat from
#     0.01 to 100 and decays past 1000; 10 sits mid-plateau.
#   wide (~520 cols, external EyeBench + text): 100 is the best single value
#     (mean R2 0.161, against 0.159 at 1000 and 0.155 at 1).
# ---------------------------------------------------------------------------

TUNED_ALPHA_NARROW = 10.0
TUNED_ALPHA_WIDE = 100.0

# Column count at which a predictor set counts as "wide". The two real sets sit
# at ~49 and ~522, so anything in this range separates them.
WIDE_SET_MIN_FEATURES = 100


def tuned_alpha(n_features: int) -> float:
    """The pre-tuned ridge penalty for a predictor set of this width."""
    return (
        TUNED_ALPHA_WIDE if n_features >= WIDE_SET_MIN_FEATURES else TUNED_ALPHA_NARROW
    )


@dataclass
class TrialLevelLinRegModel:
    """Standardized linear / ridge regression over trial-level features."""

    name: str = "trial_level_lin_reg"
    # "linear", "ridge", or a cross-validated variant in CV_MODEL_KINDS.
    model_kind: str = "ridge"
    # Fixed strength; ignored for "linear" and the CV kinds. None -> the
    # pre-tuned value for this predictor set's width (see `tuned_alpha`),
    # resolved on fit and reported as `alpha_`.
    alpha: Optional[float] = None
    # Candidate strengths for the CV kinds. None -> DEFAULT_RIDGE_ALPHAS for
    # ridge_cv, and sklearn's own alpha path for lasso_cv / elasticnet_cv
    # (whose useful scale differs from ridge's, so ridge's grid is a poor fit).
    alphas: Optional[Sequence[float]] = None
    n_cv_folds: int = 5
    l1_ratios: Sequence[float] = (0.1, 0.5, 0.7, 0.9, 0.95, 1.0)
    fill_value: float = 0.0

    model: object = field(default=None, init=False)
    scaler_: StandardScaler = field(default=None, init=False)
    feature_cols_: list[str] = field(default_factory=list, init=False)
    # Strength actually used (the CV pick for the CV kinds), set on fit.
    alpha_: Optional[float] = field(default=None, init=False)

    @property
    def is_cv_kind(self) -> bool:
        return self.model_kind in CV_MODEL_KINDS

    def _cv_splits(self, X: pd.DataFrame, y, groups) -> list:
        """Inner-CV folds: grouped by participant when groups are available."""
        n_splits = int(self.n_cv_folds)
        if groups is None:
            return list(KFold(n_splits=n_splits, shuffle=True, random_state=0).split(X))

        groups = np.asarray(groups)
        n_groups = len(np.unique(groups))
        if n_groups < n_splits:
            raise ValueError(
                f"n_cv_folds={n_splits} exceeds the {n_groups} distinct groups "
                "available for the inner split."
            )
        return list(GroupKFold(n_splits=n_splits).split(X, y, groups=groups))

    def resolve_alpha(self, n_features: int) -> float:
        """The penalty this model will use: `alpha`, or the pre-tuned default."""
        return tuned_alpha(n_features) if self.alpha is None else float(self.alpha)

    def _new_estimator(self, X=None, y=None, groups=None):
        if self.model_kind == "linear":
            return LinearRegression()
        if self.model_kind == "ridge":
            return Ridge(alpha=self.resolve_alpha(X.shape[1]))

        if not self.is_cv_kind:
            raise ValueError(f"Unsupported model_kind: {self.model_kind}")

        cv = self._cv_splits(X, y, groups)
        if self.model_kind == "ridge_cv":
            alphas = list(self.alphas) if self.alphas is not None else list(
                DEFAULT_RIDGE_ALPHAS
            )
            return RidgeCV(alphas=alphas, cv=cv, scoring="neg_mean_squared_error")
        # For the L1 kinds an int means "this many alphas along sklearn's own
        # path", which is scaled to the data -- a better default than reusing
        # ridge's grid.
        path_alphas = list(self.alphas) if self.alphas is not None else N_PATH_ALPHAS
        if self.model_kind == "lasso_cv":
            return LassoCV(
                alphas=path_alphas,
                cv=cv,
                random_state=0,
                max_iter=10_000,
            )
        return ElasticNetCV(
            alphas=path_alphas,
            l1_ratio=list(self.l1_ratios),
            cv=cv,
            random_state=0,
            max_iter=10_000,
        )

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
        groups: Optional[Sequence] = None,
    ) -> None:
        """Fit on `train_df`; `groups` (e.g. participant ids) folds the inner CV."""
        X = self._prepare_X(train_df, fit=True, feature_cols=feature_cols)
        y = pd.to_numeric(train_df[target_col], errors="coerce")

        self.model = self._new_estimator(X=X, y=y, groups=groups)
        self.model.fit(X, y)

        if self.model_kind == "linear":
            self.alpha_ = None
        else:
            # The CV estimators expose their pick as `alpha_`; a fixed Ridge
            # does not, so fall back to the resolved value.
            self.alpha_ = float(
                getattr(self.model, "alpha_", self.resolve_alpha(X.shape[1]))
            )

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
