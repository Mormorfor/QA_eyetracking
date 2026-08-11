# gbm_model.py
#
# Gradient-boosted trees for the answer reading-time regression, exposing the
# same fit/predict/get_coef_summary surface as TrialLevelLinRegModel so it drops
# into `run.evaluate_answer_rt_model` unchanged.
#
# Why this model: the ridge answers "is there a linear signal"; this answers "is
# there a nonlinear one". It can express interactions the linear model cannot --
# most obviously that hunters and gatherers may have different RT functions
# rather than just different intercepts -- and it consumes NaN natively, so the
# `fillna(0)-before-standardizing` step that turns a missing value into a fake
# z-score of -(mean/sd) does not happen here.
#
# Two places where the participant-grouped discipline has to be re-imposed by
# hand, because scikit-learn's defaults would quietly break it:
#
#   1. Early stopping. `HistGradientBoostingRegressor(early_stopping=True)`
#      carves its validation slice out at random, so the same reader lands on
#      both sides and the stopping point is chosen on leakage. Here the slice is
#      cut by participant (`GroupShuffleSplit`), the iteration count is picked
#      from the staged loss on that slice, and the final model is refit on the
#      whole training split at that count.
#   2. Feature importance. There is no `feature_importances_` on this estimator
#      and no coefficients to report, so importance is permutation importance
#      (mean drop in R2) measured on the held-out participants, from a model
#      that never saw them -- not on the data the model was fit to.

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GroupShuffleSplit, ShuffleSplit

GBM_MODEL_KINDS = ("hist_gbm",)


@dataclass
class TrialLevelGBMModel:
    """Histogram gradient boosting over trial-level features.

    Mirrors `TrialLevelLinRegModel`: `fit(train_df, target_col, feature_cols,
    groups=...)`, `predict(df)`, `get_coef_summary()`. Unlike that model it does
    not standardize or fill -- trees are scale-free and this estimator splits on
    NaN directly.
    """

    name: str = "trial_level_hist_gbm"
    model_kind: str = "hist_gbm"

    # --- boosting hyperparameters ---
    learning_rate: float = 0.06
    max_iter: int = 400
    max_leaf_nodes: int = 31
    min_samples_leaf: int = 40
    l2_regularization: float = 1.0
    max_features: float = 1.0
    loss: str = "squared_error"
    random_state: int = 0

    # --- grouped early stopping ---
    # False fixes the boosting rounds at `max_iter` instead of searching.
    early_stopping: bool = True
    validation_fraction: float = 0.2

    # --- permutation importance (the slow part; the extra refit and the
    # per-feature repredictions dominate runtime on the ~520-column set) ---
    compute_importance: bool = True
    n_permutation_repeats: int = 3

    model: object = field(default=None, init=False)
    feature_cols_: list[str] = field(default_factory=list, init=False)
    # Boosting rounds actually used (the grouped-early-stopping pick).
    n_iter_: Optional[int] = field(default=None, init=False)
    # Validation MSE per boosting round, for plotting the stopping curve.
    validation_curve_: Optional[np.ndarray] = field(default=None, init=False)
    importance_: Optional[pd.DataFrame] = field(default=None, init=False)

    # What `get_coef_summary` returns, so figures can label themselves honestly.
    coef_kind: str = field(default="permutation importance (drop in R2)", init=False)
    # No penalty to report; `run.evaluate_answer_rt_model` reads this generically.
    alpha_: Optional[float] = field(default=None, init=False)

    # ------------------------------------------------------------------
    # Data prep
    # ------------------------------------------------------------------

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
        """Numeric coercion only -- NaN is left in place for the splitter."""
        if feature_cols is None:
            if not self.feature_cols_:
                raise ValueError("feature_cols must be provided on first fit.")
            cols = list(self.feature_cols_)
        else:
            cols = self._validate_feature_cols(df, feature_cols)

        X = df[cols].copy()
        for c in X.columns:
            X[c] = pd.to_numeric(X[c], errors="coerce")

        if fit:
            self.feature_cols_ = list(cols)
        return X

    def _new_estimator(self, max_iter: int) -> HistGradientBoostingRegressor:
        return HistGradientBoostingRegressor(
            loss=self.loss,
            learning_rate=self.learning_rate,
            max_iter=int(max_iter),
            max_leaf_nodes=self.max_leaf_nodes,
            min_samples_leaf=self.min_samples_leaf,
            l2_regularization=self.l2_regularization,
            max_features=self.max_features,
            # Off unconditionally: stopping is decided by the grouped search in
            # `fit`, never by an internal random split.
            early_stopping=False,
            random_state=self.random_state,
        )

    def _grouped_holdout(self, X: pd.DataFrame, y: np.ndarray, groups):
        """Train/validation row indices for the stopping search, split by group."""
        if groups is None:
            splitter = ShuffleSplit(
                n_splits=1,
                test_size=self.validation_fraction,
                random_state=self.random_state,
            )
            return next(splitter.split(X))

        splitter = GroupShuffleSplit(
            n_splits=1,
            test_size=self.validation_fraction,
            random_state=self.random_state,
        )
        return next(splitter.split(X, y, groups=np.asarray(groups)))

    # ------------------------------------------------------------------
    # Fit / predict
    # ------------------------------------------------------------------

    def fit(
        self,
        train_df: pd.DataFrame,
        target_col: str,
        feature_cols: Sequence[str],
        groups: Optional[Sequence] = None,
    ) -> None:
        """Fit on `train_df`; `groups` (participant ids) cuts the stopping slice."""
        X = self._prepare_X(train_df, fit=True, feature_cols=feature_cols)
        y = pd.to_numeric(train_df[target_col], errors="coerce").to_numpy()

        n_iter = int(self.max_iter)
        if self.early_stopping:
            tr_idx, val_idx = self._grouped_holdout(X, y, groups)
            X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
            y_tr, y_val = y[tr_idx], y[val_idx]

            probe = self._new_estimator(self.max_iter)
            probe.fit(X_tr, y_tr)
            curve = np.array(
                [mean_squared_error(y_val, p) for p in probe.staged_predict(X_val)]
            )
            self.validation_curve_ = curve
            n_iter = int(curve.argmin()) + 1

            if self.compute_importance:
                # Importance wants the stopped model, not the fully-grown probe,
                # and wants it scored on participants it never saw.
                stopped = self._new_estimator(n_iter)
                stopped.fit(X_tr, y_tr)
                self.importance_ = self._permutation_importance(stopped, X_val, y_val)

        self.n_iter_ = n_iter
        self.model = self._new_estimator(n_iter)
        self.model.fit(X, y)

    def _permutation_importance(self, estimator, X, y) -> pd.DataFrame:
        res = permutation_importance(
            estimator,
            X,
            y,
            scoring="r2",
            n_repeats=int(self.n_permutation_repeats),
            random_state=self.random_state,
        )
        return pd.DataFrame(
            {
                "feature": list(X.columns),
                "importance": res.importances_mean,
                "importance_std": res.importances_std,
            }
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
        """Permutation importance, in the column shape the linear model returns.

        Columns are `feature` / `coef` / `abs_coef` so the same plotting and
        reporting code works for both models -- but `coef` here is an unsigned
        drop in held-out R2, not a signed effect. `coef_kind` says which.
        Empty when the model was fit with `compute_importance=False` (or with
        `early_stopping=False`, which leaves no held-out slice to measure on).
        """
        if self.model is None:
            raise RuntimeError("Model has not been fitted yet.")

        if self.importance_ is None:
            return pd.DataFrame(columns=["feature", "coef", "abs_coef"])

        out = self.importance_.rename(columns={"importance": "coef"}).copy()
        out["abs_coef"] = out["coef"].abs()
        out = out.sort_values("abs_coef", ascending=False).reset_index(drop=True)
        if top_k is not None:
            out = out.head(int(top_k))
        return out
