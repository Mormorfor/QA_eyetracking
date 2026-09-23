#logreg_model.py


from dataclasses import dataclass, field
from typing import Optional, Sequence, Literal
import warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src import constants as Con
from src.predictive_modeling.common.data_utils import (
    get_coef_summary,
    bootstrap_logreg_coef_cis,
    wald_logreg_coef_cis,
)


# ---------------------------------------------------------------------------
# The imputation, and why it is named rather than anonymous
# ---------------------------------------------------------------------------
# Two metric families follow the "exclude" convention: a word that was never
# fixated has no pupil size and no first-fixation duration, because both are
# properties *of a fixation*. So an area nobody looked at arrives here as NaN,
# and the model fills it with 0.
#
# That fill is an ASSUMPTION, NOT A MEASUREMENT, and it is provisional:
#   - for a z-scored pupil column, 0 asserts "this area had exactly this
#     participant's mean dilation" -- a specific and false claim;
#   - for a first-fixation duration, 0 asserts a fixation of length zero,
#     which does not exist.
#
# THE FILL IS DELIBERATELY BLANKET: every NaN in the feature matrix becomes 0,
# not just these two families (Diana, 2026-09-23). One rule over the whole
# matrix is the simplest thing to state in Methods and to reason about, and it
# is the method this project has always used. Kept by decision (docs/todo.md
# T3.14) on the condition that it stops being *silent*, which is what the
# counting below is for: `imputed_counts_` says how many cells and which
# columns, `imputed_mask_` says exactly which cells, so a filled 0 can always be
# told from a measured one. A stated, counted imputation is a modelling choice;
# the same one unrecorded is the violation conventions.md forbids.
#
# The markers below no longer gate the fill -- they only separate the NaN we can
# EXPLAIN (an area nobody fixated) from any we cannot, so an unexplained hole
# gets a warning on its way to becoming a 0 instead of vanishing quietly.
# Background: docs/pitfalls.md section 2. Revisit if a question-area feature
# ever enters the model -- that area is unfixated on ~30% of trials.
EXCLUDE_CONVENTION_MARKERS: tuple[str, ...] = (
    "pupil_size",
    "first_fixation_duration",
)


def is_exclude_convention_column(col: str) -> bool:
    """True if `col` belongs to a family whose 0 is not a possible measurement."""
    return any(marker in col for marker in EXCLUDE_CONVENTION_MARKERS)


def imputed_cell_counts(df: pd.DataFrame, feature_cols: Sequence[str]) -> pd.Series:
    """Per-column count of cells the model would impute, highest first.

    The Methods number. Computable without fitting anything, so the count can be
    reported for a dataset rather than recovered from a fitted model.
    """
    na = df[list(feature_cols)].isna().sum()
    return na[na > 0].sort_values(ascending=False)


@dataclass
class TrialLevelLogRegModel:
    name: str = "trial_level_log_reg"
    max_iter: int = 100000
    class_weight: str = "balanced"
    # Applied to EVERY NaN in the feature matrix -- see the comment above for
    # what the value asserts, why it is provisional, and why it is blanket.
    fill_value: float = 0.0

    model: LogisticRegression = field(default=None, init=False)
    scaler_: StandardScaler = field(default=None, init=False)
    feature_cols_: list[str] = field(default_factory=list, init=False)
    # Set on every _prepare_X call so imputed cells stay identifiable after the
    # fact -- without the mask, a genuine 0 (e.g. the 137 trials whose
    # first_encounter_avg_pupil_size_z really does equal their own mean) is
    # indistinguishable from a filled one. T3.14 point 4.
    imputed_counts_: pd.Series = field(default=None, init=False)
    imputed_mask_: pd.DataFrame = field(default=None, init=False)
    # The subset of imputed_counts_ with no documented cause. Empty on L1 and
    # KnowQA today; non-empty means something upstream is missing and the 0 is
    # covering for it.
    unexpected_imputed_: pd.Series = field(default=None, init=False)

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

        # Record what is about to be imputed, then fill EVERY remaining NaN.
        # The blanket fill is the deliberate choice (Diana, 2026-09-23): one rule
        # for the whole feature matrix is the simplest thing to state in Methods
        # and to reason about. What the rule below does NOT do is hide it --
        # every filled cell is counted and located before it disappears.
        self.imputed_mask_ = X.isna()
        self.imputed_counts_ = imputed_cell_counts(X, cols)

        # Anything outside the two families whose NaN we can *explain* gets the
        # same 0, but it is surfaced rather than absorbed: a hole in, say,
        # has_xyx means something upstream is wrong, and the fill would otherwise
        # make it invisible. Warn, don't block -- the fill is the policy.
        self.unexpected_imputed_ = self.imputed_counts_[
            [not is_exclude_convention_column(c) for c in self.imputed_counts_.index]
        ]
        if len(self.unexpected_imputed_):
            detail = ", ".join(
                f"{c} ({int(n)} of {len(X)} rows)"
                for c, n in self.unexpected_imputed_.items()
            )
            warnings.warn(
                f"Imputing 0 in columns outside {EXCLUDE_CONVENTION_MARKERS}, where "
                f"a missing value has no documented cause: {detail}. The fill is "
                "intended (docs/todo.md T3.14); an unexplained NaN is not. Worth "
                "finding out why it is missing.",
                RuntimeWarning,
                stacklevel=2,
            )

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
        y = train_df[target_col].astype(int)

        self.model = LogisticRegression(
            max_iter=self.max_iter,
            class_weight=self.class_weight,
        )
        self.model.fit(X, y)


    def predict(
        self,
        df: pd.DataFrame,
        feature_cols: Optional[Sequence[str]] = None,
    ) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model has not been fitted yet.")
        X = self._prepare_X(df, fit=False, feature_cols=feature_cols)
        return self.model.predict(X).astype(int)


    def predict_proba(
        self,
        df: pd.DataFrame,
        feature_cols: Optional[Sequence[str]] = None,
    ) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model has not been fitted yet.")
        X = self._prepare_X(df, fit=False, feature_cols=feature_cols)
        return self.model.predict_proba(X)[:, 1]

    def get_coef_summary(
            self,
            train_df: Optional[pd.DataFrame] = None,
            top_k: Optional[int] = None,
            ci_method: Literal["bootstrap", "wald", "none"] = "wald",
            ci_cluster: Literal["cluster", "row", "auto"] = "auto",
            ci: float = 0.95,
            n_boot: int = 5000,
            seed: int = 42,
            feature_cols: Optional[Sequence[str]] = None,
            target_col: str = Con.IS_CORRECT_COLUMN,
    ) -> pd.DataFrame:
        if self.model is None:
            raise RuntimeError("Model has not been fitted yet.")
        if train_df is None:
            raise ValueError("train_df is required for logistic regression coef summary.")

        X = self._prepare_X(train_df, fit=False, feature_cols=feature_cols)
        cols_used = list(X.columns)
        y = train_df[target_col].astype(int)

        out = get_coef_summary(self.model, cols_used, top_k=None)

        if ci_method == "bootstrap":
            if ci_cluster == "cluster":
                cluster_used = train_df[Con.PARTICIPANT_ID].to_numpy()
            else:
                cluster_used = None

            fit_kwargs = {
                "max_iter": self.max_iter,
                "class_weight": self.class_weight,
            }

            ci_df = bootstrap_logreg_coef_cis(
                X=X,
                y=y,
                feature_names=cols_used,
                fit_kwargs=fit_kwargs,
                n_boot=int(n_boot),
                ci=float(ci),
                seed=int(seed),
                cluster=cluster_used,
            )

            out = out.merge(
                ci_df[
                    ["feature", "ci_low", "ci_high", "or_ci_low", "or_ci_high", "sig_ci", "n_boot_ok"]
                ],
                on="feature",
                how="left",
            )

        elif ci_method == "wald":
            wald_df = wald_logreg_coef_cis(
                model=self.model,
                X=X,
                y=y,
                feature_names=cols_used,
                ci=float(ci),
            )

            out = out.merge(
                wald_df[
                    ["feature", "se", "ci_low", "ci_high", "or_ci_low", "or_ci_high", "sig_ci", "n_clusters"]
                ],
                on="feature",
                how="left",
            )

        elif ci_method == "none":
            pass
        else:
            raise ValueError(f"Unsupported ci_method: {ci_method}")

        if top_k is not None:
            out = out.sort_values("abs_coef", ascending=False).head(int(top_k))

        return out.reset_index(drop=True)