"""The paper's model: logistic regression on one row per trial."""

from dataclasses import dataclass, field
from typing import Optional, Sequence, Literal
import warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.config import columns as Con
from src.modeling.inference import get_coef_summary, bootstrap_logreg_coef_cis, wald_logreg_coef_cis


# ---------------------------------------------------------------------------
# Missing values: how the model fills them, and why the fill is counted
# ---------------------------------------------------------------------------
# Two measurement families leave out words nobody looked at, because both
# describe a fixation: pupil size and first-fixation duration. So an area
# nobody looked at arrives here as NaN, and the model fills it with 0.
#
# That 0 is an ASSUMPTION, NOT A MEASUREMENT:
#   - for a z-scored pupil column, 0 says "this area had exactly this
#     participant's average pupil size" -- a specific and false claim;
#   - for a first-fixation duration, 0 says the fixation lasted 0 ms, which
#     cannot happen.
#
# The fill covers EVERY NaN in the feature matrix, not only these two
# families: one rule for the whole matrix is the simplest to state in Methods
# and to reason about. It is never silent: `imputed_counts_` says how many
# cells were filled in each column and `imputed_mask_` says which, so a filled
# 0 can always be told from a measured one. docs/conventions.md (accepted
# deviations) has the reasoning.
#
# The markers below do not decide what is filled. They separate the NaN we can
# EXPLAIN (an area nobody looked at) from any we cannot: a NaN anywhere else
# still becomes 0, but with a warning. 
EXCLUDE_CONVENTION_MARKERS: tuple[str, ...] = (
    "pupil_size",
    "first_fixation_duration",
)


def is_exclude_convention_column(col: str) -> bool:
    """True if `col` belongs to a family whose 0 is not a possible measurement."""
    return any(marker in col for marker in EXCLUDE_CONVENTION_MARKERS)


def imputed_cell_counts(df: pd.DataFrame, feature_cols: Sequence[str]) -> pd.Series:
    """Per column, how many cells the model would fill, highest first.

    The number for Methods. Needs no fitting, so it can be reported for any
    table and feature set.
    """
    na = df[list(feature_cols)].isna().sum()
    return na[na > 0].sort_values(ascending=False)


@dataclass
class TrialLevelLogRegModel:
    """Logistic regression predicting `is_correct` from one row per trial.

    Before fitting, the feature columns are made numeric, every missing value
    is filled with `fill_value` (counted; see the note above), and each column
    is standardised with a scaler fitted on the training rows. The regression
    uses sklearn's default L2 penalty (C = 1) and `class_weight="balanced"`, so
    correct and wrong answers count equally although most answers are correct.
    The coefficients are therefore per standard deviation of each feature.
    """

    name: str = "trial_level_log_reg"
    max_iter: int = 100000
    class_weight: str = "balanced"
    # Put in place of EVERY NaN in the feature matrix; see the note above.
    fill_value: float = 0.0

    model: LogisticRegression = field(default=None, init=False)
    scaler_: StandardScaler = field(default=None, init=False)
    feature_cols_: list[str] = field(default_factory=list, init=False)
    # Set on every _prepare_X call, so the filled cells can be found afterwards:
    # without the mask, a measured 0 (a z-score can really be 0) could not be
    # told from a filled one.
    imputed_counts_: pd.Series = field(default=None, init=False)
    imputed_mask_: pd.DataFrame = field(default=None, init=False)
    # The part of imputed_counts_ with no known cause. Empty for both studies'
    # full feature sets; anything here means something upstream is missing and
    # the 0 is covering for it.
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

        # Record what is about to be filled, then fill EVERY remaining NaN. One
        # rule for the whole feature matrix is the deliberate choice; what it
        # must not do is hide the fill, so every filled cell is counted and
        # located first.
        self.imputed_mask_ = X.isna()
        self.imputed_counts_ = imputed_cell_counts(X, cols)

        # A NaN outside the two families we can explain gets the same 0, but
        # with a warning: a gap in, say, has_xyx means something upstream is
        # wrong, and the fill would otherwise hide it. Warn, don't stop -- the
        # fill is the policy.
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
                "intended; an unexplained NaN is not. Worth "
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
            # The default is the bootstrap, clustered by participant: it refits
            # the real penalised, class-weighted model on each resample, so it
            # needs no approximation, and it does not break down with few
            # participants the way the clustered Wald formula does (Study 2 has
            # 6). The callers that cannot afford ~5,000 refits ask for Wald
            # instead: crossval.py (once per fold and regime) and
            # analyses/correctness_prediction/participant_level.py (one model per
            # participant, where clustering by participant means nothing).
            ci_method: Literal["bootstrap", "wald", "none"] = "bootstrap",
            ci_cluster: Literal["cluster", "row", "auto"] = "cluster",
            ci: float = 0.95,
            n_boot: int = 5000,
            seed: int = 42,
            feature_cols: Optional[Sequence[str]] = None,
            target_col: str = Con.IS_CORRECT_COLUMN,
    ) -> pd.DataFrame:
        """Coefficient table for the fitted model, with confidence intervals.

        `ci_method`: "bootstrap" (the default), "wald" (no refitting, much
        faster) or "none". `ci_cluster`: "cluster" resamples or groups by
        participant; "row" or "auto" treats each trial on its own. `train_df`
        must be the table the model was fitted on. Coefficients are per
        standard deviation of each feature.
        """
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
            # The Wald sandwich needs no refitting, so it is the affordable option
            # inside cross-validation, where 5,000 bootstrap refits per fold and
            # regime are not.
            wald_df = wald_logreg_coef_cis(
                model=self.model,
                X=X,
                y=y,
                feature_names=cols_used,
                ci=float(ci),
                cluster=(
                    train_df[Con.PARTICIPANT_ID].to_numpy()
                    if ci_cluster == "cluster"
                    else None
                ),
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