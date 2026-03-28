#logreg_model.py


from dataclasses import dataclass, field
from typing import Optional, Sequence, Literal
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


@dataclass
class TrialLevelLogRegModel:
    name: str = "trial_level_log_reg"
    max_iter: int = 100000
    class_weight: str = "balanced"
    fill_value: float = 0.0

    model: LogisticRegression = field(default=None, init=False)
    scaler_: StandardScaler = field(default=None, init=False)
    feature_cols_: list[str] = field(default_factory=list, init=False)

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