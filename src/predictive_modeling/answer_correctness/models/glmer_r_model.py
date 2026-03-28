from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence, Dict, List

import numpy as np
import pandas as pd
import polars as pl
from pymer4.models import glmer
from sklearn.preprocessing import StandardScaler

from src import constants as Con


@dataclass
class TrialLevelGLMERModel:
    """
    Binomial mixed-effects model on an already prepared trial-level dataframe.

    Assumptions:
    - df already contains the trial-level feature columns
    - target_col exists and is binary
    - participant_col and text_col exist
    - feature_cols are passed explicitly on first fit
    """
    name: str = "trial_level_glmer"
    fill_value: float = 0.0
    optimizer_control: str = (
        "glmerControl(optimizer='bobyqa', optCtrl=list(maxfun=200000))"
    )

    model: object = field(default=None, init=False)
    scaler_: StandardScaler = field(default=None, init=False)

    raw_feature_cols_: List[str] = field(default_factory=list, init=False)
    feature_cols_: List[str] = field(default_factory=list, init=False)

    formula_: Optional[str] = field(default=None, init=False)

    rename_map_: Dict[str, str] = field(default_factory=dict, init=False)
    reverse_rename_map_: Dict[str, str] = field(default_factory=dict, init=False)

    target_col_model_: Optional[str] = field(default=None, init=False)
    participant_col_model_: Optional[str] = field(default=None, init=False)
    text_col_model_: Optional[str] = field(default=None, init=False)

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

    @staticmethod
    def _sanitize_colname(col: str) -> str:
        out = str(col)
        out = out.replace("__", "_")
        out = out.replace("-", "_")
        out = out.replace(" ", "_")
        out = out.replace("(", "_").replace(")", "_")
        out = out.replace("/", "_")
        out = out.replace("\\", "_")
        out = out.replace(".", "_")
        out = out.replace(":", "_")
        return out

    @staticmethod
    def _to_model_df(df: pd.DataFrame) -> pl.DataFrame:
        return pl.from_pandas(df.reset_index(drop=True))

    def _prepare_model_df(
        self,
        df: pd.DataFrame,
        *,
        fit: bool,
        feature_cols: Optional[Sequence[str]] = None,
        target_col: str = Con.IS_CORRECT_COLUMN,
        participant_col: str = Con.PARTICIPANT_ID,
        text_col: str = Con.TEXT_ID_WITH_Q_COLUMN,
    ) -> pd.DataFrame:
        if feature_cols is None:
            if not self.raw_feature_cols_:
                raise ValueError("feature_cols must be provided on first fit.")
            raw_cols = list(self.raw_feature_cols_)
        else:
            raw_cols = self._validate_feature_cols(df, feature_cols)

        needed = [target_col, participant_col, text_col] + list(raw_cols)
        model_df = df[needed].copy()

        for c in raw_cols:
            model_df[c] = pd.to_numeric(model_df[c], errors="coerce")
        model_df[raw_cols] = model_df[raw_cols].fillna(self.fill_value)

        model_df[target_col] = pd.to_numeric(
            model_df[target_col], errors="coerce"
        ).astype(int)
        model_df[participant_col] = model_df[participant_col].astype(str)
        model_df[text_col] = model_df[text_col].astype(str)

        if fit:
            self.raw_feature_cols_ = list(raw_cols)
            self.scaler_ = StandardScaler()
            model_df[raw_cols] = self.scaler_.fit_transform(model_df[raw_cols])
        else:
            if self.scaler_ is None:
                raise RuntimeError("Scaler has not been fitted.")
            model_df[raw_cols] = self.scaler_.transform(model_df[raw_cols])

        rename_map = {
            c: self._sanitize_colname(c)
            for c in [target_col, participant_col, text_col] + list(raw_cols)
        }

        model_df = model_df.rename(columns=rename_map)

        self.rename_map_ = dict(rename_map)
        self.reverse_rename_map_ = {v: k for k, v in rename_map.items()}

        self.target_col_model_ = rename_map[target_col]
        self.participant_col_model_ = rename_map[participant_col]
        self.text_col_model_ = rename_map[text_col]
        self.feature_cols_ = [rename_map[c] for c in raw_cols]

        return model_df

    def _build_formula(self) -> str:
        if not self.feature_cols_:
            raise RuntimeError("No fitted feature columns available.")

        fixed = " + ".join(self.feature_cols_)

        formula = (
            f"{self.target_col_model_} ~ {fixed} "
            f"+ (1|{self.participant_col_model_}) "
            f"+ (1|{self.text_col_model_})"
        )
        self.formula_ = formula
        return formula

    def fit(
        self,
        train_df: pd.DataFrame,
        target_col: str = Con.IS_CORRECT_COLUMN,
        feature_cols: Optional[Sequence[str]] = None,
        participant_col: str = Con.PARTICIPANT_ID,
        text_col: str = Con.TEXT_ID_WITH_Q_COLUMN,
    ) -> None:
        model_df = self._prepare_model_df(
            train_df,
            fit=True,
            feature_cols=feature_cols,
            target_col=target_col,
            participant_col=participant_col,
            text_col=text_col,
        )

        formula = self._build_formula()

        counts = model_df[self.target_col_model_].value_counts()
        if not {0, 1}.issubset(set(counts.index)):
            raise ValueError(
                "Target column must contain both classes 0 and 1 in training data."
            )

        w0 = 1.0 / counts[0]
        w1 = 1.0 / counts[1]
        model_df["obs_weight"] = np.where(
            model_df[self.target_col_model_] == 0,
            w0,
            w1,
        )
        model_df["obs_weight"] = model_df["obs_weight"] / model_df["obs_weight"].mean()

        self.model = glmer(
            formula=formula,
            data=self._to_model_df(model_df),
            family="binomial",
        )

        self.model.fit(
            exponentiate=False,
            summary=False,
            conf_method="wald",
            type_predict="response",
            control=self.optimizer_control,
            weights="obs_weight",
        )

    def predict_proba(
        self,
        df: pd.DataFrame,
        feature_cols: Optional[Sequence[str]] = None,
        target_col: str = Con.IS_CORRECT_COLUMN,
        participant_col: str = Con.PARTICIPANT_ID,
        text_col: str = Con.TEXT_ID_WITH_Q_COLUMN,
        use_rfx: bool = False,
    ) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model has not been fitted yet.")

        model_df = self._prepare_model_df(
            df,
            fit=False,
            feature_cols=feature_cols,
            target_col=target_col,
            participant_col=participant_col,
            text_col=text_col,
        )

        preds = self.model.predict(
            data=self._to_model_df(model_df),
            use_rfx=use_rfx,
            type_predict="response",
        )

        return np.asarray(preds).reshape(-1).astype(float)

    def predict(
        self,
        df: pd.DataFrame,
        feature_cols: Optional[Sequence[str]] = None,
        threshold: float = 0.5,
        **kwargs,
    ) -> np.ndarray:
        p = self.predict_proba(df, feature_cols=feature_cols, **kwargs)
        return (p >= threshold).astype(int)

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

        if hasattr(self.model, "result_fit") and self.model.result_fit is not None:
            out = self.model.result_fit.to_pandas()
        elif hasattr(self.model, "params") and self.model.params is not None:
            out = self.model.params.to_pandas()
        else:
            raise RuntimeError("Could not find coefficient table on fitted pymer4 model.")

        if "term" in out.columns:
            out = out.rename(columns={"term": "feature"})
        elif "index" in out.columns:
            out = out.rename(columns={"index": "feature"})
        else:
            out = out.reset_index().rename(columns={"index": "feature"})

        out["feature_raw"] = out["feature"].map(self.reverse_rename_map_).fillna(out["feature"])

        if "estimate" in out.columns:
            out["coef"] = pd.to_numeric(out["estimate"], errors="coerce")
        elif "Estimate" in out.columns:
            out["coef"] = pd.to_numeric(out["Estimate"], errors="coerce")
        else:
            out["coef"] = np.nan

        out["abs_coef"] = out["coef"].abs()

        if "conf_low" in out.columns and "conf_high" in out.columns:
            out["ci_low"] = pd.to_numeric(out["conf_low"], errors="coerce")
            out["ci_high"] = pd.to_numeric(out["conf_high"], errors="coerce")
        elif "lower_CL" in out.columns and "upper_CL" in out.columns:
            out["ci_low"] = pd.to_numeric(out["lower_CL"], errors="coerce")
            out["ci_high"] = pd.to_numeric(out["upper_CL"], errors="coerce")
        elif "2.5_ci" in out.columns and "97.5_ci" in out.columns:
            out["ci_low"] = pd.to_numeric(out["2.5_ci"], errors="coerce")
            out["ci_high"] = pd.to_numeric(out["97.5_ci"], errors="coerce")
        else:
            out["ci_low"] = np.nan
            out["ci_high"] = np.nan

        out["or"] = np.exp(out["coef"])
        out["or_ci_low"] = np.exp(out["ci_low"])
        out["or_ci_high"] = np.exp(out["ci_high"])

        if "p_value" in out.columns:
            pvals = pd.to_numeric(out["p_value"], errors="coerce")
            out["sig_ci"] = pvals < 0.05
        elif "P-val" in out.columns:
            pvals = pd.to_numeric(out["P-val"], errors="coerce")
            out["sig_ci"] = pvals < 0.05
        else:
            out["sig_ci"] = (
                    pd.notna(out["ci_low"])
                    & pd.notna(out["ci_high"])
                    & ((out["ci_low"] > 0) | (out["ci_high"] < 0))
            )

        if top_k is not None:
            out = out.sort_values("abs_coef", ascending=False).head(int(top_k))

        return out.reset_index(drop=True)

    def get_formula(self) -> str:
        if self.formula_ is None:
            raise RuntimeError("Model formula is not available before fitting.")
        return self.formula_