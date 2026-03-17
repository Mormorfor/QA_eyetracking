# answer_correctness_models.py

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Protocol, Sequence, List, Literal, Optional, Dict

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.predictive_modeling.common.data_utils import bootstrap_logreg_coef_cis, wald_logreg_coef_cis

from src import constants as Con
from src.predictive_modeling.common.data_utils import get_coef_summary as summary

from pymer4.models import glm, glmer
import polars as pl

class AnswerCorrectnessModel(Protocol):
    name: str

    def fit(self, train_df: pd.DataFrame, target_col: str, feature_cols: Optional[Sequence[str]] = None) -> None:
        ...

    def predict(self, df: pd.DataFrame, feature_cols: Optional[Sequence[str]] = None) -> np.ndarray:
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

    def fit(self, train_df: pd.DataFrame, target_col: str, feature_cols: Optional[Sequence[str]] = None) -> None:
        y = train_df[target_col].astype(int)
        counts = y.value_counts()
        self.majority_label_ = int(counts.idxmax())

    def predict(self, df: pd.DataFrame, feature_cols: Optional[Sequence[str]] = None) -> np.ndarray:
        return np.full(len(df), self.majority_label_, dtype=int)


# ---------------------------------------------------------------------
# Baseline: random prior
# ---------------------------------------------------------------------
@dataclass
class RandomPriorBaselineCorrectness:
    """
    Baseline model that predicts labels randomly according to the
    class proportions observed in the training data.
    """
    name: str = "random_prior_baseline"
    p_positive_: float = None

    def fit(
        self,
        train_df: pd.DataFrame,
        target_col: str,
        feature_cols: Optional[Sequence[str]] = None,
    ) -> None:
        y = train_df[target_col].astype(int)
        self.p_positive_ = y.mean()


    def predict(
        self,
        df: pd.DataFrame,
        feature_cols: Optional[Sequence[str]] = None,
    ) -> np.ndarray:
        n = len(df)
        return np.random.binomial(1, self.p_positive_, size=n)

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
        # Default: all <metric>__<area> combinations (no existence checks needed)
        cols: List[str] = []
        for metric in Con.AREA_METRIC_COLUMNS_MODELING:
            for area in Con.LABEL_CHOICES:
                cols.append(f"{metric}__{area}")
        return cols

    def _resolve_feature_cols(self, df: pd.DataFrame, feature_cols: Optional[Sequence[str]]) -> List[str]:
        if feature_cols is not None:
            return list(feature_cols)

        if not self.feature_cols_:
            self.feature_cols_ = self._build_feature_columns(df)
        return self.feature_cols_

    def _prepare_X(
        self,
        df: pd.DataFrame,
        *,
        fit: bool = False,
        feature_cols: Optional[Sequence[str]] = None,
    ) -> pd.DataFrame:
        cols = self._resolve_feature_cols(df, feature_cols)

        if feature_cols is not None:
            self.feature_cols_ = list(cols)

        X = df[cols].copy()
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

        return pd.DataFrame(X_scaled, columns=cols, index=df.index)


    def get_coef_summary(
            self,
            train_df: pd.DataFrame,
            top_k: int = None,
            ci_method: Literal["bootstrap", "wald", "none"] = "wald",
            ci_cluster: Literal["cluster", "row", "auto"] = "auto",
            ci: float = 0.95,
            n_boot: int = 5000,
            seed: int = 42,
            feature_cols: Optional[Sequence[str]] = None,
    ) -> pd.DataFrame:
        """
        Coef summary + optional CIs.

        ci_method:
          - "bootstrap": add bootstrap CIs
          - "wald": add wald CIs

        ci_cluster:
          - "cluster": use participant cluster IDs (if available, else row)
          - "row": Simple trial-level
        """
        if self.model is None:
            raise RuntimeError("Model has not been fitted yet.")
        X = self._prepare_X(train_df, fit=False, feature_cols=feature_cols)
        cols_used = list(X.columns)
        out = summary(self.model, cols_used, top_k)
        cluster = train_df[Con.PARTICIPANT_ID].to_numpy()

        if ci_cluster == "row":
            cluster_used = None
        elif ci_cluster == "cluster":
            cluster_used = cluster
        else:
            cluster_used = None

        if ci_method == "bootstrap":
            fit_kwargs = dict(max_iter=100000, class_weight="balanced")
            ci_df = bootstrap_logreg_coef_cis(
                X=X,
                y=train_df[Con.IS_CORRECT_COLUMN].astype(int),
                feature_names=self.feature_cols_,
                fit_kwargs=fit_kwargs,
                n_boot=int(n_boot),
                ci=float(ci),
                seed=int(seed),
                cluster=cluster_used,
            )
            out = out.merge(
                ci_df[["feature", "ci_low", "ci_high", "or_ci_low", "or_ci_high", "sig_ci", "n_boot_ok"]],
                on="feature",
                how="left",
            )

        elif ci_method == "wald":
            wald_df = wald_logreg_coef_cis(
                model=self.model,
                X=X,
                y=train_df[Con.IS_CORRECT_COLUMN].astype(int),
                feature_names=self.feature_cols_,
                ci=float(ci),
            )
            out = out.merge(
                wald_df[["feature", "se", "ci_low", "ci_high", "or_ci_low", "or_ci_high", "sig_ci", "n_clusters"]],
                on="feature",
                how="left",
            )

        else:
            pass

        if top_k is not None:
            out = out.sort_values("abs_coef", ascending=False).head(int(top_k))
        return out

    def fit(
        self,
        train_df: pd.DataFrame,
        target_col: str,
        feature_cols: Optional[Sequence[str]] = None,
    ) -> None:
        X = self._prepare_X(train_df, fit=True, feature_cols=feature_cols)
        y = train_df[target_col].astype(int)

        self.model = LogisticRegression(
            max_iter=100000,
            class_weight="balanced",
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
        return list(self.base_feature_cols_) + sorted(pref_cols)


    def _resolve_feature_cols(self, df: pd.DataFrame, feature_cols: Optional[Sequence[str]]) -> List[str]:
        if feature_cols is not None:
            return list(feature_cols)

        if not self.feature_cols_:
            self.feature_cols_ = self._build_feature_cols(df)
        return self.feature_cols_


    def _prepare_X(
            self,
            df: pd.DataFrame,
            *,
            fit: bool = False,
            feature_cols: Optional[Sequence[str]] = None,
    ) -> pd.DataFrame:
        cols = self._resolve_feature_cols(df, feature_cols)

        if feature_cols is not None:
            self.feature_cols_ = list(cols)

        X = df[cols].copy()
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

        return pd.DataFrame(X_scaled, columns=cols, index=df.index)

    def get_coef_summary(
            self,
            train_df: pd.DataFrame,
            top_k: int = None,
            ci_method: Literal["bootstrap", "wald", "none"] = "wald",
            ci_cluster: Literal["cluster", "row", "auto"] = "auto",
            ci: float = 0.95,
            n_boot: int = 5000,
            seed: int = 42,
            feature_cols: Optional[Sequence[str]] = None,
    ) -> pd.DataFrame:
        """
        Coef summary + optional CIs.

        ci_method:
          - "bootstrap": add bootstrap CIs
          - "wald": add wald CIs

        ci_cluster:
          - "cluster": use participant cluster IDs (if available, else row)
          - "row": Simple trial-level
        """
        if self.model is None:
            raise RuntimeError("Model has not been fitted yet.")
        X = self._prepare_X(train_df, fit=False, feature_cols=feature_cols)
        cols_used = list(X.columns)
        out = summary(self.model, cols_used, top_k)
        cluster = train_df[Con.PARTICIPANT_ID].to_numpy()

        if ci_cluster == "row":
            cluster_used = None
        elif ci_cluster == "cluster":
            cluster_used = cluster
        else:
            cluster_used = None

        if ci_method == "bootstrap":
            fit_kwargs = dict(max_iter=100000, class_weight="balanced")
            ci_df = bootstrap_logreg_coef_cis(
                X=X,
                y=train_df[Con.IS_CORRECT_COLUMN].astype(int),
                feature_names=self.feature_cols_,
                fit_kwargs=fit_kwargs,
                n_boot=int(n_boot),
                ci=float(ci),
                seed=int(seed),
                cluster=cluster_used,
            )
            out = out.merge(
                ci_df[["feature", "ci_low", "ci_high", "or_ci_low", "or_ci_high", "sig_ci", "n_boot_ok"]],
                on="feature",
                how="left",
            )

        elif ci_method == "wald":
            wald_df = wald_logreg_coef_cis(
                model=self.model,
                X=X,
                y=train_df[Con.IS_CORRECT_COLUMN].astype(int),
                feature_names=self.feature_cols_,
                ci=float(ci),
            )
            out = out.merge(
                wald_df[["feature", "se", "ci_low", "ci_high", "or_ci_low", "or_ci_high", "sig_ci", "n_clusters"]],
                on="feature",
                how="left",
            )

        else:
            pass

        if top_k is not None:
            out = out.sort_values("abs_coef", ascending=False).head(int(top_k))
        return out

    def fit(
            self,
            train_df: pd.DataFrame,
            target_col: str,
            feature_cols: Optional[Sequence[str]] = None,
    ) -> None:
        X = self._prepare_X(train_df, fit=True, feature_cols=feature_cols)
        y = train_df[target_col].astype(int)

        self.model = LogisticRegression(
            max_iter=100000,
            class_weight="balanced",
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

    def _resolve_feature_cols(self, df: pd.DataFrame, feature_cols: Optional[Sequence[str]]) -> List[str]:
        if feature_cols is not None:
            return list(feature_cols)

        if not self.feature_cols_:
            self.feature_cols_ = self._build_feature_cols(df)
        return self.feature_cols_

    def _prepare_X(
            self,
            df: pd.DataFrame,
            *,
            fit: bool = False,
            feature_cols: Optional[Sequence[str]] = None,
    ) -> pd.DataFrame:

        cols = self._resolve_feature_cols(df, feature_cols)

        if feature_cols is not None:
            self.feature_cols_ = list(cols)

        X = df[cols].copy()
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
            ci_method: Literal["bootstrap", "wald", "none"] = "wald",
            ci_cluster: Literal["cluster", "row", "auto"] = "auto",
            ci: float = 0.95,
            n_boot: int = 5000,
            seed: int = 42,
            feature_cols: Optional[Sequence[str]] = None,
    ) -> pd.DataFrame:
        """
        Coef summary + optional CIs.

        ci_method:
          - "bootstrap": add bootstrap CIs
          - "wald": add wald CIs

        ci_cluster:
          - "cluster": use participant cluster IDs (if available, else row)
          - "row": Simple trial-level
        """
        if self.model is None:
            raise RuntimeError("Model has not been fitted yet.")
        X = self._prepare_X(train_df, fit=False, feature_cols=feature_cols)
        cols_used = list(X.columns)

        out = summary(self.model, cols_used, top_k)
        cluster = train_df[Con.PARTICIPANT_ID].to_numpy()

        if ci_cluster == "row":
            cluster_used = None
        elif ci_cluster == "cluster":
            cluster_used = cluster
        else:
            cluster_used = None

        if ci_method == "bootstrap":
            fit_kwargs = dict(max_iter=100000, class_weight="balanced")
            ci_df = bootstrap_logreg_coef_cis(
                X=X,
                y=train_df[Con.IS_CORRECT_COLUMN].astype(int),
                feature_names=self.feature_cols_,
                fit_kwargs=fit_kwargs,
                n_boot=int(n_boot),
                ci=float(ci),
                seed=int(seed),
                cluster=cluster_used,
            )
            out = out.merge(
                ci_df[["feature", "ci_low", "ci_high", "or_ci_low", "or_ci_high", "sig_ci", "n_boot_ok"]],
                on="feature",
                how="left",
            )

        elif ci_method == "wald":
            wald_df = wald_logreg_coef_cis(
                model=self.model,
                X=X,
                y=train_df[Con.IS_CORRECT_COLUMN].astype(int),
                feature_names=self.feature_cols_,
                ci=float(ci),
            )
            out = out.merge(
                wald_df[["feature", "se", "ci_low", "ci_high", "or_ci_low", "or_ci_high", "sig_ci", "n_clusters"]],
                on="feature",
                how="left",
            )

        else:
            pass


        if top_k is not None:
            out = out.sort_values("abs_coef", ascending=False).head(int(top_k))
        return out

    def fit(
            self,
            train_df: pd.DataFrame,
            target_col: str,
            feature_cols: Optional[Sequence[str]] = None,
    ) -> None:
        X = self._prepare_X(train_df, fit=True, feature_cols=feature_cols)
        y = train_df[target_col].astype(int)

        self.model = LogisticRegression(
            max_iter=100000,
            class_weight="balanced",
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



@dataclass
class FullFeaturesCorrectnessGLMERModel:
    """
    Binomial mixed-effects model using:
      - area-metric features
      - derived features
      - pref_matching__* features

    with crossed random intercepts for:
      - participant
      - text

    """
    name: str = "full_features_correctness_glmer"
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


    def _resolve_feature_cols(
        self,
        df: pd.DataFrame,
        feature_cols: Optional[Sequence[str]],
    ) -> List[str]:
        if feature_cols is not None:
            return list(feature_cols)

        if not self.raw_feature_cols_:
            self.raw_feature_cols_ = self._build_feature_cols(df)

        return self.raw_feature_cols_


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


    def _prepare_model_df(
        self,
        df: pd.DataFrame,
        *,
        fit: bool = False,
        feature_cols: Optional[Sequence[str]] = None,
        target_col: str = Con.IS_CORRECT_COLUMN,
        participant_col: str = Con.PARTICIPANT_ID,
        text_col: str = Con.TEXT_ID_WITH_Q_COLUMN,
    ) -> pd.DataFrame:

        raw_cols = self._resolve_feature_cols(df, feature_cols)

        if feature_cols is not None:
            self.raw_feature_cols_ = list(raw_cols)

        needed = [target_col, participant_col, text_col] + list(raw_cols)
        model_df = df[needed].copy()

        for c in raw_cols:
            model_df[c] = pd.to_numeric(model_df[c], errors="coerce")
        model_df[raw_cols] = model_df[raw_cols].fillna(0.0)

        model_df[target_col] = pd.to_numeric(model_df[target_col], errors="coerce").astype(int)
        model_df[participant_col] = model_df[participant_col].astype(str)
        model_df[text_col] = model_df[text_col].astype(str)

        if fit:
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

        self.rename_map_ = {raw: san for raw, san in rename_map.items()}
        self.reverse_rename_map_ = {san: raw for raw, san in rename_map.items()}

        self.target_col_model_ = rename_map[target_col]
        self.participant_col_model_ = rename_map[participant_col]
        self.text_col_model_ = rename_map[text_col]

        self.raw_feature_cols_ = list(raw_cols)
        self.feature_cols_ = [rename_map[c] for c in raw_cols]

        return model_df


    def _build_formula(self) -> str:
        fixed = " + ".join(self.feature_cols_)

        formula = (
            f"{self.target_col_model_} ~ {fixed} "
            f"+ (1|{self.participant_col_model_}) "
            f"+ (1|{self.text_col_model_})"
        )
        self.formula_ = formula
        return formula

    @staticmethod
    def _to_model_df(df: pd.DataFrame):
        return pl.from_pandas(df.reset_index(drop=True))

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
            raise ValueError("Target column must contain both classes 0 and 1 in training data.")

        w0 = 1.0 / counts[0]
        w1 = 1.0 / counts[1]
        model_df["obs_weight"] = np.where(model_df[self.target_col_model_] == 0, w0, w1)
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
            control="glmerControl(optimizer='bobyqa', optCtrl=list(maxfun=200000))",
            weights="obs_weight",
        )

    def predict_proba(
            self,
            df: pd.DataFrame,
            target_col: str = Con.IS_CORRECT_COLUMN,
            feature_cols: Optional[Sequence[str]] = None,
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
            threshold: float = 0.5,
            **kwargs,
    ) -> np.ndarray:
        p = self.predict_proba(df, **kwargs)
        return (p >= threshold).astype(int)


    def get_coef_summary(self) -> pd.DataFrame:
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
                    pd.notna(out["ci_low"]) &
                    pd.notna(out["ci_high"]) &
                    ((out["ci_low"] > 0) | (out["ci_high"] < 0))
            )

        return out


    def get_formula(self) -> str:
        if self.formula_ is None:
            raise RuntimeError("Model formula is not available before fitting.")
        return self.formula_