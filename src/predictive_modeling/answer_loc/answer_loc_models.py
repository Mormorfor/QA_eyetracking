# answer_loc_models.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Protocol, Sequence, List
import numpy as np
import pandas as pd
from src import constants as Con

from sklearn.linear_model import LogisticRegression
import statsmodels.formula.api as smf

from src.predictive_modeling.common.features import map_last_location_to_position



class AnswerLocationModel(Protocol):
    """
    Minimal interface that all answer-location models should implement.
    """
    name: str

    def fit(self, train_df: pd.DataFrame, target_col: str) -> None:
        ...

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        ...


##################################################
# Baseline model: last location visited
##################################################

@dataclass
class LastLocationBaseline:
    name: str = "last_location"
    last_loc_col: str = Con.LAST_VISITED_LOCATION

    def fit(self, train_df: pd.DataFrame, target_col: str) -> None:
        return

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        if self.last_loc_col not in df.columns:
            raise KeyError(f"Column '{self.last_loc_col}' not found in df.")

        loc_series = df[self.last_loc_col]
        pos_series = map_last_location_to_position(loc_series)

        return pos_series.fillna(-1).astype(int).to_numpy()


##################################################
# Simplest regression model: area metrics only
##################################################

@dataclass
class AreaMetricsLogRegModel:
    """
    Multinomial logistic regression using:
      - AREA_METRIC_COLUMNS for each area in ANSWER_LABEL_CHOICES
      - optionally last visited location as a numeric feature.

    """
    name: str = "area_metrics_log_reg"
    include_last_location: bool = False

    model: LogisticRegression = field(default=None, init=False)
    feature_cols_: List[str] = field(default_factory=list, init=False)

    def _build_feature_columns(self, df: pd.DataFrame) -> List[str]:
        cols: List[str] = []

        for metric in Con.AREA_METRIC_COLUMNS:
            for area in Con.ANSWER_LABEL_CHOICES:
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

        X = temp[self.feature_cols_].fillna(0.0)
        return X



    def fit(self, train_df: pd.DataFrame, target_col: str) -> None:
        if target_col not in train_df.columns:
            raise KeyError(f"Target column '{target_col}' not found in train_df.")

        X = self._prepare_design_matrix(train_df, fit=True)
        y = train_df[target_col].astype(int)

        self.model = LogisticRegression(
            multi_class="multinomial",
            max_iter=1000,
        )
        self.model.fit(X, y)

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model has not been fitted yet.")

        X = self._prepare_design_matrix(df, fit=False)
        return self.model.predict(X).astype(int)




##################################################
# Mixed Effects Model
##################################################
@dataclass
class MixedEffectsLocationModel:
    """
    Linear mixed-effects model for answer position (0–3),
    with crossed random intercepts for participants and texts.

    - Fixed effects: area metrics per area, optionally last_loc_numeric
    - Random effects:
        (1 | participant_id)  via 'groups'
        (1 | text_id)         via variance components (vc_formula)
    """
    name: str = "mixedlm_linear"
    include_last_location: bool = False
    group_col: str = Con.PARTICIPANT_ID          # random effect: participants
    text_col: str = Con.TEXT_ID_WITH_Q_COLUMN           # random effect: texts

    model: object = field(default=None, init=False)
    fixed_cols_: List[str] = field(default_factory=list, init=False)


    def _build_fixed_effect_columns(self, df: pd.DataFrame) -> List[str]:
        cols: List[str] = []

        for metric in Con.AREA_METRIC_COLUMNS:
            for area in Con.ANSWER_LABEL_CHOICES:
                col = f"{metric}__{area}"
                if col in df.columns:
                    cols.append(col)

        if self.include_last_location and "last_loc_numeric" in df.columns:
            cols.append("last_loc_numeric")

        return cols


    def _add_last_loc_numeric(self, df: pd.DataFrame) -> pd.DataFrame:
        temp = df.copy()
        if self.include_last_location and Con.LAST_VISITED_LOCATION in temp.columns:
            temp["last_loc_numeric"] = map_last_location_to_position(
                temp[Con.LAST_VISITED_LOCATION]
            ).fillna(-1).astype(float)
        return temp



    def fit(self, train_df: pd.DataFrame, target_col: str) -> None:
        df = self._add_last_loc_numeric(train_df).copy()

        self.fixed_cols_ = self._build_fixed_effect_columns(df)

        required = [self.group_col, self.text_col, target_col] + self.fixed_cols_
        missing_cols = [c for c in required if c not in df.columns]
        if missing_cols:
            raise KeyError(f"Required column(s) missing for MixedLM: {missing_cols}")

        df = df.dropna(subset=required).reset_index(drop=True)

        if df.empty:
            raise ValueError("After dropping missing rows, no data left to fit MixedLM.")

        df[self.group_col] = df[self.group_col].astype("category")
        df[self.text_col] = df[self.text_col].astype("category")

        fixed_part = " + ".join(self.fixed_cols_) if self.fixed_cols_ else "1"
        formula = f"{target_col} ~ {fixed_part}"

        vc_formula = {self.text_col: f"0 + C({self.text_col})"}

        md = smf.mixedlm(
            formula=formula,
            data=df,
            groups=df[self.group_col],
            vc_formula=vc_formula,
            missing="raise",
        )
        self.model = md.fit(reml=False)


    def predict(self, df: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model must be fitted before prediction.")

        temp = self._add_last_loc_numeric(df)
        preds_cont = self.model.predict(temp)
        preds_int = np.clip(np.round(preds_cont), 0, 3).astype(int)
        return preds_int

