# dummy_model.py

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from src import constants as Con


@dataclass
class TrialLevelDummyModel:
    """
    Class-prior baseline for answer-correctness prediction.

    On ``fit`` it only looks at the target column and stores the training-set
    base rate of correctness ``p = mean(y_train)`` (e.g. 0.8 when 80% of the
    training trials are correct). It ignores all features.

    - ``predict_proba`` returns the constant ``p`` for every trial.
    - ``predict`` draws an independent Bernoulli(p) sample per test trial
      ("random chance" weighted by the training class distribution), so the
      expected balanced accuracy is ~0.5 regardless of class imbalance.

    The interface mirrors ``TrialLevelLogRegModel`` (fit / predict /
    predict_proba / get_coef_summary) so it can be dropped into the same
    evaluation and cross-validation pipelines.
    """

    name: str = "trial_level_dummy"
    seed: int = 42

    base_rate_: Optional[float] = field(default=None, init=False)

    def fit(
        self,
        train_df: pd.DataFrame,
        target_col: str = Con.IS_CORRECT_COLUMN,
        feature_cols: Optional[Sequence[str]] = None,
    ) -> None:
        y = pd.to_numeric(train_df[target_col], errors="coerce").astype(int)
        self.base_rate_ = float(y.mean())

    def predict_proba(
        self,
        df: pd.DataFrame,
        feature_cols: Optional[Sequence[str]] = None,
    ) -> np.ndarray:
        if self.base_rate_ is None:
            raise RuntimeError("Model has not been fitted yet.")
        return np.full(len(df), self.base_rate_, dtype=float)

    def predict(
        self,
        df: pd.DataFrame,
        feature_cols: Optional[Sequence[str]] = None,
    ) -> np.ndarray:
        if self.base_rate_ is None:
            raise RuntimeError("Model has not been fitted yet.")
        rng = np.random.RandomState(self.seed)
        draws = rng.random_sample(len(df))
        return (draws < self.base_rate_).astype(int)

    def get_coef_summary(
        self,
        train_df: Optional[pd.DataFrame] = None,
        feature_cols: Optional[Sequence[str]] = None,
        **kwargs,
    ) -> pd.DataFrame:
        """No coefficients for the dummy baseline; returns an empty summary.

        Accepts and ignores the coefficient keyword arguments
        (``top_k``, ``ci_method``, ``ci_cluster``, ``ci``, ``n_boot``,
        ``seed``, ``target_col``) so it is call-compatible with the other
        models in the evaluation pipeline.
        """
        return pd.DataFrame(
            columns=["feature", "coef", "abs_coef", "ci_low", "ci_high"]
        )
