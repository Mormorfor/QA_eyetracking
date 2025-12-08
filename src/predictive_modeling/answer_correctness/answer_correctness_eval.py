# answer_correctness_eval.py

from dataclasses import dataclass
from typing import Dict, Sequence, Callable

import numpy as np
import pandas as pd

from src import constants as Con
from src.predictive_modeling.answer_correctness.answer_correctness_data import (
    build_trial_level_with_area_metrics_for_correctness,
)
from src.predictive_modeling.answer_correctness.answer_correctness_models import (
    AnswerCorrectnessModel,
)
from src.predictive_modeling.common.data_utils import simple_train_test_split


@dataclass
class CorrectnessEvaluationResult:
    train_df: pd.DataFrame
    test_df: pd.DataFrame
    y_true: np.ndarray
    y_pred: np.ndarray
    accuracy: float
    n_test: int
    n_positive: int
    n_negative: int


def evaluate_models_on_answer_correctness(
    df: pd.DataFrame,
    models: Sequence[AnswerCorrectnessModel],
    group_cols: Sequence[str] = (Con.PARTICIPANT_ID, Con.TRIAL_ID, Con.TEXT_ID_WITH_Q_COLUMN),
    split_group_col: str = Con.PARTICIPANT_ID,
    test_size: float = 0.2,
    random_state: int = 42,
    builder_fn: Callable = build_trial_level_with_area_metrics_for_correctness,
    split_fn: Callable = simple_train_test_split,
    target_col: str = Con.IS_CORRECT_COLUMN,
) -> Dict[str, CorrectnessEvaluationResult]:
    """
    High-level evaluation pipeline for answer-correctness prediction (is_correct).
    """
    trial_df = builder_fn(
        df,
        group_cols=group_cols,
        last_loc_col=Con.LAST_VISITED_LOCATION,
    )

    if trial_df.empty:
        raise ValueError("Trial-level table is empty; cannot evaluate models.")

    if target_col not in trial_df.columns:
        raise KeyError(f"Target column '{target_col}' not found in trial-level table.")

    train_df, test_df = split_fn(
        trial_df,
        test_size=test_size,
        random_state=random_state,
        group_col=split_group_col,
    )

    y_true = test_df[target_col].astype(int).to_numpy()
    results: Dict[str, CorrectnessEvaluationResult] = {}

    for model in models:
        model.fit(train_df, target_col=target_col)
        y_pred = model.predict(test_df)

        if len(y_pred) != len(test_df):
            raise ValueError(
                f"Model '{model.name}' returned {len(y_pred)} predictions "
                f"for {len(test_df)} test trials."
            )

        y_pred = y_pred.astype(int)
        acc = float((y_true == y_pred).mean())
        n_test = int(len(test_df))
        n_pos = int((y_true == 1).sum())
        n_neg = int((y_true == 0).sum())

        results[model.name] = CorrectnessEvaluationResult(
            train_df=train_df,
            test_df=test_df,
            y_true=y_true,
            y_pred=y_pred,
            accuracy=acc,
            n_test=n_test,
            n_positive=n_pos,
            n_negative=n_neg,
        )

    return results
