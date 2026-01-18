# answer_correctness_eval.py

from dataclasses import dataclass
from typing import Dict, Sequence, Callable, List

import numpy as np
import pandas as pd

from src import constants as Con
from src.predictive_modeling.answer_correctness.answer_correctness_data import (
    build_trial_level_with_area_metrics,
)
from src.predictive_modeling.answer_correctness.answer_correctness_models import (
    AnswerCorrectnessModel,
)
from src.predictive_modeling.common.data_utils import group_vise_train_test_split


from typing import Optional

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
    coef_summary: Optional[pd.DataFrame] = None



def evaluate_models_on_answer_correctness(
    df: pd.DataFrame,
    models: Sequence[AnswerCorrectnessModel],
    group_cols: Sequence[str] = (Con.PARTICIPANT_ID, Con.TRIAL_ID),
    split_group_cols: List[str] = [Con.PARTICIPANT_ID, Con.TRIAL_ID],
    test_size: float = 0.2,
    random_state: int = 42,
    builder_fn: Callable = build_trial_level_with_area_metrics,
    split_fn: Callable = group_vise_train_test_split,
    target_col: str = Con.IS_CORRECT_COLUMN,
) -> Dict[str, CorrectnessEvaluationResult]:
    """
    High-level evaluation pipeline for answer-correctness prediction (is_correct).
    """
    trial_df = builder_fn(
        df,
        group_cols=group_cols,
    )

    train_df, test_df = split_fn(
        trial_df,
        test_size=test_size,
        random_state=random_state,
        group_cols=split_group_cols,
    )


    y_true = test_df[target_col].astype(int).to_numpy()
    results: Dict[str, CorrectnessEvaluationResult] = {}

    for model in models:
        model.fit(train_df, target_col=target_col)
        y_pred = model.predict(test_df)

        acc = float((y_true == y_pred).mean())

        coef_summary = None
        if hasattr(model, "get_coef_summary"):
            coef_summary = model.get_coef_summary(train_df)

        results[model.name] = CorrectnessEvaluationResult(
            train_df=train_df,
            test_df=test_df,
            y_true=y_true,
            y_pred=y_pred,
            accuracy=acc,
            n_test=len(test_df),
            n_positive=int((y_true == 1).sum()),
            n_negative=int((y_true == 0).sum()),
            coef_summary=coef_summary,
        )

    return results
