# answer_loc_eval.py
from dataclasses import dataclass
from typing import Dict, Sequence, Callable
import numpy as np
import pandas as pd
from src import constants as Con

from src.predictive_modeling.answer_loc.answer_loc_models import AnswerLocationModel

from src.predictive_modeling.answer_loc.answer_loc_data import (
    build_trial_level_location_table,
    group_vise_train_test_split,
)


@dataclass
class ModelEvaluationResult:
    train_df: pd.DataFrame
    test_df: pd.DataFrame
    y_true: np.ndarray
    y_pred: np.ndarray
    accuracy: float
    n_test: int


def evaluate_models_on_answer_location(
    df: pd.DataFrame,
    models: Sequence[AnswerLocationModel],
    group_cols: Sequence[str] = (Con.PARTICIPANT_ID, Con.TRIAL_ID),
    split_group_col: str = Con.PARTICIPANT_ID,
    last_loc_col: str = Con.LAST_VISITED_LOCATION,
    target_col: str = Con.SELECTED_ANSWER_POSITION_COLUMN,
    test_size: float = 0.2,
    random_state: int = 42,
    split_fn: Callable = group_vise_train_test_split,
    builder_fn: Callable = build_trial_level_location_table,
) -> Dict[str, ModelEvaluationResult]:
    """
    High-level evaluation pipeline for answer-location prediction.
    """
    trial_df = builder_fn(
        df,
        group_cols=group_cols,
        last_loc_col=last_loc_col,
        target_col=target_col,
    )

    if trial_df.empty:
        raise ValueError("Trial-level table is empty; cannot evaluate models.")

    train_df, test_df = split_fn(
        trial_df,
        test_size=test_size,
        random_state=random_state,
        group_col=split_group_col,
    )

    y_true = test_df[target_col].to_numpy()
    results: Dict[str, ModelEvaluationResult] = {}

    for model in models:
        model.fit(train_df, target_col=target_col)

        y_pred = model.predict(test_df)
        if len(y_pred) != len(test_df):
            raise ValueError(
                f"Model '{model.name}' returned {len(y_pred)} predictions "
                f"for {len(test_df)} test trials."
            )

        acc = float((y_true == y_pred).mean())

        results[model.name] = ModelEvaluationResult(
            train_df=train_df,
            test_df=test_df,
            y_true=y_true,
            y_pred=y_pred,
            accuracy=acc,
            n_test=int(len(test_df)),
        )

    return results
