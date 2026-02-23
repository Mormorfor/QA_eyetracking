# answer_correctness_eval.py

from dataclasses import dataclass
from typing import Dict, Sequence, Callable, List, Literal

import numpy as np
import pandas as pd

from src import constants as Con
from src.predictive_modeling.answer_correctness.answer_correctness_data import (
    build_trial_level_with_area_metrics,
)
from src.predictive_modeling.answer_correctness.answer_correctness_models import (
    AnswerCorrectnessModel,
)
from src.predictive_modeling.common.data_utils import (
    group_vise_train_test_split,
    leave_one_trial_out_for_participant,
)


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


@dataclass
class PerParticipantCorrectnessResult:
    participant_id: str
    per_trial_results: Dict[str, CorrectnessEvaluationResult]


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
    coef_ci_method: Literal["bootstrap", "wald", "none"] = "wald",
    coef_ci_cluster: Literal["cluster", "row", "auto"] = "row",
    coef_ci: float = 0.95,
    coef_n_boot: int = 3000,
    coef_seed: int = 42,
    coef_top_k: int = None,
) -> Dict[str, CorrectnessEvaluationResult]:
    """
    High-level evaluation pipeline for answer-correctness prediction (is_correct).
    """
    train_raw, test_raw = split_fn(
        df,
        test_size=test_size,
        random_state=random_state,
        group_cols=split_group_cols,
    )

    train_df = builder_fn(train_raw, group_cols=group_cols)
    test_df = builder_fn(test_raw, group_cols=group_cols)

    y_true = test_df[target_col].astype(int).to_numpy()
    results: Dict[str, CorrectnessEvaluationResult] = {}

    for model in models:
        model.fit(train_df, target_col=target_col)
        y_pred = model.predict(test_df)

        acc = float((y_true == y_pred).mean())

        coef_summary = None
        get_cs = getattr(model, "get_coef_summary", None)
        if callable(get_cs):
            coef_summary = get_cs(
                train_df=train_df,
                top_k=coef_top_k,
                ci_method=coef_ci_method,
                ci_cluster=coef_ci_cluster,
                ci=coef_ci,
                n_boot=coef_n_boot,
                seed=coef_seed,
            )

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


def evaluate_models_on_answer_correctness_leave_one_trial_out(
    df: pd.DataFrame,
    models: Sequence[AnswerCorrectnessModel],
    group_cols: Sequence[str] = (Con.PARTICIPANT_ID, Con.TRIAL_ID),
    participant_col: str = Con.PARTICIPANT_ID,
    trial_col: str = Con.TRIAL_ID,
    builder_fn: Callable = build_trial_level_with_area_metrics,
    split_fn: Callable = leave_one_trial_out_for_participant,
    target_col: str = Con.IS_CORRECT_COLUMN,
) -> Dict[str, Dict[str, CorrectnessEvaluationResult]]:
    """
    Evaluate each model per participant using leave-one-trial-out splitting.

    Returns:
        results[participant_id][model_name] = CorrectnessEvaluationResult
    """
    trial_df = builder_fn(df, group_cols=group_cols).copy()

    participants = (
        trial_df[participant_col]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    results: Dict[str, Dict[str, CorrectnessEvaluationResult]] = {}

    for pid in participants:
        train_df, test_df = split_fn(
            df=trial_df,
            participant_id=pid,
            participant_col=participant_col,
            trial_col=trial_col,
        )

        n_classes = train_df[target_col].dropna().astype(int).nunique()
        if n_classes < 2:
            continue

        y_true = test_df[target_col].astype(int).to_numpy()
        results[pid] = {}

        for model in models:
            model.fit(train_df, target_col=target_col)
            y_pred = model.predict(test_df)

            acc = float((y_true == y_pred).mean())
            coef_summary = None
            if hasattr(model, "get_coef_summary"):
                coef_summary = model.get_coef_summary(train_df)

            results[pid][model.name] = CorrectnessEvaluationResult(
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