from dataclasses import dataclass
from typing import Callable, Dict, Sequence, Tuple, Optional, Any

from predictive_modeling.answer_correctness.evaluation_core import CorrectnessEvaluationResult, \
    evaluate_single_model_on_prepared_split

import src.constants as Con

import pandas as pd

from predictive_modeling.answer_correctness.model_data import build_trial_level_model_df
from predictive_modeling.common.data_utils import leave_one_trial_out_for_participant
from predictive_modeling.common.feature_specs import get_full_feature_cols


@dataclass
class PerParticipantCorrectnessResult:
    participant_id: str
    per_trial_results: Dict[str, CorrectnessEvaluationResult]


def evaluate_logreg_on_answer_correctness_leave_one_trial_out(
    df: pd.DataFrame,
    *,
    model_builder: Callable[[], Any],
    participant_col: str = Con.PARTICIPANT_ID,
    trial_col: str = Con.TRIAL_ID,
    split_fn: Callable = leave_one_trial_out_for_participant,
    target_col: str = Con.IS_CORRECT_COLUMN,
    feature_cols: Optional[Sequence[str]] = None,
    pref_specs: Optional[Sequence[Tuple[str, str]]] = Con.PREF_SPECS,
    pref_extreme_mode: str = "polarity",
    keep_cols: Optional[Sequence[str]] = None,
    coef_ci_method: str = "wald",
    coef_ci_cluster: str = "row",
    coef_ci: float = 0.95,
    coef_n_boot: int = 3000,
    coef_seed: int = 42,
    coef_top_k: Optional[int] = None,
) -> Dict[str, CorrectnessEvaluationResult]:
    """
    Evaluate a logistic-regression model per participant using
    leave-one-trial-out splitting on a prepared trial-level dataframe.

    Returns
    -------
    Dict[str, CorrectnessEvaluationResult]
        results[participant_id] = evaluation result for that participant
    """
    pref_specs = pref_specs if pref_specs is not None else Con.PREF_SPECS

    trial_df = build_trial_level_model_df(
        df=df,
        pref_specs=pref_specs,
        pref_extreme_mode=pref_extreme_mode,
        keep_cols=keep_cols,
        target_col=target_col,
        include_area_features=True,
        include_derived_features=True,
        include_last_visited_answer_features=True,
    ).copy()

    feat_cols = list(feature_cols) if feature_cols is not None else list(get_full_feature_cols(trial_df))

    participants = (
        trial_df[participant_col]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    results: Dict[str, CorrectnessEvaluationResult] = {}

    for pid in participants:
        train_df, test_df = split_fn(
            df=trial_df,
            participant_id=pid,
            participant_col=participant_col,
            trial_col=trial_col,
        )

        if train_df[target_col].dropna().astype(int).nunique() < 2:
            continue

        model = model_builder()

        res = evaluate_single_model_on_prepared_split(
            model=model,
            train_df=train_df,
            test_df=test_df,
            target_col=target_col,
            feature_cols=feat_cols,
            coef_kwargs={
                "top_k": coef_top_k,
                "ci_method": coef_ci_method,
                "ci_cluster": coef_ci_cluster,
                "ci": coef_ci,
                "n_boot": coef_n_boot,
                "seed": coef_seed,
            },
        )

        results[str(pid)] = res

    return results