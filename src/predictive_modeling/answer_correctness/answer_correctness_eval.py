# answer_correctness_eval.py

from dataclasses import dataclass
from typing import Dict, Sequence, Callable, List, Literal, Optional, Mapping, Tuple, Any

import numpy as np
import pandas as pd

import statsmodels.api as sm


from src import constants as Con
from src.predictive_modeling.answer_correctness.answer_correctness_data import (
    build_trial_level_with_area_metrics,
)
from predictive_modeling.answer_correctness.models.logreg_models import (
    AnswerCorrectnessModel,
)

from src.predictive_modeling.common.data_utils import (
    group_vise_train_test_split,
    leave_one_trial_out_for_participant,
)
from src.predictive_modeling.answer_correctness.answer_correctness_data import (
    build_trial_level_all_features
)


def build_feature_trial_df(
    df: pd.DataFrame,
    feature_cols: Optional[Sequence[str]] = None,
    group_cols: Sequence[str] = (Con.PARTICIPANT_ID, Con.TRIAL_ID),
    target_col: str = Con.IS_CORRECT_COLUMN,
    pref_specs: Optional[Sequence[Tuple[str, str]]] = None,
    pref_extreme_mode: str = "polarity",
    keep_extra_cols: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """
    Build trial-level feature dataframe without fitting any model.
    """
    trial_df = build_trial_level_all_features(
        df,
        group_cols=group_cols,
        pref_specs=pref_specs,
        pref_extreme_mode=pref_extreme_mode,
        keep_cols=keep_extra_cols,
    )

    if feature_cols is None:
        return trial_df

    keep_cols: List[str] = list(group_cols)

    if target_col in trial_df.columns:
        keep_cols.append(target_col)
    if keep_extra_cols is not None:
        keep_cols.extend([c for c in keep_extra_cols if c in trial_df.columns])

    keep_cols.extend([c for c in feature_cols if c in trial_df.columns])
    keep_cols = list(dict.fromkeys(keep_cols))

    return trial_df[keep_cols].copy()


@dataclass
class CorrectnessEvaluationResult:
    train_df: pd.DataFrame
    test_df: pd.DataFrame
    y_true: np.ndarray
    y_pred: np.ndarray
    y_prob: np.ndarray
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

    feature_cols_by_model: Optional[Mapping[str, Sequence[str]]] = None,
    feature_cols: Optional[Sequence[str]] = None,
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
        feat_cols = None
        if feature_cols is not None:
            feat_cols = list(feature_cols)
        elif feature_cols_by_model is not None and model.name in feature_cols_by_model:
            model_feature_cols = feature_cols_by_model[model.name]
            feat_cols = None if model_feature_cols is None else list(model_feature_cols)

        model.fit(train_df, target_col=target_col, feature_cols=feat_cols)
        y_pred = model.predict(test_df, feature_cols=feat_cols)

        y_prob = None
        if hasattr(model, "predict_proba"):
            try:
                y_prob = model.predict_proba(test_df, feature_cols=feat_cols)
            except Exception:
                y_prob = None


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
                feature_cols=feat_cols,
            )

        results[model.name] = CorrectnessEvaluationResult(
            train_df=train_df,
            test_df=test_df,
            y_true=y_true,
            y_pred=y_pred,
            y_prob=y_prob,
            accuracy=acc,
            n_test=len(test_df),
            n_positive=int((y_true == 1).sum()),
            n_negative=int((y_true == 0).sum()),
            coef_summary=coef_summary,
        )

    return results


def evaluate_glmer_on_answer_correctness(
    df: pd.DataFrame,
    model,
    group_cols: Sequence[str] = (Con.PARTICIPANT_ID, Con.TRIAL_ID),
    split_group_cols: List[str] = [Con.PARTICIPANT_ID, Con.TRIAL_ID],
    test_size: float = 0.2,
    random_state: int = 42,
    builder_fn: Callable = None,
    split_fn: Callable = None,
    target_col: str = Con.IS_CORRECT_COLUMN,
    feature_cols: Optional[Sequence[str]] = None,
    participant_col: str = Con.PARTICIPANT_ID,
    text_col: str = Con.TEXT_ID_WITH_Q_COLUMN,
    use_rfx: bool = False,
) -> Dict[str, CorrectnessEvaluationResult]:
    """
    Evaluation pipeline for the GLMER answer-correctness model.

    Returns the same structure as evaluate_models_on_answer_correctness:
        {model_name: CorrectnessEvaluationResult}
    """

    if builder_fn is None:
        raise ValueError("builder_fn must be provided for GLMER evaluation.")
    if split_fn is None:
        raise ValueError("split_fn must be provided for GLMER evaluation.")

    train_raw, test_raw = split_fn(
        df,
        test_size=test_size,
        random_state=random_state,
        group_cols=split_group_cols,
    )

    train_df = builder_fn(train_raw, group_cols=group_cols)
    test_df = builder_fn(test_raw, group_cols=group_cols)

    y_true = test_df[target_col].astype(int).to_numpy()
    feat_cols = None if feature_cols is None else list(feature_cols)

    model.fit(
        train_df=train_df,
        target_col=target_col,
        feature_cols=feat_cols,
        participant_col=participant_col,
        text_col=text_col,
    )

    y_pred = model.predict(
        test_df,
        threshold=0.5,
        feature_cols=feat_cols,
        target_col=target_col,
        participant_col=participant_col,
        text_col=text_col,
        use_rfx=use_rfx,
    )

    y_prob = None
    if hasattr(model, "predict_proba"):
        try:
            y_prob = model.predict_proba(
                test_df,
                feature_cols=feat_cols,
                target_col=target_col,
                participant_col=participant_col,
                text_col=text_col,
                use_rfx=use_rfx,
            )
        except Exception:
            y_prob = None

    acc = float((y_true == y_pred).mean())

    coef_summary = None
    if hasattr(model, "get_coef_summary") and callable(model.get_coef_summary):
        coef_summary = model.get_coef_summary()

    results: Dict[str, CorrectnessEvaluationResult] = {
        model.name: CorrectnessEvaluationResult(
            train_df=train_df,
            test_df=test_df,
            y_true=y_true,
            y_pred=y_pred,
            y_prob=y_prob,
            accuracy=acc,
            n_test=len(test_df),
            n_positive=int((y_true == 1).sum()),
            n_negative=int((y_true == 0).sum()),
            coef_summary=coef_summary,
        )
    }

    return results


def evaluate_julia_glmer_on_answer_correctness(
    df: pd.DataFrame,
    model,
    group_cols: Sequence[str] = (Con.PARTICIPANT_ID, Con.TRIAL_ID),
    split_group_cols: List[str] = [Con.PARTICIPANT_ID, Con.TRIAL_ID],
    test_size: float = 0.2,
    random_state: int = 42,
    builder_fn: callable = None,
    split_fn: callable = None,
    target_col: str = Con.IS_CORRECT_COLUMN,
    feature_cols: Optional[Sequence[str]] = None,
    participant_col: str = Con.PARTICIPANT_ID,
    text_col: str = Con.TEXT_ID_WITH_Q_COLUMN,
    use_rfx: bool = False,
) -> Dict[str, CorrectnessEvaluationResult]:
    """
    Evaluation pipeline for the Julia GLMER answer-correctness model.

    Same structure as the old GLMER pipeline:
        {model_name: CorrectnessEvaluationResult}
    """
    if builder_fn is None:
        raise ValueError("builder_fn must be provided for GLMER evaluation.")
    if split_fn is None:
        raise ValueError("split_fn must be provided for GLMER evaluation.")

    train_raw, test_raw = split_fn(
        df,
        test_size=test_size,
        random_state=random_state,
        group_cols=split_group_cols,
    )

    train_df = builder_fn(train_raw, group_cols=group_cols)
    test_df = builder_fn(test_raw, group_cols=group_cols)

    y_true = test_df[target_col].astype(int).to_numpy()
    feat_cols = None if feature_cols is None else list(feature_cols)

    model.fit(
        train_df=train_df,
        target_col=target_col,
        feature_cols=feat_cols,
        participant_col=participant_col,
        text_col=text_col,
    )

    y_pred = model.predict(
        test_df,
        threshold=0.5,
        feature_cols=feat_cols,
        target_col=target_col,
        participant_col=participant_col,
        text_col=text_col,
        use_rfx=use_rfx,
    )

    y_prob = None
    if hasattr(model, "predict_proba"):
        try:
            y_prob = model.predict_proba(
                test_df,
                feature_cols=feat_cols,
                target_col=target_col,
                participant_col=participant_col,
                text_col=text_col,
                use_rfx=use_rfx,
            )
        except Exception:
            y_prob = None

    acc = float((y_true == y_pred).mean())

    coef_summary = None
    if hasattr(model, "get_coef_summary") and callable(model.get_coef_summary):
        coef_summary = model.get_coef_summary()

    results: Dict[str, CorrectnessEvaluationResult] = {
        model.name: CorrectnessEvaluationResult(
            train_df=train_df,
            test_df=test_df,
            y_true=y_true,
            y_pred=y_pred,
            y_prob=y_prob,
            accuracy=acc,
            n_test=len(test_df),
            n_positive=int((y_true == 1).sum()),
            n_negative=int((y_true == 0).sum()),
            coef_summary=coef_summary,
        )
    }

    return results



def fit_julia_glmer_on_answer_correctness_all(
    df: pd.DataFrame,
    model,
    group_cols: Sequence[str] = (Con.PARTICIPANT_ID, Con.TRIAL_ID),
    builder_fn: callable = None,
    target_col: str = Con.IS_CORRECT_COLUMN,
    feature_cols: Optional[Sequence[str]] = None,
    participant_col: str = Con.PARTICIPANT_ID,
    text_col: str = Con.TEXT_ID_WITH_Q_COLUMN,
) -> Dict[str, Any]:
    if builder_fn is None:
        raise ValueError("builder_fn must be provided.")

    full_df = builder_fn(df, group_cols=group_cols)
    feat_cols = None if feature_cols is None else list(feature_cols)

    model.fit(
        train_df=full_df,
        target_col=target_col,
        feature_cols=feat_cols,
        participant_col=participant_col,
        text_col=text_col,
    )

    coef_summary = model.get_coef_summary() if hasattr(model, "get_coef_summary") else None
    random_effects = model.get_random_effects() if hasattr(model, "get_random_effects") else None
    random_varcorr = (
        model.get_random_effect_variance_summary()
        if hasattr(model, "get_random_effect_variance_summary")
        else None
    )

    return {
        model.name: {
            "fit_df": full_df,
            "n_rows": len(full_df),
            "n_positive": int((full_df[target_col] == 1).sum()),
            "n_negative": int((full_df[target_col] == 0).sum()),
            "coef_summary": coef_summary,
            "random_effects": random_effects,
            "random_effect_variance_summary": random_varcorr,
        }
    }




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






