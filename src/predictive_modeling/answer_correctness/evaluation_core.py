from __future__ import annotations

from dataclasses import dataclass

from typing import Sequence, Optional, Mapping, Dict, Any
import numpy as np
import pandas as pd

from src import constants as Con


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


def evaluate_single_model_on_prepared_split(
    model,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    target_col: str = Con.IS_CORRECT_COLUMN,
    feature_cols: Sequence[str],
    fit_kwargs: Optional[dict[str, Any]] = None,
    predict_kwargs: Optional[dict[str, Any]] = None,
    predict_proba_kwargs: Optional[dict[str, Any]] = None,
    coef_kwargs: Optional[dict[str, Any]] = None,
) -> CorrectnessEvaluationResult:
    """
    Fit one model on an already prepared train_df and evaluate on test_df.

    Assumes the model implements:
    - fit(train_df, target_col=..., feature_cols=..., **fit_kwargs)
    - predict(test_df, feature_cols=..., **predict_kwargs)
    - predict_proba(test_df, feature_cols=..., **predict_proba_kwargs)
    - get_coef_summary(...)
    """
    feat_cols = list(feature_cols)
    fit_kwargs = dict(fit_kwargs or {})
    predict_kwargs = dict(predict_kwargs or {})
    predict_proba_kwargs = dict(predict_proba_kwargs or {})
    coef_kwargs = dict(coef_kwargs or {})

    y_true = test_df[target_col].astype(int).to_numpy()

    model.fit(
        train_df=train_df,
        target_col=target_col,
        feature_cols=feat_cols,
        **fit_kwargs,
    )

    y_pred = model.predict(
        test_df,
        feature_cols=feat_cols,
        **predict_kwargs,
    )
    y_pred = np.asarray(y_pred).reshape(-1).astype(int)

    y_prob = model.predict_proba(
        test_df,
        feature_cols=feat_cols,
        **predict_proba_kwargs,
    )
    y_prob = np.asarray(y_prob).reshape(-1).astype(float)

    coef_summary = model.get_coef_summary(
        train_df=train_df,
        feature_cols=feat_cols,
        **coef_kwargs,
    )

    return CorrectnessEvaluationResult(
        train_df=train_df,
        test_df=test_df,
        y_true=y_true,
        y_pred=y_pred,
        y_prob=y_prob,
        accuracy=float((y_true == y_pred).mean()),
        n_test=len(test_df),
        n_positive=int((y_true == 1).sum()),
        n_negative=int((y_true == 0).sum()),
        coef_summary=coef_summary,
    )


def evaluate_models_on_prepared_split(
    models: Sequence,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    target_col: str = Con.IS_CORRECT_COLUMN,
    feature_cols: Optional[Sequence[str]] = None,
    feature_cols_by_model: Optional[Mapping[str, Sequence[str]]] = None,
    fit_kwargs_by_model: Optional[Mapping[str, dict[str, Any]]] = None,
    predict_kwargs_by_model: Optional[Mapping[str, dict[str, Any]]] = None,
    predict_proba_kwargs_by_model: Optional[Mapping[str, dict[str, Any]]] = None,
    coef_kwargs_by_model: Optional[Mapping[str, dict[str, Any]]] = None,
) -> Dict[str, CorrectnessEvaluationResult]:
    """
    Evaluate one or more models on an already prepared train/test split.

    Use:
    - feature_cols for one shared feature set across all models, or
    - feature_cols_by_model for per-model feature sets.
    """
    results: Dict[str, CorrectnessEvaluationResult] = {}

    fit_kwargs_by_model = dict(fit_kwargs_by_model or {})
    predict_kwargs_by_model = dict(predict_kwargs_by_model or {})
    predict_proba_kwargs_by_model = dict(predict_proba_kwargs_by_model or {})
    coef_kwargs_by_model = dict(coef_kwargs_by_model or {})

    for model in models:
        model_name = model.name

        if feature_cols_by_model is not None and model_name in feature_cols_by_model:
            feat_cols = list(feature_cols_by_model[model_name])
        elif feature_cols is not None:
            feat_cols = list(feature_cols)
        else:
            raise ValueError(
                f"No feature columns provided for model '{model_name}'."
            )

        results[model_name] = evaluate_single_model_on_prepared_split(
            model=model,
            train_df=train_df,
            test_df=test_df,
            target_col=target_col,
            feature_cols=feat_cols,
            fit_kwargs=fit_kwargs_by_model.get(model_name),
            predict_kwargs=predict_kwargs_by_model.get(model_name),
            predict_proba_kwargs=predict_proba_kwargs_by_model.get(model_name),
            coef_kwargs=coef_kwargs_by_model.get(model_name),
        )

    return results


def fit_model_on_prepared_full_data(
    model,
    fit_df: pd.DataFrame,
    *,
    target_col: str = Con.IS_CORRECT_COLUMN,
    feature_cols: Optional[Sequence[str]] = None,
    fit_kwargs: Optional[dict[str, Any]] = None,
    coef_kwargs: Optional[dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Fit one model on an already prepared full dataframe (no train/test split).

    Intended for:
    - full-data coefficient inspection
    - full-data random-effects inspection
    - fit-all reporting workflows
    """
    feat_cols = None if feature_cols is None else list(feature_cols)
    fit_kwargs = dict(fit_kwargs or {})
    coef_kwargs = dict(coef_kwargs or {})

    model.fit(
        train_df=fit_df,
        target_col=target_col,
        feature_cols=feat_cols,
        **fit_kwargs,
    )

    coef_summary = model.get_coef_summary(
        train_df=fit_df,
        feature_cols=feat_cols,
        **coef_kwargs,
    )

    random_effects = model.get_random_effects()
    random_varcorr = model.get_random_effect_variance_summary()

    return {
        "fit_df": fit_df,
        "n_rows": len(fit_df),
        "n_positive": int((fit_df[target_col] == 1).sum()),
        "n_negative": int((fit_df[target_col] == 0).sum()),
        "coef_summary": coef_summary,
        "random_effects": random_effects,
        "random_effect_variance_summary": random_varcorr,
    }