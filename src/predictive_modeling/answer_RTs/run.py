# run.py
#
# Orchestration for the answer reading-time regression: build the modeling frame,
# fit one model per answer (A / B / C / D) on paragraph-only features, and report
# held-out performance. Splits are grouped by participant to avoid leakage.

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit

from src.constants import PARTICIPANT_ID, TRIAL_ID_COLS
from src.data_paths import PARAGRAPH_SPAN_FEATURES_PATH, READY_ALL_FEATURES_PATH
from src.predictive_modeling.common.prepared_dataset import PreparedTrialDataset
from src.predictive_modeling.answer_RTs.model_data import (
    ANSWER_LABELS,
    DEFAULT_TARGET_RT_METRIC,
    build_answer_rt_model_df,
    make_answer_rt_dataset,
)
from src.predictive_modeling.answer_RTs.models.gbm_model import (
    GBM_MODEL_KINDS,
    TrialLevelGBMModel,
)
from src.predictive_modeling.answer_RTs.models.linreg_model import (
    DEFAULT_RIDGE_ALPHAS,
    TrialLevelLinRegModel,
)


def build_answer_rt_model(
    model_kind: str = "ridge",
    alpha: Optional[float] = None,
    alphas: Optional[Sequence[float]] = None,
    n_cv_folds: int = 5,
    model_kwargs: Optional[Dict[str, object]] = None,
):
    """The model for a `model_kind`, with its own hyperparameters applied.

    "hist_gbm" builds the gradient-boosting model, everything else the linear
    one. `model_kwargs` is forwarded to whichever is built (e.g.
    `{"learning_rate": 0.03}` for the GBM, `{"fill_value": 0.0}` for the linear
    model), so a caller can reach a model's own knobs without this signature
    growing one parameter per estimator.
    """
    kwargs = dict(model_kwargs or {})
    if model_kind in GBM_MODEL_KINDS:
        return TrialLevelGBMModel(model_kind=model_kind, **kwargs)
    return TrialLevelLinRegModel(
        model_kind=model_kind,
        alpha=alpha,
        alphas=alphas,
        n_cv_folds=n_cv_folds,
        **kwargs,
    )


def _grouped_split(
    df: pd.DataFrame,
    test_size: float,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Train/test row indices split by participant (no participant in both)."""
    splitter = GroupShuffleSplit(
        n_splits=1, test_size=test_size, random_state=random_state
    )
    groups = df[PARTICIPANT_ID].to_numpy()
    train_idx, test_idx = next(splitter.split(df, groups=groups))
    return train_idx, test_idx


def evaluate_answer_rt_model(
    dataset: PreparedTrialDataset,
    model: Optional[TrialLevelLinRegModel] = None,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Dict[str, object]:
    """Fit on a participant-grouped train split and score on the held-out split.

    Returns a dict with the fitted model, metrics, split sizes, and the
    standardized-coefficient summary. A mean-predictor baseline RMSE is included
    for reference.
    """
    model = model or TrialLevelLinRegModel()
    df = dataset.df

    train_idx, test_idx = _grouped_split(df, test_size, random_state)
    train_df = df.iloc[train_idx]
    test_df = df.iloc[test_idx]

    # Participant ids fold whatever inner split the model does -- the alpha
    # search for the CV kinds, the early-stopping slice for the GBM -- keeping
    # it grouped the same way as the outer train/test split.
    model.fit(
        train_df,
        target_col=dataset.target_col,
        feature_cols=dataset.feature_cols,
        groups=train_df[PARTICIPANT_ID].to_numpy(),
    )

    y_test = pd.to_numeric(test_df[dataset.target_col], errors="coerce").to_numpy()
    y_pred = model.predict(test_df)

    baseline = np.full_like(y_test, train_df[dataset.target_col].mean(), dtype=float)

    metrics = {
        "target": dataset.target_col,
        "model_kind": getattr(model, "model_kind", None),
        "alpha": getattr(model, "alpha_", None),
        # Boosting rounds the GBM stopped at; None for the linear models.
        "n_iter": getattr(model, "n_iter_", None),
        "n_train": int(len(train_df)),
        "n_test": int(len(test_df)),
        "n_features": len(dataset.feature_cols),
        "r2": float(r2_score(y_test, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
        "mae": float(mean_absolute_error(y_test, y_pred)),
        "baseline_rmse": float(np.sqrt(mean_squared_error(y_test, baseline))),
    }

    # Held-out predictions, keyed for downstream plotting / regime breakdowns.
    preview_col = "question_preview"
    predictions = pd.DataFrame(
        {
            PARTICIPANT_ID: test_df[PARTICIPANT_ID].to_numpy(),
            "y_true": y_test,
            "y_pred": y_pred,
        }
    )
    if preview_col in test_df.columns:
        predictions[preview_col] = test_df[preview_col].to_numpy()

    return {
        "model": model,
        "metrics": metrics,
        "coef_summary": model.get_coef_summary(),
        # What that summary's `coef` column actually is -- signed standardized
        # coefficients for the linear models, unsigned permutation importance
        # for the GBM -- so figures can label themselves correctly.
        "coef_kind": getattr(model, "coef_kind", "standardized coefficient"),
        "predictions": predictions,
    }


def run_answer_rt_regression(
    model_df: Optional[pd.DataFrame] = None,
    answers: Sequence[str] = ANSWER_LABELS,
    rt_metric: str = DEFAULT_TARGET_RT_METRIC,
    model_kind: str = "ridge",
    alpha: Optional[float] = None,
    alphas: Optional[Sequence[float]] = None,
    n_cv_folds: int = 5,
    model_kwargs: Optional[Dict[str, object]] = None,
    feature_cols: Optional[Sequence[str]] = None,
    feature_set: Optional[str] = None,
    include_answer_text: bool = True,
    log_target: bool = False,
    test_size: float = 0.2,
    random_state: int = 42,
    paragraph_features_path: Path = PARAGRAPH_SPAN_FEATURES_PATH,
    ready_features_path: Path = READY_ALL_FEATURES_PATH,
    verbose: bool = True,
) -> Dict[str, object]:
    """Fit and evaluate one regression per answer.

    `feature_set` names the predictor set in plain words (e.g. "ours (span
    metrics)" / "external (EyeBench)"). It is carried through to the result dict
    and to every metrics row, so figures and comparison tables can say which set
    they describe instead of relying on the caller to remember.

    Defaults to a fixed ridge at the pre-tuned penalty for the predictor set's
    width (`linreg_model.tuned_alpha`), so no search runs; pass an explicit
    `alpha` to override it. `model_kind="ridge_cv"` re-runs the participant-
    grouped inner search instead -- worth doing when the predictor set or target
    has changed -- and `alphas` overrides its candidate grid.

    `model_kind="hist_gbm"` swaps the linear model for gradient-boosted trees
    (`gbm_model.TrialLevelGBMModel`), which fit interactions and nonlinearity the
    ridge cannot. `model_kwargs` reaches the chosen model's own hyperparameters.

    `log_target` regresses on log RT (see `make_answer_rt_dataset`); metrics are
    then on the log scale, so R2 is the share of variance in *log* RT explained
    and RMSE/MAE are in log units -- neither is comparable to a raw-RT run.

    Returns {"feature_set": <label>, "metrics": <DataFrame, one row per answer>,
    "results": {answer: ...}}.
    """
    if model_df is None:
        model_df = build_answer_rt_model_df(
            paragraph_features_path=paragraph_features_path,
            ready_features_path=ready_features_path,
            include_answer_text=include_answer_text,
        )

    results: Dict[str, object] = {}
    metric_rows = []
    for answer in answers:
        dataset = make_answer_rt_dataset(
            model_df,
            answer=answer,
            rt_metric=rt_metric,
            feature_cols=feature_cols,
            log_target=log_target,
            include_answer_text=include_answer_text,
        )
        model = build_answer_rt_model(
            model_kind=model_kind,
            alpha=alpha,
            alphas=alphas,
            n_cv_folds=n_cv_folds,
            model_kwargs=model_kwargs,
        )
        res = evaluate_answer_rt_model(
            dataset, model=model, test_size=test_size, random_state=random_state
        )
        results[answer] = res
        row = {"feature_set": feature_set, "answer": answer, **res["metrics"]}
        metric_rows.append(row)
        if verbose:
            m = res["metrics"]
            alpha_str = "-" if m["alpha"] is None else f"{m['alpha']:g}"
            print(
                f"answer {answer}: R2={m['r2']:.3f} "
                f"RMSE={m['rmse']:.3f} (baseline {m['baseline_rmse']:.3f}) "
                f"MAE={m['mae']:.3f}  [alpha={alpha_str}, "
                f"n_feat={m['n_features']}, n_train={m['n_train']}, "
                f"n_test={m['n_test']}]"
            )

    return {
        "feature_set": feature_set,
        "metrics": pd.DataFrame(metric_rows),
        "results": results,
    }


# ---------------------------------------------------------------------------
# Regularization sweep
# ---------------------------------------------------------------------------

def sweep_answer_rt_alphas(
    alphas: Sequence[float] = DEFAULT_RIDGE_ALPHAS,
    model_df: Optional[pd.DataFrame] = None,
    answers: Sequence[str] = ANSWER_LABELS,
    include_cv: bool = True,
    verbose: bool = True,
    **run_kwargs,
) -> pd.DataFrame:
    """Held-out metrics for a grid of fixed ridge alphas, plus the CV pick.

    A sanity check on the CV choice: it shows how flat (or not) the test metric
    is across the penalty range, so a "the CV picked alpha=1000" result can be
    read as either a real optimum or an arbitrary point on a plateau.

    Returns a tidy frame with one row per (setting, answer).
    """
    rows = []
    settings = [("ridge", a) for a in alphas]
    if include_cv:
        settings.append(("ridge_cv", None))

    for kind, alpha in settings:
        res = run_answer_rt_regression(
            model_df=model_df,
            answers=answers,
            model_kind=kind,
            alpha=alpha,
            verbose=False,
            **run_kwargs,
        )
        metrics = res["metrics"].copy()
        metrics.insert(0, "setting", kind if alpha is None else f"ridge@{alpha:g}")
        rows.append(metrics)
        if verbose:
            mean_r2 = metrics["r2"].mean()
            picked = metrics["alpha"].iloc[0]
            print(
                f"{metrics['setting'].iloc[0]:>16}: mean R2={mean_r2:+.4f} "
                f"mean RMSE={metrics['rmse'].mean():.3f} "
                f"(alpha={picked:g})"
            )

    return pd.concat(rows, ignore_index=True)


if __name__ == "__main__":
    run_answer_rt_regression()
