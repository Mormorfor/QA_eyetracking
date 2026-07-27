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
from src.predictive_modeling.answer_RTs.models.linreg_model import TrialLevelLinRegModel


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

    model.fit(train_df, target_col=dataset.target_col, feature_cols=dataset.feature_cols)

    y_test = pd.to_numeric(test_df[dataset.target_col], errors="coerce").to_numpy()
    y_pred = model.predict(test_df)

    baseline = np.full_like(y_test, train_df[dataset.target_col].mean(), dtype=float)

    metrics = {
        "target": dataset.target_col,
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
        "predictions": predictions,
    }


def run_answer_rt_regression(
    model_df: Optional[pd.DataFrame] = None,
    answers: Sequence[str] = ANSWER_LABELS,
    rt_metric: str = DEFAULT_TARGET_RT_METRIC,
    model_kind: str = "ridge",
    alpha: float = 1.0,
    feature_cols: Optional[Sequence[str]] = None,
    test_size: float = 0.2,
    random_state: int = 42,
    paragraph_features_path: Path = PARAGRAPH_SPAN_FEATURES_PATH,
    ready_features_path: Path = READY_ALL_FEATURES_PATH,
    verbose: bool = True,
) -> Dict[str, object]:
    """Fit and evaluate one regression per answer.

    Returns {"metrics": <DataFrame, one row per answer>, "results": {answer: ...}}.
    """
    if model_df is None:
        model_df = build_answer_rt_model_df(
            paragraph_features_path=paragraph_features_path,
            ready_features_path=ready_features_path,
        )

    results: Dict[str, object] = {}
    metric_rows = []
    for answer in answers:
        dataset = make_answer_rt_dataset(
            model_df, answer=answer, rt_metric=rt_metric, feature_cols=feature_cols
        )
        model = TrialLevelLinRegModel(model_kind=model_kind, alpha=alpha)
        res = evaluate_answer_rt_model(
            dataset, model=model, test_size=test_size, random_state=random_state
        )
        results[answer] = res
        row = {"answer": answer, **res["metrics"]}
        metric_rows.append(row)
        if verbose:
            m = res["metrics"]
            print(
                f"answer {answer}: R2={m['r2']:.3f} "
                f"RMSE={m['rmse']:.3f} (baseline {m['baseline_rmse']:.3f}) "
                f"MAE={m['mae']:.3f}  [n_train={m['n_train']}, n_test={m['n_test']}]"
            )

    return {"metrics": pd.DataFrame(metric_rows), "results": results}


if __name__ == "__main__":
    run_answer_rt_regression()
