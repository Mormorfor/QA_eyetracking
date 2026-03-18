from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import src.constants as Con
# ---------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------

@dataclass
class FoldRegimeEvaluationResult:
    fold_idx: int
    regime: str
    train_df: pd.DataFrame
    eval_df: pd.DataFrame
    y_true: np.ndarray
    y_pred: np.ndarray
    y_prob: Optional[np.ndarray]
    accuracy: float
    n_eval: int
    n_positive: int
    n_negative: int
    coef_summary: Optional[pd.DataFrame] = None


@dataclass
class CrossValidationRunResult:
    per_fold_results: Dict[str, Dict[int, Dict[str, FoldRegimeEvaluationResult]]]
    summary_df: pd.DataFrame
    summary_by_regime_df: pd.DataFrame
    summary_overall_df: pd.DataFrame


# ---------------------------------------------------------------------
# Fold loading / assignment
# ---------------------------------------------------------------------

def load_fold_assignment_csv(
    fold_csv_path: str | Path,
    *,
    participant_col_fold: str = "participant_id",
    text_col_fold: str = "unique_paragraph_id",
    trial_col_fold: str = "unique_trial_id",
    regime_col_fold: str = "regime",
) -> pd.DataFrame:
    """
    Load one fold CSV containing train/val/test regime assignments.
    """
    fold_df = pd.read_csv(fold_csv_path)


    #for col in [participant_col_fold, text_col_fold, trial_col_fold, regime_col_fold]:
    #        fold_df[col] = fold_df[col].astype(str).str.strip()

    return fold_df


def attach_fold_regimes(
    df: pd.DataFrame,
    fold_df: pd.DataFrame,
    *,
    df_participant_col: str = Con.PARTICIPANT_ID,
    df_text_col: str = Con.TEXT_ID_COLUMN,
    fold_participant_col: str = "participant_id",
    fold_text_col: str = "unique_paragraph_id",
    fold_regime_col: str = "regime",
) -> pd.DataFrame:
    """
    Attach regime labels using (participant_id, text_id) join ONLY.
    Includes normalization (str, strip, lower).
    Keeps only the fold columns needed for the join + regime.
    """

    out = df.copy()
    fold_df = fold_df.copy()

    out[df_participant_col] = (
        out[df_participant_col].astype(str).str.strip().str.lower()
    )
    out[df_text_col] = (
        out[df_text_col].astype(str).str.strip().str.lower()
    )

    fold_df = fold_df[
        [fold_participant_col, fold_text_col, fold_regime_col]
    ].copy()

    fold_df[fold_participant_col] = (
        fold_df[fold_participant_col].astype(str).str.strip().str.lower()
    )
    fold_df[fold_text_col] = (
        fold_df[fold_text_col].astype(str).str.strip().str.lower()
    )
    fold_df[fold_regime_col] = (
        fold_df[fold_regime_col].astype(str).str.strip()
    )

    fold_df = fold_df.drop_duplicates()

    assign_df = fold_df.rename(
        columns={
            fold_participant_col: df_participant_col,
            fold_text_col: df_text_col,
        }
    )

    out = out.merge(
        assign_df,
        on=[df_participant_col, df_text_col],
        how="inner",
    )

    return out

# ---------------------------------------------------------------------
# One-fold evaluation
# ---------------------------------------------------------------------

def evaluate_one_fold_on_regimes(
    df: pd.DataFrame,
    *,
    model_builder: Callable[[], Any],
    builder_fn: Callable[[pd.DataFrame], pd.DataFrame],
    target_col: str,
    train_regime: str = "train_train",
    eval_regimes: Optional[Sequence[str]] = None,
    coef_ci_method: str = "wald",
    coef_ci_cluster: str = "row",
    coef_ci: float = 0.95,
    coef_n_boot: int = 3000,
    coef_seed: int = 42,
    coef_top_k: Optional[int] = None,
    feature_cols: Optional[Sequence[str]] = None,
    fold_idx: int = -1,
) -> Dict[str, FoldRegimeEvaluationResult]:
    """
    Fit on train_regime and evaluate on each requested regime.
    """

    if eval_regimes is None:
        eval_regimes = [
            "val_seen_subject_unseen_item",
            "test_seen_subject_unseen_item",
            "val_unseen_subject_seen_item",
            "test_unseen_subject_seen_item",
            "val_unseen_subject_unseen_item",
            "test_unseen_subject_unseen_item",
        ]

    train_raw = df[df["regime"] == train_regime].copy()

    train_df = builder_fn(train_raw)
    results: Dict[str, FoldRegimeEvaluationResult] = {}

    for regime in eval_regimes:
        eval_raw = df[df["regime"] == regime].copy()
        eval_df = builder_fn(eval_raw)

        model = model_builder()
        model.fit(train_df, target_col=target_col, feature_cols=feature_cols)

        y_true = eval_df[target_col].astype(int).to_numpy()
        y_pred = model.predict(eval_df, feature_cols=feature_cols)
        y_prob = model.predict_proba(eval_df, feature_cols=feature_cols)

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
                feature_cols=feature_cols,
            )

        results[regime] = FoldRegimeEvaluationResult(
            fold_idx=fold_idx,
            regime=regime,
            train_df=train_df,
            eval_df=eval_df,
            y_true=y_true,
            y_pred=y_pred,
            y_prob=y_prob,
            accuracy=acc,
            n_eval=len(eval_df),
            n_positive=int((y_true == 1).sum()),
            n_negative=int((y_true == 0).sum()),
            coef_summary=coef_summary,
        )

    return results


# ---------------------------------------------------------------------
# 10-fold CV runner
# ---------------------------------------------------------------------

def run_cross_validation_on_predefined_folds(
    df: pd.DataFrame,
    *,
    fold_dir: str | Path,
    model_builders: Mapping[str, Callable[[], Any]],
    builder_fn: Callable[[pd.DataFrame], pd.DataFrame],
    target_col: str,
    n_folds: int = 10,
    fold_filename_template: str = "fold_{fold_idx}_trial_ids_by_regime.csv",
    df_participant_col: str = Con.PARTICIPANT_ID,
    df_text_col: str = Con.TEXT_ID_COLUMN,
    eval_regimes: Optional[Sequence[str]] = None,
    feature_cols_by_model: Optional[Mapping[str, Sequence[str]]] = None,
    coef_ci_method: str = "wald",
    coef_ci_cluster: str = "row",
    coef_ci: float = 0.95,
    coef_n_boot: int = 3000,
    coef_seed: int = 42,
    coef_top_k: Optional[int] = None,
) -> CrossValidationRunResult:
    """
    Run cross-validation using predefined fold assignment CSVs.
    """
    fold_dir = Path(fold_dir)

    per_fold_results: Dict[str, Dict[int, Dict[str, FoldRegimeEvaluationResult]]] = {
        model_name: {} for model_name in model_builders
    }

    rows_summary: List[Dict[str, Any]] = []

    for fold_idx in range(n_folds):
        fold_path = fold_dir / fold_filename_template.format(fold_idx=fold_idx)
        fold_assign_df = load_fold_assignment_csv(fold_path)

        df_fold = attach_fold_regimes(
            df=df,
            fold_df=fold_assign_df,
            df_participant_col=df_participant_col,
            df_text_col=df_text_col,
        )

        for model_name, model_builder in model_builders.items():
            feat_cols = None
            if feature_cols_by_model is not None and model_name in feature_cols_by_model:
                cols = feature_cols_by_model[model_name]
                feat_cols = None if cols is None else list(cols)

            fold_results = evaluate_one_fold_on_regimes(
                df_fold,
                model_builder=model_builder,
                builder_fn=builder_fn,
                target_col=target_col,
                eval_regimes=eval_regimes,
                coef_ci_method=coef_ci_method,
                coef_ci_cluster=coef_ci_cluster,
                coef_ci=coef_ci,
                coef_n_boot=coef_n_boot,
                coef_seed=coef_seed,
                coef_top_k=coef_top_k,
                feature_cols=feat_cols,
                fold_idx=fold_idx,
            )

            per_fold_results[model_name][fold_idx] = fold_results

            for regime, res in fold_results.items():
                rows_summary.append(
                    {
                        "model": model_name,
                        "fold": fold_idx,
                        "regime": regime,
                        "accuracy": res.accuracy,
                        "n_eval": res.n_eval,
                        "n_positive": res.n_positive,
                        "n_negative": res.n_negative,
                    }
                )

    summary_df = pd.DataFrame(rows_summary)

    summary_by_regime_df = (
        summary_df.groupby(["model", "regime"], as_index=False)
        .agg(
            folds=("fold", "nunique"),
            mean_accuracy=("accuracy", "mean"),
            std_accuracy=("accuracy", "std"),
            mean_n_eval=("n_eval", "mean"),
            total_n_eval=("n_eval", "sum"),
        )
        .sort_values(["model", "regime"])
        .reset_index(drop=True)
    )

    summary_overall_df = (
        summary_df.groupby(["model"], as_index=False)
        .agg(
            folds=("fold", "nunique"),
            mean_accuracy=("accuracy", "mean"),
            std_accuracy=("accuracy", "std"),
            total_n_eval=("n_eval", "sum"),
        )
        .sort_values(["model"])
        .reset_index(drop=True)
    )

    return CrossValidationRunResult(
        per_fold_results=per_fold_results,
        summary_df=summary_df,
        summary_by_regime_df=summary_by_regime_df,
        summary_overall_df=summary_overall_df,
    )


# ---------------------------------------------------------------------
# Optional helper: pooled predictions table
# ---------------------------------------------------------------------

def crossval_predictions_to_df(
    cv_result: CrossValidationRunResult,
    *,
    model_name: str,
) -> pd.DataFrame:
    """
    Flatten fold/regime predictions into one table.
    """
    rows = []

    for fold_idx, regime_dict in cv_result.per_fold_results[model_name].items():
        for regime, res in regime_dict.items():
            y_prob = res.y_prob if res.y_prob is not None else [None] * len(res.y_true)

            eval_df = res.eval_df.copy()

            for i, (idx, row) in enumerate(eval_df.iterrows()):
                rows.append(
                    {
                        "fold": fold_idx,
                        "regime": regime,
                        "row_index": idx,
                        "y_true": int(res.y_true[i]),
                        "y_pred": int(res.y_pred[i]),
                        "y_prob": None if y_prob[i] is None else float(y_prob[i]),
                    }
                )

    return pd.DataFrame(rows)