from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from IPython.display import display
from sklearn.metrics import balanced_accuracy_score

import src.constants as Con

from src.predictive_modeling.answer_correctness.model_data import build_trial_level_model_df
from src.predictive_modeling.answer_correctness.evaluation_core import evaluate_single_model_on_prepared_split
from src.predictive_modeling.common.feature_specs import get_full_feature_cols

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
    balanced_accuracy: float
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
    target_col: str,
    keep_cols: Optional[Sequence[str]] = None,
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
    train_df = build_trial_level_model_df(
        df=train_raw,
        keep_cols=keep_cols,
        target_col=target_col,
        include_area_features=True,
        include_derived_features=True,
    )

    feat_cols = list(feature_cols) if feature_cols is not None else list(get_full_feature_cols(train_df))

    results: Dict[str, FoldRegimeEvaluationResult] = {}

    for regime in eval_regimes:
        eval_raw = df[df["regime"] == regime].copy()
        eval_df = build_trial_level_model_df(
            df=eval_raw,
            keep_cols=keep_cols,
            target_col=target_col,
            include_area_features=True,
            include_derived_features=True,
        )

        model = model_builder()

        eval_res = evaluate_single_model_on_prepared_split(
            model=model,
            train_df=train_df,
            test_df=eval_df,
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

        results[regime] = FoldRegimeEvaluationResult(
            fold_idx=fold_idx,
            regime=regime,
            train_df=train_df,
            eval_df=eval_df,
            y_true=eval_res.y_true,
            y_pred=eval_res.y_pred,
            y_prob=eval_res.y_prob,
            accuracy=eval_res.accuracy,
            balanced_accuracy=float(balanced_accuracy_score(eval_res.y_true, eval_res.y_pred)),
            n_eval=eval_res.n_test,
            n_positive=eval_res.n_positive,
            n_negative=eval_res.n_negative,
            coef_summary=eval_res.coef_summary,
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
    target_col: str,
    n_folds: int = 10,
    fold_filename_template: str = "fold_{fold_idx}_trial_ids_by_regime.csv",
    df_participant_col: str = Con.PARTICIPANT_ID,
    df_text_col: str = Con.TEXT_ID_COLUMN,
    eval_regimes: Optional[Sequence[str]] = None,
    feature_cols_by_model: Optional[Mapping[str, Sequence[str]]] = None,
    keep_cols: Optional[Sequence[str]] = None,
    coef_ci_method: str = "wald",
    coef_ci_cluster: str = "row",
    coef_ci: float = 0.95,
    coef_n_boot: int = 3000,
    coef_seed: int = 42,
    coef_top_k: Optional[int] = None,
) -> CrossValidationRunResult:
    """
    Run cross-validation using predefined fold assignment CSVs.
    Stores both accuracy and balanced_accuracy in the summary tables.
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
                df=df_fold,
                model_builder=model_builder,
                target_col=target_col,
                keep_cols=keep_cols,
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
                        "balanced_accuracy": res.balanced_accuracy,
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
            mean_balanced_accuracy=("balanced_accuracy", "mean"),
            std_balanced_accuracy=("balanced_accuracy", "std"),
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
            mean_balanced_accuracy=("balanced_accuracy", "mean"),
            std_balanced_accuracy=("balanced_accuracy", "std"),
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
# Combined-folds CV runner (e.g. all_participants = hunters + gatherers)
# ---------------------------------------------------------------------

def _aggregate_cv_summary(
    rows_summary: List[Dict[str, Any]],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build the (per-row, by-regime, overall) summary tables from raw rows."""
    summary_df = pd.DataFrame(rows_summary)

    summary_by_regime_df = (
        summary_df.groupby(["model", "regime"], as_index=False)
        .agg(
            folds=("fold", "nunique"),
            mean_accuracy=("accuracy", "mean"),
            std_accuracy=("accuracy", "std"),
            mean_balanced_accuracy=("balanced_accuracy", "mean"),
            std_balanced_accuracy=("balanced_accuracy", "std"),
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
            mean_balanced_accuracy=("balanced_accuracy", "mean"),
            std_balanced_accuracy=("balanced_accuracy", "std"),
            total_n_eval=("n_eval", "sum"),
        )
        .sort_values(["model"])
        .reset_index(drop=True)
    )

    return summary_df, summary_by_regime_df, summary_overall_df


def load_combined_fold_assignment_csv(
    fold_dirs: Sequence[str | Path],
    fold_idx: int,
    *,
    fold_filename_template: str = "fold_{fold_idx}_trial_ids_by_regime.csv",
) -> pd.DataFrame:
    """
    Load and vertically stack the fold-``fold_idx`` assignment CSVs from each
    directory in ``fold_dirs``.

    Used to build an "all participants" fold by combining, e.g., the hunters
    fold_0 and the (refolded) gatherers fold_0 into a single regime-assignment
    table. Participant ids are disjoint across the two groups, so stacking the
    assignments is sufficient.
    """
    frames = []
    for fold_dir in fold_dirs:
        fold_path = Path(fold_dir) / fold_filename_template.format(fold_idx=fold_idx)
        frames.append(load_fold_assignment_csv(fold_path))
    return pd.concat(frames, ignore_index=True)


def run_cross_validation_on_combined_folds(
    df: pd.DataFrame,
    *,
    fold_dirs: Sequence[str | Path],
    model_builders: Mapping[str, Callable[[], Any]],
    target_col: str,
    n_folds: int = 10,
    fold_filename_template: str = "fold_{fold_idx}_trial_ids_by_regime.csv",
    df_participant_col: str = Con.PARTICIPANT_ID,
    df_text_col: str = Con.TEXT_ID_COLUMN,
    eval_regimes: Optional[Sequence[str]] = None,
    feature_cols_by_model: Optional[Mapping[str, Sequence[str]]] = None,
    keep_cols: Optional[Sequence[str]] = None,
    coef_ci_method: str = "wald",
    coef_ci_cluster: str = "row",
    coef_ci: float = 0.95,
    coef_n_boot: int = 3000,
    coef_seed: int = 42,
    coef_top_k: Optional[int] = None,
) -> CrossValidationRunResult:
    """
    Cross-validate using predefined folds that are pooled across several fold
    directories.

    For each ``fold_idx`` the fold-``fold_idx`` assignment CSV is loaded from
    every directory in ``fold_dirs`` and concatenated, so the same fold index
    is matched across groups (e.g. hunters fold_0 + gatherers fold_0). ``df``
    should be the matching pooled dataframe (e.g. ``all_participants``).

    Mirrors :func:`run_cross_validation_on_predefined_folds`, including the
    accuracy / balanced-accuracy summary tables.
    """
    per_fold_results: Dict[str, Dict[int, Dict[str, FoldRegimeEvaluationResult]]] = {
        model_name: {} for model_name in model_builders
    }

    rows_summary: List[Dict[str, Any]] = []

    for fold_idx in range(n_folds):
        fold_assign_df = load_combined_fold_assignment_csv(
            fold_dirs,
            fold_idx,
            fold_filename_template=fold_filename_template,
        )

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
                df=df_fold,
                model_builder=model_builder,
                target_col=target_col,
                keep_cols=keep_cols,
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
                        "balanced_accuracy": res.balanced_accuracy,
                        "n_eval": res.n_eval,
                        "n_positive": res.n_positive,
                        "n_negative": res.n_negative,
                    }
                )

    summary_df, summary_by_regime_df, summary_overall_df = _aggregate_cv_summary(
        rows_summary
    )

    return CrossValidationRunResult(
        per_fold_results=per_fold_results,
        summary_df=summary_df,
        summary_by_regime_df=summary_by_regime_df,
        summary_overall_df=summary_overall_df,
    )


def save_cross_validation_run(
    cv_out: CrossValidationRunResult,
    out_dir: str | Path,
    *,
    run_name: str = "all_participants",
    verbose: bool = True,
) -> Dict[str, Path]:
    """
    Save the summary tables of a cross-validation run as CSVs under ``out_dir``.

    Writes ``{run_name}_summary.csv`` (one row per model/fold/regime),
    ``{run_name}_summary_by_regime.csv`` and ``{run_name}_summary_overall.csv``.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "summary": out_dir / f"{run_name}_summary.csv",
        "by_regime": out_dir / f"{run_name}_summary_by_regime.csv",
        "overall": out_dir / f"{run_name}_summary_overall.csv",
    }

    cv_out.summary_df.to_csv(paths["summary"], index=False)
    cv_out.summary_by_regime_df.to_csv(paths["by_regime"], index=False)
    cv_out.summary_overall_df.to_csv(paths["overall"], index=False)

    if verbose:
        for key, path in paths.items():
            print(f"Saved {key}: {path}")

    return paths


def update_combined_cv_run(
    df: pd.DataFrame,
    *,
    out_dir: str | Path,
    run_name: str,
    fold_dirs: Sequence[str | Path],
    target_col: str,
    add_model_builders: Optional[Mapping[str, Callable[[], Any]]] = None,
    add_feature_cols_by_model: Optional[Mapping[str, Sequence[str]]] = None,
    remove_models: Optional[Sequence[str]] = None,
    n_folds: int = 10,
    fold_filename_template: str = "fold_{fold_idx}_trial_ids_by_regime.csv",
    df_participant_col: str = Con.PARTICIPANT_ID,
    df_text_col: str = Con.TEXT_ID_COLUMN,
    eval_regimes: Optional[Sequence[str]] = None,
    keep_cols: Optional[Sequence[str]] = None,
    coef_ci_method: str = "wald",
    coef_ci_cluster: str = "row",
    coef_ci: float = 0.95,
    coef_n_boot: int = 3000,
    coef_seed: int = 42,
    coef_top_k: Optional[int] = None,
    save: bool = True,
    verbose: bool = True,
) -> CrossValidationRunResult:
    """
    Incrementally update a saved combined-folds CV run, one model at a time,
    without recomputing the models you keep.

    Operates on the saved per-model/fold/regime summary rows:

    1. Load the existing ``{run_name}_summary.csv`` (empty if none yet).
    2. Drop rows for every model in ``remove_models`` *and* for every model in
       ``add_model_builders`` (so re-adding a model replaces its old rows).
    3. Cross-validate only the models in ``add_model_builders`` (reusing
       :func:`run_cross_validation_on_combined_folds`) and append their rows.
    4. Re-aggregate and (if ``save``) overwrite the saved CSVs.

    Use it e.g. to swap one feature set for another::

        update_combined_cv_run(
            df=all_participants, out_dir=..., run_name="all_participants_six_models",
            fold_dirs=[HUNTERS_FOLDS_DIR, GATHERERS_REFOLDED_DIR],
            target_col=Con.IS_CORRECT_COLUMN,
            remove_models=["total_answering_RT"],
            add_model_builders={"total_answering_RT_normalized": lambda: TrialLevelLogRegModel()},
            add_feature_cols_by_model={"total_answering_RT_normalized": cols},
        )

    Note: the returned ``per_fold_results`` holds only the newly run models
    (the kept models are restored from summary rows only) -- enough for the
    metric/plot helpers, which read ``summary_df``.
    """
    out_dir = Path(out_dir)
    summary_path = out_dir / f"{run_name}_summary.csv"

    base = pd.read_csv(summary_path) if summary_path.exists() else pd.DataFrame()

    to_drop = set(remove_models or [])
    if add_model_builders:
        to_drop |= set(add_model_builders.keys())  # re-added models replace old rows

    if not base.empty and to_drop:
        base = base[~base["model"].isin(to_drop)].copy()

    cv_new: Optional[CrossValidationRunResult] = None
    new_rows = pd.DataFrame()
    if add_model_builders:
        cv_new = run_cross_validation_on_combined_folds(
            df=df,
            fold_dirs=fold_dirs,
            model_builders=add_model_builders,
            target_col=target_col,
            n_folds=n_folds,
            fold_filename_template=fold_filename_template,
            df_participant_col=df_participant_col,
            df_text_col=df_text_col,
            eval_regimes=eval_regimes,
            feature_cols_by_model=add_feature_cols_by_model,
            keep_cols=keep_cols,
            coef_ci_method=coef_ci_method,
            coef_ci_cluster=coef_ci_cluster,
            coef_ci=coef_ci,
            coef_n_boot=coef_n_boot,
            coef_seed=coef_seed,
            coef_top_k=coef_top_k,
        )
        new_rows = cv_new.summary_df

    frames = [f for f in (base, new_rows) if not f.empty]
    if not frames:
        raise ValueError("Update produced an empty run (nothing kept, nothing added).")
    combined = pd.concat(frames, ignore_index=True)

    summary_df, summary_by_regime_df, summary_overall_df = _aggregate_cv_summary(
        combined.to_dict("records")
    )

    result = CrossValidationRunResult(
        per_fold_results=(cv_new.per_fold_results if cv_new is not None else {}),
        summary_df=summary_df,
        summary_by_regime_df=summary_by_regime_df,
        summary_overall_df=summary_overall_df,
    )

    if save:
        save_cross_validation_run(result, out_dir, run_name=run_name, verbose=verbose)
    if verbose:
        if to_drop:
            print(f"Removed/replaced: {sorted(to_drop)}")
        if add_model_builders:
            print(f"Ran: {sorted(add_model_builders.keys())}")
        print(f"Models in updated run: {sorted(summary_df['model'].unique())}")

    return result


def load_cross_validation_run(
    out_dir: str | Path,
    *,
    run_name: str = "all_participants",
) -> CrossValidationRunResult:
    """
    Rebuild a :class:`CrossValidationRunResult` from a saved run so the metric /
    plotting helpers can be used without re-running cross-validation.

    Reads ``{run_name}_summary.csv`` (written by :func:`save_cross_validation_run`)
    and recomputes the by-regime and overall summary tables from it. The
    ``per_fold_results`` field is left empty -- the saved run only stores the
    summary rows, which is all that ``summarize_cv_results_by_regime``,
    ``build_cv_model_comparison_df`` and ``show_cv_results`` need.
    """
    out_dir = Path(out_dir)
    summary_path = out_dir / f"{run_name}_summary.csv"
    if not summary_path.exists():
        raise FileNotFoundError(
            f"No saved CV summary at {summary_path}. "
            f"Run and save the cross-validation first."
        )

    rows = pd.read_csv(summary_path).to_dict("records")
    summary_df, summary_by_regime_df, summary_overall_df = _aggregate_cv_summary(rows)

    return CrossValidationRunResult(
        per_fold_results={},
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


def summarize_cv_results_by_regime(
    cv_out,
    model_name: Optional[str] = "full_features_correctness_log_reg",
    *,
    metric_col: str = "balanced_accuracy",
    test_only: bool = False,
    val_only: bool = False,
    ci: float = 0.95,
) -> pd.DataFrame:
    """
    Aggregate cross-validation results by regime across folds.

    Parameters
    ----------
    cv_out
        Output object from `run_cross_validation_on_predefined_folds`.
    model_name
        Model name to filter on. If None, keeps all models.
    metric_col
        Metric column to aggregate. Typically "balanced_accuracy" or "accuracy".
    test_only
        If True, keep only test regimes.
    val_only
        If True, keep only validation regimes.
    ci
        Confidence level for mean metric CI across folds.

    Returns
    -------
    pd.DataFrame
        One row per regime with fold-level summary statistics and CI bounds.
    """
    df = cv_out.summary_df.copy()

    if model_name is not None:
        df = df[df["model"] == model_name].copy()

    if metric_col not in df.columns:
        raise ValueError(f"metric_col='{metric_col}' not found in cv_out.summary_df")

    if test_only and val_only:
        raise ValueError("Choose only one of test_only / val_only.")

    if test_only:
        df = df[df["regime"].astype(str).str.startswith("test")].copy()
    elif val_only:
        df = df[df["regime"].astype(str).str.startswith("val")].copy()

    if df.empty:
        raise ValueError("No rows found for the requested selection.")

    z_map = {
        0.90: 1.645,
        0.95: 1.96,
        0.99: 2.576,
    }
    z = z_map.get(ci, 1.96)

    out = (
        df.groupby("regime", as_index=False)
        .agg(
            n_folds=("fold", "nunique"),
            mean_metric=(metric_col, "mean"),
            std_metric=(metric_col, "std"),
            min_metric=(metric_col, "min"),
            max_metric=(metric_col, "max"),
            mean_n_eval=("n_eval", "mean"),
            total_n_eval=("n_eval", "sum"),
        )
        .sort_values("regime")
        .reset_index(drop=True)
    )

    out["std_metric"] = out["std_metric"].fillna(0.0)
    out["se_metric"] = out["std_metric"] / np.sqrt(out["n_folds"])
    out["ci_low"] = (out["mean_metric"] - z * out["se_metric"]).clip(lower=0.0)
    out["ci_high"] = (out["mean_metric"] + z * out["se_metric"]).clip(upper=1.0)
    out["metric"] = metric_col

    return out


def build_cv_model_comparison_df(
    cv_out,
    *,
    regime: str = "test_unseen_subject_unseen_item",
    metric_col: str = "balanced_accuracy",
    ci: float = 0.95,
    models: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """
    One row per model with the cross-fold mean of ``metric_col`` and its CI on a
    single regime.

    Aggregates :func:`summarize_cv_results_by_regime` across every model in the
    run (or just ``models`` if given) and keeps only ``regime`` (default: the
    "both" test regime, unseen subject x unseen item).

    Returns
    -------
    pd.DataFrame with columns:
        model, mean_metric, std_metric, se_metric, ci_low, ci_high,
        n_folds, mean_n_eval, metric
    Sorted ascending by ``mean_metric`` (low -> high), ready for staged plotting.
    """
    if models is None:
        models = list(cv_out.summary_df["model"].unique())

    rows: List[Dict[str, Any]] = []
    for model_name in models:
        per_regime = summarize_cv_results_by_regime(
            cv_out=cv_out,
            model_name=model_name,
            metric_col=metric_col,
            ci=ci,
        )
        match = per_regime[per_regime["regime"] == regime]
        if match.empty:
            continue
        r = match.iloc[0]
        rows.append(
            {
                "model": model_name,
                "mean_metric": float(r["mean_metric"]),
                "std_metric": float(r["std_metric"]),
                "se_metric": float(r["se_metric"]),
                "ci_low": float(r["ci_low"]),
                "ci_high": float(r["ci_high"]),
                "n_folds": int(r["n_folds"]),
                "mean_n_eval": float(r["mean_n_eval"]),
                "metric": metric_col,
            }
        )

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values("mean_metric", ascending=True).reset_index(drop=True)
    return out


def show_cv_results(
    cv_out,
    model_name: Optional[str] = "full_features_correctness_log_reg",
    *,
    metric_col: str = "balanced_accuracy",
) -> Dict[str, pd.DataFrame]:
    """
    Display fold-level and aggregated CV results for a model.
    """
    summary_df = cv_out.summary_df.copy()

    if model_name is not None:
        summary_df = summary_df[summary_df["model"] == model_name].copy()

    if metric_col not in summary_df.columns:
        raise ValueError(f"metric_col='{metric_col}' not found in cv_out.summary_df")

    print("=" * 80)
    print(f"CROSS-VALIDATION RESULTS: {model_name}")
    print("=" * 80)

    print("\n1) Fold-level raw results")
    fold_level = summary_df.sort_values(["regime", "fold"]).reset_index(drop=True)
    display(fold_level)

    print(f"\n2) Aggregated by regime ({metric_col})")
    by_regime = summarize_cv_results_by_regime(
        cv_out=cv_out,
        model_name=model_name,
        metric_col=metric_col,
        test_only=False,
        val_only=False,
        ci=0.95,
    )
    display(by_regime)

    print(f"\n3) Test-only regimes ({metric_col})")
    test_only = summarize_cv_results_by_regime(
        cv_out=cv_out,
        model_name=model_name,
        metric_col=metric_col,
        test_only=True,
        val_only=False,
        ci=0.95,
    )
    display(test_only)

    print(f"\n4) Validation-only regimes ({metric_col})")
    val_only = summarize_cv_results_by_regime(
        cv_out=cv_out,
        model_name=model_name,
        metric_col=metric_col,
        test_only=False,
        val_only=True,
        ci=0.95,
    )
    display(val_only)

    print(f"\n5) Overall mean across all fold-regime evaluations ({metric_col})")
    overall = pd.DataFrame([{
        "model": model_name,
        "metric": metric_col,
        "n_rows": len(summary_df),
        "n_folds": summary_df["fold"].nunique(),
        "mean_metric": summary_df[metric_col].mean(),
        "std_metric": summary_df[metric_col].std(),
        "min_metric": summary_df[metric_col].min(),
        "max_metric": summary_df[metric_col].max(),
        "total_n_eval": summary_df["n_eval"].sum(),
    }])
    display(overall)

    return {
        "fold_level": fold_level,
        "by_regime": by_regime,
        "test_only": test_only,
        "val_only": val_only,
        "overall": overall,
    }


def plot_cv_metric_by_regime(
    cv_out,
    model_name: str = "full_features_correctness_log_reg",
    metric_col: str = "balanced_accuracy",
    ci: float = 0.95,
    test_only: bool = False,
    val_only: bool = False,
    figsize: tuple = (10, 6),
    rotate_xticks: int = 30,
):
    """
    Bar plot of mean CV metric by regime, with confidence intervals across folds.
    """
    summary = summarize_cv_results_by_regime(
        cv_out=cv_out,
        model_name=model_name,
        metric_col=metric_col,
        test_only=test_only,
        val_only=val_only,
        ci=ci,
    )

    y = summary["mean_metric"].to_numpy()
    yerr = np.vstack([
        y - summary["ci_low"].to_numpy(),
        summary["ci_high"].to_numpy() - y,
    ])

    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(
        summary["regime"],
        summary["mean_metric"],
        yerr=yerr,
        capsize=6,
    )

    pretty_metric = metric_col.replace("_", " ").title()
    ax.set_ylabel(pretty_metric)
    ax.set_xlabel("Regime")
    ax.set_title(f"{model_name}: mean CV {pretty_metric.lower()} by regime ({int(ci * 100)}% CI)")
    ax.set_ylim(0, 1)

    plt.xticks(rotation=rotate_xticks, ha="right")
    plt.tight_layout()

    return summary, fig, ax



def plot_cv_metric_by_regime_pretty(
    cv_out,
    model_name: str = "full_features_correctness_log_reg",
    metric_col: str = "balanced_accuracy",
    ci: float = 0.95,
    test_only: bool = False,
    val_only: bool = False,
    figsize: tuple = (10, 6),
):
    """
    Same as `plot_cv_metric_by_regime`, but with prettier regime labels.
    """
    pretty_names = {
        "val_seen_subject_unseen_item": "Val: seen subj,\nunseen item",
        "test_seen_subject_unseen_item": "Test: seen subj,\nunseen item",
        "val_unseen_subject_seen_item": "Val: unseen subj,\nseen item",
        "test_unseen_subject_seen_item": "Test: unseen subj,\nseen item",
        "val_unseen_subject_unseen_item": "Val: unseen subj,\nunseen item",
        "test_unseen_subject_unseen_item": "Test: unseen subj,\nunseen item",
    }

    summary, fig, ax = plot_cv_metric_by_regime(
        cv_out=cv_out,
        model_name=model_name,
        metric_col=metric_col,
        ci=ci,
        test_only=test_only,
        val_only=val_only,
        figsize=figsize,
        rotate_xticks=0,
    )

    ax.set_xticks(range(len(summary)))
    ax.set_xticklabels(
        [pretty_names.get(r, r) for r in summary["regime"]],
        rotation=0,
        ha="center",
    )
    plt.tight_layout()

    return summary, fig, ax

