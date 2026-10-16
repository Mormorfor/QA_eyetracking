"""The cross-validation loop, and saving and reloading its results.

For each stored fold (`folds.py`), each model is fitted on the fold's training
trials and scored on each held-out regime. `evaluate.py` turns the per-fold
scores into summary tables, and the tables can be saved and reloaded, so a run
does not have to be repeated to look at it again.

**Where the features come from.** Each fold and regime is a slice of one
ready-made trial table, `trial_df` (normally `load_model_ready()`).

`rebuild_features=True` instead rebuilds the features from the word-level `df`,
separately for each regime's trials. That changes only the features that
summarise a participant over several trials: the opening-scan features
(`breaks_pattern_*`, `dominance_score_*`, `strategy_distance_*`). In the
ready-made table they are worked out from all of a person's trials; rebuilt,
from that regime's trials only. The feature sets saved in
`configs/feature_sets/` do not use them; `GENERAL_FEATURES` and `ALL_FEATURES`
do. Every other feature describes one trial and comes out the same either way.

Rebuilding is slower (one build per fold, regime and model) and uses Study 1's
settings, so it is for Study 1 only.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Any, Callable, Dict, List, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score

import src.config.columns as Con
from src.features.build import build_trial_level_model_df
from src.modeling.evaluate import (
    _aggregate_cv_summary,
    evaluate_single_model_on_prepared_split,
)
from src.modeling.feature_sets import get_full_feature_cols
from src.modeling.folds import (
    REGIME_COL,
    attach_fold_regimes,
    attach_trial_level_fold_regimes,
    eval_regimes_for_split,
    load_combined_fold_assignment_csv,
    load_fold_assignment_csv,
)


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
    summary_by_novelty_df: pd.DataFrame
    summary_overall_df: pd.DataFrame

# ---------------------------------------------------------------------
# One-fold evaluation
# ---------------------------------------------------------------------

# Coefficient intervals inside cross-validation are Wald intervals, clustered by
# participant (`inference.wald_logreg_coef_cis`, which accounts for the L2
# penalty and the class weights). They are used here because they need no
# refitting: this runs once per fold and regime, where thousands of bootstrap
# refits each time would be far too slow. 
def evaluate_one_fold_on_regimes(
    df: pd.DataFrame,
    *,
    model_builder: Callable[[], Any],
    target_col: str,
    keep_cols: Optional[Sequence[str]] = None,
    train_regime: str = "train_train",
    eval_regimes: Optional[Sequence[str]] = None,
    eval_split: str = "both",
    coef_ci_method: str = "wald",
    coef_ci_cluster: str = "cluster",
    coef_ci: float = 0.95,
    coef_n_boot: int = 3000,
    coef_seed: int = 42,
    coef_top_k: Optional[int] = None,
    feature_cols: Optional[Sequence[str]] = None,
    fold_idx: int = -1,
    trial_fold_df: Optional[pd.DataFrame] = None,
) -> Dict[str, FoldRegimeEvaluationResult]:
    """
    Fit on ``train_regime`` and score on each regime in ``eval_regimes``.

    Each regime's trials are a slice of ``trial_fold_df``: this fold's part of
    a ready-made trial table, one row per trial, with a ``regime`` column (as
    made by :func:`attach_trial_level_fold_regimes`).

    With ``trial_fold_df=None`` the trial-level features are instead rebuilt
    from ``df`` (this fold's word-level frame, with a ``regime`` column)
    separately for each regime. See the top of this module for when that
    matters.
    """

    if eval_regimes is None:
        eval_regimes = eval_regimes_for_split(eval_split)

    def _regime_frame(regime: str) -> pd.DataFrame:
        if trial_fold_df is not None:
            return (
                trial_fold_df[trial_fold_df[REGIME_COL] == regime]
                .drop(columns=[REGIME_COL])
                .reset_index(drop=True)
            )
        return build_trial_level_model_df(
            df=df[df[REGIME_COL] == regime].copy(),
            keep_cols=keep_cols,
            target_col=target_col,
        )

    train_df = _regime_frame(train_regime)

    feat_cols = list(feature_cols) if feature_cols is not None else list(get_full_feature_cols(train_df))

    results: Dict[str, FoldRegimeEvaluationResult] = {}

    for regime in eval_regimes:
        eval_df = _regime_frame(regime)

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


def _check_feature_source(trial_df: Optional[pd.DataFrame], rebuild_features: bool) -> None:
    """The features come from exactly one place: `trial_df`, or the rebuild."""
    if rebuild_features and trial_df is not None:
        raise ValueError(
            "rebuild_features=True rebuilds the features from `df`; do not also "
            "pass `trial_df`."
        )
    if not rebuild_features and trial_df is None:
        raise ValueError(
            "Pass the ready-made trial table as `trial_df` (e.g. "
            "load_model_ready()), or rebuild_features=True to rebuild the "
            "features for each fold and regime."
        )


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
    eval_split: str = "both",
    feature_cols_by_model: Optional[Mapping[str, Sequence[str]]] = None,
    keep_cols: Optional[Sequence[str]] = None,
    coef_ci_method: str = "wald",
    coef_ci_cluster: str = "cluster",
    coef_ci: float = 0.95,
    coef_n_boot: int = 3000,
    coef_seed: int = 42,
    coef_top_k: Optional[int] = None,
    trial_df: Optional[pd.DataFrame] = None,
    rebuild_features: bool = False,
) -> CrossValidationRunResult:
    """
    Cross-validate on the fold files in one folder (one group's folds).
    The summary tables hold both accuracy and balanced accuracy.

    ``trial_df`` is the ready-made trial table each fold and regime is cut
    from (e.g. ``load_model_ready()``). ``df`` is used to match trials to the
    fold files' regimes, so it needs ``participant_id``, ``TRIAL_INDEX`` and
    ``text_id``; the word-level table has them.

    ``rebuild_features=True`` rebuilds the features from ``df`` for each fold
    and regime instead; ``df`` must then be the word-level table, and
    ``trial_df`` is left out. See the top of this module for when that matters.
    """
    _check_feature_source(trial_df, rebuild_features)
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

        # Sliced once per fold, not once per model: the features don't depend on
        # which model consumes them.
        trial_fold_df = (
            None
            if rebuild_features
            else attach_trial_level_fold_regimes(trial_df, df_fold, keep_cols=keep_cols)
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
                eval_split=eval_split,
                coef_ci_method=coef_ci_method,
                coef_ci_cluster=coef_ci_cluster,
                coef_ci=coef_ci,
                coef_n_boot=coef_n_boot,
                coef_seed=coef_seed,
                coef_top_k=coef_top_k,
                feature_cols=feat_cols,
                fold_idx=fold_idx,
                trial_fold_df=trial_fold_df,
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

    # The same summary tables as the other runners.
    (
        summary_df,
        summary_by_regime_df,
        summary_by_novelty_df,
        summary_overall_df,
    ) = _aggregate_cv_summary(rows_summary)

    return CrossValidationRunResult(
        per_fold_results=per_fold_results,
        summary_df=summary_df,
        summary_by_regime_df=summary_by_regime_df,
        summary_by_novelty_df=summary_by_novelty_df,
        summary_overall_df=summary_overall_df,
    )

# ---------------------------------------------------------------------
# Combined-folds CV runner (e.g. all_participants = hunters + gatherers)
# ---------------------------------------------------------------------

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
    eval_split: str = "both",
    feature_cols_by_model: Optional[Mapping[str, Sequence[str]]] = None,
    keep_cols: Optional[Sequence[str]] = None,
    coef_ci_method: str = "wald",
    coef_ci_cluster: str = "cluster",
    coef_ci: float = 0.95,
    coef_n_boot: int = 3000,
    coef_seed: int = 42,
    coef_top_k: Optional[int] = None,
    trial_df: Optional[pd.DataFrame] = None,
    rebuild_features: bool = False,
) -> CrossValidationRunResult:
    """
    Cross-validate on folds pooled from several folders -- e.g. hunters and
    gatherers together, as `notebooks/drivers/answer_corr_prediction.ipynb`
    runs it.

    For each fold number, that fold's file is read from every folder in
    ``fold_dirs`` and the files are stacked (e.g. the hunters' fold 0 plus the
    gatherers' fold 0). ``df`` and ``trial_df`` should cover the same pooled
    trials.

    Otherwise the same as :func:`run_cross_validation_on_predefined_folds`,
    including what ``df``, ``trial_df`` and ``rebuild_features`` are.
    """
    _check_feature_source(trial_df, rebuild_features)
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

        # Sliced once per fold, not once per model: the features don't depend on
        # which model consumes them.
        trial_fold_df = (
            None
            if rebuild_features
            else attach_trial_level_fold_regimes(trial_df, df_fold, keep_cols=keep_cols)
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
                eval_split=eval_split,
                coef_ci_method=coef_ci_method,
                coef_ci_cluster=coef_ci_cluster,
                coef_ci=coef_ci,
                coef_n_boot=coef_n_boot,
                coef_seed=coef_seed,
                coef_top_k=coef_top_k,
                feature_cols=feat_cols,
                fold_idx=fold_idx,
                trial_fold_df=trial_fold_df,
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

    (
        summary_df,
        summary_by_regime_df,
        summary_by_novelty_df,
        summary_overall_df,
    ) = _aggregate_cv_summary(
        rows_summary
    )

    return CrossValidationRunResult(
        per_fold_results=per_fold_results,
        summary_df=summary_df,
        summary_by_regime_df=summary_by_regime_df,
        summary_by_novelty_df=summary_by_novelty_df,
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
    Save the summary tables of a cross-validation run as CSVs in ``out_dir``.

    Writes ``{run_name}_summary.csv`` (one row per model, fold and regime),
    ``{run_name}_summary_by_regime.csv``, ``{run_name}_summary_by_novelty.csv``
    (the three numbers the paper reports) and
    ``{run_name}_summary_overall.csv``.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "summary": out_dir / f"{run_name}_summary.csv",
        "by_regime": out_dir / f"{run_name}_summary_by_regime.csv",
        "by_novelty": out_dir / f"{run_name}_summary_by_novelty.csv",
        "overall": out_dir / f"{run_name}_summary_overall.csv",
    }

    cv_out.summary_df.to_csv(paths["summary"], index=False)
    cv_out.summary_by_regime_df.to_csv(paths["by_regime"], index=False)
    cv_out.summary_by_novelty_df.to_csv(paths["by_novelty"], index=False)
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
    eval_split: str = "both",
    keep_cols: Optional[Sequence[str]] = None,
    coef_ci_method: str = "wald",
    coef_ci_cluster: str = "cluster",
    coef_ci: float = 0.95,
    coef_n_boot: int = 3000,
    coef_seed: int = 42,
    coef_top_k: Optional[int] = None,
    trial_df: Optional[pd.DataFrame] = None,
    rebuild_features: bool = False,
    save: bool = True,
    verbose: bool = True,
) -> CrossValidationRunResult:
    """
    Add, replace or remove models in a saved pooled-folds run, without
    re-running the models you keep.

    Works on the saved summary rows (one per model, fold and regime):

    1. Load the existing ``{run_name}_summary.csv`` (empty if none yet).
    2. Drop rows for every model in ``remove_models`` *and* for every model in
       ``add_model_builders`` (so re-adding a model replaces its old rows).
    3. Cross-validate only the models in ``add_model_builders`` (reusing
       :func:`run_cross_validation_on_combined_folds`) and append their rows.
    4. Re-aggregate and (if ``save``) overwrite the saved CSVs.

    Use it e.g. to swap one feature set for another::

        update_combined_cv_run(
            df=all_participants, out_dir=..., run_name="all_participants_six_models",
            fold_dirs=[HUNTERS_REFOLDED_DIR, GATHERERS_REFOLDED_DIR],
            target_col=Con.IS_CORRECT_COLUMN,
            trial_df=load_model_ready(),
            remove_models=["total_answering_RT"],
            add_model_builders={"total_answering_RT_normalized": lambda: TrialLevelLogRegModel()},
            add_feature_cols_by_model={"total_answering_RT_normalized": cols},
        )

    The returned ``per_fold_results`` holds only the newly run models (the
    kept ones come back as summary rows only) -- enough for the summary and
    plotting helpers, which read ``summary_df``.

    ``trial_df`` and ``rebuild_features`` are passed on to
    :func:`run_cross_validation_on_combined_folds`. Use the same ones the kept
    models were run with, so the old and new rows are comparable.
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
            eval_split=eval_split,
            feature_cols_by_model=add_feature_cols_by_model,
            keep_cols=keep_cols,
            coef_ci_method=coef_ci_method,
            coef_ci_cluster=coef_ci_cluster,
            coef_ci=coef_ci,
            coef_n_boot=coef_n_boot,
            coef_seed=coef_seed,
            coef_top_k=coef_top_k,
            trial_df=trial_df,
            rebuild_features=rebuild_features,
        )
        new_rows = cv_new.summary_df

    frames = [f for f in (base, new_rows) if not f.empty]
    if not frames:
        raise ValueError("Update produced an empty run (nothing kept, nothing added).")
    combined = pd.concat(frames, ignore_index=True)

    (
        summary_df,
        summary_by_regime_df,
        summary_by_novelty_df,
        summary_overall_df,
    ) = _aggregate_cv_summary(
        combined.to_dict("records")
    )

    result = CrossValidationRunResult(
        per_fold_results=(cv_new.per_fold_results if cv_new is not None else {}),
        summary_df=summary_df,
        summary_by_regime_df=summary_by_regime_df,
        summary_by_novelty_df=summary_by_novelty_df,
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
    Reload a saved run as a :class:`CrossValidationRunResult`, so the summary
    and plotting helpers can be used without re-running cross-validation.

    Reads ``{run_name}_summary.csv`` (written by
    :func:`save_cross_validation_run`) and recomputes the other summary tables
    from it. ``per_fold_results`` is left empty: a saved run keeps only the
    summary rows, which is all the summary and plotting helpers need.
    """
    out_dir = Path(out_dir)
    summary_path = out_dir / f"{run_name}_summary.csv"
    if not summary_path.exists():
        raise FileNotFoundError(
            f"No saved CV summary at {summary_path}. "
            f"Run and save the cross-validation first."
        )

    rows = pd.read_csv(summary_path).to_dict("records")
    (
        summary_df,
        summary_by_regime_df,
        summary_by_novelty_df,
        summary_overall_df,
    ) = _aggregate_cv_summary(rows)

    return CrossValidationRunResult(
        per_fold_results={},
        summary_df=summary_df,
        summary_by_regime_df=summary_by_regime_df,
        summary_by_novelty_df=summary_by_novelty_df,
        summary_overall_df=summary_overall_df,
    )

# ---------------------------------------------------------------------
# All held-out predictions in one table
# ---------------------------------------------------------------------

def crossval_predictions_to_df(
    cv_result: CrossValidationRunResult,
    *,
    model_name: str,
) -> pd.DataFrame:
    """
    One row per held-out trial, fold and regime: the true label, the predicted
    label and the predicted probability. Needs a run that was not reloaded from
    disk, since a reloaded run has no per-fold results.
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
