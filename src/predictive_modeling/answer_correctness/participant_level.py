from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Sequence, Tuple, Optional, Any

import numpy as np

from predictive_modeling.answer_correctness.evaluation_core import CorrectnessEvaluationResult, \
    evaluate_single_model_on_prepared_split

import src.constants as Con

import pandas as pd

from src.data_paths import READY_ALL_FEATURES_PATH
from predictive_modeling.answer_correctness.model_data import (
    build_trial_level_model_df,
    load_all_features,
)
from predictive_modeling.common.data_utils import (
    leave_one_trial_out_for_participant,
    iter_leave_one_trial_out_for_participant,
)
from predictive_modeling.common.feature_specs import get_full_feature_cols


def _load_or_build_full_trial_df(
    df: Optional[pd.DataFrame] = None,
    *,
    keep_cols: Optional[Sequence[str]] = None,
    target_col: str = Con.IS_CORRECT_COLUMN,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Return the full trial-level feature table, preferring the cached
    ``READY_ALL_FEATURES_PATH`` when it exists (exactly as the regular
    ``run_full_features_correctness_bundle`` pipeline does), and otherwise
    building it from the raw ``df`` with every feature group enabled.

    This keeps per-participant runs on the *same* feature table the rest of the
    correctness pipeline uses.
    """
    cache_path = Path(READY_ALL_FEATURES_PATH)
    if cache_path.exists():
        if verbose:
            print(f"Loading cached features from {cache_path}")
        return load_all_features()

    if df is None:
        raise ValueError(
            "No cached feature table found and no raw `df` provided to build one."
        )
    if verbose:
        print("No cached feature table found; building features from raw df.")
    return build_trial_level_model_df(
        df=df,
        keep_cols=keep_cols,
        target_col=target_col,
        include_area_features=True,
        include_derived_features=True,
    )


@dataclass
class PerParticipantCorrectnessResult:
    participant_id: str
    per_trial_results: Dict[str, CorrectnessEvaluationResult]


def evaluate_logreg_on_answer_correctness_leave_one_trial_out(
    df: Optional[pd.DataFrame] = None,
    *,
    model_builder: Callable[[], Any],
    trial_df: Optional[pd.DataFrame] = None,
    participant_col: str = Con.PARTICIPANT_ID,
    trial_col: str = Con.TRIAL_ID,
    split_fn: Callable = leave_one_trial_out_for_participant,
    target_col: str = Con.IS_CORRECT_COLUMN,
    feature_cols: Optional[Sequence[str]] = None,
    keep_cols: Optional[Sequence[str]] = None,
    coef_ci_method: str = "wald",
    coef_ci_cluster: str = "row",
    coef_ci: float = 0.95,
    coef_n_boot: int = 3000,
    coef_seed: int = 42,
    coef_top_k: Optional[int] = None,
    verbose: bool = True,
) -> Dict[str, CorrectnessEvaluationResult]:
    """
    Evaluate a logistic-regression model per participant using a single random
    leave-one-trial-out split per participant.

    The trial-level feature table is sourced the same way as the regular
    ``run_full_features_correctness_bundle`` pipeline: pass a prepared
    ``trial_df`` (e.g. ``load_all_features()``) to use it directly, otherwise
    the cached ``READY_ALL_FEATURES_PATH`` is loaded when present and only
    rebuilt from the raw ``df`` as a fallback.

    Returns
    -------
    Dict[str, CorrectnessEvaluationResult]
        results[participant_id] = evaluation result for that participant
    """
    if trial_df is None:
        trial_df = _load_or_build_full_trial_df(
            df=df, keep_cols=keep_cols, target_col=target_col, verbose=verbose
        )
    trial_df = trial_df.copy()

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


def evaluate_logreg_on_answer_correctness_full_loo(
    df: Optional[pd.DataFrame] = None,
    *,
    model_builder: Callable[[], Any],
    trial_df: Optional[pd.DataFrame] = None,
    participant_col: str = Con.PARTICIPANT_ID,
    trial_col: str = Con.TRIAL_ID,
    target_col: str = Con.IS_CORRECT_COLUMN,
    feature_cols: Optional[Sequence[str]] = None,
    keep_cols: Optional[Sequence[str]] = None,
    min_trials: int = 2,
    coef_ci_method: str = "wald",
    coef_ci_cluster: str = "row",
    coef_ci: float = 0.95,
    coef_n_boot: int = 3000,
    coef_seed: int = 42,
    coef_top_k: Optional[int] = None,
    verbose: bool = True,
) -> Dict[str, CorrectnessEvaluationResult]:
    """
    Per-participant *full* leave-one-trial-out cross-validation.

    For each participant, every trial is held out as the single-row test set
    exactly once (train on all that participant's remaining trials), and the
    held-out predictions are pooled into one ``CorrectnessEvaluationResult`` per
    participant:

    - ``y_true`` / ``y_pred`` / ``y_prob`` are the pooled held-out predictions
      across all LOO folds (one entry per evaluable trial), so ``accuracy`` is
      the participant's leave-one-out accuracy.
    - ``coef_summary`` comes from a single logistic-regression fit on *all* of
      the participant's trials (the same coefficients-from-a-full-fit convention
      the rest of the pipeline uses for reporting).
    - ``train_df`` is the participant's full trial table; ``test_df`` is the
      concatenation of the held-out rows that were actually evaluated.

    Feature handling matches ``run_full_features_correctness_bundle``: the
    feature table is the cached ``READY_ALL_FEATURES_PATH`` (or a prepared
    ``trial_df`` you pass in), and the default feature set is
    ``get_full_feature_cols(trial_df)``. The model itself (fit / predict /
    predict_proba / coefficient CIs) is driven exactly as in the regular
    prepared-split evaluation.

    Folds whose training set collapses to a single class are skipped (a
    logistic regression cannot be fit on one class); participants with fewer
    than ``min_trials`` trials, or with only a single outcome class overall, are
    skipped entirely. A per-participant summary of skips is printed when
    ``verbose``.

    Returns
    -------
    Dict[str, CorrectnessEvaluationResult]
        results[participant_id] = pooled leave-one-out result for that
        participant.
    """
    if trial_df is None:
        trial_df = _load_or_build_full_trial_df(
            df=df, keep_cols=keep_cols, target_col=target_col, verbose=verbose
        )
    trial_df = trial_df.copy()

    feat_cols = (
        list(feature_cols)
        if feature_cols is not None
        else list(get_full_feature_cols(trial_df))
    )

    coef_kwargs = {
        "top_k": coef_top_k,
        "ci_method": coef_ci_method,
        "ci_cluster": coef_ci_cluster,
        "ci": coef_ci,
        "n_boot": coef_n_boot,
        "seed": coef_seed,
    }

    # Iterate the participant column's own values (no str-coercion) so the
    # membership test inside the LOO iterator matches the frame's dtype.
    participants = pd.unique(trial_df[participant_col].dropna())

    results: Dict[str, CorrectnessEvaluationResult] = {}

    for pid in participants:
        df_p = trial_df[trial_df[participant_col] == pid]
        n_trials = int(df_p[trial_col].dropna().nunique())

        if n_trials < min_trials:
            if verbose:
                print(f"[{pid}] skipped: only {n_trials} trial(s) (< {min_trials}).")
            continue

        if df_p[target_col].dropna().astype(int).nunique() < 2:
            if verbose:
                print(f"[{pid}] skipped: only one outcome class across all trials.")
            continue

        y_true_parts: list[np.ndarray] = []
        y_pred_parts: list[np.ndarray] = []
        y_prob_parts: list[np.ndarray] = []
        test_row_frames: list[pd.DataFrame] = []
        n_skipped_folds = 0

        for train_fold, test_fold, _test_trial in iter_leave_one_trial_out_for_participant(
            df=df_p,
            participant_id=pid,
            participant_col=participant_col,
            trial_col=trial_col,
        ):
            if train_fold[target_col].dropna().astype(int).nunique() < 2:
                # A logistic regression needs both classes in the train fold.
                n_skipped_folds += 1
                continue

            model = model_builder()
            model.fit(
                train_df=train_fold,
                target_col=target_col,
                feature_cols=feat_cols,
            )

            y_true_parts.append(test_fold[target_col].astype(int).to_numpy())
            y_pred_parts.append(
                np.asarray(model.predict(test_fold, feature_cols=feat_cols))
                .reshape(-1)
                .astype(int)
            )
            y_prob_parts.append(
                np.asarray(model.predict_proba(test_fold, feature_cols=feat_cols))
                .reshape(-1)
                .astype(float)
            )
            test_row_frames.append(test_fold)

        if not y_true_parts:
            if verbose:
                print(
                    f"[{pid}] skipped: no evaluable LOO folds "
                    f"({n_skipped_folds} single-class train fold(s))."
                )
            continue

        y_true = np.concatenate(y_true_parts)
        y_pred = np.concatenate(y_pred_parts)
        y_prob = np.concatenate(y_prob_parts)
        test_df_all = pd.concat(test_row_frames, axis=0)

        # Coefficients from a single fit on all of the participant's trials,
        # matching the pipeline's "coefficients from a full-data fit" convention.
        coef_model = model_builder()
        coef_model.fit(
            train_df=df_p,
            target_col=target_col,
            feature_cols=feat_cols,
        )
        coef_summary = coef_model.get_coef_summary(
            train_df=df_p,
            feature_cols=feat_cols,
            **coef_kwargs,
        )

        results[str(pid)] = CorrectnessEvaluationResult(
            train_df=df_p,
            test_df=test_df_all,
            y_true=y_true,
            y_pred=y_pred,
            y_prob=y_prob,
            accuracy=float((y_true == y_pred).mean()),
            n_test=int(len(y_true)),
            n_positive=int((y_true == 1).sum()),
            n_negative=int((y_true == 0).sum()),
            coef_summary=coef_summary,
        )

        if verbose:
            print(
                f"[{pid}] LOO acc={results[str(pid)].accuracy:.3f} "
                f"over {len(y_true)} trial(s)"
                + (f", {n_skipped_folds} fold(s) skipped" if n_skipped_folds else "")
            )

    return results