"""The headline model fitted on one participant at a time.

Each participant gets their own logistic regression, tested on trials it was not
trained on: one random trial (`evaluate_logreg_on_answer_correctness_leave_one_trial_out`),
or every trial in turn (`evaluate_logreg_on_answer_correctness_full_loo`, which
`person_variance/` runs). This shows how well the model reads each individual, and
which features it relies on for each of them.

Both use the saved model-ready table when it exists, so the features are the ones the
rest of the analysis uses. Participants with very few wrong answers score worst (there
are few wrong answers to learn from); they are filtered out in `person_variance/`, not
here.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Sequence, Tuple, Optional, Any

import numpy as np

from src.modeling.evaluate import CorrectnessEvaluationResult, \
    evaluate_single_model_on_prepared_split

import src.config.columns as Con

import pandas as pd

from src.config.datasets import READY_ALL_FEATURES_PATH
from src.features.build import build_trial_level_model_df, load_model_ready
from src.modeling.folds import leave_one_trial_out_for_participant, iter_leave_one_trial_out_for_participant
from src.modeling.feature_sets import get_full_feature_cols


def _load_or_build_full_trial_df(
    df: Optional[pd.DataFrame] = None,
    *,
    keep_cols: Optional[Sequence[str]] = None,
    target_col: str = Con.IS_CORRECT_COLUMN,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    The trial table: the saved model-ready table (`load_model_ready`) if it exists,
    otherwise built from the word-level `df`.

    The saved table is used as it is, whatever `df` holds; `keep_cols` and
    `target_col` apply only when building.
    """
    cache_path = Path(READY_ALL_FEATURES_PATH)
    if cache_path.exists():
        if verbose:
            print(f"Loading cached features from {cache_path}")
        return load_model_ready()

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
    )


@dataclass
class PerParticipantCorrectnessResult:
    """One participant's results, trial by trial. Not used by the functions here."""
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
    For each participant, fit the model on all their trials but one and test it on
    that one.

    The held-out trial is drawn at random and no seed is passed, so it changes from
    run to run. `split_fn` makes each split (default
    `modeling/folds.py::leave_one_trial_out_for_participant`).

    `trial_df` is the trial table to use; without it, the saved model-ready table is
    loaded, or built from `df` if there is none. `feature_cols` defaults to every
    feature the table has (`get_full_feature_cols`). Coefficient intervals default
    to Wald, which needs no refitting, with each trial counted on its own: within
    one person, clustering by participant means nothing.

    Participant ids are compared as text, so `trial_df` must hold them as text (as
    `load_model_ready` does). A participant whose training trials are all one
    outcome is skipped without a message: a logistic regression needs both.

    Returns `{participant_id: CorrectnessEvaluationResult}`.
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
    For each participant, leave one trial out at a time: every trial is held out
    once, the model is fitted on the participant's other trials, and the held-out
    predictions are pooled into one result per participant.

    In each result, `y_true`, `y_pred` and `y_prob` hold one entry per trial that
    was tested, so `accuracy` is the participant's leave-one-out accuracy.
    `coef_summary` comes from one more fit, on all of the participant's trials, as
    the paper's coefficients come from one fit on all the data. `train_df` is the
    participant's whole table; `test_df` holds the trials that were tested.

    The trial table, the feature set and the coefficient intervals are chosen as in
    `evaluate_logreg_on_answer_correctness_leave_one_trial_out`.

    Skipped: a participant with fewer than `min_trials` trials or with only one
    outcome, and any held-out trial whose training trials are all one outcome (a
    logistic regression needs both); a participant left with no trial to test is
    skipped too. With `verbose`, the skips are printed, along with each
    participant's accuracy.

    Returns `{participant_id: CorrectnessEvaluationResult}`.
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

    # The ids as the table holds them, not turned into text, so that the
    # comparison inside the splitter matches the table's type.
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
                # A logistic regression needs both outcomes among its training trials.
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

        # Coefficients from one fit on all of the participant's trials, as the
        # paper's come from one fit on all the data.
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