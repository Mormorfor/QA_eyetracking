"""Fit the correctness model separately for each participant, or reload a saved run.

For each participant, every trial is predicted once by a model trained on that
person's other trials only -- one trial left out at a time, "LOO" in the names
here. What is left out is a trial, never a person. The predictions are pooled into
one `CorrectnessEvaluationResult` per participant, so its `accuracy` is the share
of that person's trials their own model got right. Their coefficients come from one
more fit, on all of their trials.

The model is the headline model's (`TrialLevelLogRegModel`), and it should be given
the headline model's feature table, `load_model_ready()`. The fitting is done by
`participant_level.py::evaluate_logreg_on_answer_correctness_full_loo`. A
participant whose answers are all correct (or all wrong) gets no model, and a trial
whose removal leaves only one outcome to train on is never scored. On Study 1, 356
of the 360 participants get a result.

One model per trial per participant makes the run slow, so the result is saved to
`reports/_cache/per_person_corr_loo_results/results_by_pid__<tag>.pkl`, one file
per feature-set tag.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Sequence

import pandas as pd

import src.modeling.feature_sets as FG
from src.modeling.models.logreg_model import (
    TrialLevelLogRegModel,
)
from src.analyses.correctness_prediction.participant_level import (
    evaluate_logreg_on_answer_correctness_full_loo,
)
from src.config.outputs import PER_PERSON_LOO_RESULTS_DIR

# The key each participant's result sits under in the nested shape that the shared
# per-participant helpers take (see `to_nested_results`).
MODEL_NAME = "trial_level_log_reg"

# The headline model's ten features, plus whether the area looked at just before
# the confirming press was the correct answer or the question: 12 in all.
DEFAULT_FEATURE_COLS = FG.SELECT_1_COLS + FG.LAST_CONFIRM_COMPACT
DEFAULT_FEATURE_SET_TAG = "select_1_plus_last_confirm_compact"


def results_cache_path(feature_set_tag: str = DEFAULT_FEATURE_SET_TAG) -> Path:
    """Where the per-participant run for this feature-set tag is saved."""
    return Path(PER_PERSON_LOO_RESULTS_DIR) / f"results_by_pid__{feature_set_tag}.pkl"


def run_or_load_per_person_loo(
    trial_df: Optional[pd.DataFrame] = None,
    *,
    feature_cols: Optional[Sequence[str]] = None,
    feature_set_tag: str = DEFAULT_FEATURE_SET_TAG,
    force_recompute: bool = False,
    model_builder: Callable[[], Any] = TrialLevelLogRegModel,
    coef_ci_method: str = "wald",
    coef_ci_cluster: str = "row",
    verbose: bool = True,
) -> Dict[str, Any]:
    """Run the per-participant models, or reload them if this tag was run before.

    `trial_df` is the trial-level feature table, normally `load_model_ready()`. It
    is needed only when the run is computed. `feature_cols` defaults to
    `DEFAULT_FEATURE_COLS`.

    The saved file is found by `feature_set_tag` alone. If it exists it is loaded
    and returned, whatever the other arguments say -- so give a new tag whenever the
    features, the model or the interval settings change, or the old run comes back.
    `force_recompute` runs again and overwrites the file.

    Returns `{participant_id: CorrectnessEvaluationResult}`: the pooled predictions
    for that person's scored trials, and the coefficient table of the fit on all of
    their trials (95% Wald intervals by default).
    """
    feat_cols = list(feature_cols) if feature_cols is not None else list(DEFAULT_FEATURE_COLS)

    cache_path = results_cache_path(feature_set_tag)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"Using {len(feat_cols)} features:")
        for c in feat_cols:
            print("  -", c)

    if cache_path.exists() and not force_recompute:
        with open(cache_path, "rb") as f:
            results_by_pid = pickle.load(f)
        if verbose:
            print(f"\nLoaded {len(results_by_pid)} participants from cache: {cache_path}")
        return results_by_pid

    if trial_df is None:
        raise ValueError(
            f"No cached run at {cache_path} and no `trial_df` given to compute one."
        )

    results_by_pid = evaluate_logreg_on_answer_correctness_full_loo(
        trial_df=trial_df,
        model_builder=model_builder,
        feature_cols=feat_cols,
        coef_ci_method=coef_ci_method,
        coef_ci_cluster=coef_ci_cluster,
        verbose=verbose,
    )
    with open(cache_path, "wb") as f:
        pickle.dump(results_by_pid, f)
    if verbose:
        print(f"\nEvaluated {len(results_by_pid)} participants; saved to {cache_path}")
    return results_by_pid


def to_nested_results(
    results_by_pid: Mapping[str, Any],
    model_name: str = MODEL_NAME,
) -> Dict[str, Dict[str, Any]]:
    """Put each result one level down, under `model_name`: the
    `{pid: {model_name: result}}` shape that the shared per-participant helpers in
    `plots/feature_ranks.py` take."""
    return {pid: {model_name: res} for pid, res in results_by_pid.items()}
