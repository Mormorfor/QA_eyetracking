"""Run (or reload) the per-person full leave-one-trial-out correctness runs.

For **each participant**, every trial is held out once as the test set while the
model trains on that participant's remaining trials; the held-out predictions are
pooled into one :class:`CorrectnessEvaluationResult` per participant, so
``accuracy`` is the participant's leave-one-out accuracy.

Feature handling is intentionally identical to the regular
``run_full_features_correctness_bundle`` pipeline: the feature table is the
cached ``READY_ALL_FEATURES_PATH`` (``load_all_features()``), and the feature set
is passed through exactly as ``feature_cols=`` in a bundle run. Per-participant
coefficients come from a single full-data fit on that participant's trials (the
pipeline's coefficients-from-a-full-fit convention).

The run is expensive (one logistic regression per trial per participant), so the
resulting ``results_by_pid`` dict is pickled under
:data:`~src.data_paths.PER_PERSON_LOO_RESULTS_DIR`, keyed by the feature-set tag
so different feature sets get their own cache file.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Sequence

import pandas as pd

import predictive_modeling.answer_correctness.feature_groups as FG
from predictive_modeling.answer_correctness.models.logreg_model import (
    TrialLevelLogRegModel,
)
from predictive_modeling.answer_correctness.participant_level import (
    evaluate_logreg_on_answer_correctness_full_loo,
)
from src.data_paths import PER_PERSON_LOO_RESULTS_DIR

# Key the per-participant results are stored under in the nested
# ``{pid: {model_name: result}}`` shape the shared coefficient / clustering
# utilities expect.
MODEL_NAME = "trial_level_log_reg"

# The pinned feature set: the "selection" group plus the compact
# last-label-before-confirm features (12 features).
DEFAULT_FEATURE_COLS = FG.SELECT_1_COLS + FG.LAST_CONFIRM_COMPACT
DEFAULT_FEATURE_SET_TAG = "select_1_plus_last_confirm_compact"

# Where saved figures are mirrored to, when a plot is called with ``save=True``.
PAPER_DIRS = ["papers/correctness_prediction"]


def results_cache_path(feature_set_tag: str = DEFAULT_FEATURE_SET_TAG) -> Path:
    """Cache file for one feature set's per-person leave-one-out results."""
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
    """Full leave-one-trial-out per participant, cached on disk.

    Parameters
    ----------
    trial_df
        Prepared trial-level feature table (e.g. ``load_all_features()``). Only
        needed when the run is actually computed; on a cache hit it is ignored.
    feature_cols
        Feature set to train on; defaults to :data:`DEFAULT_FEATURE_COLS`.
    feature_set_tag
        Names the cache file (and, downstream, saved figures). Change it
        whenever ``feature_cols`` changes, or a stale cache will be reused.
    force_recompute
        Re-run even when a cache file exists (and overwrite it).

    Returns
    -------
    dict
        ``results[participant_id] = CorrectnessEvaluationResult`` (pooled
        held-out predictions + a full-fit ``coef_summary``).
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
    """Wrap ``{pid: result}`` into the ``{pid: {model_name: result}}`` shape the
    shared per-participant coefficient / clustering utilities expect."""
    return {pid: {model_name: res} for pid, res in results_by_pid.items()}
