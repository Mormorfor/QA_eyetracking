"""Train-on-L1 / test-on-new-data correctness comparison, split by knowledge regime.

The new (question-answering) experiment assigns each article a *knowledge regime*
(no / partial / full knowledge). This module fits the L1 correctness model on **all**
of the L1 data and evaluates it separately on each regime of the new data, so we can
compare how well the model transfers to each condition.

The regime lives in the raw IA report (``regime`` column) but is dropped during
feature building, so it is re-attached to the model-ready features on the trial keys
``(participant_id, TRIAL_INDEX)`` before splitting.

Everything is parameterized by path/DataFrame, so the same entry point works for the
current three-participant test run and, unchanged, for the full experiment once its
features are built — just point ``new_features`` / ``regime_source`` at the real data
(or pass the loaded frames in directly).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import pandas as pd

from src import constants as Con
from src.data_paths import (
    READY_ALL_FEATURES_PATH,
    NEW_EXP_FEATURES_PATH,
    NEW_EXP_IA_ANSWERS_PATH,
)
from src.predictive_modeling.answer_correctness.model_data import load_all_features
from src.predictive_modeling.answer_correctness.run_model_bundles import (
    run_cross_dataset_correctness_bundle,
)
from src.predictive_modeling.answer_correctness.answer_correctness_viz import (
    correctness_results_to_summary_df,
)

# Default regime column name in the raw IA report.
REGIME_COL = "regime"

# Regime labels that are not real knowledge conditions (practice / unassigned).
EXCLUDED_REGIMES = frozenset({"unset", "none", "nan", ""})

# Label used for the pooled (all-regimes-together) baseline row / run.
POOLED_LABEL = "all"

# Ordering hint so discovered regimes come out no -> partial -> full.
_REGIME_ORDER_HINT = ("no", "partial", "full")

# Columns surfaced in the trimmed comparison table (others are kept in the full frame).
COMPARISON_VIEW_COLS = (
    "regime",
    "n_test",
    "n_positive",
    "n_negative",
    "accuracy",
    "balanced_accuracy",
    "macro_f1",
    "weighted_f1",
    "f1_class_0",
    "f1_class_1",
    "roc_auc",
)

DataFrameOrPath = Union[pd.DataFrame, str, Path]


def _normalize_regime_series(s: pd.Series) -> pd.Series:
    """Lower-case and strip a regime column so labels match regardless of casing."""
    return s.astype(str).str.strip().str.lower()


def _regime_sort_key(regime: str):
    """Sort discovered regimes as no -> partial -> full, unknowns last (alphabetical)."""
    r = regime.lower()
    for i, hint in enumerate(_REGIME_ORDER_HINT):
        if hint in r:
            return (0, i, r)
    return (1, 0, r)


def load_regime_map(
    ia_source: DataFrameOrPath,
    *,
    regime_col: str = REGIME_COL,
    participant_col: str = Con.PARTICIPANT_ID,
    trial_col: str = Con.TRIAL_ID,
) -> pd.DataFrame:
    """Build a per-trial ``(participant_id, TRIAL_INDEX, regime)`` mapping.

    ``ia_source`` is the raw/cleaned IA report (a DataFrame or a CSV path). The regime
    is an article-level label, so one value is kept per trial. ``participant_id`` is
    lower-cased to match the feature tables (which lower-case it during cleaning).
    """
    if isinstance(ia_source, pd.DataFrame):
        ia = ia_source[[participant_col, trial_col, regime_col]].copy()
    else:
        ia = pd.read_csv(
            ia_source,
            low_memory=False,
            usecols=[participant_col, trial_col, regime_col],
        )

    ia[participant_col] = _normalize_regime_series(ia[participant_col])
    ia[regime_col] = _normalize_regime_series(ia[regime_col])

    return (
        ia.dropna(subset=[regime_col])
        .drop_duplicates(subset=[participant_col, trial_col])
        .reset_index(drop=True)
    )


def attach_regime(
    features_df: pd.DataFrame,
    regime_source: Optional[DataFrameOrPath],
    *,
    regime_col: str = REGIME_COL,
    participant_col: str = Con.PARTICIPANT_ID,
    trial_col: str = Con.TRIAL_ID,
) -> pd.DataFrame:
    """Return ``features_df`` with a normalized ``regime`` column.

    If the column is already present it is just normalized in place. Otherwise the
    regime is merged on the trial keys from ``regime_source`` (a per-trial mapping,
    a full IA report, or a CSV path).
    """
    df = features_df.copy()

    if regime_col in df.columns:
        df[regime_col] = _normalize_regime_series(df[regime_col])
        return df

    if regime_source is None:
        raise ValueError(
            f"'{regime_col}' is not in features_df and no regime_source was given "
            "to merge it from."
        )

    regime_map = load_regime_map(
        regime_source,
        regime_col=regime_col,
        participant_col=participant_col,
        trial_col=trial_col,
    )[[participant_col, trial_col, regime_col]]

    return df.merge(regime_map, on=[participant_col, trial_col], how="left")


def _resolve_features(
    frame_or_path: DataFrameOrPath,
) -> pd.DataFrame:
    """Load a model-ready feature table from a path, or pass a DataFrame through."""
    if isinstance(frame_or_path, pd.DataFrame):
        return frame_or_path
    return load_all_features(frame_or_path)


def _resolve_regimes(
    new_features: pd.DataFrame,
    regimes: Optional[Sequence[str]],
    *,
    regime_col: str,
    excluded: frozenset,
) -> List[str]:
    """Return the ordered list of regimes to evaluate.

    If ``regimes`` is given, it is used verbatim (already the desired order).
    Otherwise regimes are discovered from the data (excluding practice/unassigned
    labels) and ordered no -> partial -> full, unknowns last.
    """
    if regimes is not None:
        return [r.strip().lower() for r in regimes]

    observed = (
        new_features[regime_col]
        .dropna()
        .unique()
        .tolist()
    )
    discovered = [r for r in observed if r not in excluded]
    return sorted(discovered, key=_regime_sort_key)


def run_regime_split_comparison(
    train_df: DataFrameOrPath = READY_ALL_FEATURES_PATH,
    new_features: DataFrameOrPath = NEW_EXP_FEATURES_PATH,
    regime_source: Optional[DataFrameOrPath] = NEW_EXP_IA_ANSWERS_PATH,
    *,
    regimes: Optional[Sequence[str]] = None,
    regime_col: str = REGIME_COL,
    excluded_regimes: frozenset = EXCLUDED_REGIMES,
    include_pooled: bool = True,
    feature_cols: Optional[Sequence[str]] = None,
    save: bool = False,
    close: bool = False,
    verbose: bool = True,
    **bundle_kwargs: Any,
) -> Dict[str, Any]:
    """Train the L1 correctness model on all of L1, test it per knowledge regime.

    Fits ``TrialLevelLogRegModel`` on the entire ``train_df`` (L1) and evaluates it,
    via :func:`run_cross_dataset_correctness_bundle`, separately on each regime of the
    new data — plus a pooled ``all`` baseline when ``include_pooled`` is set. Returns a
    side-by-side comparison table so the transfer to each condition can be compared.

    Parameters
    ----------
    train_df :
        L1 model-ready features, or a path to load them from (default: the cached L1
        ``READY_ALL_FEATURES_PATH``).
    new_features :
        New-experiment model-ready features, or a path (default:
        ``NEW_EXP_FEATURES_PATH``). If it already carries a ``regime`` column, that is
        used directly and ``regime_source`` is ignored.
    regime_source :
        Where to read the per-trial regime from when ``new_features`` lacks it: a raw
        IA report DataFrame, a per-trial mapping, or a CSV path (default: the new-exp
        cleaned IA report). Set to ``None`` if ``new_features`` already has the column.
    regimes :
        Explicit regimes to evaluate, in the desired order. ``None`` (default)
        discovers them from the data, dropping ``excluded_regimes`` and ordering
        no -> partial -> full.
    include_pooled :
        Also run/report the pooled ``all`` baseline over every new-data trial.
    feature_cols :
        Passed through to the bundle. ``None`` uses L1's full feature set; any feature
        absent from the new data is dropped automatically.
    save, close, **bundle_kwargs :
        Forwarded to :func:`run_cross_dataset_correctness_bundle` (plot/CSV saving,
        figure closing, ``dpi``, ``paper_dirs``, ...).

    Returns
    -------
    dict with:
        ``comparison`` : trimmed one-row-per-regime metrics table (see
            :data:`COMPARISON_VIEW_COLS`).
        ``comparison_full`` : the same rows with every summary column.
        ``outputs`` : ``{label -> bundle result dict}`` for each run.
        ``new_features`` : the new features with the ``regime`` column attached.
        ``regimes`` : the ordered regimes that were evaluated.
    """
    l1_features = _resolve_features(train_df)
    new_feats = _resolve_features(new_features)
    new_feats = attach_regime(new_feats, regime_source, regime_col=regime_col)

    eval_regimes = _resolve_regimes(
        new_feats, regimes, regime_col=regime_col, excluded=excluded_regimes
    )

    if verbose:
        print(f"train (L1): {l1_features.shape} | test (new): {new_feats.shape}")
        print("\nnew-data trials per regime:")
        print(new_feats[regime_col].value_counts(dropna=False).to_string())
        print(f"\nregimes evaluated: {eval_regimes}")

    # Build the list of (label, test-subset) runs: pooled baseline first, then regimes.
    splits: List[tuple[str, pd.DataFrame]] = []
    if include_pooled:
        splits.append((POOLED_LABEL, new_feats))
    splits.extend(
        (r, new_feats.loc[new_feats[regime_col] == r]) for r in eval_regimes
    )

    summaries: List[pd.DataFrame] = []
    outputs: Dict[str, Any] = {}

    for label, subset in splits:
        test_subset = subset.drop(columns=[regime_col])
        n = len(test_subset)
        if n == 0:
            if verbose:
                print(f"[skip] no trials for regime '{label}'")
            continue

        if verbose:
            print(f"\n{'=' * 72}")
            print(f"Train L1  /  Test new [{label}]  —  {n} trials")
            print(f"{'=' * 72}")

        out = run_cross_dataset_correctness_bundle(
            train_df=l1_features,
            test_df=test_subset,
            feature_cols=feature_cols,
            save=save,
            close=close,
            split_tag=f"trainL1_test_{label.replace(' ', '_')}",
            title_prefix=f"[{label}]",
            **bundle_kwargs,
        )
        outputs[label] = out

        model_name = next(iter(out["results"]))
        summary = correctness_results_to_summary_df(
            out["results"],
            run_identifier=label,
            trained_feature_cols_by_model={model_name: out["feature_cols"]},
        )
        summary.insert(0, "regime", label)
        summaries.append(summary)

    if not summaries:
        raise ValueError("No non-empty regime subsets to evaluate.")

    comparison_full = pd.concat(summaries, ignore_index=True)
    view_cols = [c for c in COMPARISON_VIEW_COLS if c in comparison_full.columns]
    comparison = comparison_full[view_cols].copy()

    return {
        "comparison": comparison,
        "comparison_full": comparison_full,
        "outputs": outputs,
        "new_features": new_feats,
        "regimes": eval_regimes,
    }
