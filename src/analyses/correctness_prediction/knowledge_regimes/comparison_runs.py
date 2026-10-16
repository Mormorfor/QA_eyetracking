"""The L1-trained correctness model scored separately on each Study 2 knowledge regime.

`run_regime_split_comparison` fits the correctness model on every L1 trial and tests it
on the Study 2 trials of each regime in turn (no, partial and full knowledge), and on all
of them pooled (`all`), through the standard train-on-L1 / test-on-new-data bundle
(`run.py::run_cross_dataset_correctness_bundle`). It returns one row of scores per run,
side by side. `confidence_correlation` builds on its per-trial predictions.

**The default data is Study 2's first pilot, not KnowQA.** `NEW_EXP_FEATURES_PATH` and
`NEW_EXP_IA_ANSWERS_PATH` are the `testrun_qa` tables (3 participants, 36 trials). Pass
KnowQA's feature table and IA report to run on the real data.

The regime is read from the feature table's `regime` column. Only if the table has none
is it merged in from the IA report, on `(participant_id, TRIAL_INDEX)`.

**The per-participant features are pooled over the regimes, on purpose.** The opening-scan
features (`dominance_score_*`, `breaks_pattern_*`) are worked out once from each person's
whole set of trials -- all three regimes and, in KnowQA, all their sessions (145 trials) --
and the table is then sliced by regime. That keeps training and test alike: the L1 model
learned these features as a trait of the person over all of their trials (about 54 each).
Rebuilt per regime, the same column would describe about 48 trials of one condition at test
time: a different quantity under the same name. Per-regime scores answer another question,
whether a person's opening scan changes with what they know, and they can differ a lot:
built with `save_model_ready(..., pattern_scope_by=["regime"])`, participant 4000's
`dominance_score_no_q` is 0.224 under full knowledge and 0.630 under partial.
`pitfalls.md` §3 has more on scope.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Any, Dict, List, Sequence, Union

import pandas as pd

from src.config import columns as Con
from src.config.datasets import READY_ALL_FEATURES_PATH, NEW_EXP_FEATURES_PATH, NEW_EXP_IA_ANSWERS_PATH
from src.features.build import load_model_ready
from src.analyses.correctness_prediction.run import (
    run_cross_dataset_correctness_bundle,
)
from src.analyses.correctness_prediction.knowledge_regimes.descriptives import (
    SHOWCASE_FEATURE_COLS,
)
from src.analyses.correctness_prediction.plots.results import correctness_results_to_summary_df

# The regime column, in both the IA report and the feature table.
REGIME_COL = "regime"

# Regime labels that are not a knowledge condition, such as "unset". Left out of the
# per-regime runs only: the pooled run still includes those trials, as it does trials
# with no regime at all.
EXCLUDED_REGIMES = frozenset({"unset", "none", "nan", ""})

# Label of the pooled run: every regime together.
POOLED_LABEL = "all"

# The words `_regime_sort_key` looks for in a regime label, in the intended order.
_REGIME_ORDER_HINT = ("no", "partial", "full")

# Columns of the short comparison table; `comparison_full` keeps every column.
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
    """`s` as stripped, lower-case text, so labels match whatever their spacing or case."""
    return s.astype(str).str.strip().str.lower()


def _regime_sort_key(regime: str):
    """Sort key meant to order the regimes no -> partial -> full, unknown labels last.

    It looks for the words in that order, and "no" is inside "knowledge", so every
    "... knowledge" label matches "no" first: the three regimes in fact sort
    alphabetically (full, no, partial).
    """
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
    """One row per trial: `(participant_id, TRIAL_INDEX, regime)`, from the IA report.

    `ia_source` is the IA report, as a DataFrame or a CSV path. It repeats the regime on
    every word row of a trial; the first is kept. `participant_id` and `regime` are
    stripped and lower-cased, as Study 2's cleaning does for the ids, so they match the
    feature table. Trials with no regime are left out.
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
    """`features_df` with a stripped, lower-case `regime` column.

    If the table already has the column, it is only normalised. Otherwise it is merged in
    from `regime_source` (an IA report or a per-trial map, as a DataFrame or a CSV path)
    on `(participant_id, TRIAL_INDEX)`. The merge keeps every trial and does not check
    coverage: a trial the source does not cover gets no regime, and then appears only in
    the pooled run.
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
    return load_model_ready(frame_or_path)


def _resolve_regimes(
    new_features: pd.DataFrame,
    regimes: Optional[Sequence[str]],
    *,
    regime_col: str,
    excluded: frozenset,
) -> List[str]:
    """The regimes to evaluate, in order.

    Given `regimes`: those labels, stripped and lower-cased, in the order given.
    Otherwise every regime in the data except those in `excluded`, sorted by
    `_regime_sort_key`.
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
    feature_cols: Optional[Sequence[str]] = SHOWCASE_FEATURE_COLS,
    save: Optional[bool] = None,
    close: bool = False,
    verbose: bool = True,
    **bundle_kwargs: Any,
) -> Dict[str, Any]:
    """Fit the correctness model on all of L1 and test it on each Study 2 regime.

    Each regime's trials of `new_features`, and all of them pooled (`all`, with
    `include_pooled`), go through `run_cross_dataset_correctness_bundle` in turn: the
    model (`TrialLevelLogRegModel`) is fitted on every trial of `train_df` and scored on
    that subset. Every run fits the same model on the same data, so the runs differ only
    in the trials they are scored on. A regime with no trials is skipped, with a message.

    `new_features` and `regime_source` default to the first pilot; see the module
    docstring. `regime_source` is read only when `new_features` has no `regime` column.
    `regimes` sets which regimes run, and in what order; by default they are taken from
    the data, minus `excluded_regimes`, in the order `_regime_sort_key` gives.
    `feature_cols` defaults to the Study 1 model's features
    (`descriptives.SHOWCASE_FEATURE_COLS`), the same as `descriptives.fit_l1_model`, so
    both parts of the Study 2 analysis use one model. `None` uses every L1 feature
    (`get_full_feature_cols`); the bundle drops any feature the Study 2 table lacks,
    printing which. `save`, `close` and
    `bundle_kwargs` go to the bundle, which saves each run's figures and summary under
    `reports/correctness_prediction/`.

    Returns a dict: `comparison` (one row per run, the columns in `COMPARISON_VIEW_COLS`),
    `comparison_full` (every summary column), `outputs` (`{run label: bundle result}`),
    `new_features` (with `regime` attached) and `regimes` (the regimes run). The
    comparison tables are returned, not saved.
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

    # The runs as (label, test trials): the pooled run first, then the regimes.
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
