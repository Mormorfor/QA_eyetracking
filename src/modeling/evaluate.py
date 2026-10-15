"""Fitting and scoring a model on one split, and the cross-validation summary tables.

`evaluate_single_model_on_prepared_split` fits one model on a training table
and scores it on a test table; how the split was made is decided elsewhere
(`folds.py`, `crossval.py`). `collect_logreg_coef_summaries` fits once on all
the data, for the coefficients the paper reports. The second half of the file
turns per-fold scores into the summary tables.
"""


from __future__ import annotations

from dataclasses import dataclass

from typing import Sequence, Optional, Mapping, Dict, Any, List, Tuple
import numpy as np
import pandas as pd

from src.config import columns as Con
# `parse_regime` splits a fold-regime name into (split, novelty); the summaries
# below group on both. It lives with the fold vocabulary it belongs to.
from src.modeling.folds import parse_regime


# ---------------------------------------------------------------------------
# Results, and a container for a model-ready table
# ---------------------------------------------------------------------------

@dataclass
class PreparedTrialDataset:
    """
    A trial table, with which of its columns are features, target and ids.
    Used by the answer-reading-time exploration.
    """
    df: pd.DataFrame
    feature_cols: List[str]
    target_col: str
    id_cols: List[str]


@dataclass
class CorrectnessEvaluationResult:
    """What one fit-and-score returns: the held-out labels, predictions and
    probabilities, the accuracy and counts, and the coefficient table."""
    train_df: pd.DataFrame
    test_df: pd.DataFrame
    y_true: np.ndarray
    y_pred: np.ndarray
    y_prob: np.ndarray
    accuracy: float
    n_test: int
    n_positive: int
    n_negative: int
    coef_summary: Optional[pd.DataFrame] = None


def evaluate_single_model_on_prepared_split(
    model,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    target_col: str = Con.IS_CORRECT_COLUMN,
    feature_cols: Sequence[str],
    fit_kwargs: Optional[dict[str, Any]] = None,
    predict_kwargs: Optional[dict[str, Any]] = None,
    predict_proba_kwargs: Optional[dict[str, Any]] = None,
    coef_kwargs: Optional[dict[str, Any]] = None,
) -> CorrectnessEvaluationResult:
    """
    Fit one model on ``train_df`` and score it on ``test_df``.

    The model must have:
    - fit(train_df, target_col=..., feature_cols=..., **fit_kwargs)
    - predict(test_df, feature_cols=..., **predict_kwargs)
    - predict_proba(test_df, feature_cols=..., **predict_proba_kwargs)
    - get_coef_summary(...)
    """
    feat_cols = list(feature_cols)
    fit_kwargs = dict(fit_kwargs or {})
    predict_kwargs = dict(predict_kwargs or {})
    predict_proba_kwargs = dict(predict_proba_kwargs or {})
    coef_kwargs = dict(coef_kwargs or {})

    y_true = test_df[target_col].astype(int).to_numpy()

    model.fit(
        train_df=train_df,
        target_col=target_col,
        feature_cols=feat_cols,
        **fit_kwargs,
    )

    y_pred = model.predict(
        test_df,
        feature_cols=feat_cols,
        **predict_kwargs,
    )
    y_pred = np.asarray(y_pred).reshape(-1).astype(int)

    y_prob = model.predict_proba(
        test_df,
        feature_cols=feat_cols,
        **predict_proba_kwargs,
    )
    y_prob = np.asarray(y_prob).reshape(-1).astype(float)

    coef_summary = model.get_coef_summary(
        train_df=train_df,
        feature_cols=feat_cols,
        **coef_kwargs,
    )

    return CorrectnessEvaluationResult(
        train_df=train_df,
        test_df=test_df,
        y_true=y_true,
        y_pred=y_pred,
        y_prob=y_prob,
        accuracy=float((y_true == y_pred).mean()),
        n_test=len(test_df),
        n_positive=int((y_true == 1).sum()),
        n_negative=int((y_true == 0).sum()),
        coef_summary=coef_summary,
    )


def evaluate_models_on_prepared_split(
    models: Sequence,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    target_col: str = Con.IS_CORRECT_COLUMN,
    feature_cols: Optional[Sequence[str]] = None,
    feature_cols_by_model: Optional[Mapping[str, Sequence[str]]] = None,
    fit_kwargs_by_model: Optional[Mapping[str, dict[str, Any]]] = None,
    predict_kwargs_by_model: Optional[Mapping[str, dict[str, Any]]] = None,
    predict_proba_kwargs_by_model: Optional[Mapping[str, dict[str, Any]]] = None,
    coef_kwargs_by_model: Optional[Mapping[str, dict[str, Any]]] = None,
) -> Dict[str, CorrectnessEvaluationResult]:
    """
    Fit and score several models on the same train/test split.

    Use:
    - feature_cols for one shared feature set across all models, or
    - feature_cols_by_model for per-model feature sets.
    """
    results: Dict[str, CorrectnessEvaluationResult] = {}

    fit_kwargs_by_model = dict(fit_kwargs_by_model or {})
    predict_kwargs_by_model = dict(predict_kwargs_by_model or {})
    predict_proba_kwargs_by_model = dict(predict_proba_kwargs_by_model or {})
    coef_kwargs_by_model = dict(coef_kwargs_by_model or {})

    for model in models:
        model_name = model.name

        if feature_cols_by_model is not None and model_name in feature_cols_by_model:
            feat_cols = list(feature_cols_by_model[model_name])
        elif feature_cols is not None:
            feat_cols = list(feature_cols)
        else:
            raise ValueError(
                f"No feature columns provided for model '{model_name}'."
            )

        results[model_name] = evaluate_single_model_on_prepared_split(
            model=model,
            train_df=train_df,
            test_df=test_df,
            target_col=target_col,
            feature_cols=feat_cols,
            fit_kwargs=fit_kwargs_by_model.get(model_name),
            predict_kwargs=predict_kwargs_by_model.get(model_name),
            predict_proba_kwargs=predict_proba_kwargs_by_model.get(model_name),
            coef_kwargs=coef_kwargs_by_model.get(model_name),
        )

    return results


def collect_logreg_coef_summaries(
    trial_df: pd.DataFrame,
    feature_sets: Mapping[str, Sequence[str]],
    *,
    target_col: str = Con.IS_CORRECT_COLUMN,
    model_builder=None,
    ci_method: str = "bootstrap",
    ci_cluster: str = "cluster",
    ci: float = 0.95,
) -> Dict[str, pd.DataFrame]:
    """
    Fit the logistic regression once on all of ``trial_df`` for each feature
    set, and return each fit's coefficient table. The paper's coefficient
    figures come from here.

    Cross-validation runs reloaded from disk keep no per-fold coefficient
    tables, so coefficients for reporting come from this single fit. An empty
    feature set (e.g. the dummy baseline) gives an empty table.

    **The intervals default to the bootstrap, clustered by participant.** Each
    resample draws whole participants and refits the actual model, so the L2
    penalty, the class weights and the grouping of trials within people are all
    accounted for without approximation. (``ci_cluster="auto"`` would give a
    row bootstrap instead, which is not what the name suggests.)

    Cost: about 70 s per feature set on Study 1, at the default 5,000
    resamples. Cross-validation uses the Wald intervals instead (`crossval.py`),
    because it would otherwise bootstrap inside every fold; those coefficients
    are not what the paper reports.

    Returns
    -------
    Dict[str, pd.DataFrame]
        ``{model_name: coef_summary}``, each exactly what
        ``TrialLevelLogRegModel.get_coef_summary`` produces (feature, coef,
        odds_ratio, abs_coef, ci_low, ci_high, or_ci_low, or_ci_high, sig_ci,
        n_boot_ok).
    """
    # Imported lazily to avoid a module-level import cycle.
    from src.modeling.models.logreg_model import (
        TrialLevelLogRegModel,
    )

    if model_builder is None:
        model_builder = lambda: TrialLevelLogRegModel()

    summaries: Dict[str, pd.DataFrame] = {}
    for name, cols in feature_sets.items():
        cols = list(cols) if cols is not None else []
        if not cols:
            summaries[name] = pd.DataFrame()
            continue

        model = model_builder()
        model.fit(train_df=trial_df, target_col=target_col, feature_cols=cols)
        summaries[name] = model.get_coef_summary(
            train_df=trial_df,
            feature_cols=cols,
            ci_method=ci_method,
            ci_cluster=ci_cluster,
            ci=ci,
            target_col=target_col,
        )

    return summaries


def fit_julia_mixed_model_on_prepared_full_data(
    model,
    fit_df: pd.DataFrame,
    *,
    target_col: str = Con.IS_CORRECT_COLUMN,
    feature_cols: Optional[Sequence[str]] = None,
    fit_kwargs: Optional[dict[str, Any]] = None,
    coef_kwargs: Optional[dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Fit a mixed-effects model on all of ``fit_df`` (no split), and return its
    fixed-effect coefficients, its random effects per participant and per text,
    and the variance summary -- how much of the correctness signal sits with
    the person and how much with the item.

    Only for the Julia mixed-effects model
    (`explorations/mixed_models/models.py::TrialLevelJuliaGLMERModel`): it
    calls `get_random_effects()` and `get_random_effect_variance_summary()`,
    which only that model has. Passing the logistic regression raises
    `AttributeError`, on purpose -- a model with no random effects has nothing
    to return here. For a full-data fit of the logistic regression, use
    `collect_logreg_coef_summaries` above.

    Mixed effects are not part of the current paper (`research-context.md` §6);
    this is exploration code.
    """
    feat_cols = None if feature_cols is None else list(feature_cols)
    fit_kwargs = dict(fit_kwargs or {})
    coef_kwargs = dict(coef_kwargs or {})

    model.fit(
        train_df=fit_df,
        target_col=target_col,
        feature_cols=feat_cols,
        **fit_kwargs,
    )

    coef_summary = model.get_coef_summary(
        train_df=fit_df,
        feature_cols=feat_cols,
        **coef_kwargs,
    )

    random_effects = model.get_random_effects()
    random_varcorr = model.get_random_effect_variance_summary()

    return {
        "fit_df": fit_df,
        "n_rows": len(fit_df),
        "n_positive": int((fit_df[target_col] == 1).sum()),
        "n_negative": int((fit_df[target_col] == 0).sum()),
        "coef_summary": coef_summary,
        "random_effects": random_effects,
        "random_effect_variance_summary": random_varcorr,
    }

# ===========================================================================
# Cross-validation summary tables
#
# These turn per-fold, per-regime scores into the reported tables. Folds score
# different numbers of trials, so means are weighted by how many trials each
# fold scored (`n_eval`); the plain mean is kept beside it as
# `unweighted_mean_*`.
# ===========================================================================

def _weighted_fold_means(
    frame: pd.DataFrame,
    keys: List[str],
    value_cols: Sequence[str] = ("accuracy", "balanced_accuracy"),
    weight_col: str = "n_eval",
) -> pd.DataFrame:
    """
    Mean of a per-fold score, weighted by how many trials each fold scored.

    Fold sizes differ within a regime -- on hunters the `both` cell has 78-120
    trials per fold, and `seen_subject_unseen_item` 702-1080 -- so a plain mean
    would let a fold of 78 trials count as much as one of 120.

    **Exact for `accuracy`, an approximation for `balanced_accuracy`.** Pooling
    accuracy over folds is the weighted mean. Balanced accuracy is the mean of
    sensitivity and specificity, so pooling it exactly needs each fold's
    confusion counts, which the summary rows do not have. Weighting by `n_eval`
    goes the right way -- bigger folds are better estimates -- but is not the
    same as scoring all held-out trials at once.
    """
    rows = []
    for key_vals, g in frame.groupby(keys, sort=False):
        if not isinstance(key_vals, tuple):
            key_vals = (key_vals,)
        w = g[weight_col].astype(float)
        total = w.sum()
        rec = dict(zip(keys, key_vals))
        for col in value_cols:
            rec[f"mean_{col}"] = (
                float((g[col].astype(float) * w).sum() / total)
                if total else float("nan")
            )
        rows.append(rec)
    return pd.DataFrame(rows)


_SUMMARY_AGG = dict(
    folds=("fold", "nunique"),
    # The `mean_*` columns come from `_weighted_fold_means`; the plain means
    # are kept here as `unweighted_mean_*`, so the two can be compared.
    unweighted_mean_accuracy=("accuracy", "mean"),
    std_accuracy=("accuracy", "std"),
    unweighted_mean_balanced_accuracy=("balanced_accuracy", "mean"),
    std_balanced_accuracy=("balanced_accuracy", "std"),
    mean_n_eval=("n_eval", "mean"),
    total_n_eval=("n_eval", "sum"),
)


def _aggregate_cv_summary(
    rows_summary: List[Dict[str, Any]],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Build the four summary tables from the per-(model, fold, regime) rows.

    Each regime name is split into its two parts, ``split`` and ``novelty``
    (see `folds.py`), so results can be grouped by either:

    ========================  ===================================================
    table                     one row per / use
    ========================  ===================================================
    ``summary_df``            model x fold x regime -- the raw rows, with
                              ``split`` and ``novelty`` added
    ``summary_by_regime_df``  model x regime -- the finest summary
    ``summary_by_novelty_df`` model x novelty -- **the three numbers the paper
                              reports**, pooling whichever splits were evaluated
    ``summary_overall_df``    model x split. It averages over the three
                              novelties, which ask different questions, so it is
                              only a check that a run worked, not a result
    ========================  ===================================================

    ``mean_accuracy`` / ``mean_balanced_accuracy`` are weighted by ``n_eval``;
    the plain means are kept as ``unweighted_mean_*``. The spread (``std_*``),
    and the standard error in :func:`summarize_cv_results_by_regime`, are not
    corrected for the folds sharing most of their training data, so they
    understate the uncertainty. There is no agreed correction, so they are left
    as they are and described as such.
    """
    summary_df = pd.DataFrame(rows_summary)

    parsed = summary_df["regime"].map(parse_regime)
    summary_df["split"] = [p[0] for p in parsed]
    summary_df["novelty"] = [p[1] for p in parsed]

    def _by(keys: List[str]) -> pd.DataFrame:
        agg = summary_df.groupby(keys, as_index=False).agg(**_SUMMARY_AGG)
        weighted = _weighted_fold_means(summary_df, keys)
        return (
            agg.merge(weighted, on=keys, how="left", validate="one_to_one")
            .sort_values(keys)
            .reset_index(drop=True)
        )

    summary_by_regime_df = _by(["model", "regime"])
    summary_by_novelty_df = _by(["model", "novelty"])
    summary_overall_df = _by(["model", "split"])

    # Record what each frame pooled, so a saved table can be read back later
    # without having to guess which splits went into it.
    summary_by_novelty_df["splits_pooled"] = "+".join(
        sorted(summary_df["split"].unique())
    )
    summary_overall_df["novelties_pooled"] = "+".join(
        sorted(summary_df["novelty"].unique())
    )

    return (
        summary_df,
        summary_by_regime_df,
        summary_by_novelty_df,
        summary_overall_df,
    )

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

    # Only three levels are tabulated. Any other stops with an error, so a
    # figure can never be drawn at one level and labelled with another (plot
    # titles are built from `ci`).
    z_map = {
        0.90: 1.645,
        0.95: 1.96,
        0.99: 2.576,
    }
    if ci not in z_map:
        raise ValueError(
            f"ci must be one of {sorted(z_map)}, got {ci!r}. These are the only "
            f"levels with a tabulated z here; anything else would have been "
            f"drawn at z=1.96 and labelled with the level you asked for."
        )
    z = z_map[ci]

    out = (
        df.groupby("regime", as_index=False)
        .agg(
            n_folds=("fold", "nunique"),
            unweighted_mean_metric=(metric_col, "mean"),
            std_metric=(metric_col, "std"),
            min_metric=(metric_col, "min"),
            max_metric=(metric_col, "max"),
            mean_n_eval=("n_eval", "mean"),
            total_n_eval=("n_eval", "sum"),
        )
        .sort_values("regime")
        .reset_index(drop=True)
    )

    # Weighted by how many trials each fold scored (see _weighted_fold_means).
    weighted = _weighted_fold_means(df, ["regime"], value_cols=(metric_col,))
    out = out.merge(
        weighted.rename(columns={f"mean_{metric_col}": "mean_metric"}),
        on="regime", how="left", validate="one_to_one",
    )

    # With a single fold the spread is undefined (NaN); it is set to 0, so the
    # interval below shrinks to the mean.
    out["std_metric"] = out["std_metric"].fillna(0.0)
    # NOT corrected for the overlap between folds: they share most of their
    # training data, so this understates the uncertainty and the bars are
    # narrower than they should be. There is no agreed estimator to switch to,
    # so it is left as it is and said so.
    out["se_metric"] = out["std_metric"] / np.sqrt(out["n_folds"])
    # Scores are proportions, so the interval is cut to [0, 1].
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

    Runs :func:`summarize_cv_results_by_regime` for every model in the run (or
    just ``models`` if given) and keeps only ``regime`` (default: the "both"
    test regime, new person and new paragraph). A model with no results for
    that regime is left out.

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
