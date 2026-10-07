"""Running a model bundle: build the split, fit, score, save, plot.

Was `answer_correctness/run_model_bundles.py`.

**It tried to live in `modeling/` during stage E and came back.** The run half is generic --
build the split, fit, score, save -- but the plot half draws *this* analysis's figures
(`plots/coefficients`, `plots/feature_ranks`, `plots/probabilities`, `plots/results`), so
putting the whole module below `analyses/` only moved the layering violation into
`modeling/`. It belongs with the figures it draws.

⚠️ **Two parked modules still import from here**, which is `explorations/` -> `analyses/`
and the one direction the layering rule does not allow:
`explorations/feature_search/column_options.py` (sweeping column sets) and
`explorations/mixed_models/run.py` (the Julia/R bundles). Both want the *run* half and get
the *plot* half with it.

**The fix is to split run from plot** -- `modeling/bundles.py` runs and saves numbers, the
plot orchestration stays here and is passed in or called after. That is a signature change
on the paper's main run path, so it is flagged rather than done (stage E, 2026-10-07).
"""

from __future__ import annotations
from pathlib import Path
from typing import Optional, Sequence, Tuple, List, Dict, Any
import pandas as pd
from src.config import columns as Con
from src.config.datasets import READY_ALL_FEATURES_PATH
from src.modeling.folds import group_vise_train_test_split
from src.modeling.feature_sets import get_full_feature_cols
from src.features.build import build_trial_level_model_df, load_model_ready
from src.modeling.evaluate import (
    evaluate_models_on_prepared_split, fit_julia_mixed_model_on_prepared_full_data,
)
from src.modeling.models.logreg_model import (
    TrialLevelLogRegModel,
)
from src.analyses.correctness_prediction.plots.coefficients import (
    plot_coef_summary_barh,
)
from src.analyses.correctness_prediction.plots.feature_ranks import (
    plot_feature_correlation_heatmap,
)
from src.analyses.correctness_prediction.plots.probabilities import (
    plot_predicted_probability_hist,
)
from src.analyses.correctness_prediction.plots.results import (
    correctness_results_to_summary_df,
    show_correctness_model_results,
)
from src.lib.plotting.confusion import plot_confusion_heatmap
from src.lib.plotting.output import save_output


def _split_tag(test_regimes: Sequence[str]) -> str:
    return "+".join(test_regimes)


def _build_full_trial_df(
    df: pd.DataFrame,
    keep_cols: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    return build_trial_level_model_df(
        df=df,
        keep_cols=keep_cols,
        target_col=Con.IS_CORRECT_COLUMN,
    )


def _load_or_build_full_trial_df(
    df: pd.DataFrame,
    *,
    keep_cols: Optional[Sequence[str]] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Return the full trial-level feature DataFrame.

    Loads from READY_ALL_FEATURES_PATH if it exists; otherwise builds via
    `_build_full_trial_df`. — re-run
    `save_model_ready` to refresh the cache). Any `keep_cols` not already
    present in the cached frame are merged from `df` on (participant, trial).
    """
    cache_path = Path(READY_ALL_FEATURES_PATH)
    if cache_path.exists():
        if verbose:
            print(f"Loading cached features from {cache_path}")
        trial_df = load_model_ready()
        keep = list(keep_cols) if keep_cols is not None else []
        missing = [c for c in keep if c not in trial_df.columns]
        if missing:
            extra = (
                df[[Con.PARTICIPANT_ID, Con.TRIAL_ID] + missing]
                .drop_duplicates(subset=[Con.PARTICIPANT_ID, Con.TRIAL_ID])
            )
            trial_df = trial_df.merge(
                extra, on=[Con.PARTICIPANT_ID, Con.TRIAL_ID], how="left"
            )
        return trial_df

    return _build_full_trial_df(
        df=df,
        keep_cols=keep_cols,
    )


def build_train_test_trial_dfs(
    df: pd.DataFrame,
    test_regimes: Sequence[str],
    *,
    test_split: str = "test",
    fold: Optional[int] = None,
    sources: Sequence[str] = ("hunters", "gatherers"),

    keep_cols: Optional[Sequence[str]] = None,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    train_raw, test_raw, split_info = group_vise_train_test_split(
        df,
        test_regimes=test_regimes,
        test_split=test_split,
        fold=fold,
        sources=sources,
        random_state=random_state,
    )

    if Path(READY_ALL_FEATURES_PATH).exists():
        trial_full = _load_or_build_full_trial_df(
            df=df,
            keep_cols=keep_cols,
        )
        key_cols = [Con.PARTICIPANT_ID, Con.TRIAL_ID]
        train_keys = train_raw[key_cols].drop_duplicates()
        test_keys = test_raw[key_cols].drop_duplicates()
        train_df = trial_full.merge(train_keys, on=key_cols, how="inner")
        test_df = trial_full.merge(test_keys, on=key_cols, how="inner")
        return train_df, test_df, split_info

    train_df = build_trial_level_model_df(
        df=train_raw,
        keep_cols=keep_cols,
        target_col=Con.IS_CORRECT_COLUMN,
    )

    test_df = build_trial_level_model_df(
        df=test_raw,
        keep_cols=keep_cols,
        target_col=Con.IS_CORRECT_COLUMN,
    )

    return train_df, test_df, split_info


def _resolve_feature_cols(
    train_df: pd.DataFrame,
    feature_cols: Optional[Sequence[str]],
) -> List[str]:
    if feature_cols is not None:
        return list(feature_cols)
    return list(get_full_feature_cols(train_df))


def _save_summary_csv(
    *,
    results: Dict[str, Any],
    model_name: str,
    trained_feature_cols: Sequence[str],
    base_dir: str,
    run_identifier: str,
    to_paper,
    formula: Optional[str] = None,
):
    summary_df = correctness_results_to_summary_df(
        results,
        run_identifier=run_identifier,
        trained_feature_cols_by_model={model_name: list(trained_feature_cols)},
    )

    if formula is not None:
        summary_df["formula"] = formula

    return save_output(
        None,
        analysis="correctness_prediction",
        plot="model_summary",
        tables={"summary": summary_df},
        save=True,
        to_paper=to_paper,
        subdir=base_dir,
        run=run_identifier,
        model=model_name,
    ).paths


def _titled(core: str, title_prefix: str = "") -> str:
    """Prepend an optional context label (e.g. a knowledge regime) to a plot title."""
    prefix = title_prefix.strip()
    return f"{prefix} — {core}" if prefix else core


def _plot_confusions(
    *,
    y_true,
    y_pred,
    model_name: str,
    base_dir: str,
    save: bool,
    to_paper,
    close: bool,
    title_prefix: str = "",
):
    _, _, cm_paths = plot_confusion_heatmap(
        y_true=y_true,
        y_pred=y_pred,
        labels=(0, 1),
        normalize=True,
        title=_titled(f"{model_name} – normalized confusion", title_prefix),
        save=save,
        subdir=f"{base_dir}/confusion",
        plot="confusion_matrix",
        model=model_name,
        to_paper=to_paper,
        close=close,
    )

    _, _, cm_paths2 = plot_confusion_heatmap(
        y_true=y_true,
        y_pred=y_pred,
        labels=(0, 1),
        normalize=False,
        title=_titled(f"{model_name} – un-normalized confusion", title_prefix),
        save=save,
        subdir=f"{base_dir}/confusion",
        plot="confusion_matrix",
        model=model_name,
        to_paper=to_paper,
        close=close,
    )

    return cm_paths, cm_paths2


def _plot_prob_hist(
    *,
    y_true,
    y_prob,
    model_name: str,
    base_dir: str,
    save: bool,
    to_paper,
    dpi: int,
    close: bool,
    title_prefix: str = "",
):
    if y_prob is None:
        return []
    _, _, prob_paths = plot_predicted_probability_hist(
        y_true=y_true,
        y_prob=y_prob,
        title=_titled(
            f"{model_name} – predicted probability by true outcome", title_prefix
        ),
        save=save,
        subdir=f"{base_dir}/predicted_probabilities",
        plot="predicted_probability_hist",
        model=model_name,
        to_paper=to_paper,
        dpi=dpi,
        close=close,
    )
    return prob_paths


def _plot_coef_summaries(
    *,
    coef_summary: pd.DataFrame,
    model_name: str,
    base_dir: str,
    save: bool,
    to_paper,
    dpi: int,
    close: bool,
    figsize: Optional[Tuple[int, int]] = None,
    title_prefix: str = "",
):
    coef_paths = []
    coef_sig_paths = []

    if coef_summary is not None and not coef_summary.empty:
        _, _, coef_paths = plot_coef_summary_barh(
            coef_summary=coef_summary,
            value_col="coef",
            model_name=model_name,
            title=_titled(f"{model_name} – coefficients", title_prefix),
            save=save,
            subdir=f"{base_dir}/coefficients",
            plot="coefficients",
            model=model_name,
            coefs="all",
            to_paper=to_paper,
            dpi=dpi,
            close=close,
            significant_only=False,
            figsize=figsize,
        )

        _, _, coef_sig_paths = plot_coef_summary_barh(
            coef_summary=coef_summary,
            value_col="coef",
            model_name=model_name,
            title=_titled(f"{model_name} – significant coefficients", title_prefix),
            save=save,
            subdir=f"{base_dir}/coefficients",
            plot="coefficients",
            model=model_name,
            coefs="significant",
            to_paper=to_paper,
            dpi=dpi,
            close=close,
            significant_only=True,
            figsize=figsize,
        )

    return coef_paths, coef_sig_paths


def _plot_feature_corr(
    *,
    trial_df: pd.DataFrame,
    corr_feature_cols: Sequence[str],
    base_dir: str,
    save: bool,
    to_paper,
    dpi: int,
    close: bool,
    title_prefix: str = "",
):
    _, _, corr_paths = plot_feature_correlation_heatmap(
        trial_df,
        feature_cols=list(corr_feature_cols),
        figsize=(30, 30),
        method="pearson",
        cluster_order=True,
        title=_titled("Feature correlation (pearson) – cluster-ordered", title_prefix),
        save=save,
        subdir=f"{base_dir}/diagnostics/feature_correlation",
        plot="feature_correlation",
        order="clustered",
        n=len(corr_feature_cols),
        to_paper=to_paper,
        dpi=dpi,
        close=close,
    )
    return corr_paths


def run_full_features_correctness_bundle(
    df: pd.DataFrame,
    test_regimes: Sequence[str],
    *,
    test_split: str = "test",
    fold: Optional[int] = None,
    sources: Sequence[str] = ("hunters", "gatherers"),
    feature_cols: Optional[Sequence[str]] = None,
    coef_ci_method: str = "wald",
    coef_ci_cluster: str = "row",
    save: Optional[bool] = None,
    to_paper=None,
    dpi: int = 300,
    close: bool = False,
    subdir: Optional[str] = None,
    run_identifier: str = "",
    random_state: int = 42,
    coef_figsize: Optional[Tuple[int, int]] = None,
    title_prefix: str = "",
) -> Dict[str, Any]:
    model = TrialLevelLogRegModel()
    model_name = model.name
    model_family = "logreg"

    split_tag = _split_tag(test_regimes)
    # Browsable folder under reports/correctness_prediction/; the filename still
    # carries every facet, so the folder is navigation, not identity.
    base_dir = "/".join(x for x in (split_tag or "full_fit", model_family, subdir) if x)

    train_df, test_df, split_info = build_train_test_trial_dfs(
        df=df,
        test_regimes=test_regimes,
        test_split=test_split,
        fold=fold,
        sources=sources,
        keep_cols=None,
        random_state=random_state,
    )

    feat_cols = _resolve_feature_cols(train_df, feature_cols)

    results = evaluate_models_on_prepared_split(
        models=[model],
        train_df=train_df,
        test_df=test_df,
        target_col=Con.IS_CORRECT_COLUMN,
        feature_cols=feat_cols,
        coef_kwargs_by_model={
            model_name: {
                "ci_method": coef_ci_method,
                "ci_cluster": coef_ci_cluster,
            }
        },
    )

    show_correctness_model_results(results)
    res = results[model_name]

    summary_paths = None
    if save:
        summary_paths = _save_summary_csv(
            results=results,
            model_name=model_name,
            trained_feature_cols=model.feature_cols_,
            base_dir=base_dir,
            run_identifier=run_identifier,
            to_paper=to_paper,
            formula=None,
        )

    cm_paths, cm_paths2 = _plot_confusions(
        y_true=res.y_true,
        y_pred=res.y_pred,
        model_name=model_name,
        base_dir=base_dir,
        save=save,
        to_paper=to_paper,
        close=close,
        title_prefix=title_prefix,
    )

    prob_hist_paths = _plot_prob_hist(
        y_true=res.y_true,
        y_prob=getattr(res, "y_prob", None),
        model_name=model_name,
        base_dir=base_dir,
        save=save,
        to_paper=to_paper,
        dpi=dpi,
        close=close,
        title_prefix=title_prefix,
    )

    coef_paths, coef_sig_paths = _plot_coef_summaries(
        coef_summary=res.coef_summary,
        model_name=model_name,
        base_dir=base_dir,
        save=save,
        to_paper=to_paper,
        dpi=dpi,
        close=close,
        figsize=coef_figsize,
        title_prefix=title_prefix,
    )

    trial_df = _load_or_build_full_trial_df(
        df=df,
        keep_cols=None,
    )

    corr_paths = _plot_feature_corr(
        trial_df=trial_df,
        corr_feature_cols=model.feature_cols_,
        base_dir=base_dir,
        save=save,
        to_paper=to_paper,
        dpi=dpi,
        close=close,
        title_prefix=title_prefix,
    )

    return {
        "results": results,
        "train_df": train_df,
        "test_df": test_df,
        "trial_df": trial_df,
        "split_tag": split_tag,
        "split_info": split_info,
        "base_subdir": base_dir,
        "summary_csv": summary_paths,
        "paths": {
            "confusion_norm": cm_paths,
            "confusion_unnorm": cm_paths2,
            "predicted_probability_hist": prob_hist_paths,
            "coef_all": coef_paths,
            "coef_significant": coef_sig_paths,
            "correlation": corr_paths,
        },
    }


def run_cross_dataset_correctness_bundle(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    feature_cols: Optional[Sequence[str]] = None,
    coef_ci_method: str = "wald",
    coef_ci_cluster: str = "row",
    save: Optional[bool] = None,
    to_paper=None,
    dpi: int = 300,
    close: bool = False,
    subdir: Optional[str] = None,
    split_tag: str = "trainL1_testnew",
    run_identifier: str = "",
    coef_figsize: Optional[Tuple[int, int]] = None,
    title_prefix: str = "",
) -> Dict[str, Any]:
    """
    Train a logistic-regression correctness model on the ENTIRE ``train_df``
    (no train/test split) and evaluate it on a separate ``test_df``.

    ``train_df`` and ``test_df`` are already-prepared trial-level feature tables
    (e.g. produced by ``save_model_ready`` / ``build_trial_level_model_df``).
    Intended for train-on-L1 / test-on-new-data: the model is fit on all of
    ``train_df`` and its accuracy/coefficients/confusion are reported on
    ``test_df``.

    ``feature_cols`` selects the features to use (default: the full feature set of
    ``train_df``). Any feature not present in ``test_df`` is dropped with a notice,
    since the model requires every feature column to exist in both frames (e.g.
    paragraph-region RT/TFD, which a no-paragraph experiment lacks).
    """
    model = TrialLevelLogRegModel()
    model_name = model.name
    model_family = "logreg"

    # Browsable folder under reports/correctness_prediction/; the filename still
    # carries every facet, so the folder is navigation, not identity.
    base_dir = "/".join(x for x in (split_tag or "full_fit", model_family, subdir) if x)

    # Resolve features from the training frame, then keep only those the test
    # frame also has (the model validates that every feature column is present).
    feat_cols = _resolve_feature_cols(train_df, feature_cols)
    missing_in_test = [c for c in feat_cols if c not in test_df.columns]
    if missing_in_test:
        print(
            f"[cross-dataset] dropping {len(missing_in_test)} feature(s) absent "
            f"from the test set: {missing_in_test}"
        )
        feat_cols = [c for c in feat_cols if c in test_df.columns]

    results = evaluate_models_on_prepared_split(
        models=[model],
        train_df=train_df,
        test_df=test_df,
        target_col=Con.IS_CORRECT_COLUMN,
        feature_cols=feat_cols,
        coef_kwargs_by_model={
            model_name: {
                "ci_method": coef_ci_method,
                "ci_cluster": coef_ci_cluster,
            }
        },
    )

    show_correctness_model_results(results)
    res = results[model_name]

    summary_paths = None
    if save:
        summary_paths = _save_summary_csv(
            results=results,
            model_name=model_name,
            trained_feature_cols=model.feature_cols_,
            base_dir=base_dir,
            run_identifier=run_identifier,
            to_paper=to_paper,
            formula=None,
        )

    cm_paths, cm_paths2 = _plot_confusions(
        y_true=res.y_true,
        y_pred=res.y_pred,
        model_name=model_name,
        base_dir=base_dir,
        save=save,
        to_paper=to_paper,
        close=close,
        title_prefix=title_prefix,
    )

    prob_hist_paths = _plot_prob_hist(
        y_true=res.y_true,
        y_prob=getattr(res, "y_prob", None),
        model_name=model_name,
        base_dir=base_dir,
        save=save,
        to_paper=to_paper,
        dpi=dpi,
        close=close,
        title_prefix=title_prefix,
    )

    coef_paths, coef_sig_paths = _plot_coef_summaries(
        coef_summary=res.coef_summary,
        model_name=model_name,
        base_dir=base_dir,
        save=save,
        to_paper=to_paper,
        dpi=dpi,
        close=close,
        figsize=coef_figsize,
        title_prefix=title_prefix,
    )

    # Correlation heatmap over the features the model was actually trained on,
    # computed on the training (L1) frame.
    corr_paths = _plot_feature_corr(
        trial_df=train_df,
        corr_feature_cols=model.feature_cols_,
        base_dir=base_dir,
        save=save,
        to_paper=to_paper,
        dpi=dpi,
        close=close,
        title_prefix=title_prefix,
    )

    return {
        "results": results,
        "train_df": train_df,
        "test_df": test_df,
        "feature_cols": list(model.feature_cols_),
        "split_tag": split_tag,
        "base_subdir": base_dir,
        "summary_csv": summary_paths,
        "paths": {
            "confusion_norm": cm_paths,
            "confusion_unnorm": cm_paths2,
            "predicted_probability_hist": prob_hist_paths,
            "coef_all": coef_paths,
            "coef_significant": coef_sig_paths,
            "correlation": corr_paths,
        },
    }
