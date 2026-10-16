"""The headline model on one train/test split: fit, score, save, and draw its figures.

`run_full_features_correctness_bundle` takes one stored fold (`modeling/folds.py`),
fits the logistic regression on the fold's training trials, scores it on the held-out
regimes asked for, prints the scores, and draws the model's figures: the confusion
matrix, the predicted probabilities, the coefficients and the correlations between
the features. `run_cross_dataset_correctness_bundle` does the same with a training
table and a separate test table, e.g. trained on Study 1 and tested on Study 2.

This is one split, not cross-validation. The accuracy the paper reports comes from
all ten folds (`modeling/crossval.py`), and its coefficients from one fit on all the
data (`modeling/evaluate.py::collect_logreg_coef_summaries`).

Everything is saved under `reports/correctness_prediction/{figures,tables}/`, in
`<held-out regimes>/logreg/<subdir>/`.

Two modules in `explorations/` import from this one: `feature_search/column_options.py`
and `mixed_models/run.py`.
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
    """The output folder's name: the held-out regimes joined by "+" ("new_item+both")."""
    return "+".join(test_regimes)


def _build_full_trial_df(
    df: pd.DataFrame,
    keep_cols: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Build the trial table from the word-level `df`, with `is_correct` as the target."""
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
    The trial table: the saved model-ready table (`load_model_ready`) if it exists,
    otherwise built from `df`.

    The saved table is used as it is, whatever `df` holds; rebuild it with
    `features/build.py::save_model_ready` when the data change. Any `keep_cols` it
    lacks are joined in from `df` by participant and trial.
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
    """
    Split `df` by one stored fold and return the training and test trial tables,
    plus the split's details.

    `df` is the word-level frame (`features/build.py::load_model_input`). The split is
    `modeling/folds.py::group_vise_train_test_split`, reading the fold files of the
    groups in `sources`: training is the fold's `train_train` trials, test is the
    trials of `test_regimes` in `test_split` ("test", "val" or "both"). The fold is
    `fold`, or else one drawn with `random_state` -- fold 0 with the default 42.

    If the saved model-ready table exists, both tables are cut from it by
    participant and trial. Its per-participant opening-scan features were worked
    out from all of a person's trials, so under `new_item`, where the same people
    are in training and test, the training rows carry information from the test
    trials (`pitfalls.md` §3). Without the saved table, each side's features are
    built from its own trials only.
    """
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
    """`feature_cols` if given, else every feature `train_df` has."""
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
    """Save the summary table as `model_summary`; always writes, so callers check `save`."""
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
    """The confusion matrix twice: as shares of each true outcome, and as counts."""
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
    """The histogram of predicted probabilities; nothing if there are none."""
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
    """Coefficient bars twice: every feature, and those whose interval excludes 0."""
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
    """Pearson correlations between `corr_feature_cols` over `trial_df`, cluster-ordered."""
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
    """
    Fit the logistic regression on one fold's training trials, score it on the
    `test_regimes` trials, and save the scores and figures.

    The split is made by `build_train_test_trial_dfs` (see there for which fold,
    and where the features come from). `feature_cols` defaults to every feature
    the training table has (`get_full_feature_cols`).

    The scores are printed. The figures: the confusion matrix (as shares and as
    counts), a histogram of the predicted probabilities on the test trials, the
    coefficients (all, and those whose interval excludes 0), and the correlations
    between the model's features over all trials. Only the summary table's file
    name carries `run_identifier`, so a second run with the same regimes and
    `subdir` overwrites the first one's figures.

    The summary table is written only when `save=True` is passed; the figures
    follow `save`, or the session default (`set_output_defaults`) when it is None.

    The coefficient intervals come from the fit on the training trials and are, by
    default, Wald intervals with each trial counted on its own
    (`coef_ci_method="wald"`, `coef_ci_cluster="row"`): quick, and not the paper's
    coefficients.

    `title_prefix` goes in front of every figure title. Returns a dict with the
    results, the training, test and full trial tables, the split's details, the
    output folder and the saved paths.
    """
    model = TrialLevelLogRegModel()
    model_name = model.name
    model_family = "logreg"

    split_tag = _split_tag(test_regimes)
    # The output folder under reports/correctness_prediction/. The held-out regimes
    # are named only here, not in the file names.
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
    Fit the logistic regression on all of `train_df` and score it on a separate
    `test_df` -- e.g. trained on Study 1 and tested on Study 2.

    Both are trial tables, one row per trial, as `features/build.py::save_model_ready`
    makes them. `feature_cols` defaults to every feature `train_df` has. Those that
    `test_df` lacks are dropped, with a printed note, because the model needs every
    feature in both tables (Study 2 has no paragraph screen, so no paragraph
    features).

    Scores, figures and saving as in `run_full_features_correctness_bundle`, in
    `<split_tag>/logreg/<subdir>/`; the feature correlations are over `train_df`.
    """
    model = TrialLevelLogRegModel()
    model_name = model.name
    model_family = "logreg"

    # The output folder under reports/correctness_prediction/. `split_tag` is named
    # only here, not in the file names.
    base_dir = "/".join(x for x in (split_tag or "full_fit", model_family, subdir) if x)

    # Keep only the features the test table also has: the model stops if one is
    # missing.
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

    # The correlations of the features the model used, over the training table.
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
