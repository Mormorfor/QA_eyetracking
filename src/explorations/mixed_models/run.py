"""Mixed-model runs: the paper's fit-score-save cycle, with the Julia GLMM. **Parked.**

`run_full_features_correctness_julia_glmer_bundle` is `analyses/correctness_prediction/run.py`'s
single-split bundle with the mixed model (`models.py`) in place of the logistic regression:
one stored fold, fit, score, print, figures. `run_full_features_correctness_julia_glmer_fit_all`
fits on every trial and also returns the random effects and their variances.

Both take an unfitted `TrialLevelJuliaGLMERModel`, already configured -- text level,
prediction rule, class weights -- or build the default one. The split, the summary table and
the figures come from the paper module and its `plots/`, so the outputs land beside the
logistic regression's, in `reports/correctness_prediction/{figures,tables}/<held-out regimes
or full_fit>/julia/`.

Cross-validation needs nothing from here: `modeling/crossval.py` takes the model as it takes
the logistic regression (`overview.ipynb` beside this file does that).
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple

import pandas as pd

from src.config import columns as Con
from src.modeling.evaluate import (
    evaluate_models_on_prepared_split,
    fit_julia_mixed_model_on_prepared_full_data,
)
from src.explorations.mixed_models.models import TrialLevelJuliaGLMERModel
from src.explorations.mixed_models.plots import (
    plot_random_effects_barh,
    plot_random_effects_distribution,
    summarize_random_effects,
)
from src.analyses.correctness_prediction.plots.results import show_correctness_model_results
from src.lib.plotting.output import save_output
from src.analyses.correctness_prediction.run import (
    _load_or_build_full_trial_df,
    _plot_coef_summaries,
    _plot_confusions,
    _plot_prob_hist,
    _resolve_feature_cols,
    _save_summary_csv,
    _split_tag,
    build_train_test_trial_dfs,
)

# The output folder under reports/correctness_prediction/ that keeps these apart from the
# logistic regression's ("logreg").
MODEL_FAMILY = "julia"


def _fit_report(model: TrialLevelJuliaGLMERModel) -> Dict[str, Any]:
    """One row describing a fitted model: formula, weights, how the fit ended, the SDs."""
    row: Dict[str, Any] = {
        "formula": model.get_formula(),
        "class_weight": str(model.class_weight),
        "predict_with": model.predict_with,
        "optimizer_returnvalue": model.optimizer_returnvalue_,
        "converged": model.converged_,
        "is_singular": model.is_singular_,
        "n_train": model.n_train_,
    }
    for _, vc in model.get_variance_components().iterrows():
        row[f"sd__{vc['group']}__{vc['term']}"] = vc["sd"]
    return row


def run_full_features_correctness_julia_glmer_bundle(
    df: pd.DataFrame,
    test_regimes: Sequence[str],
    *,
    model: Optional[TrialLevelJuliaGLMERModel] = None,
    test_split: str = "test",
    fold: Optional[int] = None,
    sources: Sequence[str] = ("hunters", "gatherers"),
    feature_cols: Optional[Sequence[str]] = None,
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
    Fit the mixed model on one fold's training trials, score it on the `test_regimes`
    trials, print the scores and draw the figures.

    The same split and the same figures as the paper's
    `run_full_features_correctness_bundle`: confusion matrix (shares and counts), the
    predicted probabilities on the test trials, and the coefficients (all, and those
    whose interval excludes 0). The intervals are the mixed model's own Wald intervals.
    The feature-correlation figure is left out: it describes the features, not the
    model, and the paper bundle draws it.

    `model` is an unfitted `TrialLevelJuliaGLMERModel`; its `predict_with` decides how
    the test trials are scored. The summary table is written only with `save=True`.
    """
    model = model if model is not None else TrialLevelJuliaGLMERModel()
    model_name = model.name

    split_tag = _split_tag(test_regimes)
    base_dir = "/".join(x for x in (split_tag or "full_fit", MODEL_FAMILY, subdir) if x)

    train_df, test_df, split_info = build_train_test_trial_dfs(
        df=df,
        test_regimes=test_regimes,
        test_split=test_split,
        fold=fold,
        sources=sources,
        keep_cols=[model.text_col],
        random_state=random_state,
    )

    feat_cols = _resolve_feature_cols(train_df, feature_cols)

    results = evaluate_models_on_prepared_split(
        models=[model],
        train_df=train_df,
        test_df=test_df,
        target_col=Con.IS_CORRECT_COLUMN,
        feature_cols=feat_cols,
    )

    show_correctness_model_results(results)
    res = results[model_name]
    fit_report = _fit_report(model)
    print(f"Formula: {fit_report['formula']}")
    print(
        f"Optimiser: {fit_report['optimizer_returnvalue']}   "
        f"singular: {fit_report['is_singular']}"
    )

    summary_paths = None
    if save:
        summary_paths = _save_summary_csv(
            results=results,
            model_name=model_name,
            trained_feature_cols=model.feature_cols_,
            base_dir=base_dir,
            run_identifier=run_identifier,
            to_paper=to_paper,
            formula=fit_report["formula"],
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
        y_prob=res.y_prob,
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

    return {
        "results": results,
        "model": model,
        "fit_report": fit_report,
        "train_df": train_df,
        "test_df": test_df,
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
        },
    }


def run_full_features_correctness_julia_glmer_fit_all(
    df: pd.DataFrame,
    *,
    model: Optional[TrialLevelJuliaGLMERModel] = None,
    feature_cols: Optional[Sequence[str]] = None,
    save: Optional[bool] = None,
    to_paper=None,
    dpi: int = 300,
    close: bool = False,
    subdir: Optional[str] = None,
    top_n_rfx: int = 30,
    run_identifier: str = "",
    coef_figsize: Optional[Tuple[int, int]] = None,
) -> Dict[str, Any]:
    """
    Fit the mixed model once on every trial, and draw its coefficients and random effects.

    Returns the coefficient table (Wald intervals), the random effects per participant and
    per text, the variance components, and a one-row summary of the fit (formula, weights,
    how the optimiser stopped, whether the fit is singular, the random-effect SDs). For
    each grouping factor: a histogram of its random intercepts and the `top_n_rfx` largest
    in absolute size. The tables are written only with `save=True`.
    """
    model = model if model is not None else TrialLevelJuliaGLMERModel()
    model_name = f"{model.name}_fit_all"
    base_dir = "/".join(x for x in ("full_fit", MODEL_FAMILY, subdir) if x)

    fit_df = _load_or_build_full_trial_df(df=df, keep_cols=[model.text_col])
    feat_cols = _resolve_feature_cols(fit_df, feature_cols)

    res = fit_julia_mixed_model_on_prepared_full_data(
        model=model,
        fit_df=fit_df,
        target_col=Con.IS_CORRECT_COLUMN,
        feature_cols=feat_cols,
    )

    variance_df = model.get_variance_components()
    summary_df = pd.DataFrame([{
        "run_identifier": run_identifier,
        "model_name": model_name,
        "n_rows": int(res["n_rows"]),
        "n_positive": int(res["n_positive"]),
        "n_negative": int(res["n_negative"]),
        **_fit_report(model),
        "n_features": len(model.feature_cols_),
        "trained_feature_cols": " | ".join(model.feature_cols_),
    }])
    print(f"Formula: {model.get_formula()}")
    print(res["random_effect_variance_summary"])

    summary_paths = None
    if save:
        summary_paths = save_output(
            None,
            analysis="correctness_prediction",
            plot="model_summary_fit_all",
            tables={"summary": summary_df, "variance_components": variance_df},
            save=save,
            to_paper=to_paper,
            subdir=base_dir,
            model=model_name,
        ).paths

    coef_paths, coef_sig_paths = _plot_coef_summaries(
        coef_summary=res["coef_summary"],
        model_name=model_name,
        base_dir=base_dir,
        save=save,
        to_paper=to_paper,
        dpi=dpi,
        close=close,
        figsize=coef_figsize,
    )

    random_effects = res["random_effects"]
    rfx_paths: Dict[str, Dict[str, Any]] = {}
    rfx_summary_frames = []
    for group, table in random_effects.items():
        level = "participant" if group == model.participant_col else "text"
        rfx_summary_frames.append(summarize_random_effects(table, group_name=group))

        _, _, dist_paths = plot_random_effects_distribution(
            table,
            effect_col="random_intercept",
            title=f"{model_name} – {level} random intercepts",
            save=save,
            subdir=f"{base_dir}/random_effects",
            plot="random_effects_distribution",
            model=model_name,
            level=level,
            to_paper=to_paper,
            dpi=dpi,
            close=close,
        )
        _, _, bar_paths = plot_random_effects_barh(
            table,
            id_col=group,
            effect_col="random_intercept",
            title=f"{model_name} – largest {level} random intercepts",
            top_n=top_n_rfx,
            save=save,
            subdir=f"{base_dir}/random_effects",
            plot="random_effects_barh",
            model=model_name,
            level=level,
            top=top_n_rfx,
            to_paper=to_paper,
            dpi=dpi,
            close=close,
        )
        rfx_paths[group] = {"distribution": dist_paths, "top_abs": bar_paths}

    random_effects_summary_df = pd.concat(rfx_summary_frames, ignore_index=True)

    rfx_summary_paths = None
    if save:
        rfx_summary_paths = save_output(
            None,
            analysis="correctness_prediction",
            plot="random_effects_summary",
            tables={"summary": random_effects_summary_df},
            save=save,
            to_paper=to_paper,
            subdir=f"{base_dir}/random_effects",
            model=model_name,
        ).paths

    return {
        "result": res,
        "model": model,
        "fit_df": fit_df,
        "base_subdir": base_dir,
        "summary_df": summary_df,
        "summary_csv": summary_paths,
        "variance_components": variance_df,
        "random_effects": random_effects,
        "random_effects_summary_df": random_effects_summary_df,
        "paths": {
            "coef_all": coef_paths,
            "coef_significant": coef_sig_paths,
            "random_effects": rfx_paths,
            "random_effects_summary": rfx_summary_paths,
        },
    }
