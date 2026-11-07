"""What only the mixed model has: random effects and their variances.

Everything the mixed model shares with the logistic regression runs through the same
code, with `model_builder` set to the mixed model: cross-validation
(`modeling/crossval.py`), one split and new data (`../run.py`), the model comparison
(`../comparison.py`), Study 2 (`../knowledge_regimes/`). This module adds:

* `run_glmm_fit_all` -- one fit on every trial, with the coefficients, the random
  intercept of every participant and item, and their spread, drawn and saved;
* `fit_report` -- one row describing a fitted model: its formula, weights, how the
  optimiser stopped, and the random-effect SDs;
* `variance_summary` -- the random-effect SDs of several fits side by side, with the
  share of the latent variance each grouping takes.

Outputs go under `reports/correctness_prediction/{figures,tables}/glmm/full_fit/`.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from src.config import columns as Con
from src.lib.plotting.output import save_output
from src.modeling.evaluate import fit_julia_mixed_model_on_prepared_full_data
from src.modeling.models.glmm_model import TrialLevelJuliaGLMERModel
from src.analyses.correctness_prediction.run import (
    _load_or_build_full_trial_df,
    _plot_coef_summaries,
    _resolve_feature_cols,
)
from src.analyses.correctness_prediction.mixed_model.plots import (
    plot_random_effects_barh,
    plot_random_effects_distribution,
    summarize_random_effects,
)


def fit_report(model: TrialLevelJuliaGLMERModel) -> Dict[str, Any]:
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


def variance_summary(models: Mapping[str, TrialLevelJuliaGLMERModel]) -> pd.DataFrame:
    """The random intercepts' SDs of several fitted models, one row per model.

    Beside each SD, that grouping's share of the variance on the latent (log-odds) scale:
    its variance over the sum of both random-intercept variances and pi^2 / 3, the fixed
    variance of the logistic distribution. This is the usual latent-scale intraclass
    correlation for a logistic model. Fits with class weights have enlarged variances, so
    read the shares from an unweighted fit.
    """
    rows = {}
    for name, model in models.items():
        sd = model.get_variance_components().set_index("group")["sd"]
        var_p = sd[model.participant_col] ** 2
        var_i = sd[model.text_col] ** 2
        latent = var_p + var_i + np.pi ** 2 / 3
        rows[name] = {
            "class_weight": str(model.class_weight),
            "sd_participant": sd[model.participant_col],
            "sd_item": sd[model.text_col],
            "share_participant": var_p / latent,
            "share_item": var_i / latent,
            "converged": model.converged_,
            "singular": model.is_singular_,
        }
    return pd.DataFrame(rows).T


def run_glmm_fit_all(
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

    `df` is the word-level frame, as for the other runs; the trial table is the saved
    model-ready one. `model` is an unfitted `TrialLevelJuliaGLMERModel` (the default one if
    not given).

    Returns the coefficient table (Wald intervals), the random intercept of every
    participant and item, the variance components, and a one-row summary of the fit
    (`fit_report`). Figures: the coefficients (all, and those whose interval excludes 0)
    and, for each grouping, a histogram of its random intercepts and the `top_n_rfx`
    largest in absolute size. The tables are written only with `save=True`.
    """
    model = model if model is not None else TrialLevelJuliaGLMERModel()
    model_name = f"{model.name}_fit_all"
    base_dir = "/".join(x for x in (model.family, "full_fit", subdir) if x)

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
        **fit_report(model),
        "n_features": len(model.feature_cols_),
        "trained_feature_cols": " | ".join(model.feature_cols_),
    }])

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
        level = "participant" if group == model.participant_col else "item"
        rfx_summary_frames.append(summarize_random_effects(table, group_name=group))

        _, _, dist_paths = plot_random_effects_distribution(
            table,
            effect_col="random_intercept",
            title=f"{level.capitalize()} random intercepts ({len(table)} {level}s)",
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
            title=f"The {top_n_rfx} largest {level} random intercepts",
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
