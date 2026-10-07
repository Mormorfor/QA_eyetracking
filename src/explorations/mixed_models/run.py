"""Mixed-effects bundles: the same evaluate-and-save cycle, Julia/R backend. **Parked.**

Split out of `run_model_bundles.py` in stage E. They share every helper with the logistic
bundles, which is why those moved down to `modeling/bundles.py` rather than being
duplicated here -- leaving these in `analyses/` had made `analyses/` import `explorations/`.
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
from src.explorations.mixed_models.models import (
    TrialLevelJuliaGLMERModel,
)
from src.explorations.mixed_models.plots import (
    plot_random_effects_barh,
    plot_random_effects_distribution,
    summarize_random_effects,
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
from src.analyses.correctness_prediction.run import (
    _load_or_build_full_trial_df,
    _plot_coef_summaries,
    _plot_confusions,
    _plot_feature_corr,
    _resolve_feature_cols,
    _save_summary_csv,
    _split_tag,
    build_train_test_trial_dfs,
)


def run_full_features_correctness_julia_glmer_bundle(
    df: pd.DataFrame,
    test_regimes: Sequence[str],
    *,
    test_split: str = "test",
    fold: Optional[int] = None,
    sources: Sequence[str] = ("hunters", "gatherers"),
    feature_cols: Optional[Sequence[str]] = None,
    save: Optional[bool] = None,
    to_paper=None,
    dpi: int = 300,
    close: bool = False,
    subdir: Optional[str] = None,
    use_rfx: bool = False,
    run_identifier: str = "",
    random_state: int = 42,
    participant_effects_mode: str = "slopes",
    text_effects_mode: str = "slopes",
) -> Dict[str, Any]:
    model = TrialLevelJuliaGLMERModel()
    model.participant_effects_mode = participant_effects_mode
    model.text_effects_mode = text_effects_mode

    model_name = model.name
    model_family = "julia"

    split_tag = _split_tag(test_regimes)
    # Browsable folder under reports/correctness_prediction/; the filename still
    # carries every facet, so the folder is navigation, not identity.
    base_dir = "/".join(x for x in (split_tag or "full_fit", model_family, subdir) if x)

    keep_cols = [Con.TEXT_ID_WITH_Q_COLUMN]

    train_df, test_df, split_info = build_train_test_trial_dfs(
        df=df,
        test_regimes=test_regimes,
        test_split=test_split,
        fold=fold,
        sources=sources,
        keep_cols=keep_cols,
        random_state=random_state,
    )

    feat_cols = _resolve_feature_cols(train_df, feature_cols)

    results = evaluate_models_on_prepared_split(
        models=[model],
        train_df=train_df,
        test_df=test_df,
        target_col=Con.IS_CORRECT_COLUMN,
        feature_cols=feat_cols,
        fit_kwargs_by_model={
            model_name: {
                "participant_col": Con.PARTICIPANT_ID,
                "text_col": Con.TEXT_ID_WITH_Q_COLUMN,
            }
        },
        predict_kwargs_by_model={
            model_name: {
                "target_col": Con.IS_CORRECT_COLUMN,
                "participant_col": Con.PARTICIPANT_ID,
                "text_col": Con.TEXT_ID_WITH_Q_COLUMN,
                "use_rfx": use_rfx,
            }
        },
        predict_proba_kwargs_by_model={
            model_name: {
                "target_col": Con.IS_CORRECT_COLUMN,
                "participant_col": Con.PARTICIPANT_ID,
                "text_col": Con.TEXT_ID_WITH_Q_COLUMN,
                "use_rfx": use_rfx,
            }
        },
    )

    show_correctness_model_results(results)
    res = results[model_name]

    formula = model.get_formula()
    print(f"Model formula: {formula}")

    summary_paths = None
    if save:
        summary_paths = _save_summary_csv(
            results=results,
            model_name=model_name,
            trained_feature_cols=model.feature_cols_raw_,
            base_dir=base_dir,
            run_identifier=run_identifier,
            to_paper=to_paper,
            formula=formula,
        )

    cm_paths, cm_paths2 = _plot_confusions(
        y_true=res.y_true,
        y_pred=res.y_pred,
        model_name=model_name,
        base_dir=base_dir,
        save=save,
        to_paper=to_paper,
        close=close,
    )

    coef_paths, coef_sig_paths = _plot_coef_summaries(
        coef_summary=res.coef_summary,
        model_name=model_name,
        base_dir=base_dir,
        save=save,
        to_paper=to_paper,
        dpi=dpi,
        close=close,
    )

    trial_df = _load_or_build_full_trial_df(
        df=df,
        keep_cols=keep_cols,
    )

    corr_paths = _plot_feature_corr(
        trial_df=trial_df,
        corr_feature_cols=model.feature_cols_raw_,
        base_dir=base_dir,
        save=save,
        to_paper=to_paper,
        dpi=dpi,
        close=close,
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
        "formula": formula,
        "paths": {
            "confusion_norm": cm_paths,
            "confusion_unnorm": cm_paths2,
            "coef_all": coef_paths,
            "coef_significant": coef_sig_paths,
            "correlation": corr_paths,
        },
    }


def run_full_features_correctness_julia_glmer_fit_all(
    df: pd.DataFrame,
    feature_cols: Optional[Sequence[str]] = None,
    save: Optional[bool] = None,
    to_paper=None,
    dpi: int = 300,
    close: bool = False,
    subdir: Optional[str] = None,
    top_n_rfx: int = 30,
    run_identifier: str = "",
    participant_effects_mode: str = "slopes",
    text_effects_mode: str = "slopes",
) -> Dict[str, Any]:
    model = TrialLevelJuliaGLMERModel()
    model.participant_effects_mode = participant_effects_mode
    model.text_effects_mode = text_effects_mode

    model_name = f"{model.name}_fit_all"

    base_dir = "/".join(x for x in ("full_fit", "julia", subdir) if x)

    fit_df = _load_or_build_full_trial_df(
        df=df,
        keep_cols=[Con.TEXT_ID_WITH_Q_COLUMN],
    )

    feat_cols = _resolve_feature_cols(fit_df, feature_cols)

    res = fit_julia_mixed_model_on_prepared_full_data(
        model=model,
        fit_df=fit_df,
        target_col=Con.IS_CORRECT_COLUMN,
        feature_cols=feat_cols,
        fit_kwargs={
            "participant_col": Con.PARTICIPANT_ID,
            "text_col": Con.TEXT_ID_WITH_Q_COLUMN,
        },
    )

    formula = model.get_formula()
    print(f"Model formula: {formula}")

    trained_feature_cols = list(model.feature_cols_raw_)

    summary_df = pd.DataFrame([{
        "run_identifier": run_identifier,
        "model_name": model_name,
        "n_rows": int(res["n_rows"]),
        "n_positive": int(res["n_positive"]),
        "n_negative": int(res["n_negative"]),
        "formula": formula,
        "participant_effects_mode": participant_effects_mode,
        "text_effects_mode": text_effects_mode,
        "random_effect_variance_summary": res["random_effect_variance_summary"],
        "n_features": len(trained_feature_cols),
        "trained_feature_cols": " | ".join(trained_feature_cols),
    }])

    summary_paths = None
    if save:
        summary_paths = save_output(
            None,
            analysis="correctness_prediction",
            plot="model_summary_fit_all",
            tables={"summary": summary_df},
            save=save,
            to_paper=to_paper,
            subdir=base_dir,
            model=model_name,
        ).paths

    coef_paths = []
    coef_sig_paths = []

    coef_summary = res["coef_summary"]
    if coef_summary is not None and not coef_summary.empty:
        _, _, coef_paths = plot_coef_summary_barh(
            coef_summary=coef_summary,
            value_col="coef",
            model_name=model_name,
            title=f"{model_name} – coefficients",
            save=save,
            subdir=f"{base_dir}/coefficients",
            plot="coefficients",
            model=model_name,
            coefs="all",
            to_paper=to_paper,
            dpi=dpi,
            close=close,
            significant_only=False,
        )

        _, _, coef_sig_paths = plot_coef_summary_barh(
            coef_summary=coef_summary,
            value_col="coef",
            model_name=model_name,
            title=f"{model_name} – significant coefficients",
            save=save,
            subdir=f"{base_dir}/coefficients",
            plot="coefficients",
            model=model_name,
            coefs="significant",
            to_paper=to_paper,
            dpi=dpi,
            close=close,
            significant_only=True,
        )

    corr_paths = _plot_feature_corr(
        trial_df=fit_df,
        corr_feature_cols=model.feature_cols_raw_,
        base_dir=base_dir,
        save=save,
        to_paper=to_paper,
        dpi=dpi,
        close=close,
    )

    random_effects = res["random_effects"] or {}
    rfx_paths = {}
    rfx_summary_frames = []

    if Con.PARTICIPANT_ID in random_effects:
        part_df = random_effects[Con.PARTICIPANT_ID]
        rfx_summary_frames.append(
            summarize_random_effects(part_df, group_name=Con.PARTICIPANT_ID)
        )

        _, _, part_dist_paths = plot_random_effects_distribution(
            part_df,
            effect_col="random_intercept",
            title=f"{model_name} – participant random-effects distribution",
            save=save,
            subdir=f"{base_dir}/random_effects",
            plot="random_effects_distribution",
            model=model_name,
            level="participant",
            to_paper=to_paper,
            dpi=dpi,
            close=close,
        )

        _, _, part_bar_paths = plot_random_effects_barh(
            part_df,
            id_col=Con.PARTICIPANT_ID,
            effect_col="random_intercept",
            title=f"{model_name} – strongest participant random effects",
            top_n=top_n_rfx,
            save=save,
            subdir=f"{base_dir}/random_effects",
            plot="random_effects_barh",
            model=model_name,
            level="participant",
            top=top_n_rfx,
            to_paper=to_paper,
            dpi=dpi,
            close=close,
        )

        rfx_paths[Con.PARTICIPANT_ID] = {
            "distribution": part_dist_paths,
            "top_abs": part_bar_paths,
        }

    if Con.TEXT_ID_WITH_Q_COLUMN in random_effects:
        text_df = random_effects[Con.TEXT_ID_WITH_Q_COLUMN]
        rfx_summary_frames.append(
            summarize_random_effects(text_df, group_name=Con.TEXT_ID_WITH_Q_COLUMN)
        )

        _, _, text_dist_paths = plot_random_effects_distribution(
            text_df,
            effect_col="random_intercept",
            title=f"{model_name} – text random-effects distribution",
            save=save,
            subdir=f"{base_dir}/random_effects",
            plot="random_effects_distribution",
            model=model_name,
            level="text",
            to_paper=to_paper,
            dpi=dpi,
            close=close,
        )

        _, _, text_bar_paths = plot_random_effects_barh(
            text_df,
            id_col=Con.TEXT_ID_WITH_Q_COLUMN,
            effect_col="random_intercept",
            title=f"{model_name} – strongest text random effects",
            top_n=top_n_rfx,
            save=save,
            subdir=f"{base_dir}/random_effects",
            plot="random_effects_barh",
            model=model_name,
            level="text",
            top=top_n_rfx,
            to_paper=to_paper,
            dpi=dpi,
            close=close,
        )

        rfx_paths[Con.TEXT_ID_WITH_Q_COLUMN] = {
            "distribution": text_dist_paths,
            "top_abs": text_bar_paths,
        }

    random_effects_summary_df = (
        pd.concat(rfx_summary_frames, ignore_index=True)
        if rfx_summary_frames else pd.DataFrame()
    )

    rfx_summary_paths = None
    if save and not random_effects_summary_df.empty:
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
        "fit_df": fit_df,
        "base_subdir": base_dir,
        "summary_csv": summary_paths,
        "formula": formula,
        "random_effects": random_effects,
        "random_effects_summary_df": random_effects_summary_df,
        "paths": {
            "coef_all": coef_paths,
            "coef_significant": coef_sig_paths,
            "correlation": corr_paths,
            "random_effects": rfx_paths,
            "random_effects_summary": rfx_summary_paths,
        },
    }
