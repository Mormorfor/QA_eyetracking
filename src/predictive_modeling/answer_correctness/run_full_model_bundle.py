from __future__ import annotations

from typing import Sequence, Tuple, Optional, List, Dict, Any
import pandas as pd

from src import constants as Con

from src.predictive_modeling.answer_correctness.answer_correctness_data import (
    build_trial_level_all_features,
)
from predictive_modeling.answer_correctness.models.logreg_models import (
    FullFeaturesCorrectnessLogRegModel,
)

from predictive_modeling.answer_correctness.models.julia_model_old import (
    FullFeaturesCorrectnessJuliaGLMERModel,
)

from src.predictive_modeling.answer_correctness.answer_correctness_eval import (
    evaluate_models_on_answer_correctness,
    evaluate_glmer_on_answer_correctness,
    evaluate_julia_glmer_on_answer_correctness, fit_julia_glmer_on_answer_correctness_all
)
from src.predictive_modeling.common.viz_utils import (
    plot_confusion_heatmap,
)

from src.predictive_modeling.answer_correctness.answer_correctness_viz import (
    show_correctness_model_results,
    correctness_results_to_summary_df,
)
from src.viz.plot_output import save_df_csv, _answer_correctness_rel_dir

from predictive_modeling.answer_correctness.models.logreg_models import (
    FullFeaturesCorrectnessGLMERModel
)

from src.predictive_modeling.common.data_utils import (
    group_vise_train_test_split,
)

from predictive_modeling.answer_correctness.answer_correctness_viz import plot_coef_summary_barh, \
    plot_feature_correlation_heatmap, plot_random_effects_distribution, plot_random_effects_barh, summarize_random_effects


def _split_tag(split_group_cols: Sequence[str]) -> str:
    return "+".join(split_group_cols)



def run_full_features_correctness_bundle(
    df: pd.DataFrame,
    split_group_cols: Sequence[str],
    feature_cols: Optional[Sequence[str]] = None,
    group_cols: Sequence[str] = (Con.PARTICIPANT_ID, Con.TRIAL_ID),
    pref_specs: Optional[Sequence[Tuple[str, str]]] = Con.PREF_SPECS,
    pref_extreme_mode: str = "polarity",
    coef_ci_method: str = "wald",
    coef_ci_cluster: str = "row",
    save: bool = True,
    paper_dirs: Optional[List[str]] = None,
    dpi: int = 300,
    close: bool = False,
    subdir: Optional[str] = None,
    run_identifier: str = "",
) -> Dict[str, Any]:

    model = FullFeaturesCorrectnessLogRegModel()
    model_name = "full_features_correctness_log_reg"
    model_family = "logreg"

    split_tag = _split_tag(split_group_cols)
    base_dir = _answer_correctness_rel_dir(
        model_family=model_family,
        subdir=subdir,
        split_tag=split_tag,
    )

    feature_cols_by_model = (
        {model_name: list(feature_cols)}
        if feature_cols is not None
        else None
    )

    def builder_fn_local(d: pd.DataFrame, group_cols=group_cols):
        return build_trial_level_all_features(
            d,
            group_cols=group_cols,
            pref_specs=pref_specs,
            pref_extreme_mode=pref_extreme_mode,
        )

    results = evaluate_models_on_answer_correctness(
        df=df,
        models=[model],
        group_cols=tuple(group_cols),
        split_group_cols=list(split_group_cols),
        builder_fn=builder_fn_local,
        coef_ci_method=coef_ci_method,
        coef_ci_cluster=coef_ci_cluster,
        feature_cols_by_model=feature_cols_by_model,
    )

    show_correctness_model_results(results)

    summary_paths = None
    if save:
        trained_feature_cols_by_model = {
            model_name: list(getattr(model, "feature_cols_", []) or [])
        }

        summary_df = correctness_results_to_summary_df(
            results,
            run_identifier=run_identifier,
            trained_feature_cols_by_model=trained_feature_cols_by_model,
        )
        summary_paths = save_df_csv(
            summary_df,
            rel_dir=base_dir,
            filename="model_summary",
            paper_dirs=paper_dirs,
        )

    res = results[model_name]

    fig_cm, cm_df, cm_paths = plot_confusion_heatmap(
        y_true=res.y_true,
        y_pred=res.y_pred,
        labels=(0, 1),
        normalize=True,
        title=f"{model_name} – normalized confusion",
        save=save,
        rel_dir=f"{base_dir}/confusion",
        filename=f"{model_name}_norm_confusion",
        paper_dirs=paper_dirs,
        close=close,
    )

    fig_cm2, cm_df2, cm_paths2 = plot_confusion_heatmap(
        y_true=res.y_true,
        y_pred=res.y_pred,
        labels=(0, 1),
        normalize=False,
        title=f"{model_name} – un-normalized confusion",
        save=save,
        rel_dir=f"{base_dir}/confusion",
        filename=f"{model_name}_unnorm_confusion",
        paper_dirs=paper_dirs,
        close=close,
    )

    coef_paths = []
    coef_sig_paths = []

    if res.coef_summary is not None and not res.coef_summary.empty:
        fig_all, coef_df, coef_paths = plot_coef_summary_barh(
            coef_summary=res.coef_summary,
            value_col="coef",
            model_name=model_name,
            title=f"{model_name} – coefficients",
            save=save,
            rel_dir=f"{base_dir}/coefficients",
            filename=f"{model_name}_coef_all",
            paper_dirs=paper_dirs,
            dpi=dpi,
            close=close,
            significant_only=False,
        )

        fig_sig, coef_sig_df, coef_sig_paths = plot_coef_summary_barh(
            coef_summary=res.coef_summary,
            value_col="coef",
            model_name=model_name,
            title=f"{model_name} – significant coefficients",
            save=save,
            rel_dir=f"{base_dir}/coefficients",
            filename=f"{model_name}_coef_significant",
            paper_dirs=paper_dirs,
            dpi=dpi,
            close=close,
            significant_only=True,
        )

    trial_df = build_trial_level_all_features(
        df,
        group_cols=group_cols,
        pref_specs=pref_specs,
        pref_extreme_mode=pref_extreme_mode,
    )

    corr_feature_cols = list(model.feature_cols_)

    fig_corr, corr_df, corr_paths = plot_feature_correlation_heatmap(
        trial_df,
        feature_cols=list(corr_feature_cols),
        figsize=(30, 30),
        method="pearson",
        cluster_order=True,
        save=save,
        rel_dir=f"{base_dir}/diagnostics/feature_correlation",
        filename=f"feature_corr_clustered_n{len(corr_feature_cols)}",
        paper_dirs=paper_dirs,
        dpi=dpi,
        close=close,
    )

    return {
        "results": results,
        "paths": {
            "confusion_norm": cm_paths,
            "confusion_unnorm": cm_paths2,
            "coef_all": coef_paths,
            "coef_significant": coef_sig_paths,
            "correlation": corr_paths,
        },
        "trial_df": trial_df,
        "split_tag": split_tag,
        "base_rel_dir": base_dir,
        "summary_csv": summary_paths,
    }



def run_full_features_correctness_glmer_bundle(
    df: pd.DataFrame,
    split_group_cols: Sequence[str],
    feature_cols: Optional[Sequence[str]] = None,
    group_cols: Sequence[str] = (Con.PARTICIPANT_ID, Con.TRIAL_ID),
    pref_specs: Optional[Sequence[Tuple[str, str]]] = None,
    pref_extreme_mode: str = "polarity",
    save: bool = True,
    paper_dirs: Optional[List[str]] = None,
    dpi: int = 300,
    close: bool = False,
    subdir: Optional[str] = None,
    use_rfx: bool = False,
    run_identifier: str = "",
) -> Dict[str, Any]:

    model = FullFeaturesCorrectnessGLMERModel()
    model_name = model.name
    model_family = "glmer"

    split_tag = _split_tag(split_group_cols)
    base_dir = _answer_correctness_rel_dir(
        model_family=model_family,
        subdir=subdir,
        split_tag=split_tag,
    )

    def builder_fn_local(d: pd.DataFrame, group_cols=group_cols):
        return build_trial_level_all_features(
            d,
            group_cols=group_cols,
            pref_specs=pref_specs,
            pref_extreme_mode=pref_extreme_mode,
            keep_cols=[Con.TEXT_ID_WITH_Q_COLUMN],
        )

    results = evaluate_glmer_on_answer_correctness(
        df=df,
        model=model,
        group_cols=tuple(group_cols),
        split_group_cols=list(split_group_cols),
        builder_fn=builder_fn_local,
        split_fn=group_vise_train_test_split,
        target_col=Con.IS_CORRECT_COLUMN,
        feature_cols=feature_cols,
        participant_col=Con.PARTICIPANT_ID,
        text_col=Con.TEXT_ID_WITH_Q_COLUMN,
        use_rfx=use_rfx,
    )

    show_correctness_model_results(results)

    res = results[model_name]
    formula = model.get_formula() if hasattr(model, "get_formula") else None
    print(f"Model formula: {formula}")

    summary_paths = None
    if save:
        trained_feature_cols_by_model = {
            model_name: list(getattr(model, "raw_feature_cols_", []) or [])
        }

        summary_df = correctness_results_to_summary_df(
            results,
            run_identifier=run_identifier,
            trained_feature_cols_by_model=trained_feature_cols_by_model,
        )
        summary_df["formula"] = formula

        summary_paths = save_df_csv(
            summary_df,
            rel_dir=base_dir,
            filename="model_summary",
            paper_dirs=paper_dirs,
        )

    fig_cm, cm_df, cm_paths = plot_confusion_heatmap(
        y_true=res.y_true,
        y_pred=res.y_pred,
        labels=(0, 1),
        normalize=True,
        title=f"{model_name} – normalized confusion",
        save=save,
        rel_dir=f"{base_dir}/confusion",
        filename=f"{model_name}_norm_confusion",
        paper_dirs=paper_dirs,
        close=close,
    )

    fig_cm2, cm_df2, cm_paths2 = plot_confusion_heatmap(
        y_true=res.y_true,
        y_pred=res.y_pred,
        labels=(0, 1),
        normalize=False,
        title=f"{model_name} – un-normalized confusion",
        save=save,
        rel_dir=f"{base_dir}/confusion",
        filename=f"{model_name}_unnorm_confusion",
        paper_dirs=paper_dirs,
        close=close,
    )

    coef_paths = []
    coef_sig_paths = []

    coef_summary = res.coef_summary
    if coef_summary is not None and not coef_summary.empty:
        fig_all, coef_df, coef_paths = plot_coef_summary_barh(
            coef_summary=coef_summary,
            value_col="coef",
            model_name=model_name,
            title=f"{model_name} – coefficients",
            save=save,
            rel_dir=f"{base_dir}/coefficients",
            filename=f"{model_name}_coef_all",
            paper_dirs=paper_dirs,
            dpi=dpi,
            close=close,
            significant_only=False,
        )

        fig_sig, coef_sig_df, coef_sig_paths = plot_coef_summary_barh(
            coef_summary=coef_summary,
            value_col="coef",
            model_name=model_name,
            title=f"{model_name} – significant coefficients",
            save=save,
            rel_dir=f"{base_dir}/coefficients",
            filename=f"{model_name}_coef_significant",
            paper_dirs=paper_dirs,
            dpi=dpi,
            close=close,
            significant_only=True,
        )

    trial_df = build_trial_level_all_features(
        df,
        group_cols=group_cols,
        pref_specs=pref_specs,
        pref_extreme_mode=pref_extreme_mode,
        keep_cols=[Con.TEXT_ID_WITH_Q_COLUMN],
    )

    corr_feature_cols = list(model.raw_feature_cols_)

    fig_corr, corr_df, corr_paths = plot_feature_correlation_heatmap(
        trial_df,
        feature_cols=list(corr_feature_cols),
        figsize=(30, 30),
        method="pearson",
        cluster_order=True,
        save=save,
        rel_dir=f"{base_dir}/diagnostics/feature_correlation",
        filename=f"feature_corr_clustered_n{len(corr_feature_cols)}",
        paper_dirs=paper_dirs,
        dpi=dpi,
        close=close,
    )

    return {
        "results": results,
        "paths": {
            "confusion_norm": cm_paths,
            "confusion_unnorm": cm_paths2,
            "coef_all": coef_paths,
            "coef_significant": coef_sig_paths,
            "correlation": corr_paths,
        },
        "trial_df": trial_df,
        "split_tag": split_tag,
        "base_rel_dir": base_dir,
        "summary_csv": summary_paths,
        "formula": formula,
    }



def run_full_features_correctness_julia_glmer_bundle(
    df: pd.DataFrame,
    split_group_cols: Sequence[str],
    feature_cols: Optional[Sequence[str]] = None,
    group_cols: Sequence[str] = (Con.PARTICIPANT_ID, Con.TRIAL_ID),
    pref_specs: Optional[Sequence[Tuple[str, str]]] = None,
    pref_extreme_mode: str = "polarity",
    save: bool = True,
    paper_dirs: Optional[List[str]] = None,
    dpi: int = 300,
    close: bool = False,
    subdir: Optional[str] = None,
    use_rfx: bool = False,
    run_identifier: str = "",
) -> Dict[str, Any]:

    model = FullFeaturesCorrectnessJuliaGLMERModel()
    model_name = model.name
    model_family = "julia"

    split_tag = _split_tag(split_group_cols)
    base_dir = _answer_correctness_rel_dir(
        model_family=model_family,
        subdir=subdir,
        split_tag=split_tag,
    )

    def builder_fn_local(d: pd.DataFrame, group_cols=group_cols):
        return build_trial_level_all_features(
            d,
            group_cols=group_cols,
            pref_specs=pref_specs,
            pref_extreme_mode=pref_extreme_mode,
            keep_cols=[Con.TEXT_ID_WITH_Q_COLUMN],
        )

    results = evaluate_julia_glmer_on_answer_correctness(
        df=df,
        model=model,
        group_cols=tuple(group_cols),
        split_group_cols=list(split_group_cols),
        builder_fn=builder_fn_local,
        split_fn=group_vise_train_test_split,
        target_col=Con.IS_CORRECT_COLUMN,
        feature_cols=feature_cols,
        participant_col=Con.PARTICIPANT_ID,
        text_col=Con.TEXT_ID_WITH_Q_COLUMN,
        use_rfx=use_rfx,
    )

    show_correctness_model_results(results)

    res = results[model_name]
    formula = model.get_formula() if hasattr(model, "get_formula") else None
    print(f"Model formula: {formula}")

    summary_paths = None
    if save:
        trained_feature_cols_by_model = {
            model_name: list(getattr(model, "raw_feature_cols_", []) or [])
        }

        summary_df = correctness_results_to_summary_df(
            results,
            run_identifier=run_identifier,
            trained_feature_cols_by_model=trained_feature_cols_by_model,
        )
        summary_df["formula"] = formula

        summary_paths = save_df_csv(
            summary_df,
            rel_dir=base_dir,
            filename="model_summary",
            paper_dirs=paper_dirs,
        )

    fig_cm, cm_df, cm_paths = plot_confusion_heatmap(
        y_true=res.y_true,
        y_pred=res.y_pred,
        labels=(0, 1),
        normalize=True,
        title=f"{model_name} – normalized confusion",
        save=save,
        rel_dir=f"{base_dir}/confusion",
        filename=f"{model_name}_norm_confusion",
        paper_dirs=paper_dirs,
        close=close,
    )

    fig_cm2, cm_df2, cm_paths2 = plot_confusion_heatmap(
        y_true=res.y_true,
        y_pred=res.y_pred,
        labels=(0, 1),
        normalize=False,
        title=f"{model_name} – un-normalized confusion",
        save=save,
        rel_dir=f"{base_dir}/confusion",
        filename=f"{model_name}_unnorm_confusion",
        paper_dirs=paper_dirs,
        close=close,
    )

    coef_paths = []
    coef_sig_paths = []

    coef_summary = res.coef_summary
    if coef_summary is not None and not coef_summary.empty:
        fig_all, coef_df, coef_paths = plot_coef_summary_barh(
            coef_summary=coef_summary,
            value_col="coef",
            model_name=model_name,
            title=f"{model_name} – coefficients",
            save=save,
            rel_dir=f"{base_dir}/coefficients",
            filename=f"{model_name}_coef_all",
            paper_dirs=paper_dirs,
            dpi=dpi,
            close=close,
            significant_only=False,
        )

        fig_sig, coef_sig_df, coef_sig_paths = plot_coef_summary_barh(
            coef_summary=coef_summary,
            value_col="coef",
            model_name=model_name,
            title=f"{model_name} – significant coefficients",
            save=save,
            rel_dir=f"{base_dir}/coefficients",
            filename=f"{model_name}_coef_significant",
            paper_dirs=paper_dirs,
            dpi=dpi,
            close=close,
            significant_only=True,
        )

    trial_df = build_trial_level_all_features(
        df,
        group_cols=group_cols,
        pref_specs=pref_specs,
        pref_extreme_mode=pref_extreme_mode,
        keep_cols=[Con.TEXT_ID_WITH_Q_COLUMN],
    )

    corr_feature_cols = list(model.raw_feature_cols_)

    fig_corr, corr_df, corr_paths = plot_feature_correlation_heatmap(
        trial_df,
        feature_cols=list(corr_feature_cols),
        figsize=(30, 30),
        method="pearson",
        cluster_order=True,
        save=save,
        rel_dir=f"{base_dir}/diagnostics/feature_correlation",
        filename=f"feature_corr_clustered_n{len(corr_feature_cols)}",
        paper_dirs=paper_dirs,
        dpi=dpi,
        close=close,
    )

    return {
        "results": results,
        "paths": {
            "confusion_norm": cm_paths,
            "confusion_unnorm": cm_paths2,
            "coef_all": coef_paths,
            "coef_significant": coef_sig_paths,
            "correlation": corr_paths,
        },
        "trial_df": trial_df,
        "split_tag": split_tag,
        "base_rel_dir": base_dir,
        "summary_csv": summary_paths,
        "formula": formula,
    }



def run_full_features_correctness_julia_glmer_fit_all(
    df: pd.DataFrame,
    feature_cols: Optional[Sequence[str]] = None,
    group_cols: Sequence[str] = (Con.PARTICIPANT_ID, Con.TRIAL_ID),
    pref_specs: Optional[Sequence[Tuple[str, str]]] = None,
    pref_extreme_mode: str = "polarity",
    save: bool = True,
    paper_dirs: Optional[List[str]] = None,
    dpi: int = 300,
    close: bool = False,
    subdir: Optional[str] = None,
    top_n_rfx: int = 30,
    run_identifier: str = "",
) -> Dict[str, Any]:

    model = FullFeaturesCorrectnessJuliaGLMERModel()
    model_name = f"{model.name}_fit_all"

    base_dir = _answer_correctness_rel_dir(
        model_family="julia",
        subdir=subdir,
        fit_all=True,
    )

    def builder_fn_local(d: pd.DataFrame, group_cols=group_cols):
        return build_trial_level_all_features(
            d,
            group_cols=group_cols,
            pref_specs=pref_specs,
            pref_extreme_mode=pref_extreme_mode,
            keep_cols=[Con.TEXT_ID_WITH_Q_COLUMN],
        )

    results = fit_julia_glmer_on_answer_correctness_all(
        df=df,
        model=model,
        group_cols=tuple(group_cols),
        builder_fn=builder_fn_local,
        target_col=Con.IS_CORRECT_COLUMN,
        feature_cols=feature_cols,
        participant_col=Con.PARTICIPANT_ID,
        text_col=Con.TEXT_ID_WITH_Q_COLUMN,
    )

    res = results[model.name]
    formula = model.get_formula() if hasattr(model, "get_formula") else None
    print(f"Model formula: {formula}")

    trained_feature_cols = list(getattr(model, "raw_feature_cols_", []) or [])

    summary_df = pd.DataFrame([{
        "run_identifier": run_identifier,
        "model_name": model_name,
        "n_rows": int(res["n_rows"]),
        "n_positive": int(res["n_positive"]),
        "n_negative": int(res["n_negative"]),
        "formula": formula,
        "random_effect_variance_summary": res["random_effect_variance_summary"],
        "n_features": len(trained_feature_cols),
        "trained_feature_cols": " | ".join(trained_feature_cols),
    }])

    summary_paths = None
    if save:
        summary_paths = save_df_csv(
            summary_df,
            rel_dir=base_dir,
            filename="model_summary_fit_all",
            paper_dirs=paper_dirs,
        )

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
            rel_dir=f"{base_dir}/coefficients",
            filename=f"{model_name}_coef_all",
            paper_dirs=paper_dirs,
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
            rel_dir=f"{base_dir}/coefficients",
            filename=f"{model_name}_coef_significant",
            paper_dirs=paper_dirs,
            dpi=dpi,
            close=close,
            significant_only=True,
        )

    fit_df = res["fit_df"]
    corr_feature_cols = list(model.raw_feature_cols_)

    _, _, corr_paths = plot_feature_correlation_heatmap(
        fit_df,
        feature_cols=corr_feature_cols,
        figsize=(30, 30),
        method="pearson",
        cluster_order=True,
        save=save,
        rel_dir=f"{base_dir}/diagnostics/feature_correlation",
        filename=f"feature_corr_clustered_n{len(corr_feature_cols)}",
        paper_dirs=paper_dirs,
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
            rel_dir=f"{base_dir}/random_effects",
            filename=f"{model_name}_participant_distribution",
            paper_dirs=paper_dirs,
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
            rel_dir=f"{base_dir}/random_effects",
            filename=f"{model_name}_participant_top{top_n_rfx}",
            paper_dirs=paper_dirs,
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
            rel_dir=f"{base_dir}/random_effects",
            filename=f"{model_name}_text_distribution",
            paper_dirs=paper_dirs,
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
            rel_dir=f"{base_dir}/random_effects",
            filename=f"{model_name}_text_top{top_n_rfx}",
            paper_dirs=paper_dirs,
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
        rfx_summary_paths = save_df_csv(
            random_effects_summary_df,
            rel_dir=f"{base_dir}/random_effects",
            filename="random_effects_summary",
            paper_dirs=paper_dirs,
        )

    return {
        "results": results,
        "paths": {
            "coef_all": coef_paths,
            "coef_significant": coef_sig_paths,
            "correlation": corr_paths,
            "random_effects": rfx_paths,
            "random_effects_summary": rfx_summary_paths,
        },
        "fit_df": fit_df,
        "base_rel_dir": base_dir,
        "summary_csv": summary_paths,
        "formula": formula,
        "random_effects": random_effects,
        "random_effects_summary_df": random_effects_summary_df,
    }
