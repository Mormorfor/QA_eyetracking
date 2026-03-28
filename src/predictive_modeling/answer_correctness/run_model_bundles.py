from __future__ import annotations

from typing import Sequence, Tuple, Optional, List, Dict, Any

import pandas as pd

from src import constants as Con

from src.predictive_modeling.common.data_utils import group_vise_train_test_split
from src.predictive_modeling.common.feature_specs import get_full_feature_cols

from src.predictive_modeling.answer_correctness.model_data import (
    build_trial_level_model_df,
)
from src.predictive_modeling.answer_correctness.evaluation_core import (
    evaluate_models_on_prepared_split, fit_model_on_prepared_full_data,
)

from src.predictive_modeling.answer_correctness.models.logreg_model import (
    TrialLevelLogRegModel,
)

# from src.predictive_modeling.answer_correctness.models.glmer_r_model import (
#     TrialLevelGLMERModel,
# )

from src.predictive_modeling.answer_correctness.models.julia_model import (
    TrialLevelJuliaGLMERModel,
)

from src.predictive_modeling.answer_correctness.answer_correctness_viz import (
    show_correctness_model_results,
    correctness_results_to_summary_df,
    plot_coef_summary_barh,
    plot_feature_correlation_heatmap,
    plot_random_effects_distribution,
    summarize_random_effects, plot_random_effects_barh,
)

from src.predictive_modeling.common.viz_utils import plot_confusion_heatmap
from src.viz.plot_output import save_df_csv, _answer_correctness_rel_dir



def _split_tag(split_group_cols: Sequence[str]) -> str:
    return "+".join(split_group_cols)


def build_train_test_trial_dfs(
    df: pd.DataFrame,
    split_group_cols: Sequence[str],
    pref_specs: Optional[Sequence[Tuple[str, str]]] = Con.PREF_SPECS,
    pref_extreme_mode: str = "polarity",
    keep_cols: Optional[Sequence[str]] = None,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_raw, test_raw = group_vise_train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        group_cols=list(split_group_cols),
    )

    train_df = build_trial_level_model_df(
        df=train_raw,
        pref_specs=pref_specs,
        pref_extreme_mode=pref_extreme_mode,
        keep_cols=keep_cols,
        target_col=Con.IS_CORRECT_COLUMN,
        include_area_features=True,
        include_derived_features=True,
        include_last_visited_features=True,
    )

    test_df = build_trial_level_model_df(
        df=test_raw,
        pref_specs=pref_specs,
        pref_extreme_mode=pref_extreme_mode,
        keep_cols=keep_cols,
        target_col=Con.IS_CORRECT_COLUMN,
        include_area_features=True,
        include_derived_features=True,
        include_last_visited_features=True,
    )

    return train_df, test_df


def _build_full_trial_df(
    df: pd.DataFrame,
    pref_specs: Optional[Sequence[Tuple[str, str]]] = Con.PREF_SPECS,
    pref_extreme_mode: str = "polarity",
    keep_cols: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    return build_trial_level_model_df(
        df=df,
        pref_specs=pref_specs,
        pref_extreme_mode=pref_extreme_mode,
        keep_cols=keep_cols,
        target_col=Con.IS_CORRECT_COLUMN,
        include_area_features=True,
        include_derived_features=True,
        include_last_visited_features=True,
    )


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
    paper_dirs: Optional[List[str]],
    formula: Optional[str] = None,
):
    summary_df = correctness_results_to_summary_df(
        results,
        run_identifier=run_identifier,
        trained_feature_cols_by_model={model_name: list(trained_feature_cols)},
    )

    if formula is not None:
        summary_df["formula"] = formula

    return save_df_csv(
        summary_df,
        rel_dir=base_dir,
        filename="model_summary",
        paper_dirs=paper_dirs,
    )


def _plot_confusions(
    *,
    y_true,
    y_pred,
    model_name: str,
    base_dir: str,
    save: bool,
    paper_dirs: Optional[List[str]],
    close: bool,
):
    _, _, cm_paths = plot_confusion_heatmap(
        y_true=y_true,
        y_pred=y_pred,
        labels=(0, 1),
        normalize=True,
        title=f"{model_name} – normalized confusion",
        save=save,
        rel_dir=f"{base_dir}/confusion",
        filename=f"{model_name}_norm_confusion",
        paper_dirs=paper_dirs,
        close=close,
    )

    _, _, cm_paths2 = plot_confusion_heatmap(
        y_true=y_true,
        y_pred=y_pred,
        labels=(0, 1),
        normalize=False,
        title=f"{model_name} – un-normalized confusion",
        save=save,
        rel_dir=f"{base_dir}/confusion",
        filename=f"{model_name}_unnorm_confusion",
        paper_dirs=paper_dirs,
        close=close,
    )

    return cm_paths, cm_paths2


def _plot_coef_summaries(
    *,
    coef_summary: pd.DataFrame,
    model_name: str,
    base_dir: str,
    save: bool,
    paper_dirs: Optional[List[str]],
    dpi: int,
    close: bool,
):
    coef_paths = []
    coef_sig_paths = []

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

    return coef_paths, coef_sig_paths


def _plot_feature_corr(
    *,
    trial_df: pd.DataFrame,
    corr_feature_cols: Sequence[str],
    base_dir: str,
    save: bool,
    paper_dirs: Optional[List[str]],
    dpi: int,
    close: bool,
):
    _, _, corr_paths = plot_feature_correlation_heatmap(
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
    return corr_paths


def run_full_features_correctness_bundle(
    df: pd.DataFrame,
    split_group_cols: Sequence[str],
    feature_cols: Optional[Sequence[str]] = None,
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
    test_size: float = 0.2,
    random_state: int = 42,
) -> Dict[str, Any]:
    model = TrialLevelLogRegModel()
    model_name = model.name
    model_family = "logreg"

    split_tag = _split_tag(split_group_cols)
    base_dir = _answer_correctness_rel_dir(
        model_family=model_family,
        subdir=subdir,
        split_tag=split_tag,
    )

    train_df, test_df = build_train_test_trial_dfs(
        df=df,
        split_group_cols=split_group_cols,
        pref_specs=pref_specs,
        pref_extreme_mode=pref_extreme_mode,
        keep_cols=None,
        test_size=test_size,
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
            paper_dirs=paper_dirs,
            formula=None,
        )

    cm_paths, cm_paths2 = _plot_confusions(
        y_true=res.y_true,
        y_pred=res.y_pred,
        model_name=model_name,
        base_dir=base_dir,
        save=save,
        paper_dirs=paper_dirs,
        close=close,
    )

    coef_paths, coef_sig_paths = _plot_coef_summaries(
        coef_summary=res.coef_summary,
        model_name=model_name,
        base_dir=base_dir,
        save=save,
        paper_dirs=paper_dirs,
        dpi=dpi,
        close=close,
    )

    trial_df = _build_full_trial_df(
        df=df,
        pref_specs=pref_specs,
        pref_extreme_mode=pref_extreme_mode,
        keep_cols=None,
    )

    corr_paths = _plot_feature_corr(
        trial_df=trial_df,
        corr_feature_cols=model.feature_cols_,
        base_dir=base_dir,
        save=save,
        paper_dirs=paper_dirs,
        dpi=dpi,
        close=close,
    )

    return {
        "results": results,
        "train_df": train_df,
        "test_df": test_df,
        "trial_df": trial_df,
        "split_tag": split_tag,
        "base_rel_dir": base_dir,
        "summary_csv": summary_paths,
        "paths": {
            "confusion_norm": cm_paths,
            "confusion_unnorm": cm_paths2,
            "coef_all": coef_paths,
            "coef_significant": coef_sig_paths,
            "correlation": corr_paths,
        },
    }


# def run_full_features_correctness_glmer_bundle(
#     df: pd.DataFrame,
#     split_group_cols: Sequence[str],
#     feature_cols: Optional[Sequence[str]] = None,
#     pref_specs: Optional[Sequence[Tuple[str, str]]] = Con.PREF_SPECS,
#     pref_extreme_mode: str = "polarity",
#     save: bool = True,
#     paper_dirs: Optional[List[str]] = None,
#     dpi: int = 300,
#     close: bool = False,
#     subdir: Optional[str] = None,
#     use_rfx: bool = False,
#     run_identifier: str = "",
#     test_size: float = 0.2,
#     random_state: int = 42,
# ) -> Dict[str, Any]:
#     model = TrialLevelGLMERModel()
#     model_name = model.name
#     model_family = "glmer"
#
#     split_tag = _split_tag(split_group_cols)
#     base_dir = _answer_correctness_rel_dir(
#         model_family=model_family,
#         subdir=subdir,
#         split_tag=split_tag,
#     )
#
#     keep_cols = [Con.TEXT_ID_WITH_Q_COLUMN]
#
#     train_df, test_df = build_train_test_trial_dfs(
#         df=df,
#         split_group_cols=split_group_cols,
#         pref_specs=pref_specs,
#         pref_extreme_mode=pref_extreme_mode,
#         keep_cols=keep_cols,
#         test_size=test_size,
#         random_state=random_state,
#     )
#
#     feat_cols = _resolve_feature_cols(train_df, feature_cols)
#
#     results = evaluate_models_on_prepared_split(
#         models=[model],
#         train_df=train_df,
#         test_df=test_df,
#         target_col=Con.IS_CORRECT_COLUMN,
#         feature_cols=feat_cols,
#         fit_kwargs_by_model={
#             model_name: {
#                 "participant_col": Con.PARTICIPANT_ID,
#                 "text_col": Con.TEXT_ID_WITH_Q_COLUMN,
#             }
#         },
#         predict_kwargs_by_model={
#             model_name: {
#                 "target_col": Con.IS_CORRECT_COLUMN,
#                 "participant_col": Con.PARTICIPANT_ID,
#                 "text_col": Con.TEXT_ID_WITH_Q_COLUMN,
#                 "use_rfx": use_rfx,
#             }
#         },
#         predict_proba_kwargs_by_model={
#             model_name: {
#                 "target_col": Con.IS_CORRECT_COLUMN,
#                 "participant_col": Con.PARTICIPANT_ID,
#                 "text_col": Con.TEXT_ID_WITH_Q_COLUMN,
#                 "use_rfx": use_rfx,
#             }
#         },
#     )
#
#     show_correctness_model_results(results)
#     res = results[model_name]
#
#     formula = model.get_formula()
#     print(f"Model formula: {formula}")
#
#     summary_paths = None
#     if save:
#         summary_paths = _save_summary_csv(
#             results=results,
#             model_name=model_name,
#             trained_feature_cols=model.raw_feature_cols_,
#             base_dir=base_dir,
#             run_identifier=run_identifier,
#             paper_dirs=paper_dirs,
#             formula=formula,
#         )
#
#     cm_paths, cm_paths2 = _plot_confusions(
#         y_true=res.y_true,
#         y_pred=res.y_pred,
#         model_name=model_name,
#         base_dir=base_dir,
#         save=save,
#         paper_dirs=paper_dirs,
#         close=close,
#     )
#
#     coef_paths, coef_sig_paths = _plot_coef_summaries(
#         coef_summary=res.coef_summary,
#         model_name=model_name,
#         base_dir=base_dir,
#         save=save,
#         paper_dirs=paper_dirs,
#         dpi=dpi,
#         close=close,
#     )
#
#     trial_df = _build_full_trial_df(
#         df=df,
#         pref_specs=pref_specs,
#         pref_extreme_mode=pref_extreme_mode,
#         keep_cols=keep_cols,
#     )
#
#     corr_paths = _plot_feature_corr(
#         trial_df=trial_df,
#         corr_feature_cols=model.raw_feature_cols_,
#         base_dir=base_dir,
#         save=save,
#         paper_dirs=paper_dirs,
#         dpi=dpi,
#         close=close,
#     )
#
#     return {
#         "results": results,
#         "train_df": train_df,
#         "test_df": test_df,
#         "trial_df": trial_df,
#         "split_tag": split_tag,
#         "base_rel_dir": base_dir,
#         "summary_csv": summary_paths,
#         "formula": formula,
#         "paths": {
#             "confusion_norm": cm_paths,
#             "confusion_unnorm": cm_paths2,
#             "coef_all": coef_paths,
#             "coef_significant": coef_sig_paths,
#             "correlation": corr_paths,
#         },
#     }


def run_full_features_correctness_julia_glmer_bundle(
    df: pd.DataFrame,
    split_group_cols: Sequence[str],
    feature_cols: Optional[Sequence[str]] = None,
    pref_specs: Optional[Sequence[Tuple[str, str]]] = Con.PREF_SPECS,
    pref_extreme_mode: str = "polarity",
    save: bool = True,
    paper_dirs: Optional[List[str]] = None,
    dpi: int = 300,
    close: bool = False,
    subdir: Optional[str] = None,
    use_rfx: bool = False,
    run_identifier: str = "",
    test_size: float = 0.2,
    random_state: int = 42,
    participant_effects_mode: str = "slopes",
    text_effects_mode: str = "slopes",
) -> Dict[str, Any]:
    model = TrialLevelJuliaGLMERModel()
    model.participant_effects_mode = participant_effects_mode
    model.text_effects_mode = text_effects_mode

    model_name = model.name
    model_family = "julia"

    split_tag = _split_tag(split_group_cols)
    base_dir = _answer_correctness_rel_dir(
        model_family=model_family,
        subdir=subdir,
        split_tag=split_tag,
    )

    keep_cols = [Con.TEXT_ID_WITH_Q_COLUMN]

    train_df, test_df = build_train_test_trial_dfs(
        df=df,
        split_group_cols=split_group_cols,
        pref_specs=pref_specs,
        pref_extreme_mode=pref_extreme_mode,
        keep_cols=keep_cols,
        test_size=test_size,
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
            paper_dirs=paper_dirs,
            formula=formula,
        )

    cm_paths, cm_paths2 = _plot_confusions(
        y_true=res.y_true,
        y_pred=res.y_pred,
        model_name=model_name,
        base_dir=base_dir,
        save=save,
        paper_dirs=paper_dirs,
        close=close,
    )

    coef_paths, coef_sig_paths = _plot_coef_summaries(
        coef_summary=res.coef_summary,
        model_name=model_name,
        base_dir=base_dir,
        save=save,
        paper_dirs=paper_dirs,
        dpi=dpi,
        close=close,
    )

    trial_df = _build_full_trial_df(
        df=df,
        pref_specs=pref_specs,
        pref_extreme_mode=pref_extreme_mode,
        keep_cols=keep_cols,
    )

    corr_paths = _plot_feature_corr(
        trial_df=trial_df,
        corr_feature_cols=model.feature_cols_raw_,
        base_dir=base_dir,
        save=save,
        paper_dirs=paper_dirs,
        dpi=dpi,
        close=close,
    )

    return {
        "results": results,
        "train_df": train_df,
        "test_df": test_df,
        "trial_df": trial_df,
        "split_tag": split_tag,
        "base_rel_dir": base_dir,
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
    pref_specs: Optional[Sequence[Tuple[str, str]]] = Con.PREF_SPECS,
    pref_extreme_mode: str = "polarity",
    save: bool = True,
    paper_dirs: Optional[List[str]] = None,
    dpi: int = 300,
    close: bool = False,
    subdir: Optional[str] = None,
    top_n_rfx: int = 30,
    run_identifier: str = "",
    participant_effects_mode: str = "slopes",
    text_effects_mode: str = "slopes",
) -> Dict[str, Any]:
    pref_specs = pref_specs if pref_specs is not None else Con.PREF_SPECS

    model = TrialLevelJuliaGLMERModel()
    model.participant_effects_mode = participant_effects_mode
    model.text_effects_mode = text_effects_mode

    model_name = f"{model.name}_fit_all"

    base_dir = _answer_correctness_rel_dir(
        model_family="julia",
        subdir=subdir,
        fit_all=True,
    )

    fit_df = _build_full_trial_df(
        df=df,
        pref_specs=pref_specs,
        pref_extreme_mode=pref_extreme_mode,
        keep_cols=[Con.TEXT_ID_WITH_Q_COLUMN],
    )

    feat_cols = _resolve_feature_cols(fit_df, feature_cols)

    res = fit_model_on_prepared_full_data(
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

    corr_paths = _plot_feature_corr(
        trial_df=fit_df,
        corr_feature_cols=model.feature_cols_raw_,
        base_dir=base_dir,
        save=save,
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
        "result": res,
        "fit_df": fit_df,
        "base_rel_dir": base_dir,
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