from __future__ import annotations

from typing import Sequence, Tuple, Optional, List, Dict, Any
import pandas as pd

from src import constants as Con

from src.predictive_modeling.answer_correctness.answer_correctness_data import (
    build_trial_level_all_features,
)
from src.predictive_modeling.answer_correctness.answer_correctness_models import (
    FullFeaturesCorrectnessLogRegModel,
)
from src.predictive_modeling.answer_correctness.answer_correctness_eval import (
    evaluate_models_on_answer_correctness,
)
from src.predictive_modeling.common.viz_utils import (
    plot_confusion_heatmap,
)
from src.predictive_modeling.answer_correctness.answer_correctness_viz import (
    plot_coef_summary_barh,
    plot_feature_correlation_heatmap,
)

from src.predictive_modeling.answer_correctness.answer_correctness_viz import (
    show_correctness_model_results,
    correctness_results_to_summary_df,
)
from src.viz.plot_output import save_df_csv


def _split_tag(split_group_cols: Sequence[str]) -> str:
    return "+".join(split_group_cols)


def _base_rel_dir(split_tag: str, subdir: Optional[str]) -> str:
    """
    answer_correctness/<split_tag>[/<subdir>]
    """
    if subdir is None or str(subdir).strip() == "":
        return f"answer_correctness/{split_tag}"
    return f"answer_correctness/{split_tag}/{subdir}"


def run_full_features_correctness_bundle(
    df: pd.DataFrame,
    split_group_cols: Sequence[str],
    feature_cols: Optional[Sequence[str]] = None,
    group_cols: Sequence[str] = (Con.PARTICIPANT_ID, Con.TRIAL_ID),
    pref_specs: Optional[Sequence[Tuple[str, str]]] = None,
    pref_extreme_mode: str = "polarity",
    coef_ci_method: str = "wald",
    coef_ci_cluster: str = "row",
    save: bool = True,
    paper_dirs: Optional[List[str]] = None,
    dpi: int = 300,
    close: bool = False,
    subdir: Optional[str] = None,
) -> Dict[str, Any]:

    model = FullFeaturesCorrectnessLogRegModel()
    model_name = "full_features_correctness_log_reg"

    split_tag = _split_tag(split_group_cols)
    base_dir = _base_rel_dir(split_tag, subdir)

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

    # save CSV summary next to plots
    summary_paths = None
    if save:
        summary_df = correctness_results_to_summary_df(results)
        summary_paths = save_df_csv(
            summary_df,
            rel_dir=base_dir,
            filename="model_summary",
            paper_dirs=paper_dirs,
        )

    res = results[model_name]

    # ------------------------
    # Confusion matrix
    # ------------------------
    fig_cm, cm_df, cm_paths = plot_confusion_heatmap(
        y_true=res.y_true,
        y_pred=res.y_pred,
        labels=(0, 1),
        normalize=True,
        title=f"{model_name} – normalized confusion",
        save=save,
        rel_dir=f"{base_dir}/confusion",
        filename=f"{model_name}_normalized_confusion",
        paper_dirs=paper_dirs,
        close=close,
    )

    coef_paths = []
    coef_sig_paths = []

    # ------------------------
    # Coefficient plots
    # ------------------------
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

    # ------------------------
    # Correlation matrix
    # ------------------------
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
            "confusion": cm_paths,
            "coef_all": coef_paths,
            "coef_significant": coef_sig_paths,
            "correlation": corr_paths,
        },
        "trial_df": trial_df,
        "split_tag": split_tag,
        "base_rel_dir": base_dir,
        "summary_csv": summary_paths,
    }