from __future__ import annotations

from typing import Sequence, Tuple, Optional, List, Dict, Any
import pandas as pd
import numpy as np

from src import constants as Con

from src.predictive_modeling.answer_correctness.answer_correctness_data import (
    build_trial_level_all_features,
)
from src.predictive_modeling.answer_correctness.answer_correctness_models import (
    FullFeaturesCorrectnessLogRegModel,
)
from src.predictive_modeling.answer_correctness.answer_correctness_eval import (
    evaluate_models_on_answer_correctness,
    evaluate_glmer_on_answer_correctness
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

from src.predictive_modeling.answer_correctness.answer_correctness_models import (
    FullFeaturesCorrectnessGLMERModel
)

from src.predictive_modeling.common.data_utils import (
    group_vise_train_test_split,
)

from sklearn.metrics import (
    precision_recall_fscore_support,
    balanced_accuracy_score,
    roc_auc_score,
    average_precision_score,
)

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
            "confusion_norm": cm_paths,
            "confusion_unnorm"  : cm_paths2,
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
) -> Dict[str, Any]:

    model = FullFeaturesCorrectnessGLMERModel()
    model_name = model.name

    split_tag = _split_tag(split_group_cols)
    base_dir = _base_rel_dir(split_tag, subdir)

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

    # ------------------------
    # Save CSV summary
    # ------------------------
    summary_paths = None
    if save:
        y_true = np.asarray(res.y_true)
        y_pred = np.asarray(res.y_pred)
        y_prob = None if res.y_prob is None else np.asarray(res.y_prob)

        prec, rec, f1, support = precision_recall_fscore_support(
            y_true,
            y_pred,
            labels=[0, 1],
            average=None,
            zero_division=0,
        )

        macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
            y_true,
            y_pred,
            labels=[0, 1],
            average="macro",
            zero_division=0,
        )

        weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(
            y_true,
            y_pred,
            labels=[0, 1],
            average="weighted",
            zero_division=0,
        )

        bal_acc = balanced_accuracy_score(y_true, y_pred)

        roc_auc = None
        avg_prec = None
        if y_prob is not None:
            try:
                roc_auc = float(roc_auc_score(y_true, y_prob))
            except Exception:
                roc_auc = None
            try:
                avg_prec = float(average_precision_score(y_true, y_prob))
            except Exception:
                avg_prec = None

        summary_df = pd.DataFrame([{
            "model_name": model_name,
            "accuracy": float(res.accuracy),
            "balanced_accuracy": float(bal_acc),
            "n_test": int(res.n_test),
            "n_positive": int(res.n_positive),
            "n_negative": int(res.n_negative),

            "precision_0": float(prec[0]),
            "recall_0": float(rec[0]),
            "f1_0": float(f1[0]),
            "support_0": int(support[0]),

            "precision_1": float(prec[1]),
            "recall_1": float(rec[1]),
            "f1_1": float(f1[1]),
            "support_1": int(support[1]),

            "macro_precision": float(macro_p),
            "macro_recall": float(macro_r),
            "macro_f1": float(macro_f1),

            "weighted_precision": float(weighted_p),
            "weighted_recall": float(weighted_r),
            "weighted_f1": float(weighted_f1),

            "roc_auc": roc_auc,
            "average_precision": avg_prec,
            "formula": formula,
        }])

        summary_paths = save_df_csv(
            summary_df,
            rel_dir=base_dir,
            filename="model_summary",
            paper_dirs=paper_dirs,
        )

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

    # ------------------------
    # Coefficient plots
    # ------------------------
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

    # ------------------------
    # Correlation matrix
    # ------------------------
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
            "confusion_unnorm"  : cm_paths2,
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