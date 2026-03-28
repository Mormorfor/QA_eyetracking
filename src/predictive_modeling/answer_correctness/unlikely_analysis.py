from dataclasses import dataclass
from typing import Optional, Sequence, Mapping, Dict, Any
import numpy as np
import pandas as pd
from IPython.core.display_functions import display
from matplotlib import pyplot as plt

from src import constants as Con


def build_prediction_overview_df(
    result,
    *,
    target_col: str = Con.IS_CORRECT_COLUMN,
    prob_col: str = "pred_prob_correct",
    pred_col: str = "pred_label",
    true_col: str = "true_label",
) -> pd.DataFrame:
    """
    Combine test_df with predictions from a CorrectnessEvaluationResult.
    """
    df = result.test_df.copy().reset_index(drop=True)

    if len(df) != len(result.y_true):
        raise ValueError("Length mismatch between test_df and y_true.")
    if len(df) != len(result.y_pred):
        raise ValueError("Length mismatch between test_df and y_pred.")
    if len(df) != len(result.y_prob):
        raise ValueError("Length mismatch between test_df and y_prob.")

    df[true_col] = np.asarray(result.y_true).astype(int)
    df[pred_col] = np.asarray(result.y_pred).astype(int)
    df[prob_col] = np.asarray(result.y_prob).astype(float)

    if target_col in df.columns:
        df["_target_matches_result"] = (
            df[target_col].astype(int).to_numpy() == df[true_col].to_numpy()
        )
    else:
        df[target_col] = df[true_col]
        df["_target_matches_result"] = True

    df["is_correct_prediction"] = (df[true_col] == df[pred_col]).astype(int)

    # "Unexpectedness" in the direction of the true label
    # wrong item with high p(correct) -> large unexpectedness
    # right item with low p(correct) -> large unexpectedness
    df["unexpectedness"] = np.where(
        df[true_col] == 1,
        1.0 - df[prob_col],
        df[prob_col],
    )

    df["prob_error"] = np.abs(df[true_col] - df[prob_col])

    return df



def get_unlikely_items(
    result,
    *,
    target_col: str = Con.IS_CORRECT_COLUMN,
    prob_col: str = "pred_prob_correct",
    high_thresh: float = 0.80,
    low_thresh: float = 0.20,
    sort_output: bool = True,
) -> dict[str, pd.DataFrame]:
    """
    Extract the two 'unlikely' groups from one CorrectnessEvaluationResult:
    1. high predicted probability of correctness, but actually wrong
    2. low predicted probability of correctness, but actually right
    """
    df = build_prediction_overview_df(
        result,
        target_col=target_col,
        prob_col=prob_col,
    )

    high_prob_wrong = df[
        (df[target_col].astype(int) == 0) & (df[prob_col] >= high_thresh)
    ].copy()

    low_prob_right = df[
        (df[target_col].astype(int) == 1) & (df[prob_col] <= low_thresh)
    ].copy()

    if sort_output:
        high_prob_wrong = high_prob_wrong.sort_values(prob_col, ascending=False)
        low_prob_right = low_prob_right.sort_values(prob_col, ascending=True)

    return {
        "overview_df": df,
        "high_prob_wrong": high_prob_wrong,
        "low_prob_right": low_prob_right,
    }



def summarize_unlikely_items(
    result,
    *,
    target_col: str = Con.IS_CORRECT_COLUMN,
    prob_col: str = "pred_prob_correct",
    high_thresh: float = 0.80,
    low_thresh: float = 0.20,
) -> pd.DataFrame:
    """
    Return a compact numeric overview of unlikely items.
    """
    out = get_unlikely_items(
        result,
        target_col=target_col,
        prob_col=prob_col,
        high_thresh=high_thresh,
        low_thresh=low_thresh,
    )

    df = out["overview_df"]
    hpw = out["high_prob_wrong"]
    lpr = out["low_prob_right"]

    n_total = len(df)
    n_right = int((df[target_col] == 1).sum())
    n_wrong = int((df[target_col] == 0).sum())

    return pd.DataFrame([
        {
            "group": "high_prob_wrong",
            "n": len(hpw),
            "pct_of_all": len(hpw) / n_total if n_total else np.nan,
            "pct_of_wrong_items": len(hpw) / n_wrong if n_wrong else np.nan,
            "mean_prob_correct": hpw[prob_col].mean() if len(hpw) else np.nan,
            "min_prob_correct": hpw[prob_col].min() if len(hpw) else np.nan,
            "max_prob_correct": hpw[prob_col].max() if len(hpw) else np.nan,
        },
        {
            "group": "low_prob_right",
            "n": len(lpr),
            "pct_of_all": len(lpr) / n_total if n_total else np.nan,
            "pct_of_right_items": len(lpr) / n_right if n_right else np.nan,
            "mean_prob_correct": lpr[prob_col].mean() if len(lpr) else np.nan,
            "min_prob_correct": lpr[prob_col].min() if len(lpr) else np.nan,
            "max_prob_correct": lpr[prob_col].max() if len(lpr) else np.nan,
        },
    ])



def get_most_extreme_unlikely_items(
    result,
    *,
    target_col: str = Con.IS_CORRECT_COLUMN,
    prob_col: str = "pred_prob_correct",
    top_n: int = 20,
) -> dict[str, pd.DataFrame]:
    """
    Get the most extreme unlikely items in both directions.
    """
    df = build_prediction_overview_df(
        result,
        target_col=target_col,
        prob_col=prob_col,
    )

    high_prob_wrong = (
        df[df[target_col].astype(int) == 0]
        .sort_values(prob_col, ascending=False)
        .head(top_n)
        .copy()
    )

    low_prob_right = (
        df[df[target_col].astype(int) == 1]
        .sort_values(prob_col, ascending=True)
        .head(top_n)
        .copy()
    )

    return {
        "most_high_prob_wrong": high_prob_wrong,
        "most_low_prob_right": low_prob_right,
    }



def summarize_unlikely_by_group(
    result,
    *,
    group_col: str,
    target_col: str = Con.IS_CORRECT_COLUMN,
    prob_col: str = "pred_prob_correct",
    high_thresh: float = 0.80,
    low_thresh: float = 0.20,
    min_n: int = 5,
) -> pd.DataFrame:
    """
    Summarize unlikely items by participant, text, question, etc.
    """
    df = build_prediction_overview_df(
        result,
        target_col=target_col,
        prob_col=prob_col,
    )

    df["high_prob_wrong"] = (
        (df[target_col].astype(int) == 0) & (df[prob_col] >= high_thresh)
    ).astype(int)

    df["low_prob_right"] = (
        (df[target_col].astype(int) == 1) & (df[prob_col] <= low_thresh)
    ).astype(int)

    out = (
        df.groupby(group_col)
        .agg(
            n_items=(target_col, "size"),
            actual_accuracy=(target_col, "mean"),
            mean_prob_correct=(prob_col, "mean"),
            n_high_prob_wrong=("high_prob_wrong", "sum"),
            n_low_prob_right=("low_prob_right", "sum"),
        )
        .reset_index()
    )

    out["pct_high_prob_wrong"] = out["n_high_prob_wrong"] / out["n_items"]
    out["pct_low_prob_right"] = out["n_low_prob_right"] / out["n_items"]

    out = out[out["n_items"] >= min_n].copy()

    return out.sort_values(
        ["n_high_prob_wrong", "n_low_prob_right", "n_items"],
        ascending=[False, False, False],
    ).reset_index(drop=True)



def compare_unlikely_feature_means(
    result,
    *,
    feature_cols: Sequence[str],
    target_col: str = Con.IS_CORRECT_COLUMN,
    prob_col: str = "pred_prob_correct",
    high_thresh: float = 0.80,
    low_thresh: float = 0.20,
) -> dict[str, pd.DataFrame]:
    """
    Compare feature means of unlikely groups against the full test set.
    """
    out = get_unlikely_items(
        result,
        target_col=target_col,
        prob_col=prob_col,
        high_thresh=high_thresh,
        low_thresh=low_thresh,
    )

    full_df = out["overview_df"]
    hpw = out["high_prob_wrong"]
    lpr = out["low_prob_right"]

    full_means = full_df[list(feature_cols)].mean(numeric_only=True)
    result_dict = {}

    if len(hpw):
        hpw_means = hpw[list(feature_cols)].mean(numeric_only=True)
        result_dict["high_prob_wrong_feature_diff"] = (
            pd.DataFrame({
                "overall_mean": full_means,
                "group_mean": hpw_means,
                "diff_vs_overall": hpw_means - full_means,
            })
            .sort_values("diff_vs_overall", key=lambda s: s.abs(), ascending=False)
            .reset_index()
            .rename(columns={"index": "feature"})
        )

    if len(lpr):
        lpr_means = lpr[list(feature_cols)].mean(numeric_only=True)
        result_dict["low_prob_right_feature_diff"] = (
            pd.DataFrame({
                "overall_mean": full_means,
                "group_mean": lpr_means,
                "diff_vs_overall": lpr_means - full_means,
            })
            .sort_values("diff_vs_overall", key=lambda s: s.abs(), ascending=False)
            .reset_index()
            .rename(columns={"index": "feature"})
        )

    return result_dict




def summarize_unlikely_items_with_viz(
    result,
    *,
    target_col: str = Con.IS_CORRECT_COLUMN,
    prob_col: str = "pred_prob_correct",
    high_thresh: float = 0.80,
    low_thresh: float = 0.20,
    participant_col: str = Con.PARTICIPANT_ID,
    text_col: str = Con.TEXT_ID_WITH_Q_COLUMN,
    top_n_groups: int = 15,
    figsize_hist: tuple = (8, 5),
    figsize_bar: tuple = (10, 6),
):
    """
    Summarize and visualize unlikely items from one CorrectnessEvaluationResult.

    Shows:
    1. Histogram of predicted probabilities by true label
    2. Bar chart of counts in the two unlikely groups
    3. Top participants with unlikely items
    4. Top texts/questions with unlikely items

    Returns a dict with the main derived dataframes.
    """
    unlikely = get_unlikely_items(
        result,
        target_col=target_col,
        prob_col=prob_col,
        high_thresh=high_thresh,
        low_thresh=low_thresh,
    )

    df = unlikely["overview_df"].copy()
    high_prob_wrong = unlikely["high_prob_wrong"].copy()
    low_prob_right = unlikely["low_prob_right"].copy()

    # -----------------------------
    # 1. Numeric overview
    # -----------------------------
    n_total = len(df)
    n_right = int((df[target_col] == 1).sum())
    n_wrong = int((df[target_col] == 0).sum())

    overview_df = pd.DataFrame([
        {
            "group": "high_prob_wrong",
            "n": len(high_prob_wrong),
            "pct_of_all": len(high_prob_wrong) / n_total if n_total else np.nan,
            "pct_of_wrong_items": len(high_prob_wrong) / n_wrong if n_wrong else np.nan,
            "mean_prob_correct": high_prob_wrong[prob_col].mean() if len(high_prob_wrong) else np.nan,
            "mean_unexpectedness": high_prob_wrong["unexpectedness"].mean() if len(high_prob_wrong) else np.nan,
        },
        {
            "group": "low_prob_right",
            "n": len(low_prob_right),
            "pct_of_all": len(low_prob_right) / n_total if n_total else np.nan,
            "pct_of_right_items": len(low_prob_right) / n_right if n_right else np.nan,
            "mean_prob_correct": low_prob_right[prob_col].mean() if len(low_prob_right) else np.nan,
            "mean_unexpectedness": low_prob_right["unexpectedness"].mean() if len(low_prob_right) else np.nan,
        },
    ])

    print("\n=== OVERVIEW ===")
    display(overview_df)

    # -----------------------------
    # 2. Histogram of predicted probabilities by true label
    # -----------------------------
    fig1, ax1 = plt.subplots(figsize=figsize_hist)

    df_right = df[df[target_col] == 1]
    df_wrong = df[df[target_col] == 0]

    ax1.hist(df_right[prob_col], bins=30, alpha=0.6, label="Actually right (true=1)")
    ax1.hist(df_wrong[prob_col], bins=30, alpha=0.6, label="Actually wrong (true=0)")

    ax1.axvline(high_thresh, linestyle="--", linewidth=1, label=f"high_thresh={high_thresh:.2f}")
    ax1.axvline(low_thresh, linestyle="--", linewidth=1, label=f"low_thresh={low_thresh:.2f}")

    ax1.set_title("Predicted probability of correctness by true outcome")
    ax1.set_xlabel("Predicted probability of correctness")
    ax1.set_ylabel("Count")
    ax1.legend()
    plt.tight_layout()
    plt.show()

    # -----------------------------
    # 3. Group counts
    # -----------------------------
    group_counts_df = pd.DataFrame({
        "group": ["high_prob_wrong", "low_prob_right"],
        "count": [len(high_prob_wrong), len(low_prob_right)],
    })

    fig2, ax2 = plt.subplots(figsize=(6, 4))
    ax2.bar(group_counts_df["group"], group_counts_df["count"])
    ax2.set_title("Counts of unlikely items")
    ax2.set_ylabel("Count")
    plt.tight_layout()
    plt.show()

    # -----------------------------
    # 4. Top participants
    # -----------------------------
    participant_summary_df = None
    if participant_col in df.columns:
        participant_flags = df.copy()
        participant_flags["high_prob_wrong_flag"] = (
            (participant_flags[target_col] == 0) & (participant_flags[prob_col] >= high_thresh)
        ).astype(int)
        participant_flags["low_prob_right_flag"] = (
            (participant_flags[target_col] == 1) & (participant_flags[prob_col] <= low_thresh)
        ).astype(int)

        participant_summary_df = (
            participant_flags.groupby(participant_col)
            .agg(
                n_items=(target_col, "size"),
                n_high_prob_wrong=("high_prob_wrong_flag", "sum"),
                n_low_prob_right=("low_prob_right_flag", "sum"),
                mean_prob_correct=(prob_col, "mean"),
                actual_accuracy=(target_col, "mean"),
            )
            .reset_index()
        )

        participant_summary_df["pct_high_prob_wrong"] = (
            participant_summary_df["n_high_prob_wrong"] / participant_summary_df["n_items"]
        )
        participant_summary_df["pct_low_prob_right"] = (
            participant_summary_df["n_low_prob_right"] / participant_summary_df["n_items"]
        )

        participant_summary_df = participant_summary_df.sort_values(
            ["n_high_prob_wrong", "n_low_prob_right", "n_items"],
            ascending=[False, False, False],
        ).reset_index(drop=True)

        print("\n=== TOP PARTICIPANTS ===")
        display(participant_summary_df.head(top_n_groups))

        top_participants_plot_df = participant_summary_df.head(top_n_groups).copy()
        top_participants_plot_df = top_participants_plot_df.sort_values(
            "n_high_prob_wrong", ascending=True
        )

        fig3, ax3 = plt.subplots(figsize=figsize_bar)
        ax3.barh(
            top_participants_plot_df[participant_col].astype(str),
            top_participants_plot_df["n_high_prob_wrong"],
            label="high_prob_wrong",
            alpha=0.8,
        )
        ax3.barh(
            top_participants_plot_df[participant_col].astype(str),
            top_participants_plot_df["n_low_prob_right"],
            left=top_participants_plot_df["n_high_prob_wrong"],
            label="low_prob_right",
            alpha=0.8,
        )
        ax3.set_title("Top participants by unlikely item counts")
        ax3.set_xlabel("Count")
        ax3.set_ylabel(participant_col)
        ax3.legend()
        plt.tight_layout()
        plt.show()

    # -----------------------------
    # 5. Top texts/questions
    # -----------------------------
    text_summary_df = None
    if text_col in df.columns:
        text_flags = df.copy()
        text_flags["high_prob_wrong_flag"] = (
            (text_flags[target_col] == 0) & (text_flags[prob_col] >= high_thresh)
        ).astype(int)
        text_flags["low_prob_right_flag"] = (
            (text_flags[target_col] == 1) & (text_flags[prob_col] <= low_thresh)
        ).astype(int)

        text_summary_df = (
            text_flags.groupby(text_col)
            .agg(
                n_items=(target_col, "size"),
                n_high_prob_wrong=("high_prob_wrong_flag", "sum"),
                n_low_prob_right=("low_prob_right_flag", "sum"),
                mean_prob_correct=(prob_col, "mean"),
                actual_accuracy=(target_col, "mean"),
            )
            .reset_index()
        )

        text_summary_df["pct_high_prob_wrong"] = (
            text_summary_df["n_high_prob_wrong"] / text_summary_df["n_items"]
        )
        text_summary_df["pct_low_prob_right"] = (
            text_summary_df["n_low_prob_right"] / text_summary_df["n_items"]
        )

        text_summary_df = text_summary_df.sort_values(
            ["n_high_prob_wrong", "n_low_prob_right", "n_items"],
            ascending=[False, False, False],
        ).reset_index(drop=True)

        print("\n=== TOP TEXTS / QUESTIONS ===")
        display(text_summary_df.head(top_n_groups))

        top_texts_plot_df = text_summary_df.head(top_n_groups).copy()
        top_texts_plot_df = top_texts_plot_df.sort_values(
            "n_high_prob_wrong", ascending=True
        )

        fig4, ax4 = plt.subplots(figsize=figsize_bar)
        ax4.barh(
            top_texts_plot_df[text_col].astype(str),
            top_texts_plot_df["n_high_prob_wrong"],
            label="high_prob_wrong",
            alpha=0.8,
        )
        ax4.barh(
            top_texts_plot_df[text_col].astype(str),
            top_texts_plot_df["n_low_prob_right"],
            left=top_texts_plot_df["n_high_prob_wrong"],
            label="low_prob_right",
            alpha=0.8,
        )
        ax4.set_title("Top texts/questions by unlikely item counts")
        ax4.set_xlabel("Count")
        ax4.set_ylabel(text_col)
        ax4.legend()
        plt.tight_layout()
        plt.show()

    # -----------------------------
    # 6. Most extreme rows
    # -----------------------------
    print("\n=== MOST EXTREME HIGH-PROB-WRONG ===")
    display(
        high_prob_wrong.sort_values(prob_col, ascending=False).head(20)
    )

    print("\n=== MOST EXTREME LOW-PROB-RIGHT ===")
    display(
        low_prob_right.sort_values(prob_col, ascending=True).head(20)
    )

    return {
        "overview_df": df,
        "overview_summary_df": overview_df,
        "high_prob_wrong_df": high_prob_wrong,
        "low_prob_right_df": low_prob_right,
        "participant_summary_df": participant_summary_df,
        "text_summary_df": text_summary_df,
        "group_counts_df": group_counts_df,
    }



def analyze_unlikely_feature_patterns(
    result,
    *,
    feature_cols,
    target_col="is_correct",
    prob_col="pred_prob_correct",
    high_thresh=0.80,
    low_thresh=0.20,
    top_k_features=15,
    show_distributions=False,
):
    """
    Analyze whether unlikely cases differ systematically in feature space.

    Produces:
    - Feature deviation tables
    - Bar plots of strongest differences
    - Optional distribution plots
    """

    unlikely = get_unlikely_items(
        result,
        target_col=target_col,
        prob_col=prob_col,
        high_thresh=high_thresh,
        low_thresh=low_thresh,
    )

    df = unlikely["overview_df"]
    hpw = unlikely["high_prob_wrong"]
    lpr = unlikely["low_prob_right"]

    # -----------------------------
    # 1. Compute feature means
    # -----------------------------
    full_means = df[feature_cols].mean(numeric_only=True)

    hpw_means = hpw[feature_cols].mean(numeric_only=True) if len(hpw) else None
    lpr_means = lpr[feature_cols].mean(numeric_only=True) if len(lpr) else None

    results = {}

    # -----------------------------
    # 2. Build comparison tables
    # -----------------------------
    if hpw_means is not None:
        hpw_df = pd.DataFrame({
            "feature": feature_cols,
            "overall_mean": full_means.values,
            "hpw_mean": hpw_means.values,
        })
        hpw_df["diff_vs_overall"] = hpw_df["hpw_mean"] - hpw_df["overall_mean"]
        hpw_df["abs_diff"] = hpw_df["diff_vs_overall"].abs()

        hpw_df = hpw_df.sort_values("abs_diff", ascending=False).reset_index(drop=True)

        print("\n=== HIGH PROB WRONG: FEATURE DEVIATIONS ===")
        display(hpw_df.head(30))

        results["hpw_feature_diff"] = hpw_df

    if lpr_means is not None:
        lpr_df = pd.DataFrame({
            "feature": feature_cols,
            "overall_mean": full_means.values,
            "lpr_mean": lpr_means.values,
        })
        lpr_df["diff_vs_overall"] = lpr_df["lpr_mean"] - lpr_df["overall_mean"]
        lpr_df["abs_diff"] = lpr_df["diff_vs_overall"].abs()

        lpr_df = lpr_df.sort_values("abs_diff", ascending=False).reset_index(drop=True)

        print("\n=== LOW PROB RIGHT: FEATURE DEVIATIONS ===")
        display(lpr_df.head(30))

        results["lpr_feature_diff"] = lpr_df

    # -----------------------------
    # 3. Visualization: top features
    # -----------------------------
    def plot_top_features(df_diff, title):
        top_df = df_diff.head(top_k_features).copy()
        top_df = top_df.sort_values("diff_vs_overall", ascending=True)

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.barh(top_df["feature"], top_df["diff_vs_overall"])
        ax.set_title(title)
        ax.set_xlabel("Difference vs overall mean")
        plt.tight_layout()
        plt.show()

    if "hpw_feature_diff" in results:
        plot_top_features(
            results["hpw_feature_diff"],
            "High-prob-wrong: strongest feature deviations"
        )

    if "lpr_feature_diff" in results:
        plot_top_features(
            results["lpr_feature_diff"],
            "Low-prob-right: strongest feature deviations"
        )

    # -----------------------------
    # 4. Optional: feature distributions
    # -----------------------------
    if show_distributions and "hpw_feature_diff" in results:
        print("\n=== DISTRIBUTIONS (TOP FEATURES, HPW) ===")
        for feat in results["hpw_feature_diff"]["feature"].head(5):
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.hist(df[feat], bins=30, alpha=0.5, label="all")
            ax.hist(hpw[feat], bins=30, alpha=0.5, label="high_prob_wrong")
            ax.set_title(feat)
            ax.legend()
            plt.tight_layout()
            plt.show()

    if show_distributions and "lpr_feature_diff" in results:
        print("\n=== DISTRIBUTIONS (TOP FEATURES, LPR) ===")
        for feat in results["lpr_feature_diff"]["feature"].head(5):
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.hist(df[feat], bins=30, alpha=0.5, label="all")
            ax.hist(lpr[feat], bins=30, alpha=0.5, label="low_prob_right")
            ax.set_title(feat)
            ax.legend()
            plt.tight_layout()
            plt.show()

    return results




def analyze_unlikely_feature_patterns_with_scaling(
    result,
    *,
    model,  # <-- pass the fitted model
    feature_cols,
    target_col="is_correct",
    prob_col="pred_prob_correct",
    high_thresh=0.80,
    low_thresh=0.20,
    top_k_features=15,
):
    """
    Compare feature deviations:
    1) raw (original units)
    2) standardized (model space, using scaler_)

    This lets you distinguish:
    - behavioral magnitude (raw)
    - model relevance (scaled)
    """

    unlikely = get_unlikely_items(
        result,
        target_col=target_col,
        prob_col=prob_col,
        high_thresh=high_thresh,
        low_thresh=low_thresh,
    )

    df = unlikely["overview_df"]
    hpw = unlikely["high_prob_wrong"]
    lpr = unlikely["low_prob_right"]

    # -----------------------------
    # 1. RAW FEATURES (as before)
    # -----------------------------
    def compute_diff(df_all, df_group, feature_cols):
        overall = df_all[feature_cols].mean(numeric_only=True)
        group = df_group[feature_cols].mean(numeric_only=True)

        out = pd.DataFrame({
            "feature": feature_cols,
            "overall_mean": overall.values,
            "group_mean": group.values,
        })
        out["diff"] = out["group_mean"] - out["overall_mean"]
        out["abs_diff"] = out["diff"].abs()
        return out.sort_values("abs_diff", ascending=False).reset_index(drop=True)

    raw_results = {}
    if len(hpw):
        raw_results["hpw"] = compute_diff(df, hpw, feature_cols)
    if len(lpr):
        raw_results["lpr"] = compute_diff(df, lpr, feature_cols)

    # -----------------------------
    # 2. STANDARDIZED FEATURES
    # -----------------------------
    if model.scaler_ is None:
        raise RuntimeError("Model scaler_ not fitted.")

    X_scaled = model.scaler_.transform(df[feature_cols])
    X_scaled = pd.DataFrame(X_scaled, columns=feature_cols, index=df.index)

    X_scaled_hpw = X_scaled.loc[hpw.index] if len(hpw) else None
    X_scaled_lpr = X_scaled.loc[lpr.index] if len(lpr) else None

    scaled_results = {}

    def compute_scaled_diff(X_all, X_group):
        overall = X_all.mean()
        group = X_group.mean()

        out = pd.DataFrame({
            "feature": X_all.columns,
            "overall_mean_scaled": overall.values,
            "group_mean_scaled": group.values,
        })
        out["diff_scaled"] = out["group_mean_scaled"] - out["overall_mean_scaled"]
        out["abs_diff_scaled"] = out["diff_scaled"].abs()
        return out.sort_values("abs_diff_scaled", ascending=False).reset_index(drop=True)

    if X_scaled_hpw is not None:
        scaled_results["hpw"] = compute_scaled_diff(X_scaled, X_scaled_hpw)

    if X_scaled_lpr is not None:
        scaled_results["lpr"] = compute_scaled_diff(X_scaled, X_scaled_lpr)

    # -----------------------------
    # 3. PLOTTING
    # -----------------------------
    def plot(df_diff, col, title):
        top = df_diff.head(top_k_features).copy()
        top = top.sort_values(col, ascending=True)

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.barh(top["feature"], top[col])
        ax.set_title(title)
        ax.set_xlabel(col)
        plt.tight_layout()
        plt.show()

    print("\n=== RAW (BEHAVIORAL SPACE) ===")
    if "hpw" in raw_results:
        display(raw_results["hpw"].head(20))
        plot(raw_results["hpw"], "diff", "HPW (raw)")

    if "lpr" in raw_results:
        display(raw_results["lpr"].head(20))
        plot(raw_results["lpr"], "diff", "LPR (raw)")

    print("\n=== STANDARDIZED (MODEL SPACE) ===")
    if "hpw" in scaled_results:
        display(scaled_results["hpw"].head(20))
        plot(scaled_results["hpw"], "diff_scaled", "HPW (scaled)")

    if "lpr" in scaled_results:
        display(scaled_results["lpr"].head(20))
        plot(scaled_results["lpr"], "diff_scaled", "LPR (scaled)")

    return {
        "raw": raw_results,
        "scaled": scaled_results,
    }