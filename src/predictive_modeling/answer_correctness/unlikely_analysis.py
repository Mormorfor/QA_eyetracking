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




def summarize_feature_means_for_probability_outcome_groups(
    result,
    *,
    feature_cols: list[str],
    target_col: str = Con.IS_CORRECT_COLUMN,
    prob_col: str = "pred_prob_correct",
    high_thresh: float = 0.80,
    low_thresh: float = 0.20,
    include_counts: bool = True,
) -> pd.DataFrame:
    """
    For each requested feature, compute mean values in four groups:

    - HP_W: high predicted probability, actually wrong
    - HP_R: high predicted probability, actually right
    - LP_W: low predicted probability, actually wrong
    - LP_R: low predicted probability, actually right
    """
    unlikely = get_unlikely_items(
        result,
        target_col=target_col,
        prob_col=prob_col,
        high_thresh=high_thresh,
        low_thresh=low_thresh,
    )

    df = unlikely["overview_df"].copy()

    hp_w_mask = (df[prob_col] >= high_thresh) & (df[target_col] == 0)
    hp_r_mask = (df[prob_col] >= high_thresh) & (df[target_col] == 1)
    lp_w_mask = (df[prob_col] <= low_thresh) & (df[target_col] == 0)
    lp_r_mask = (df[prob_col] <= low_thresh) & (df[target_col] == 1)

    rows = []
    for feature in feature_cols:
        row = {
            "feature": feature,
            "mean_HP_W": df.loc[hp_w_mask, feature].mean(),
            "mean_HP_R": df.loc[hp_r_mask, feature].mean(),
            "mean_LP_W": df.loc[lp_w_mask, feature].mean(),
            "mean_LP_R": df.loc[lp_r_mask, feature].mean(),
        }

        if include_counts:
            row.update({
                "n_HP_W": int(hp_w_mask.sum()),
                "n_HP_R": int(hp_r_mask.sum()),
                "n_LP_W": int(lp_w_mask.sum()),
                "n_LP_R": int(lp_r_mask.sum()),
            })

        rows.append(row)

    summary_df = pd.DataFrame(rows)

    return summary_df



def rank_feature_separation_between_probability_outcome_groups(
    feature_group_means_df: pd.DataFrame,
):
    df = feature_group_means_df.copy()

    df["abs_HP_W_minus_LP_W"] = (
        df["mean_HP_W"] - df["mean_LP_W"]
    ).abs()

    df["abs_LP_R_minus_HP_R"] = (
        df["mean_LP_R"] - df["mean_HP_R"]
    ).abs()

    top_hpw_lpw = (
        df.sort_values("abs_HP_W_minus_LP_W", ascending=False)
        [["feature", "mean_HP_W", "mean_LP_W", "abs_HP_W_minus_LP_W"]]
        .reset_index(drop=True)
    )

    top_lpr_hpr = (
        df.sort_values("abs_LP_R_minus_HP_R", ascending=False)
        [["feature", "mean_LP_R", "mean_HP_R", "abs_LP_R_minus_HP_R"]]
        .reset_index(drop=True)
    )

    return {
        "full_df": df,
        "top_hpw_lpw": top_hpw_lpw,
        "top_lpr_hpr": top_lpr_hpr,
    }



def plot_feature_group_separation(
    feature_group_means_df,
    top_n: int = 20,
    figsize=(10, 6),
):
    df = feature_group_means_df.copy()

    df["abs_HP_W_minus_LP_W"] = (
        df["mean_HP_W"] - df["mean_LP_W"]
    ).abs()

    df["abs_LP_R_minus_HP_R"] = (
        df["mean_LP_R"] - df["mean_HP_R"]
    ).abs()

    # --- HP-W vs LP-W ---
    top1 = (
        df.sort_values("abs_HP_W_minus_LP_W", ascending=False)
        .head(top_n)
        .sort_values("abs_HP_W_minus_LP_W", ascending=True)
    )

    plt.figure(figsize=figsize)
    plt.barh(top1["feature"], top1["abs_HP_W_minus_LP_W"])
    plt.title("Top feature differences: |HP-W − LP-W|")
    plt.xlabel("Absolute difference")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.show()

    # --- LP-R vs HP-R ---
    top2 = (
        df.sort_values("abs_LP_R_minus_HP_R", ascending=False)
        .head(top_n)
        .sort_values("abs_LP_R_minus_HP_R", ascending=True)
    )

    plt.figure(figsize=figsize)
    plt.barh(top2["feature"], top2["abs_LP_R_minus_HP_R"])
    plt.title("Top feature differences: |LP-R − HP-R|")
    plt.xlabel("Absolute difference")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.show()

