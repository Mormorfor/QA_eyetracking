# visualisations_scan_patterns.py

import os

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from src import constants as Con


def plot_correctness_by_group(
    df: pd.DataFrame,
    label: str = "Group",
    uniform_rel_range = None,
    metric = None,
    save: bool = False,
    output_root: str = "../reports/plots/scan_patterns",
) -> pd.DataFrame:

    """
    Create a barplot comparing correctness rates across:
        - matching
        - uniform
        - mismatch

    Expects columns:
        - 'pref_group'
        - Con.IS_CORRECT_COLUMN
    """
    metric_str = metric.replace("_", " ").title() if metric else "Eye-Tracking Metric"
    thr_str = (
        f"Uniform threshold = {uniform_rel_range:.2f}"
        if uniform_rel_range is not None
        else "Uniform threshold: n/a"
    )

    group_order = ["matching", "uniform", "mismatch"]
    df = df[df["pref_group"].isin(group_order)].copy()
    df["pref_group"] = pd.Categorical(
        df["pref_group"], categories=group_order, ordered=True
    )

    summary = (
        df.groupby("pref_group")[Con.IS_CORRECT_COLUMN]
        .agg(correct_rate="mean", n_trials="count")
        .reset_index()
    )
    summary["correct_rate"] = summary["correct_rate"].round(3)

    # Plot
    plt.figure(figsize=(7, 5))
    sns.barplot(
        data=df,
        x="pref_group",
        y=Con.IS_CORRECT_COLUMN,
        estimator="mean",
        errorbar=("ci", 95),
        capsize=0.15,
    )

    plt.title(
        f"{label}: Correctness by Trial Preference Group\n"
        f"(Preference based on {metric_str}; {thr_str})"
    )
    plt.xlabel("Preference Group")
    plt.ylabel("Correctness Rate (proportion correct)")
    plt.ylim(0, 1)
    plt.tight_layout()

    if save:
        os.makedirs(output_root, exist_ok=True)
        metric_tag = metric if metric else "unknown_metric"
        thr_tag = f"uniform{uniform_rel_range:.2f}" if uniform_rel_range is not None else "uniformNA"
        fname = f"{label.lower()}_correctness_{metric_tag}_{thr_tag}.png"
        plt.savefig(os.path.join(output_root, fname), dpi=300)
    plt.show()

    return summary
