# src/viz/visualisations_preference_correctness_tests.py

from __future__ import annotations

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def _ensure_str_thr(df: pd.DataFrame, thr_col: str = "uniform_rel_range") -> pd.DataFrame:
    df = df.copy()
    df[thr_col] = df[thr_col].astype(float)
    df["thr_str"] = df[thr_col].map(lambda x: f"{x:.2f}")
    return df


def plot_pairwise_correctness_significance_heatmaps(
    stats_summary: pd.DataFrame,
    group: str,
    use_p: str = "p_fdr",
    alpha: float = 0.05,
    thresholds=(0.10, 0.15, 0.20),
    save: bool = False,
    output_root: str = "../reports/plots/preference_correctness_tests",
):
    df = stats_summary[stats_summary["group"] == group].copy()
    df["uniform_rel_range"] = df["uniform_rel_range"].astype(float).round(2)

    pair_col = "comparison" if "comparison" in df.columns else "pair"
    pairs = sorted(df[pair_col].unique())
    metrics = sorted(df["metric"].unique())
    thresholds = [float(t) for t in thresholds]

    for pair in pairs:
        sub = df[df[pair_col] == pair].copy()

        pivot_p = sub.pivot(index="metric", columns="uniform_rel_range", values=use_p)

        # force full grid (THIS fixes the “tiny black square” plots)
        pivot_p = pivot_p.reindex(index=metrics, columns=thresholds)

        sig = (pivot_p < alpha).fillna(False)

        # no applymap; no deprecation
        annot = sig.replace({True: "✓", False: ""}).astype(str)

        plt.figure(figsize=(7.5, max(2.5, 0.55 * len(metrics))))
        ax = sns.heatmap(
            sig.astype(int),
            annot=annot,
            fmt="",
            cbar=False,
            vmin=0,
            vmax=1,
            linewidths=0.5,
        )
        ax.set_title(f"{group}: significance ({use_p} < {alpha}) — {pair}")
        ax.set_xlabel("uniform_rel_range")
        ax.set_ylabel("metric")
        plt.tight_layout()

        if save:
            import os
            os.makedirs(output_root, exist_ok=True)
            plt.savefig(
                os.path.join(output_root, f"{group}__sig__{pair}__{use_p}.png"),
                dpi=300
            )
        plt.show()



def plot_pairwise_correctness_effect_heatmaps(
    stats_summary: pd.DataFrame,
    group: str,
    effect_col: str = "delta_correctness",
    signif_col: str = "p_fdr",
    alpha: float = 0.05,
    thresholds=(0.10, 0.15, 0.20),
    save: bool = False,
    output_root: str = "../reports/plots/preference_correctness_tests",
):
    df = stats_summary[stats_summary["group"] == group].copy()
    df["uniform_rel_range"] = df["uniform_rel_range"].astype(float).round(2)

    pair_col = "comparison" if "comparison" in df.columns else "pair"
    pairs = sorted(df[pair_col].unique())

    metrics = sorted(df["metric"].unique())
    thresholds = [float(t) for t in thresholds]

    def _fmt_cell(effect, p):
        if pd.isna(effect):
            return ""
        s = f"{effect:+.3f}"
        if pd.notna(p) and p < alpha:
            s += "*"
        return s

    for pair in pairs:
        sub = df[df[pair_col] == pair].copy()
        pivot_eff = sub.pivot(index="metric", columns="uniform_rel_range", values=effect_col)
        pivot_p   = sub.pivot(index="metric", columns="uniform_rel_range", values=signif_col)

        # force full grid
        pivot_eff = pivot_eff.reindex(index=metrics, columns=thresholds)
        pivot_p   = pivot_p.reindex(index=metrics, columns=thresholds)

        # IMPORTANT: annotation dataframe as object dtype
        annot = pd.DataFrame("", index=pivot_eff.index, columns=pivot_eff.columns, dtype=object)

        for r in pivot_eff.index:
            for c in pivot_eff.columns:
                annot.loc[r, c] = _fmt_cell(pivot_eff.loc[r, c], pivot_p.loc[r, c])

        plt.figure(figsize=(7.5, max(2.5, 0.55 * len(metrics))))
        ax = sns.heatmap(
            pivot_eff,
            annot=annot,
            fmt="",
            center=0,
            linewidths=0.5,
            cbar=True,
        )
        ax.set_title(f"{group}: effect size ({effect_col}) — {pair}  (* = {signif_col} < {alpha})")
        ax.set_xlabel("uniform_rel_range")
        ax.set_ylabel("metric")
        plt.tight_layout()

        if save:
            import os
            os.makedirs(output_root, exist_ok=True)
            plt.savefig(
                os.path.join(output_root, f"{group}__effect__{pair}__{effect_col}.png"),
                dpi=300
            )
        plt.show()
