# src/predictive_modeling/answer_correctness/answer_correctness_participant_similarity_centroids.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


@dataclass
class ClusterCentroidResult:
    centroids: pd.DataFrame
    counts: pd.Series
    top_features: pd.DataFrame


def compute_cluster_centroids(
    X: pd.DataFrame,
    labels: pd.Series,
    *,
    agg: str = "mean",
    top_k: int = 10,
) -> ClusterCentroidResult:
    """
    Compute cluster centroids from participant vectors.
    """
    labels = labels.loc[X.index]

    df = X.copy()
    df["__cluster__"] = labels.values

    if agg == "mean":
        centroids = df.groupby("__cluster__").mean(numeric_only=True)
    elif agg == "median":
        centroids = df.groupby("__cluster__").median(numeric_only=True)
    else:
        raise ValueError(f"Unknown agg: {agg}")

    counts = labels.value_counts().sort_index()

    rows = []
    for c in centroids.index:
        s = centroids.loc[c].dropna()
        top = s.abs().sort_values(ascending=False).head(int(top_k))
        for feat in top.index:
            rows.append(
                {
                    "cluster": int(c),
                    "feature": feat,
                    "centroid": float(s[feat]),
                    "abs_centroid": float(abs(s[feat])),
                }
            )
    top_df = pd.DataFrame(rows).sort_values(["cluster", "abs_centroid"], ascending=[True, False])

    return ClusterCentroidResult(
        centroids=centroids,
        counts=counts,
        top_features=top_df,
    )


def plot_cluster_centroids_barh(
    centroids: pd.DataFrame,
    counts: Optional[pd.Series] = None,
    *,
    top_k: int = 12,
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[str] = None,
) -> None:
    """
    Plot per-cluster centroid profiles as horizontal bar charts.

    - Selects top_k features by absolute centroid within each cluster.
    """
    import os


    for c in centroids.index:
        s = centroids.loc[c].dropna()
        top_feats = s.abs().sort_values(ascending=False).head(int(top_k)).index
        s_plot = s.loc[top_feats].sort_values()

        n_bars = len(s_plot)
        row_height = 0.35
        min_h = 4
        max_h = 20
        height = min(max(min_h, n_bars * row_height), max_h)

        fig, ax = plt.subplots(figsize=(figsize[0], height))
        ax.barh(s_plot.index.astype(str), s_plot.values)
        ax.axvline(0, linewidth=1)

        n_txt = ""
        if counts is not None and int(c) in counts.index:
            n_txt = f" (n={int(counts.loc[int(c)])})"
        if title:
            ax.set_title(f"{title} – cluster {int(c)}{n_txt}")
        else:
            ax.set_title(f"Cluster {int(c)}{n_txt}")

        ax.set_xlabel("centroid value")
        ax.set_ylabel("feature/family")
        plt.tight_layout()

        if save_path is not None:
            out = save_path.format(cluster=int(c))
            os.makedirs(os.path.dirname(out), exist_ok=True)
            fig.savefig(out, dpi=200, bbox_inches="tight")

        #plt.close(fig)


