# src/predictive_modeling/answer_correctness/answer_correctness_participant_clusters_viz.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Literal

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import squareform

import umap


# ---------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------

LinkageMethod = Literal["average", "complete", "single", "ward"]
CutMode = Literal["n_clusters", "distance"]


# ---------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------

@dataclass
class ParticipantClusteringResult:
    linkage_Z: np.ndarray
    labels: pd.Series


# ---------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------

def hierarchical_cluster_participants(
    distance_matrix: pd.DataFrame,
    linkage_method: LinkageMethod = "average",
    cut_mode: CutMode = "n_clusters",
    n_clusters: int = 5,
    distance_threshold: Optional[float] = None,
) -> ParticipantClusteringResult:
    """
    Hierarchical clustering from a square distance matrix.
    """
    condensed = squareform(distance_matrix.values, checks=False)
    Z = linkage(condensed, method=linkage_method)

    if cut_mode == "n_clusters":
        lab = fcluster(Z, t=int(n_clusters), criterion="maxclust")
    else:
        lab = fcluster(Z, t=float(distance_threshold), criterion="distance")

    labels = pd.Series(lab, index=distance_matrix.index, name="cluster")
    return ParticipantClusteringResult(linkage_Z=Z, labels=labels)


# ---------------------------------------------------------------------
# Dendrogram plot
# ---------------------------------------------------------------------

def plot_participant_dendrogram(
    clustering: ParticipantClusteringResult,
    figsize: Tuple[int, int] = (12, 6),
    title: Optional[str] = None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=figsize)
    dendrogram(
        clustering.linkage_Z,
        labels=[str(x) for x in clustering.labels.index.tolist()],
        leaf_rotation=90,
        leaf_font_size=8,
        ax=ax,
    )
    if title:
        ax.set_title(title)
    ax.set_ylabel("Distance")
    plt.tight_layout()

    if save_path is not None:
        import os
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")

    return fig


# ---------------------------------------------------------------------
# UMAP
# ---------------------------------------------------------------------

def compute_participant_umap(
    coef_matrix: pd.DataFrame,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    metric: str = "cosine",
) -> pd.DataFrame:
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=int(n_neighbors),
        min_dist=float(min_dist),
        metric=metric,
    )
    emb = reducer.fit_transform(coef_matrix.values)
    return pd.DataFrame(emb, index=coef_matrix.index, columns=["x", "y"])


def plot_participant_umap(
    umap_xy: pd.DataFrame,
    labels: Optional[pd.Series] = None,
    figsize: Tuple[int, int] = (8, 6),
    title: Optional[str] = None,
    annotate: bool = False,
    save_path: Optional[str] = None,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=figsize)

    if labels is None:
        ax.scatter(umap_xy["x"], umap_xy["y"])
    else:
        for c in sorted(labels.unique()):
            idx = labels[labels == c].index
            ax.scatter(umap_xy.loc[idx, "x"], umap_xy.loc[idx, "y"], label=f"cluster {c}")
        ax.legend(loc="best")

    if annotate:
        for pid, row in umap_xy.iterrows():
            ax.text(row["x"], row["y"], str(pid), fontsize=7)

    if title:
        ax.set_title(title)
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    plt.tight_layout()

    if save_path is not None:
        import os
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")

    return fig
