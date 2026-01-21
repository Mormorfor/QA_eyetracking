# src/predictive_modeling/answer_correctness/answer_correctness_participant_similarity_viz.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Tuple, Literal

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import squareform

import umap


LinkageMethod = Literal["average", "complete", "single", "ward"]
CutMode = Literal["n_clusters", "distance"]


@dataclass
class ParticipantClusteringResult:
    linkage_Z: np.ndarray
    labels: pd.Series
    umap_df: Optional[pd.DataFrame] = None


def hierarchical_cluster_participants(
    distance_matrix: pd.DataFrame,
    linkage_method: LinkageMethod = "average",
    cut_mode: CutMode = "n_clusters",
    n_clusters: int = 5,
    distance_threshold: Optional[float] = None,
) -> ParticipantClusteringResult:
    """
    Hierarchical clustering from a square distance matrix.

    cut_mode:
      - "n_clusters": cut to exactly n_clusters
      - "distance": cut by distance threshold (provide distance_threshold)

    Returns linkage + per-participant cluster labels.
    """
    D = distance_matrix.values
    condensed = squareform(D, checks=False)

    Z = linkage(condensed, method=linkage_method)

    if cut_mode == "n_clusters":
        labels = fcluster(Z, t=int(n_clusters), criterion="maxclust")
    elif cut_mode == "distance":
        if distance_threshold is None:
            raise ValueError("distance_threshold must be provided when cut_mode='distance'")
        labels = fcluster(Z, t=float(distance_threshold), criterion="distance")
    else:
        raise ValueError(f"Unknown cut_mode: {cut_mode}")

    labels_s = pd.Series(labels, index=distance_matrix.index, name="cluster")
    return ParticipantClusteringResult(linkage_Z=Z, labels=labels_s)


def plot_participant_dendrogram(
    clustering: ParticipantClusteringResult,
    figsize: Tuple[int, int] = (12, 6),
    title: Optional[str] = None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot dendrogram for participant clustering.
    """
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


def compute_participant_umap(
    coef_matrix: pd.DataFrame,
    random_state: int = 42,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    metric: str = "cosine",
) -> pd.DataFrame:
    """
    Compute a 2D UMAP embedding from participant coefficient vectors.
    Returns DataFrame with index=participant_id and columns ['x','y'].
    """
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=int(n_neighbors),
        min_dist=float(min_dist),
        metric=metric,
        random_state=int(random_state),
    )
    X = coef_matrix.values
    emb = reducer.fit_transform(X)
    return pd.DataFrame(emb, index=coef_matrix.index, columns=["x", "y"])


def plot_participant_umap(
    umap_xy: pd.DataFrame,
    labels: Optional[pd.Series] = None,
    figsize: Tuple[int, int] = (8, 6),
    title: Optional[str] = None,
    annotate: bool = False,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Scatter plot of UMAP coordinates.
    """
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
