# src/predictive_modeling/answer_correctness/answer_correctness_participant_cluster_metrics.py

from __future__ import annotations

from typing import Dict, Literal

import numpy as np
import pandas as pd

from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_distances, euclidean_distances

from scipy.cluster.hierarchy import linkage, fcluster, cophenet
from scipy.spatial.distance import squareform


# ---------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------

Metric = Literal["cosine", "euclidean"]
LinkageMethod = Literal["average", "complete", "single", "ward"]


# ---------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------

def compute_distance_matrix(X: pd.DataFrame, metric: Metric = "cosine") -> pd.DataFrame:
    """
    Compute participant x participant distances from vectors (rows).
    """
    A = X.values
    if metric == "cosine":
        D = cosine_distances(A)
    elif metric == "euclidean":
        D = euclidean_distances(A)
    else:
        raise ValueError(f"Unknown metric: {metric}")

    return pd.DataFrame(D, index=X.index, columns=X.index)


def hierarchical_labels_from_distance(
    distance_matrix: pd.DataFrame,
    linkage_method: LinkageMethod = "average",
    n_clusters: int = 5,
) -> np.ndarray:
    """
    Hierarchical clustering on a square distance matrix -> flat labels.
    """
    condensed = squareform(distance_matrix.values, checks=False)
    Z = linkage(condensed, method=linkage_method)
    return fcluster(Z, t=int(n_clusters), criterion="maxclust")


# ---------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------

def evaluate_k_range_silhouette(
    X: pd.DataFrame,
    distance_metric: Metric = "cosine",
    linkage_method: LinkageMethod = "average",
    k_range: range = range(2, 11),
) -> pd.DataFrame:
    """
    For each k:
      - hierarchical cluster (using distance_metric)
      - silhouette on X (using distance_metric)

    Returns: columns ['k', 'silhouette'].
    """
    dist_df = compute_distance_matrix(X, metric=distance_metric)

    out = []
    for k in k_range:
        labels = hierarchical_labels_from_distance(dist_df, linkage_method, int(k))
        sil = silhouette_score(X.values, labels, metric=distance_metric)
        out.append({"k": int(k), "silhouette": float(sil)})

    return pd.DataFrame(out)


def evaluate_k_range_davies_bouldin(
    X: pd.DataFrame,
    linkage_method: LinkageMethod = "average",
    k_range: range = range(2, 11),
    normalize_rows_l2: bool = False,
) -> pd.DataFrame:
    """
    Davies–Bouldin index (lower is better). Uses Euclidean geometry internally.
    """
    A = X.values
    if normalize_rows_l2:
        A = normalize(A)

    dist_df = pd.DataFrame(euclidean_distances(A), index=X.index, columns=X.index)

    out = []
    for k in k_range:
        labels = hierarchical_labels_from_distance(dist_df, linkage_method, int(k))
        db = davies_bouldin_score(A, labels)
        out.append({"k": int(k), "davies_bouldin": float(db)})

    return pd.DataFrame(out)


def cophenetic_correlation(
    distance_matrix: pd.DataFrame,
    linkage_method: LinkageMethod = "average",
) -> float:
    """
    Dendrogram fidelity: correlation between original distances and cophenetic distances.
    Higher is better.
    """
    condensed = squareform(distance_matrix.values, checks=False)
    Z = linkage(condensed, method=linkage_method)
    coph_corr, _ = cophenet(Z, condensed)
    return float(coph_corr)
