# src/predictive_modeling/answer_correctness/answer_correctness_participant_similarity_metrics.py

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Literal

import numpy as np
import pandas as pd

from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.preprocessing import normalize

from scipy.cluster.hierarchy import linkage, fcluster, cophenet
from scipy.spatial.distance import squareform


Metric = Literal["cosine", "euclidean"]
LinkageMethod = Literal["average", "complete", "single", "ward"]


def _cluster_labels_from_distance(
    distance_matrix: pd.DataFrame,
    linkage_method: LinkageMethod,
    n_clusters: int,
) -> np.ndarray:
    """hierarchical clustering on a square distance matrix -> flat labels."""
    D = distance_matrix.values
    condensed = squareform(D, checks=False)
    Z = linkage(condensed, method=linkage_method)
    labels = fcluster(Z, t=int(n_clusters), criterion="maxclust")
    return labels


def evaluate_k_range_silhouette(
    X: pd.DataFrame,
    distance_metric: Metric = "cosine",
    linkage_method: LinkageMethod = "average",
    k_range: range = range(2, 11),
) -> pd.DataFrame:
    """
    For each k in k_range:
      - cluster participants hierarchically
      - compute silhouette score on X using the chosen distance metric

    Returns a DataFrame with columns: k, silhouette
    """
    from sklearn.metrics.pairwise import cosine_distances, euclidean_distances

    if distance_metric == "cosine":
        D = cosine_distances(X.values)
    else:
        D = euclidean_distances(X.values)

    dist_df = pd.DataFrame(D, index=X.index, columns=X.index)

    out = []
    for k in k_range:
        labels = _cluster_labels_from_distance(dist_df, linkage_method=linkage_method, n_clusters=int(k))
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
    Davies–Bouldin index (lower is better). Uses Euclidean distances internally.
    """
    X_arr = X.values
    if normalize_rows_l2:
        X_arr = normalize(X_arr)

    from sklearn.metrics.pairwise import euclidean_distances
    D = euclidean_distances(X_arr)
    dist_df = pd.DataFrame(D, index=X.index, columns=X.index)

    out = []
    for k in k_range:
        labels = _cluster_labels_from_distance(dist_df, linkage_method=linkage_method, n_clusters=int(k))
        db = davies_bouldin_score(X_arr, labels)
        out.append({"k": int(k), "davies_bouldin": float(db)})

    return pd.DataFrame(out)


def cophenetic_correlation(
    distance_matrix: pd.DataFrame,
    linkage_method: LinkageMethod = "average",
) -> float:
    """
    Cophenetic correlation: how well the dendrogram preserves the original distances.
    Higher is better.
    """
    D = distance_matrix.values
    condensed = squareform(D, checks=False)
    Z = linkage(condensed, method=linkage_method)
    coph_corr, _ = cophenet(Z, condensed)
    return float(coph_corr)


def permutation_silhouette_baseline(
    X: pd.DataFrame,
    distance_metric: Metric = "cosine",
    linkage_method: LinkageMethod = "average",
    n_clusters: int = 5,
    n_permutations: int = 100,
    random_state: int = 42,
) -> Dict[str, float]:
    """
    Computes silhouette on real data, then compares to a null distribution where
    each feature column is independently permuted across participants (destroying
    participant-level structure but keeping each feature's distribution).

    Returns:
      dict with real_silhouette, null_mean, null_std, z_score
    """
    rng = np.random.default_rng(int(random_state))

    from sklearn.metrics.pairwise import cosine_distances, euclidean_distances
    if distance_metric == "cosine":
        D_real = cosine_distances(X.values)
    else:
        D_real = euclidean_distances(X.values)

    dist_df = pd.DataFrame(D_real, index=X.index, columns=X.index)
    labels_real = _cluster_labels_from_distance(dist_df, linkage_method=linkage_method, n_clusters=int(n_clusters))
    real_sil = float(silhouette_score(X.values, labels_real, metric=distance_metric))

    null_sils = []
    X_arr = X.values.copy()

    for _ in range(int(n_permutations)):
        X_perm = X_arr.copy()
        for j in range(X_perm.shape[1]):
            rng.shuffle(X_perm[:, j])

        if distance_metric == "cosine":
            Dp = cosine_distances(X_perm)
        else:
            Dp = euclidean_distances(X_perm)

        distp_df = pd.DataFrame(Dp, index=X.index, columns=X.index)
        labels_p = _cluster_labels_from_distance(distp_df, linkage_method=linkage_method, n_clusters=int(n_clusters))
        sil_p = float(silhouette_score(X_perm, labels_p, metric=distance_metric))
        null_sils.append(sil_p)

    null_mean = float(np.mean(null_sils))
    null_std = float(np.std(null_sils)) if float(np.std(null_sils)) > 0 else 1e-9
    z = (real_sil - null_mean) / null_std

    return {
        "real_silhouette": real_sil,
        "null_mean": null_mean,
        "null_std": null_std,
        "z_score": float(z),
    }
