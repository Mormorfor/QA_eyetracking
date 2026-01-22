# src/predictive_modeling/answer_correctness/answer_correctness_participant_clustering.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd

from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

from src.predictive_modeling.answer_correctness.answer_correctness_participant_similarity_metrics import (
    compute_distance_matrix,
)

Metric = Literal["cosine", "euclidean"]
LinkageMethod = Literal["average", "complete", "single", "ward"]


# ---------------------------------------------------------------------
# Agreement metrics
# ---------------------------------------------------------------------

def clustering_agreement(
    labels_a: pd.Series,
    labels_b: pd.Series,
) -> Dict[str, float]:
    """
    Compare two cluster labelings (same participants).
    Returns ARI and NMI.
    """
    idx = labels_a.index.intersection(labels_b.index)
    a = labels_a.loc[idx].astype(int).to_numpy()
    b = labels_b.loc[idx].astype(int).to_numpy()

    return {
        "ari": float(adjusted_rand_score(a, b)),
        "nmi": float(normalized_mutual_info_score(a, b)),
    }


# ---------------------------------------------------------------------
# Hierarchical clustering wrapper (returns Series labels)
# ---------------------------------------------------------------------

def cluster_hierarchical_from_distance(
    D: pd.DataFrame,
    linkage_method: LinkageMethod = "average",
    n_clusters: int = 5,
) -> pd.Series:
    """
    Hierarchical clustering from a square distance matrix -> flat labels.
    """
    condensed = squareform(D.values, checks=False)
    Z = linkage(condensed, method=linkage_method)
    lab = fcluster(Z, t=int(n_clusters), criterion="maxclust")
    return pd.Series(lab, index=D.index, name=f"hier_{linkage_method}_k{n_clusters}")


def cluster_hierarchical(
    X: pd.DataFrame,
    metric: Metric = "cosine",
    linkage_method: LinkageMethod = "average",
    n_clusters: int = 5,
) -> pd.Series:
    """
    Convenience: compute distance then hierarchical cluster.
    """
    if linkage_method == "ward" and metric != "euclidean":
        raise ValueError("Ward linkage should be used with Euclidean geometry (metric='euclidean').")

    D = compute_distance_matrix(X, metric=metric)
    return cluster_hierarchical_from_distance(D, linkage_method=linkage_method, n_clusters=n_clusters)


# ---------------------------------------------------------------------
# K-medoids (PAM) on a precomputed distance matrix
# ---------------------------------------------------------------------

@dataclass
class KMedoidsResult:
    labels: pd.Series
    medoid_indices: List[int]          # indices into D.index
    medoid_ids: List[Any]              # participant IDs
    inertia: float                     # sum of distances to assigned medoid


def _assign_to_medoids(D: np.ndarray, medoids: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Assign each point to its closest medoid.
    Returns labels (medoid slot index 0..k-1) and inertia (sum min distances).
    """
    dist_to_medoids = D[:, medoids]  # shape (n, k)
    assign = np.argmin(dist_to_medoids, axis=1)
    inertia = float(np.sum(dist_to_medoids[np.arange(D.shape[0]), assign]))
    return assign, inertia


def _pam_update_medoids(D: np.ndarray, assign: np.ndarray, k: int, medoids: np.ndarray) -> np.ndarray:
    """
    For each cluster, choose the point minimizing sum of distances within that cluster.
    """
    new_medoids = medoids.copy()
    n = D.shape[0]

    for ci in range(k):
        members = np.where(assign == ci)[0]
        if len(members) == 0:
            # empty cluster: keep medoid as is
            continue
        # cost for choosing each member as medoid = sum distances to all members
        subD = D[np.ix_(members, members)]
        costs = np.sum(subD, axis=1)
        best_member = members[int(np.argmin(costs))]
        new_medoids[ci] = best_member

    return new_medoids


def cluster_kmedoids_pam_from_distance(
    D_df: pd.DataFrame,
    n_clusters: int = 5,
    n_init: int = 20,
    max_iter: int = 200,
    random_state: int = 42,
) -> KMedoidsResult:
    """
    Partitioning Around Medoids (PAM) using a precomputed distance matrix.

    - Works with cosine distances nicely.
    - Returns participant-level labels (1..k), medoids, and inertia.
    """
    rng = np.random.default_rng(int(random_state))
    D = np.asarray(D_df.values, dtype=float)
    n = D.shape[0]
    k = int(n_clusters)

    if k < 2 or k > n:
        raise ValueError(f"n_clusters must be in [2, n]. Got {k} with n={n}.")

    best_inertia = np.inf
    best_medoids = None
    best_assign = None

    for _ in range(int(n_init)):
        medoids = rng.choice(n, size=k, replace=False)
        assign, inertia = _assign_to_medoids(D, medoids)

        for _it in range(int(max_iter)):
            new_medoids = _pam_update_medoids(D, assign, k, medoids)
            new_assign, new_inertia = _assign_to_medoids(D, new_medoids)

            if np.array_equal(new_medoids, medoids):
                # converged
                medoids = new_medoids
                assign = new_assign
                inertia = new_inertia
                break

            medoids = new_medoids
            assign = new_assign
            inertia = new_inertia

        if inertia < best_inertia:
            best_inertia = inertia
            best_medoids = medoids.copy()
            best_assign = assign.copy()

    # labels as 1..k (like fcluster)
    labels = pd.Series(best_assign + 1, index=D_df.index, name=f"kmedoids_k{k}")

    medoid_ids = [D_df.index[i] for i in best_medoids.tolist()]

    return KMedoidsResult(
        labels=labels,
        medoid_indices=best_medoids.tolist(),
        medoid_ids=medoid_ids,
        inertia=float(best_inertia),
    )


def cluster_kmedoids_pam(
    X: pd.DataFrame,
    metric: Metric = "cosine",
    n_clusters: int = 5,
    n_init: int = 20,
    max_iter: int = 200,
    random_state: int = 42,
) -> KMedoidsResult:
    """
    Convenience: compute distance then PAM k-medoids.
    """
    D = compute_distance_matrix(X, metric=metric)
    return cluster_kmedoids_pam_from_distance(
        D,
        n_clusters=n_clusters,
        n_init=n_init,
        max_iter=max_iter,
        random_state=random_state,
    )


# ---------------------------------------------------------------------
# Robustness grid runner
# ---------------------------------------------------------------------

def run_clustering_robustness_grid(
    X: pd.DataFrame,
    *,
    n_clusters: int = 5,
    base: Tuple[str, str] = ("hier", "average"),   # ("hier", linkage) or ("kmedoids", "")
    metrics: Iterable[Metric] = ("cosine", "euclidean"),
    linkages: Iterable[LinkageMethod] = ("average", "complete"),
    kmedoids_metrics: Iterable[Metric] = ("cosine",),
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Runs a small set of clustering configs and compares each to a chosen base clustering.

    Returns a DataFrame with:
      algo, metric, linkage, ari_vs_base, nmi_vs_base, (and medoid info for kmedoids if used)
    """
    results = []

    # --- base labels
    if base[0] == "hier":
        base_labels = cluster_hierarchical(X, metric="cosine", linkage_method=base[1], n_clusters=n_clusters)
        base_name = f"hier_{base[1]}_cosine"
    elif base[0] == "kmedoids":
        base_res = cluster_kmedoids_pam(X, metric="cosine", n_clusters=n_clusters, random_state=random_state)
        base_labels = base_res.labels
        base_name = "kmedoids_cosine"
    else:
        raise ValueError("base must be ('hier', <linkage>) or ('kmedoids','').")

    # --- hierarchical configs
    for m in metrics:
        for L in linkages:
            if L == "ward" and m != "euclidean":
                continue
            try:
                lab = cluster_hierarchical(X, metric=m, linkage_method=L, n_clusters=n_clusters)
                agree = clustering_agreement(base_labels, lab)
                results.append({
                    "algo": "hierarchical",
                    "metric": m,
                    "linkage": L,
                    "n_clusters": int(n_clusters),
                    "base": base_name,
                    **agree,
                })
            except Exception as e:
                results.append({
                    "algo": "hierarchical",
                    "metric": m,
                    "linkage": L,
                    "n_clusters": int(n_clusters),
                    "base": base_name,
                    "ari": np.nan,
                    "nmi": np.nan,
                    "error": str(e),
                })

    # --- k-medoids configs
    for m in kmedoids_metrics:
        try:
            km = cluster_kmedoids_pam(X, metric=m, n_clusters=n_clusters, random_state=random_state)
            agree = clustering_agreement(base_labels, km.labels)
            results.append({
                "algo": "kmedoids_pam",
                "metric": m,
                "linkage": "",
                "n_clusters": int(n_clusters),
                "base": base_name,
                "medoids": ",".join(map(str, km.medoid_ids)),
                "inertia": float(km.inertia),
                **agree,
            })
        except Exception as e:
            results.append({
                "algo": "kmedoids_pam",
                "metric": m,
                "linkage": "",
                "n_clusters": int(n_clusters),
                "base": base_name,
                "ari": np.nan,
                "nmi": np.nan,
                "error": str(e),
            })

    df = pd.DataFrame(results)
    cols = ["algo", "metric", "linkage", "n_clusters", "base", "ari", "nmi"]
    extra = [c for c in df.columns if c not in cols]
    return df[cols + extra]
