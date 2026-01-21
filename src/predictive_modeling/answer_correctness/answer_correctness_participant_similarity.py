# src/predictive_modeling/answer_correctness/answer_correctness_participant_similarity.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional, Tuple, Literal, Dict, List

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_distances, euclidean_distances

from src import constants as Con


Metric = Literal["cosine", "euclidean"]


@dataclass
class ParticipantClusteringInputs:
    """
    Container for participant-level coefficient representations.

    coef_matrix:
        rows = participants
        cols = features
        values = coefficients
    coef_matrix_z:
        z-scored version across participants (per feature column)
    """
    coef_matrix: pd.DataFrame
    coef_matrix_z: pd.DataFrame



def build_participant_coef_matrix(
    results_by_pid: Mapping[Any, Mapping[str, Any]],
    model_name: str,
    coef_col: str = "coef",
    participant_col_name: str = Con.PARTICIPANT_ID,
    fill_value: float = 0.0,
    drop_all_zero_features: bool = True,
) -> pd.DataFrame:
    """
    Build a participant x feature coefficient matrix from nested evaluation results.

    Expects:
        results_by_pid[pid][model_name].coef_summary is a DataFrame with columns:
            - 'feature'
            - coef_col (e.g., 'coef' or 'standardized_coef')

    Notes:
      - Some participants/models may be missing -> skipped.
      - Missing features for a participant are filled with `fill_value` (default 0.0).
      - If drop_all_zero_features is True, removes columns that are all zeros across participants.
    """
    rows: List[Dict[str, float]] = []
    pids: List[Any] = []

    for pid, model_dict in results_by_pid.items():
        res = model_dict[model_name]
        coef_df = getattr(res, "coef_summary", None)

        tmp = coef_df[["feature", coef_col]].copy()
        tmp = tmp.dropna(subset=["feature", coef_col])
        tmp[coef_col] = pd.to_numeric(tmp[coef_col], errors="coerce")
        tmp = tmp.dropna(subset=[coef_col])

        series = tmp.drop_duplicates(subset=["feature"], keep="last").set_index("feature")[coef_col]

        row_dict = series.to_dict()
        rows.append(row_dict)
        pids.append(pid)


    mat = pd.DataFrame(rows, index=pids).fillna(float(fill_value))
    mat.index.name = participant_col_name

    if drop_all_zero_features:
        nonzero_cols = (mat.abs().sum(axis=0) > 0)
        mat = mat.loc[:, nonzero_cols]

    mat = mat.reindex(sorted(mat.columns), axis=1)
    return mat


def zscore_coef_matrix_across_participants(
    coef_matrix: pd.DataFrame,
    with_mean: bool = True,
    with_std: bool = True,
) -> pd.DataFrame:
    """
    Z-score each feature column across participants.

    Returns a DataFrame with same shape/index/columns.
    """
    scaler = StandardScaler(with_mean=with_mean, with_std=with_std)
    z = scaler.fit_transform(coef_matrix.values)
    return pd.DataFrame(z, index=coef_matrix.index, columns=coef_matrix.columns)


def build_participant_clustering_inputs(
    results_by_pid: Mapping[Any, Mapping[str, Any]],
    model_name: str,
    coef_col: str = "coef",
    fill_value: float = 0.0,
    drop_all_zero_features: bool = True,
    zscore: bool = True,
) -> ParticipantClusteringInputs:
    """
    Convenience wrapper:
      - builds participant x feature coefficient matrix
      - optionally z-scores across participants
    """
    mat = build_participant_coef_matrix(
        results_by_pid=results_by_pid,
        model_name=model_name,
        coef_col=coef_col,
        fill_value=fill_value,
        drop_all_zero_features=drop_all_zero_features,
    )

    if zscore:
        mat_z = zscore_coef_matrix_across_participants(mat)
    else:
        mat_z = mat.copy()

    return ParticipantClusteringInputs(coef_matrix=mat, coef_matrix_z=mat_z)



def compute_participant_distance_matrix(
    coef_matrix: pd.DataFrame,
    metric: Metric = "cosine",
) -> pd.DataFrame:
    """
    Compute participant x participant distance matrix from coefficient vectors.

    Returns square DataFrame with same index/columns = participant IDs.
    """
    X = coef_matrix.values
    if metric == "cosine":
        D = cosine_distances(X)
    elif metric == "euclidean":
        D = euclidean_distances(X)
    else:
        raise ValueError(f"Unsupported metric: {metric}")

    return pd.DataFrame(D, index=coef_matrix.index, columns=coef_matrix.index)


#
# def compute_participant_cosine_similarity(
#     coef_matrix: pd.DataFrame,
# ) -> pd.DataFrame:
#     """
#     cosine similarity = 1 - cosine distance.
#     """
#     D = compute_participant_distance_matrix(coef_matrix, metric="cosine")
#     S = 1.0 - D
#     return S



def get_top_features_for_participant(
    coef_matrix: pd.DataFrame,
    participant_id: Any,
    top_k: int = 15,
    by: Literal["abs", "pos", "neg"] = "abs",
) -> pd.Series:
    """
    return the strongest features for a participant from a coefficient matrix.

    by:
      - "abs": largest absolute coefficients
      - "pos": largest positive coefficients
      - "neg": most negative coefficients (sorted ascending)
    """
    if participant_id not in coef_matrix.index:
        raise KeyError(f"participant_id {participant_id!r} not in coef_matrix index.")

    s = coef_matrix.loc[participant_id]

    if by == "abs":
        return s.reindex(s.abs().sort_values(ascending=False).head(top_k).index)
    if by == "pos":
        return s.sort_values(ascending=False).head(top_k)
    if by == "neg":
        return s.sort_values(ascending=True).head(top_k)

    raise ValueError(f"Unsupported 'by' value: {by}")
