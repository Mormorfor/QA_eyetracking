# src/predictive_modeling/answer_correctness/answer_correctness_participant_similarity.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional, Dict, List, Callable, Literal

import pandas as pd
from sklearn.preprocessing import StandardScaler

from src import constants as Con


# ---------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------

Metric = Literal["cosine", "euclidean"]
FamilyRule = Callable[[str], bool]
FamilyAgg = Literal["sum_abs", "mean_abs", "sum_signed", "mean_signed"]


# ---------------------------------------------------------------------
# Core container
# ---------------------------------------------------------------------

@dataclass
class ParticipantClusteringInputs:
    """
    coef_matrix:
        participant x (feature or family)
    coef_matrix_z:
        z-scored across participants (per column)
    """
    coef_matrix: pd.DataFrame
    coef_matrix_z: pd.DataFrame


# ---------------------------------------------------------------------
# Coefficient matrix building
# ---------------------------------------------------------------------

def build_participant_coef_matrix(
    results_by_pid: Mapping[Any, Mapping[str, Any]],
    model_name: str,
    coef_col: str = "coef",
    participant_col_name: str = Con.PARTICIPANT_ID,
    fill_value: float = 0.0,
    drop_all_zero_features: bool = True,
) -> pd.DataFrame:
    """
    Build participant x feature coefficient matrix from results_by_pid.

    """
    rows: List[Dict[str, float]] = []
    pids: List[Any] = []

    for pid, model_dict in results_by_pid.items():
        res = model_dict[model_name]
        coef_df = res.coef_summary

        tmp = coef_df[["feature", coef_col]].copy()
        tmp = tmp.dropna(subset=["feature", coef_col])
        tmp[coef_col] = pd.to_numeric(tmp[coef_col], errors="coerce")
        tmp = tmp.dropna(subset=[coef_col])

        s = (
            tmp.drop_duplicates(subset=["feature"], keep="last")
            .set_index("feature")[coef_col]
        )

        rows.append(s.to_dict())
        pids.append(pid)

    mat = pd.DataFrame(rows, index=pids).fillna(float(fill_value))
    mat.index.name = participant_col_name

    if drop_all_zero_features:
        mat = mat.loc[:, (mat.abs().sum(axis=0) > 0)]

    return mat.reindex(sorted(mat.columns), axis=1)


def zscore_across_participants(
    mat: pd.DataFrame,
    with_mean: bool = True,
    with_std: bool = True,
) -> pd.DataFrame:
    """
    Z-score each column across participants.
    """
    scaler = StandardScaler(with_mean=with_mean, with_std=with_std)
    z = scaler.fit_transform(mat.values)
    return pd.DataFrame(z, index=mat.index, columns=mat.columns)


def build_participant_clustering_inputs(
    results_by_pid: Mapping[Any, Mapping[str, Any]],
    model_name: str,
    coef_col: str = "coef",
    zscore: bool = True,
) -> ParticipantClusteringInputs:
    """
    Builds participant x feature coef matrix (+ optional z-scored version).
    """
    mat = build_participant_coef_matrix(
        results_by_pid=results_by_pid,
        model_name=model_name,
        coef_col=coef_col,
    )
    mat_z = zscore_across_participants(mat) if zscore else mat.copy()
    return ParticipantClusteringInputs(coef_matrix=mat, coef_matrix_z=mat_z)


# ---------------------------------------------------------------------
# Feature-family definitions + aggregation
# ---------------------------------------------------------------------

def default_feature_families() -> Dict[str, FamilyRule]:
    """
    Family rules based on your feature naming conventions.
    First match wins (insertion order matters).
    """
    return {
        "matching":   lambda f: f.startswith("pref_matching__"),
        "pupil":      lambda f: "pupil" in f.lower(),
        "dwell":      lambda f: "dwell" in f.lower(),
        "skip":       lambda f: "skip" in f.lower(),
        "fixation":   lambda f: "fix" in f.lower(),
        "sequence":   lambda f: f in {"seq_len", "has_xyx", "has_xyxy"},
        "other":      lambda f: True,
    }


def assign_feature_family(feature: str, families: Dict[str, FamilyRule]) -> str:
    for fam, rule in families.items():
        if rule(feature):
            return fam
    return "other"


def build_participant_family_coef_matrix(
    coef_matrix: pd.DataFrame,
    families: Optional[Dict[str, FamilyRule]] = None,
    agg: FamilyAgg = "sum_abs",
    include_counts: bool = False,
) -> pd.DataFrame:
    """
    Collapse participant x feature -> participant x family.
    """
    families = families or default_feature_families()
    f2fam = {f: assign_feature_family(f, families) for f in coef_matrix.columns}

    fam_names = list(dict.fromkeys(f2fam.values()))  # stable
    out = pd.DataFrame(index=coef_matrix.index)

    for fam in fam_names:
        cols = [c for c, ff in f2fam.items() if ff == fam]
        block = coef_matrix[cols]

        if agg == "sum_abs":
            out[fam] = block.abs().sum(axis=1)
        elif agg == "mean_abs":
            out[fam] = block.abs().mean(axis=1)
        elif agg == "sum_signed":
            out[fam] = block.sum(axis=1)
        elif agg == "mean_signed":
            out[fam] = block.mean(axis=1)
        else:
            raise ValueError(f"Unknown agg: {agg}")

        if include_counts:
            out[f"{fam}__n_features"] = len(cols)

    return out.reindex(sorted(out.columns), axis=1)


def build_participant_family_clustering_inputs(
    results_by_pid: Mapping[Any, Mapping[str, Any]],
    model_name: str,
    coef_col: str = "coef",
    families: Optional[Dict[str, FamilyRule]] = None,
    family_agg: FamilyAgg = "sum_abs",
    zscore: bool = True,
) -> ParticipantClusteringInputs:
    """
    Builds participant x family coef matrix (+ optional z-scored version).
    """
    feat_mat = build_participant_coef_matrix(
        results_by_pid=results_by_pid,
        model_name=model_name,
        coef_col=coef_col,
    )
    fam_mat = build_participant_family_coef_matrix(
        coef_matrix=feat_mat,
        families=families,
        agg=family_agg,
    )
    fam_z = zscore_across_participants(fam_mat) if zscore else fam_mat.copy()
    return ParticipantClusteringInputs(coef_matrix=fam_mat, coef_matrix_z=fam_z)
