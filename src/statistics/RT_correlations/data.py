"""
Loading the frames the correlation maps run on.

Three frames, built the same way from the same code path: all participants,
hunters only, gatherers only. Each is a cached model-ready feature CSV derived
from the corresponding processed CSV by
`answer_correctness.model_data.save_all_features`:

    all_participants.csv  ->  L1_model_ready_all_features.csv
    hunters.csv           ->  L1_model_ready_hunters_features.csv
    gatherers.csv         ->  L1_model_ready_gatherers_features.csv

`load_features` builds a cache that is missing and rebuilds one on request, for
any of the three. Rebuilding is not cheap -- the processed CSVs are 1.5-3 GB --
so it only happens when the cache is absent or `rebuild=True`.
"""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import pandas as pd

from src.data_paths import (
    ALL_PARTICIPANTS_PROCESSED_PATH,
    GATHERERS_PROCESSED_PATH,
    HUNTERS_PROCESSED_PATH,
    L1_BASED_DATA_DIR,
    READY_ALL_FEATURES_PATH,
)

__all__ = [
    "FEATURE_SOURCES",
    "GROUP_NAMES",
    "load_features",
    "load_all_features",
    "load_group_features",
]


class FeatureSource(NamedTuple):
    """Where a frame is cached, and what it is built from if the cache is gone."""

    processed: Path  # the big processed CSV
    features: Path  # the cached model-ready CSV


def _features_path(name: str) -> Path:
    return L1_BASED_DATA_DIR / f"L1_model_ready_{name}_features.csv"


# `_features_path("all")` is READY_ALL_FEATURES_PATH; the constant is used
# directly so the dependency on data_paths stays visible.
FEATURE_SOURCES: dict[str, FeatureSource] = {
    "all": FeatureSource(ALL_PARTICIPANTS_PROCESSED_PATH, READY_ALL_FEATURES_PATH),
    "hunters": FeatureSource(HUNTERS_PROCESSED_PATH, _features_path("hunters")),
    "gatherers": FeatureSource(GATHERERS_PROCESSED_PATH, _features_path("gatherers")),
}

GROUP_NAMES = ["hunters", "gatherers"]


def load_features(
    name: str = "all", rebuild: bool = False, verbose: bool = True
) -> pd.DataFrame:
    """Trial-level model-ready features for one of `FEATURE_SOURCES`.

    Parameters
    ----------
    name : "all", "hunters" or "gatherers".
    rebuild : regenerate the cached CSV from the processed one even if it
        exists. Reads a 1.5-3 GB CSV and takes a while.

    Returns
    -------
    DataFrame, one row per (participant_id, TRIAL_INDEX).
    """
    if name not in FEATURE_SOURCES:
        raise KeyError(
            f"unknown feature set {name!r}; expected one of {list(FEATURE_SOURCES)}"
        )
    source = FEATURE_SOURCES[name]

    if rebuild or not source.features.exists():
        # Imported lazily: building is rare and pulls in the modelling stack.
        from src.predictive_modeling.answer_correctness.model_data import (
            save_all_features,
        )

        if verbose:
            print(f"building {name} features from {source.processed}")
        save_all_features(
            pd.read_csv(source.processed), output_path=source.features
        )

    out = pd.read_csv(source.features)
    if verbose:
        print(f"{name}: {out.shape}")
    return out


def load_all_features(rebuild: bool = False, verbose: bool = True) -> pd.DataFrame:
    """Features for all participants."""
    return load_features("all", rebuild=rebuild, verbose=verbose)


def load_group_features(
    rebuild: bool = False, verbose: bool = True
) -> dict[str, pd.DataFrame]:
    """Features per reading regime.

    Returns
    -------
    {"hunters": DataFrame, "gatherers": DataFrame}
    """
    return {
        name: load_features(name, rebuild=rebuild, verbose=verbose)
        for name in GROUP_NAMES
    }
