"""
Loading the trial tables the correlation maps run on.

Three tables, one row per trial, all built the same way: all participants,
hunters only, gatherers only. They are Study 1's model-ready tables, the same
files `scripts/build_dataset.py` writes (under `data/datasets/l1_onestop/`):

    all        features/model_ready.csv
    hunters    features/model_ready_hunters.csv
    gatherers  features/model_ready_gatherers.csv

`load_features` reads one. If it is missing, or on `rebuild=True`, it first
builds it from that group's word, area and trial tables
(`features.build.load_model_input`, then `save_model_ready`) and saves it in
place. Building takes seconds.
"""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import pandas as pd

from src.config.datasets import dataset as _dataset

__all__ = [
    "FEATURE_SOURCES",
    "GROUP_NAMES",
    "load_features",
    "load_all_participants_features",
    "load_group_features",
]


class FeatureSource(NamedTuple):
    """Which group a table holds, and where it is saved."""

    processed: "str | None"  # the group whose tables it is built from; None = everyone
    features: Path  # the saved model-ready CSV


def _features_path(name: str) -> Path:
    """Where the model-ready table for `name` ("all", "hunters", "gatherers") is saved."""
    ds = _dataset("l1")
    return ds.model_ready if name == "all" else ds.features / f"model_ready_{name}.csv"


#: The three tables `load_features` can load: which group each holds, and where
#: it is saved.
FEATURE_SOURCES: dict[str, FeatureSource] = {
    "all": FeatureSource(None, _features_path("all")),
    "hunters": FeatureSource("hunters", _features_path("hunters")),
    "gatherers": FeatureSource("gatherers", _features_path("gatherers")),
}

GROUP_NAMES = ["hunters", "gatherers"]


def load_features(
    name: str = "all", rebuild: bool = False, verbose: bool = True
) -> pd.DataFrame:
    """The model-ready table for `name` ("all", "hunters" or "gatherers"): one
    row per trial, with every feature.

    A missing table is built and saved first; `rebuild=True` rebuilds and saves
    it even if it exists. It is built with the same call as in
    `scripts/build_dataset.py` and saved over the same file, and the "all" table
    is the one the rest of the project reads, so a rebuild here replaces it for
    every other analysis too.
    """
    if name not in FEATURE_SOURCES:
        raise KeyError(
            f"unknown feature set {name!r}; expected one of {list(FEATURE_SOURCES)}"
        )
    source = FEATURE_SOURCES[name]

    if rebuild or not source.features.exists():
        # Imported only when needed: building is rare and pulls in the
        # feature-building code.
        from src.features.build import save_model_ready

        from src.features.build import load_model_input

        if verbose:
            print(f"building {name} features from the grain tables")
        save_model_ready(
            load_model_input(_dataset("l1"), facet=source.processed, verbose=verbose),
            output_path=source.features,
        )

    out = pd.read_csv(source.features)
    if verbose:
        print(f"{name}: {out.shape}")
    return out


def load_all_participants_features(
    rebuild: bool = False, verbose: bool = True
) -> pd.DataFrame:
    """The table for every participant, both groups: `load_features("all")`.

    This is `features/model_ready.csv`, the same file
    `features.build.load_model_ready` reads by default.
    """
    return load_features("all", rebuild=rebuild, verbose=verbose)


def load_group_features(
    rebuild: bool = False, verbose: bool = True
) -> dict[str, pd.DataFrame]:
    """The two groups' tables: `{"hunters": DataFrame, "gatherers": DataFrame}`."""
    return {
        name: load_features(name, rebuild=rebuild, verbose=verbose)
        for name in GROUP_NAMES
    }
