"""
Loading the frames the correlation maps run on.

Three frames, built the same way from the same code path: all participants,
hunters only, gatherers only. Each is a cached model-ready feature CSV derived
from the corresponding processed CSV by
`features.build.save_model_ready`:

    interim/all_participants.csv  ->  features/model_ready.csv
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

from src.config.datasets import dataset as _dataset

__all__ = [
    "FEATURE_SOURCES",
    "GROUP_NAMES",
    "load_features",
    "load_all_participants_features",
    "load_group_features",
]


class FeatureSource(NamedTuple):
    """Where a frame is cached, and what it is built from if the cache is gone."""

    processed: "str | None"  # the group facet to load, or None for all
    features: Path  # the cached model-ready CSV


def _features_path(name: str) -> Path:
    """Where the cached model-ready table for one group lives.

    **Corrected 2026-10-13.** This returned
    `L1_model_ready_{name}_features.csv` at the dataset root -- the
    pre-restructure name, which stage G renamed to `features/model_ready_*.csv`.
    Nothing noticed because the miss is silent: `load_features` treats an absent
    cache as "build it", so every call to `load_features("hunters")` was
    rebuilding the group table from the 1.5 GB `hunters.csv` instead of reading a
    20 MB cache that was sitting right there under a different name.
    """
    ds = _dataset("l1")
    return ds.model_ready if name == "all" else ds.features / f"model_ready_{name}.csv"


#: What each frame is built from, and where it is cached.
#:
#: The sources are the **grain tables**, not the broadcast ones. `load_model_input`
#: reads `interest_areas` + `areas_answers` + `trials` and rejoins them, which is
#: the same frame `all_participants.csv` held -- verified equivalent on all 121
#: shared columns, 0 value / null / dtype differences -- at 4 s instead of 64 s,
#: and without the 5.8 GB those three files took.
FEATURE_SOURCES: dict[str, FeatureSource] = {
    "all": FeatureSource(None, _features_path("all")),
    "hunters": FeatureSource("hunters", _features_path("hunters")),
    "gatherers": FeatureSource("gatherers", _features_path("gatherers")),
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
    """The pooled frame -- every participant, both groups.

    Named in full to pair with `load_group_features` and to stop colliding with
    `features.build.load_model_ready`, which reads a different artifact: this one
    is one of three group-scoped caches, that one is a dataset's model-ready
    table. Renamed 2026-10-07 (stage D step 7).
    """
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
