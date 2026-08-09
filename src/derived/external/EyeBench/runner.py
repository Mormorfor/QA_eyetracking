"""Driver for the lab's EyeBench trial-level paragraph feature extraction.

The extraction itself lives in `src/external/EyeBench/` (vendored source plus our
adapter `paragraph_trial_features.py`); this module is only the orchestration
that used to sit in the notebook: pick the participants, decide between the
cached CSV and a rebuild, and read back the feature -> model-family key files.

A full build reads two multi-GB reports and takes ~1.5 h, so
`get_paragraph_trial_features` loads `PARAGRAPH_TRIAL_FEATURES_PATH` whenever it
exists and only ever writes it when `rebuild=True` is passed explicitly.

Usage:
    from src.derived.external.EyeBench.runner import (
        get_paragraph_trial_features,
        feature_key_summary,
    )

    trial_features = get_paragraph_trial_features()
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import pandas as pd

from src.data_paths import (
    PARAGRAPH_TRIAL_FEATURES_DIR,
    PARAGRAPH_TRIAL_FEATURES_PATH,
    READY_ALL_FEATURES_PATH,
)
from src.external.EyeBench.paragraph_trial_features import (
    build_paragraph_trial_level_features,
    load_paragraph_trial_level_features,
    save_paragraph_trial_level_features,
)

__all__ = [
    "get_paragraph_trial_features",
    "quick_test_participants",
    "load_feature_keys",
    "feature_key_summary",
]

# Where a `quick_test` run writes its feature-key files, so a smoke run cannot
# clobber the key files that belong to the cached full extraction.
QUICK_TEST_SUBDIR = "_quick_test"

FEATURE_KEY_FILES = (
    "ia_trial_level_feature_keys.csv",
    "fixation_trial_level_feature_keys.csv",
)


def quick_test_participants(
    n: int, participants_path: Path = READY_ALL_FEATURES_PATH
) -> list[str]:
    """The first `n` participant ids, for a fast smoke run of the extraction."""
    ids = pd.read_csv(Path(participants_path), usecols=["participant_id"])[
        "participant_id"
    ].unique()
    return sorted(ids)[:n]


def get_paragraph_trial_features(
    rebuild: bool = False,
    quick_test: int = 0,
    include_fixation_features: bool = True,
    fix_ptb_pos_double_mapping: bool = True,
    reconstruct_missing_columns: bool = True,
    participants: Optional[Sequence[str]] = None,
    cache_path: Path = PARAGRAPH_TRIAL_FEATURES_PATH,
    features_dir: Path = PARAGRAPH_TRIAL_FEATURES_DIR,
    verbose: bool = True,
) -> pd.DataFrame:
    """Trial-level paragraph reading features, one row per (participant, trial).

    Reads the cached CSV when there is one; otherwise runs the extraction (~1.5 h
    on the full reports) and caches the result.

    Args:
        rebuild: re-run the extraction and overwrite `cache_path` even though it
            already exists. The only code path that writes over the cache.
        quick_test: >0 restricts the run to that many participants and neither
            reads nor writes the cache -- a smoke run. Its feature-key files go
            to `features_dir / QUICK_TEST_SUBDIR` so the real ones survive.
        include_fixation_features: False gives the IA-only half, much faster,
            without the `fix_feature_*` / gaze-entropy / `gsf` columns.
        fix_ptb_pos_double_mapping: True gives the 80 `ptb_pos_*` columns real
            values; False reproduces the source pipeline, where they come out
            all-zero because `ptb_pos` is mapped to numbers twice.
        reconstruct_missing_columns: fill the IA columns the source feature list
            wants but our report names differently or lacks (+35 columns).
        participants: restrict a (non-quick-test) run to these participant ids.
        cache_path: CSV holding the built feature table.
        features_dir: directory the extraction writes its two
            `*_trial_level_feature_keys.csv` files to.

    Returns:
        The feature table. Built runs are indexed by (participant_id,
        TRIAL_INDEX); a run loaded from cache carries those as plain columns.
    """
    build_kwargs = dict(
        include_fixation_features=include_fixation_features,
        fix_ptb_pos_double_mapping=fix_ptb_pos_double_mapping,
        reconstruct_missing_columns=reconstruct_missing_columns,
    )
    cache_path = Path(cache_path)
    features_dir = Path(features_dir)

    if quick_test:
        features = build_paragraph_trial_level_features(
            participants=quick_test_participants(quick_test),
            processed_data_path=features_dir / QUICK_TEST_SUBDIR,
            verbose=verbose,
            **build_kwargs,
        )
    elif rebuild or not cache_path.exists():
        features = save_paragraph_trial_level_features(
            output_path=cache_path,
            participants=participants,
            processed_data_path=features_dir,
            verbose=verbose,
            **build_kwargs,
        )
    else:
        features = load_paragraph_trial_level_features(cache_path)

    if verbose:
        print("trial-level paragraph features:", features.shape)
    return features


def load_feature_keys(features_dir: Path = PARAGRAPH_TRIAL_FEATURES_DIR) -> pd.DataFrame:
    """The feature -> model-family tables written alongside the extraction.

    One row per feature, with `source` ("ia" / "fixation") added from the file it
    came from. Missing files are skipped, so an IA-only run still works.
    """
    features_dir = Path(features_dir)
    tables = [
        pd.read_csv(features_dir / name).assign(source=name.split("_")[0])
        for name in FEATURE_KEY_FILES
        if (features_dir / name).exists()
    ]
    if not tables:
        raise FileNotFoundError(
            f"no *_trial_level_feature_keys.csv in {features_dir}; run the "
            "extraction first"
        )
    return pd.concat(tables, ignore_index=True)


def feature_key_summary(
    features_dir: Path = PARAGRAPH_TRIAL_FEATURES_DIR,
    feature_keys: Optional[pd.DataFrame] = None,
) -> pd.Series:
    """How many features each (source, model family) contributes."""
    if feature_keys is None:
        feature_keys = load_feature_keys(features_dir)
    return feature_keys.groupby(["source", "feature_type"]).size().rename("n_features")
