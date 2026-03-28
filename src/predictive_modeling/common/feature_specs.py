# feature_specs.py

from typing import List
import pandas as pd

from src import constants as Con


DERIVED_BASE_FEATURES = [
    "seq_len",
    "has_xyx",
    "has_xyxy",
    "trial_mean_dwell",
]


def get_area_feature_cols(df: pd.DataFrame) -> List[str]:
    """
    Return area-based feature columns that are present in df.

    Includes:
      - <metric>__<area>
      - <metric>__correct
      - <metric>__wrong_mean
      - <metric>__contrast
      - <metric>__distance_furthest
      - <metric>__distance_closest
    """
    cols: List[str] = []

    area_labels = list(Con.LABEL_CHOICES)
    derived_suffixes = [
        Con.CORRECT_SUFFIX,
        Con.WRONG_MEAN_SUFFIX,
        Con.CONTRAST_SUFFIX,
        Con.DISTANCE_FURTHEST_SUFFIX,
        Con.DISTANCE_CLOSEST_SUFFIX,
    ]

    for metric in Con.AREA_METRIC_COLUMNS_MODELING:
        for area in area_labels:
            col = f"{metric}__{area}"
            if col in df.columns:
                cols.append(col)

        for suffix in derived_suffixes:
            col = f"{metric}__{suffix}"
            if col in df.columns:
                cols.append(col)

    return cols


def get_derived_feature_cols(df: pd.DataFrame) -> List[str]:
    """
    Return derived trial-level feature columns that are present in df.
    """
    cols: List[str] = []

    cols.extend([c for c in DERIVED_BASE_FEATURES if c in df.columns])
    cols.extend(sorted(c for c in df.columns if c.startswith("pref_matching__")))

    return cols


def get_last_visited_feature_cols(df: pd.DataFrame) -> List[str]:
    """
    Return one-hot last-visited feature columns.
    """
    return sorted(c for c in df.columns if c.startswith("last_visited_"))


def get_full_feature_cols(df: pd.DataFrame) -> List[str]:
    """
    Return the full model feature set:
      - area features
      - derived features
      - last visited one-hot features
    """
    cols: List[str] = []
    cols.extend(get_area_feature_cols(df))
    cols.extend(get_derived_feature_cols(df))
    cols.extend(get_last_visited_feature_cols(df))
    return cols