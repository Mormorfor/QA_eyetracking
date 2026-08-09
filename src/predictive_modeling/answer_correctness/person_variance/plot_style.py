"""Shared colours and label helpers for the per-person variance figures.

Every figure in this package encodes the same two meanings with the same two
colours -- blue = positive / "helps", orange = negative / "hurts" -- so a bar,
a box and a stacked share all read the same way across sections.
"""

from __future__ import annotations

from typing import Iterable, List, Sequence

# Direction encoding, shared by every figure in the package.
POS_COLOR = "#4C72B0"   # positive coefficient / association ("helps")
NEG_COLOR = "#DD8452"   # negative coefficient / association ("hurts")
ZERO_COLOR = "#D9D9D9"  # ~zero coefficient (feature inactive for that person)

# Confusion quadrants (see ``mistake_types``): hits in cool colours, errors warm.
QUAD_COLORS = {
    "TP": "#4C72B0",
    "TN": "#55A868",
    "FP": "#DD8452",
    "FN": "#C44E52",
}


def signed_bar_colors(values: Iterable[float]) -> List[str]:
    """Orange for negative values, blue for zero/positive ones."""
    return [NEG_COLOR if v < 0 else POS_COLOR for v in values]


def clean_feature_labels(features: Sequence[str]) -> List[str]:
    """``area_dwell_proportion__correct`` -> ``area_dwell_proportion . correct``."""
    return [str(f).replace("__", " · ") for f in features]
