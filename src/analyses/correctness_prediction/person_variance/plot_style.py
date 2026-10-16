"""Colours and label helpers shared by the figures in this folder.

Wherever colour marks the sign of a coefficient or a correlation, blue is positive
("helps") and orange negative ("hurts"), so a bar, a box and a stacked share read
the same way in every figure. In two figures the same colours only tell two series
apart: the two accuracy histograms, and the sensitivity and specificity bars. The
figures drawn by the shared helpers in `plots/` use their own colours.
"""

from __future__ import annotations

from typing import Iterable, List, Sequence

# Sign colours.
POS_COLOR = "#4C72B0"   # positive coefficient / association ("helps")
NEG_COLOR = "#DD8452"   # negative coefficient / association ("hurts")
ZERO_COLOR = "#D9D9D9"  # zero coefficient (feature never varied for that person)

# Confusion quadrants (see `mistake_types`): hits in cool colours, errors warm.
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
    """Axis labels with `__` shown as ` · `: `skip_rate__correct` -> `skip_rate · correct`."""
    return [str(f).replace("__", " · ") for f in features]
