"""
Which columns make up a correlation map, and how to label them.

A map has the paragraph's three parts on the rows and the four answers and the
question on the columns, for one metric (`RT` or `TFD`) and one scaling
(`normalized` = divided by the area's number of words, `pure` = not divided).
The column names follow `{metric}_{scaling}_{area}`, e.g. `RT_normalized_critical`.
"""

from __future__ import annotations

from typing import Sequence

__all__ = [
    "REGIONS",
    "ANSWERS",
    "METRICS",
    "SCALINGS",
    "region_col",
    "answer_col",
    "region_cols",
    "answer_cols",
    "map_cell",
    "pretty_labels",
]

REGIONS = ["distractor", "critical", "outside"]
ANSWERS = ["answer_A", "answer_B", "answer_C", "answer_D", "question"]
METRICS = ["RT", "TFD"]
SCALINGS = ["normalized", "pure"]

# Stripped from a column name before it becomes an axis label. The dwell
# proportions (`proportions.py`) are not one of the METRICS x SCALINGS columns
# but label the same rows and columns, so their prefix is stripped too.
_PREFIXES = tuple(f"{m}_{s}_" for m in METRICS for s in SCALINGS) + (
    "area_dwell_proportion__",
)


def _col(metric: str, part: str, scaling: str = "normalized") -> str:
    return f"{metric}_{scaling}_{part}"


def region_col(region: str, metric: str = "RT", scaling: str = "normalized") -> str:
    """One row of a map, e.g. `region_col("critical")` -> `RT_normalized_critical`."""
    return _col(metric, region, scaling)


def answer_col(answer: str, metric: str = "RT", scaling: str = "normalized") -> str:
    """One column of a map, e.g. `answer_col("answer_A")` -> `RT_normalized_answer_A`."""
    return _col(metric, answer, scaling)


def region_cols(metric: str = "RT", scaling: str = "normalized") -> list[str]:
    """The rows of a map: the three paragraph parts, e.g. `RT_normalized_distractor`."""
    return [_col(metric, r, scaling) for r in REGIONS]


def answer_cols(metric: str = "RT", scaling: str = "normalized") -> list[str]:
    """The columns of a map: four answers and the question, e.g. `RT_normalized_answer_A`."""
    return [_col(metric, a, scaling) for a in ANSWERS]


def map_cell(
    region: str, answer: str, metric: str = "RT", scaling: str = "normalized"
) -> tuple[str, str]:
    """The key of one cell of a map, as `compare_cells` takes it.

    >>> map_cell("critical", "answer_A")
    ('RT_normalized_critical', 'RT_normalized_answer_A')
    """
    return _col(metric, region, scaling), _col(metric, answer, scaling)


def pretty_labels(cols: Sequence[str]) -> list[str]:
    """Axis labels: drop the metric prefix, write `answer_A` as `ans A`, `_` as a space."""
    out = []
    for c in cols:
        s = str(c)
        for pref in _PREFIXES:
            s = s.replace(pref, "")
        out.append(s.replace("answer_", "ans ").replace("_", " "))
    return out
