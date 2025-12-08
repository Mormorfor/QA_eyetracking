# features.py

from typing import Sequence
import pandas as pd
from src import constants as Con


def map_last_location_to_position(
    series: pd.Series,
    choices: Sequence[str] = Con.AREA_LABEL_CHOICES,
) -> pd.Series:
    """
    Map last_area_visited_loc values to numeric positions 0–3.
    index 1 → answer position 0, index 2 → 1, index 3 → 2, index 4 → 3.
    """
    mapping = {}
    for idx, label in enumerate(choices):
        if idx == 0:
            continue
        mapping[label] = idx - 1
    return series.map(mapping)
