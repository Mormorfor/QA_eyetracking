# answer_loc_models.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Protocol, Sequence
import numpy as np
import pandas as pd
from src import constants as Con


class AnswerLocationModel(Protocol):
    """
    Minimal interface that all answer-location models should implement.
    """
    name: str

    def fit(self, train_df: pd.DataFrame, target_col: str) -> None:
        ...

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        ...


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


@dataclass
class LastLocationBaseline:
    name: str = "last_location"
    last_loc_col: str = Con.LAST_VISITED_LOCATION

    def fit(self, train_df: pd.DataFrame, target_col: str) -> None:
        # nothing to learn for this baseline
        return

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        if self.last_loc_col not in df.columns:
            raise KeyError(f"Column '{self.last_loc_col}' not found in df.")

        loc_series = df[self.last_loc_col]
        pos_series = map_last_location_to_position(loc_series)

        return pos_series.fillna(-1).astype(int).to_numpy()
