# prepared_dataset.py

from dataclasses import dataclass
from typing import List
import pandas as pd


@dataclass
class PreparedTrialDataset:
    """
    Container for a ready-to-model trial-level dataset.
    """
    df: pd.DataFrame
    feature_cols: List[str]
    target_col: str
    id_cols: List[str]