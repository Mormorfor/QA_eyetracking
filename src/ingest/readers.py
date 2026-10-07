"""Loading the raw interest-area reports."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config.datasets import IA_ANSWERS_PATH, IA_PARAGRAPH_PATH


def load_raw_answers_data(ia_a_path: Path = IA_ANSWERS_PATH):
    """
    Load raw interest area level answers data from CSV file.
    """
    return pd.read_csv(ia_a_path, engine="python")


def load_raw_paragraphs_data(ia_p_path: Path = IA_PARAGRAPH_PATH):
    """
    Load raw interest area level paragraphs data from CSV file.
    """
    return pd.read_csv(ia_p_path, engine="python")
