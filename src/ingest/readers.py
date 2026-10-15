"""Load the raw interest-area reports (one row per word)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config.datasets import IA_ANSWERS_PATH, IA_PARAGRAPH_PATH


def load_raw_answers_data(ia_a_path: Path = IA_ANSWERS_PATH, usecols=None):
    """Load the answer screen's interest-area report.

    `usecols` is the list of columns to read; `None` reads all of them. The list
    is worked out by `ingest/build.py` (from `registry.raw_whitelist`) and passed
    in, so that this file does not depend on the registry.
    """
    return pd.read_csv(ia_a_path, engine="python", usecols=usecols)


def load_raw_paragraphs_data(ia_p_path: Path = IA_PARAGRAPH_PATH):
    """Load the paragraph screen's interest-area report (every column)."""
    return pd.read_csv(ia_p_path, engine="python")
