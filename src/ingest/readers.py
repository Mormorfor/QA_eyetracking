"""Loading the raw interest-area reports."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config.datasets import IA_ANSWERS_PATH, IA_PARAGRAPH_PATH


def load_raw_answers_data(ia_a_path: Path = IA_ANSWERS_PATH, usecols=None):
    """
    Load raw interest area level answers data from CSV file.

    `usecols` is the read-time column whitelist (stage D section 3.4, proved and
    switched on 2026-10-13). `None` reads every column, which is what the
    two-run comparison in step 2 of the grain proposal needs in order to stay
    repeatable -- so this stays a parameter rather than becoming unconditional.

    The whitelist itself is **not built here.** `ingest/registry.py` owns it, and
    registry imports `features/`; building it here would drag that dependency
    into a primitive and break the one layering rule this package actually
    enforces -- only the orchestration modules reach downstream. So `build.main`
    computes it and passes it in.
    """
    return pd.read_csv(ia_a_path, engine="python", usecols=usecols)


def load_raw_paragraphs_data(ia_p_path: Path = IA_PARAGRAPH_PATH):
    """
    Load raw interest area level paragraphs data from CSV file.
    """
    return pd.read_csv(ia_p_path, engine="python")
