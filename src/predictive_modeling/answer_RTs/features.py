# features.py
#
# Paragraph-span features for the answer reading-time (answer_RTs) regression.
#
# THIS MODULE NO LONGER BUILDS ANYTHING. It is a compatibility surface.
#
# The extraction it used to contain moved to `src/derived/paragraph_prep.py` on
# 2026-09-23 (`todo.md` T6.1 and T1.7). Two reasons:
#
# 1. **It was in the wrong place.** Paragraph feature extraction lived inside a
#    *modelling* module for a strand that is parked, while being load-bearing
#    for two things that are not: the correctness model's paragraph dwell
#    proportions, and `statistics/RT_correlations` -- the current text-QA
#    analysis. Retiring `answer_RTs/` as a modelling strand must not retire the
#    extraction with it.
# 2. **It was a second implementation of the eight per-area metrics.** The
#    answer screen had its own set in `data_prep/data_csv_generation.py`. That
#    duplication is how the two came to disagree on
#    `mean_first_fixation_duration` (T3.6) while this file asserted they "line up
#    1:1". There is now one implementation, `derived/area_metrics.py`,
#    parameterized by the grouping column.
#
# The names below are re-exported so existing callers keep working unchanged --
# `answer_RTs/model_data.py` and `statistics/RT_correlations/proportions.py`
# both import from here.

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from src.data_paths import PARAGRAPH_SPAN_FEATURES_PATH
from src.derived.paragraph_prep import (  # noqa: F401  (re-exported)
    PARAGRAPH_METRIC_COLUMNS,
    QUESTION_PREVIEW_COL,
    SPAN_COL as PARAGRAPH_SPAN_COL,
    SPANS as PARAGRAPH_SPANS,
    build_paragraph_features,
    save_paragraph_features,
)


def build_trial_level_paragraph_features(*args, **kwargs) -> pd.DataFrame:
    """Deprecated alias for `derived.paragraph_prep.build_paragraph_features`."""
    return build_paragraph_features(*args, **kwargs)


def load_paragraph_features(
    path: Path = PARAGRAPH_SPAN_FEATURES_PATH,
) -> pd.DataFrame:
    """Load the cached paragraph features produced by `save_paragraph_features`.

    Kept here (rather than re-exported) because reading the cache pulls in no
    part of the paragraph pipeline, and both callers want only this.
    """
    return pd.read_csv(path)


if __name__ == "__main__":
    save_paragraph_features()
