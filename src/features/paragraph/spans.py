"""The paragraph screen: its raw reports in, one row of features per trial out.

Study 1 only. Each paragraph is split into three parts: "critical" (holds the
answer), "distractor" (supports a wrong answer) and "outside" (the rest). For
each part this measures the same things as on the answer screen, with the same
code (`area_metrics.py`, `reading_times.py`):

    <metric>__<part>     e.g. area_dwell_proportion__critical
    RT_* / TFD_* / TimeSinceOffset_* per part

The paragraph fixation report is read once (`fixations.load_fixations`) and
used for the pupil baseline -- each person's own, from their fixations on this
screen -- the visit counts and the reading times.

`save_paragraph_features` builds the table and saves it to
`features/paragraph_spans.csv`. The answer-screen pipeline reads nothing from
the paragraph reports; the model table joins this file in by trial.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import columns as C
from src.config.screens import PUPIL_RAW_COLS, screen
from src.config.columns import TRIAL_ID_COLS
from src.config.datasets import (
    FIX_PARAGRAPH_PATH,
    IA_PARAGRAPH_PATH,
    PARAGRAPH_SPAN_FEATURES_PATH,
    PUPIL_BASELINE_PARAGRAPH_PATH,
)
from src.features import area_metrics as am
from src.features.fixations import load_fixations, match_key_types
from src.features.sequences import build_area_sequences
from src.features.pupil import compute_participant_pupil_stats, prepare_screen_pupil
from src.features.reading_times import build_screen_rt_tfd
from src.features.build import build_area_metric_pivot


# The screen this module builds (see config/screens.py), and short names for
# its area column and its three parts.
PARAGRAPH_SCREEN = screen("paragraph")
SPAN_COL = PARAGRAPH_SCREEN.area_col
SPANS = PARAGRAPH_SCREEN.regions

QUESTION_PREVIEW_COL = C.QUESTION_PREVIEW_COLUMN

# The raw pupil columns both screens z-score (defined in config/screens.py).
_PUPIL_RAW_COLS = PUPIL_RAW_COLS

# The 14 columns of the paragraph word report this pipeline reads. The report is
# 5.6 GB; reading all of it would need well over 12 GB of memory, so only these
# columns are read.
IA_USECOLS = [
    C.PARTICIPANT_ID,
    C.TRIAL_ID,
    SPAN_COL,
    C.INTEREST_AREA_ID,
    C.IA_DWELL_TIME,
    C.IA_FIXATIONS_COUNT,
    C.IA_FIRST_FIXATION_DURATION,
    QUESTION_PREVIEW_COL,
    *_PUPIL_RAW_COLS,
    # needed by reading_times.compute_reading_times for the span-based RT
    C.IA_FIRST_FIXATION_TIME,
    C.IA_LAST_FIXATION_TIME,
    C.IA_LAST_FIXATION_DURATION,
]

PARAGRAPH_METRIC_COLUMNS = list(C.AREA_METRIC_COLUMNS_MODELING)


# ---------------------------------------------------------------------------
# The paragraph fixation report
# ---------------------------------------------------------------------------

# The columns of the paragraph fixation report this pipeline reads.
FIXATION_COLUMNS = [
    C.CURRENT_FIX_START,
    C.CURRENT_FIX_DURATION,
    C.CURRENT_FIX_INTEREST_AREA_ID,
    C.NEAREST_IA,
    C.CURRENT_FIX_PUPIL_SIZE,
]


# ---------------------------------------------------------------------------
# Preparation
# ---------------------------------------------------------------------------

def prepare_paragraph_ia(
    paragraph_ia: pd.DataFrame,
    pupil_stats: pd.DataFrame,
) -> pd.DataFrame:
    """Turn the paragraph word columns into numbers and add z-scored pupil
    columns, using this screen's own baseline. See `pupil.prepare_screen_pupil`."""
    return prepare_screen_pupil(paragraph_ia, pupil_stats, _PUPIL_RAW_COLS)


def span_visit_counts(
    df: pd.DataFrame,
    fixations: pd.DataFrame,
    verbose: bool = True,
) -> pd.DataFrame:
    """How many times each paragraph part was entered, one row per trial and part.

    Builds each trial's sequence of parts from the paragraph `fixations` with
    the same function the answer screen uses (`sequences.build_area_sequences`),
    merges repeats, and counts. No opening entry is dropped: this screen has no
    question.
    """
    seqs = build_area_sequences(
        df,
        [SPAN_COL],
        fixations=fixations,
        drop_leading_question=False,
        verbose=verbose,
    )
    rows = []
    for _, r in seqs.iterrows():
        counts = am.visit_counts_from_sequence(am.collapse_runs(r[SPAN_COL]), SPANS)
        for span in SPANS:
            rows.append(
                {
                    C.PARTICIPANT_ID: r[C.PARTICIPANT_ID],
                    C.TRIAL_ID: r[C.TRIAL_ID],
                    SPAN_COL: span,
                    C.NUM_LABEL_VISITS: counts[span],
                }
            )
    if verbose:
        print(f"  span visits: {len(seqs):,} trial(s)")
    return pd.DataFrame(rows)


def build_span_metrics(
    df: pd.DataFrame,
    fixations: pd.DataFrame,
    verbose: bool = True,
) -> pd.DataFrame:
    """Every per-area measurement for the paragraph, one row per trial and part.

    Uses `area_metrics.build_area_metrics`, as the answer screen does, with this
    screen's own visit counts.
    """
    return am.build_area_metrics(
        df,
        PARAGRAPH_SCREEN,
        visit_counts=span_visit_counts(df, fixations, verbose=verbose),
        metric_cols=PARAGRAPH_METRIC_COLUMNS,
    )


# ---------------------------------------------------------------------------
# Reading times
# ---------------------------------------------------------------------------

def build_paragraph_rt_tfd(
    paragraph_ia: pd.DataFrame,
    fixations: pd.DataFrame,
    include_run_based_rt: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    """The three time measures for each paragraph part, one row per trial.

    `fixations` is the paragraph fixation table. Uses
    `reading_times.build_screen_rt_tfd`, as the answer screen does.
    """
    return build_screen_rt_tfd(
        paragraph_ia,
        PARAGRAPH_SCREEN,
        fixations=fixations,
        include_run_based_rt=include_run_based_rt,
        verbose=verbose,
    )


# ---------------------------------------------------------------------------
# The pipeline
# ---------------------------------------------------------------------------

def build_paragraph_features(
    paragraph_ia: pd.DataFrame | None = None,
    paragraph_ia_path: Path = IA_PARAGRAPH_PATH,
    fixations_path: Path = FIX_PARAGRAPH_PATH,
    pupil_baseline_path: Path = PUPIL_BASELINE_PARAGRAPH_PATH,
    include_rt_tfd: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    """Build the paragraph feature table from the raw paragraph reports.

    One row per trial: `<metric>__<part>` for every measurement, the reading
    times per part (unless `include_rt_tfd=False`), and `question_preview`.

    The paragraph pupil baseline is computed here, from all of each person's
    fixations in the paragraph report, with the same function as the answer
    screen's (`pupil.compute_participant_pupil_stats`). It is saved to
    `pupil_baseline_path` as a record of what was used (None skips saving), and
    never read back from that file.
    """
    if verbose:
        print("Reading the paragraph fixation report...")
    fixations = load_fixations(fixations_path, FIXATION_COLUMNS, verbose=verbose)
    pupil_stats = compute_participant_pupil_stats(fixations)

    # Saved under its own name: this screen's baseline differs from the answer
    # screen's (different brightness, different task).
    if pupil_baseline_path is not None:
        pupil_baseline_path = Path(pupil_baseline_path)
        pupil_baseline_path.parent.mkdir(parents=True, exist_ok=True)
        pupil_stats.to_csv(pupil_baseline_path, index=False)
        if verbose:
            print(
                f"  paragraph pupil baseline: {len(pupil_stats)} participants "
                f"-> {pupil_baseline_path}"
            )

    if paragraph_ia is None:
        if verbose:
            print(f"Reading {len(IA_USECOLS)} columns of {paragraph_ia_path}...")
        paragraph_ia = pd.read_csv(
            paragraph_ia_path, usecols=IA_USECOLS, low_memory=False
        )
    if verbose:
        print(f"  {len(paragraph_ia):,} paragraph interest areas")
    fixations = match_key_types(fixations, paragraph_ia)

    prepared = prepare_paragraph_ia(paragraph_ia, pupil_stats)

    span_metrics = build_span_metrics(prepared, fixations, verbose=verbose)
    features = build_area_metric_pivot(
        df=span_metrics, area_col=SPAN_COL, metric_cols=PARAGRAPH_METRIC_COLUMNS
    )

    preview = (
        prepared[list(TRIAL_ID_COLS) + [QUESTION_PREVIEW_COL]]
        .groupby(list(TRIAL_ID_COLS), as_index=False)
        .first()
    )
    preview[QUESTION_PREVIEW_COL] = (
        preview[QUESTION_PREVIEW_COL].astype("boolean").astype("Int64")
    )
    features = features.merge(preview, on=list(TRIAL_ID_COLS), how="left")

    if include_rt_tfd:
        rt = build_paragraph_rt_tfd(prepared, fixations, verbose=verbose)
        before = len(features)
        features = features.merge(rt, on=list(TRIAL_ID_COLS), how="left")
        assert len(features) == before, (
            f"paragraph RT/TFD join changed the row count: {before} -> {len(features)}"
        )

    return features


def save_paragraph_features(
    output_path: Path = PARAGRAPH_SPAN_FEATURES_PATH,
    verbose: bool = True,
    **kwargs,
) -> pd.DataFrame:
    """Build the paragraph feature table and save it."""
    features = build_paragraph_features(verbose=verbose, **kwargs)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(output_path, index=False)
    if verbose:
        print(
            f"Saved {len(features)} trials x {len(features.columns)} cols "
            f"to {output_path}"
        )
    return features


if __name__ == "__main__":
    save_paragraph_features()


def load_paragraph_features(
    path: Path = PARAGRAPH_SPAN_FEATURES_PATH,
) -> pd.DataFrame:
    """Read the paragraph feature table saved by `save_paragraph_features`."""
    return pd.read_csv(path)
