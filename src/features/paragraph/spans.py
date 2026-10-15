"""The paragraph screen: its raw reports in, one row of features per trial out.

Study 1 only. Each paragraph is split into three parts: "critical" (holds the
answer), "distractor" (supports a wrong answer) and "outside" (the rest). For
each part this measures the same things as on the answer screen, with the same
code (`area_metrics.py`, `reading_times.py`):

    <metric>__<part>     e.g. area_dwell_proportion__critical
    RT_* / TFD_* / TimeSinceOffset_* per part

The pupil baseline is each person's own, from their fixations on this screen.

`save_paragraph_features` builds the table and saves it to
`features/paragraph_spans.csv`. The answer-screen pipeline reads nothing from
the paragraph reports; the model table joins this file in by trial.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
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
from src.features.sequences import build_area_sequences
from src.features.pupil import (
    prepare_screen_pupil,
    scale_pupil_area_to_mm,
    zscore_pupil_by_participant,
)
from src.features.reading_times import (
    PARAGRAPH_AREA_COL,
    PARAGRAPH_REGIONS,
    build_screen_rt_tfd,
    compute_reading_times,
    compute_run_based_rt_from_fixations,
    load_paragraph_fixations,
)
from src.features.build import build_area_metric_pivot

def _key(pid, tid) -> tuple[str, str]:
    """A (participant, trial) key as text that matches across the two reports.

    The fixation report is read as text and the word report with guessed types,
    so the same trial can be "5" on one side and 5 or 5.0 on the other. Without
    this, nothing would match, the nearest-word lists would quietly come back
    empty, and every missed fixation would be dropped. Both sides use this.
    """
    def one(v):
        t = str(v).strip()
        if t.endswith(".0") and t[:-2].lstrip("-").isdigit():
            t = t[:-2]
        return t

    return (one(pid), one(tid))


# The screen this module builds (see config/screens.py), and short names for
# its area column and its three parts.
PARAGRAPH_SCREEN = screen("paragraph")
SPAN_COL = PARAGRAPH_SCREEN.area_col
SPANS = PARAGRAPH_SCREEN.regions

QUESTION_PREVIEW_COL = C.QUESTION_PREVIEW_COLUMN

# The raw pupil columns both screens z-score (defined in config/screens.py).
_PUPIL_RAW_COLS = PUPIL_RAW_COLS

# The 15 columns of the paragraph word report this pipeline reads. The report is
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
    C.INTEREST_AREA_FIXATION_SEQUENCE,
    QUESTION_PREVIEW_COL,
    *_PUPIL_RAW_COLS,
    # needed by compute_reading_times for the span-based RT
    "IA_FIRST_FIXATION_TIME",
    "IA_LAST_FIXATION_TIME",
    "IA_LAST_FIXATION_DURATION",
]

PARAGRAPH_METRIC_COLUMNS = list(C.AREA_METRIC_COLUMNS_MODELING)


# ---------------------------------------------------------------------------
# One streaming pass over the fixation report
# ---------------------------------------------------------------------------

def scan_paragraph_fixations(
    fixations_path: Path = FIX_PARAGRAPH_PATH,
    chunksize: int = 2_000_000,
    verbose: bool = True,
) -> tuple[pd.DataFrame, dict]:
    """Read the paragraph fixation report once, collecting two things.

    The report is several GB, so both come from a single pass, read in chunks:

    1. **the pupil baseline** -- each person's average and SD of pupil size
       (mm), built up chunk by chunk. Gives the same result as
       `pupil.compute_participant_pupil_stats` would on the whole file.
    2. **the nearest-word lists** -- for each trial, the nearest word of every
       fixation that missed all words, in order, used to place those
       fixations. A `None` marks a fixation with no nearest word; it is kept so
       the list stays in step with the fixations (see
       `area_metrics.nearest_ia_queue`).

    Returns (pupil baseline, {(participant_id, TRIAL_INDEX): [nearest ids]}).
    """
    usecols = [
        C.PARTICIPANT_ID,
        C.TRIAL_ID,
        C.CURRENT_FIX_PUPIL_SIZE,
        C.CURRENT_FIX_INTEREST_AREAS,
        C.NEAREST_IA,
    ]
    acc: dict = {}
    nearest: dict = {}

    reader = pd.read_csv(
        fixations_path, usecols=usecols, chunksize=chunksize, dtype=str
    )
    for chunk in reader:
        # --- pupil accumulators ---
        mm = scale_pupil_area_to_mm(chunk[C.CURRENT_FIX_PUPIL_SIZE])
        ok = mm.notna()
        if ok.any():
            g = pd.DataFrame(
                {C.PARTICIPANT_ID: chunk.loc[ok, C.PARTICIPANT_ID], "v": mm[ok]}
            ).groupby(C.PARTICIPANT_ID)["v"]
            part = pd.DataFrame(
                {"n": g.size(), "s": g.sum(), "ss": g.apply(lambda x: (x**2).sum())}
            )
            for pid, row in part.iterrows():
                cur = np.array([row["n"], row["s"], row["ss"]], dtype="float64")
                prev = acc.get(pid)
                acc[pid] = cur if prev is None else prev + cur

        # --- off-area fixations, in order, per trial ---
        areas = chunk[C.CURRENT_FIX_INTEREST_AREAS].apply(
            am._parse_interest_area_list
        )
        off = chunk[areas.apply(len) == 0]
        if len(off):
            for (pid, tid), g2 in off.groupby(
                [C.PARTICIPANT_ID, C.TRIAL_ID], sort=False
            ):
                # The same function the answer screen uses, so both build these
                # lists the same way.
                nearest.setdefault(_key(pid, tid), []).extend(
                    am.nearest_ia_queue(g2)
                )

    if not acc:
        raise ValueError(
            f"No usable pupil values in {fixations_path} -- cannot build a "
            "paragraph pupil baseline."
        )

    out = pd.DataFrame(
        [(pid, n, s, ss) for pid, (n, s, ss) in acc.items()],
        columns=[C.PARTICIPANT_ID, "n", "s", "ss"],
    )
    out["pupil_mean"] = out["s"] / out["n"]
    var = (out["ss"] - out["n"] * out["pupil_mean"] ** 2) / (out["n"] - 1)
    out["pupil_sd"] = np.sqrt(var.clip(lower=0))
    if verbose:
        print(
            f"  paragraph baseline: {len(out)} participants, "
            f"{int(out['n'].sum()):,} fixations; "
            f"off-area fixations on {len(nearest):,} trials"
        )
    return out[[C.PARTICIPANT_ID, "pupil_mean", "pupil_sd"]], nearest


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
    nearest_by_trial: dict,
    verbose: bool = True,
) -> pd.DataFrame:
    """How many times each paragraph part was entered, one row per trial and part.

    Builds each trial's sequence of parts with the same function the answer
    screen uses (`sequences.build_area_sequences`), merges repeats, and counts.
    No opening-question fixation is dropped: this screen has no question.
    """
    seqs = build_area_sequences(
        df,
        [SPAN_COL],
        nearest_by_trial=nearest_by_trial,
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
        print(
            f"  span visits: {len(seqs):,} trial(s); off-area fixation queue covers "
            f"{len(nearest_by_trial):,}"
        )
    return pd.DataFrame(rows)


def build_span_metrics(
    df: pd.DataFrame,
    nearest_by_trial: dict,
    verbose: bool = True,
) -> pd.DataFrame:
    """Every per-area measurement for the paragraph, one row per trial and part.

    Uses `area_metrics.build_area_metrics`, as the answer screen does, with this
    screen's own visit counts.
    """
    return am.build_area_metrics(
        df,
        PARAGRAPH_SCREEN,
        visit_counts=span_visit_counts(df, nearest_by_trial, verbose=verbose),
        metric_cols=PARAGRAPH_METRIC_COLUMNS,
    )


# ---------------------------------------------------------------------------
# Reading times
# ---------------------------------------------------------------------------

def build_paragraph_rt_tfd(
    paragraph_ia: pd.DataFrame,
    fixations_path: Path = FIX_PARAGRAPH_PATH,
    include_run_based_rt: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    """The three time measures for each paragraph part, one row per trial.

    Uses `reading_times.build_screen_rt_tfd`, as the answer screen does.
    """
    return build_screen_rt_tfd(
        paragraph_ia,
        PARAGRAPH_SCREEN,
        fixations=load_paragraph_fixations(fixations_path, verbose=verbose)
        if include_run_based_rt
        else None,
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

    The paragraph pupil baseline is computed here and saved to
    `pupil_baseline_path` as a record of what was used (None skips saving). It
    is never read back from that file.
    """
    if verbose:
        print("Scanning the paragraph fixation report...")
    pupil_stats, nearest_by_trial = scan_paragraph_fixations(
        fixations_path, verbose=verbose
    )

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

    prepared = prepare_paragraph_ia(paragraph_ia, pupil_stats)

    span_metrics = build_span_metrics(prepared, nearest_by_trial, verbose=verbose)
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
        rt = build_paragraph_rt_tfd(
            prepared, fixations_path=fixations_path, verbose=verbose
        )
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
