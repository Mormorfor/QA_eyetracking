# src/features/paragraph/spans.py
#
# The paragraph screen's own preprocessing pipeline: raw paragraph reports in,
# one trial-level paragraph feature table out.
#
# WHY THIS EXISTS (`todo.md` T6.1)
# -------------------------------
# Paragraph features used to be built in two unrelated places, neither of which
# was about paragraphs:
#
#   * the per-span eye-movement metrics lived inside
#     `predictive_modeling/answer_RTs/features.py` -- a *modelling* module for a
#     strand that is parked, even though the extraction it contains is
#     load-bearing for the correctness model and for the text-QA analysis;
#   * the per-span RT / TFD / TimeSinceOffset columns were produced by the
#     ANSWER pipeline, which opened the paragraph IA and fixation reports in the
#     middle of preparing the answer screen and merged the results into the
#     answer table.
#
# The second is the one that hurt. It made "prepare the QA data" depend on
# multi-GB paragraph reports, it threaded an `include_paragraph` flag through
# four call layers purely because KnowQA has no paragraph screen, and it joined
# the two screens with `how="inner"`, so a trial with answer data but no
# paragraph data vanished from the RT table without a word.
#
# Now: paragraph reports -> this module -> one paragraph feature table, joined
# to anything that wants it at feature-construction time, on
# (participant_id, TRIAL_INDEX), with coverage asserted. The answer pipeline
# reads no paragraph input at all.
#
# SAME MECHANICS AS THE ANSWER SCREEN
# -----------------------------------
# Every metric comes from `features/area_metrics.py`, the one implementation both
# screens share, differing only in the grouping column (`auxiliary_span_type`
# here, `area_label` there). The RT/TFD definitions come from
# `features/reading_times.py`, which was already parameterized by area column.
# The pupil baseline follows the project rule -- per person, computed over THIS
# screen's fixations (`todo.md` T3.20).

from __future__ import annotations

# The `sys.path.insert(PROJECT_ROOT)` header this file used to carry was removed
# on 2026-10-06. It was correct at `src/derived/paragraph_prep.py`, but stage C
# moved this file one level deeper, so its `parents[2]` resolved to `src/` rather
# than the repo root -- putting `src/` back on the import path, which is the
# stdlib-shadowing hazard stage A removed (docs/pitfalls.md section 6). Nothing
# replaces it: the project is run from the repo root, which is already on the path.
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import columns as C
from src.config.screens import PUPIL_RAW_COLS, screen
from src.config.columns import TRIAL_ID_COLS
from src.config.datasets import FIX_PARAGRAPH_PATH, IA_PARAGRAPH_PATH, PARAGRAPH_SPAN_FEATURES_PATH
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
    """Normalize a (participant, trial) key to strings that match across reports.

    The fixation report is streamed with `dtype=str` while the IA report is read
    with inferred dtypes, so the same trial can arrive as `"5"` on one side and
    `5` (or `5.0`) on the other. A mismatch here would not raise -- the
    nearest-interest-area queue would simply come back empty and the fallback
    would silently do nothing, which is exactly the behaviour T1.7 is removing.
    So both sides go through this, and `build_span_metrics` reports the hit rate.
    """
    def one(v):
        t = str(v).strip()
        if t.endswith(".0") and t[:-2].lstrip("-").isdigit():
            t = t[:-2]
        return t

    return (one(pid), one(tid))


# The screen this module builds. Everything that used to make this file "the
# paragraph version" of the answer pipeline now lives on this record
# (`config/screens.py`) -- grouping column, regions, and the three switches that
# differ. SPAN_COL / SPANS are kept as local aliases because the code below reads
# better with them.
PARAGRAPH_SCREEN = screen("paragraph")
SPAN_COL = PARAGRAPH_SCREEN.area_col
SPANS = PARAGRAPH_SCREEN.regions

QUESTION_PREVIEW_COL = C.QUESTION_PREVIEW_COLUMN

# The pupil columns both screens z-score. Declared once, in `config/screens.py`,
# beside the rest of the screen vocabulary -- a local copy here is exactly the
# drift stage D step 2b removed.
_PUPIL_RAW_COLS = PUPIL_RAW_COLS

# The 15 columns of the 157-column paragraph IA report this pipeline reads. The
# report is 5.6 GB; reading it whole needs well over 12 GB in pandas and simply
# does not fit on the project's machine. Naming the columns is what makes a
# paragraph rebuild runnable at all.
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
    """Stream the paragraph fixation report once, returning what needs it.

    Two things come out of the same pass, because the report is several GB and
    reading it twice is the only alternative:

    1. **the pupil baseline** -- per-participant mean/SD in mm, accumulated as
       count / sum / sum-of-squares per chunk and combined at the end. Exact, and
       memory-flat. `ddof=1` matches pandas `.std()`, so it agrees with
       `compute_participant_pupil_stats` to floating-point precision.
    2. **the nearest-interest-area queue** -- for each trial, the
       `CURRENT_FIX_NEAREST_INTEREST_AREA` of the fixations that landed outside
       every interest area, in order. `area_metrics.resolve_fixation_sequence`
       uses it to place those fixations rather than drop them, which is what the
       answer screen has always done and the paragraph screen never did.

    Returns (pupil_stats, {(participant_id, TRIAL_INDEX): [nearest ids...]}).
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
                ids = pd.to_numeric(g2[C.NEAREST_IA], errors="coerce").dropna()
                nearest.setdefault(_key(pid, tid), []).extend(ids.astype(int).tolist())

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
    """Coerce the measure columns and add the z-scored pupil columns.

    Now a one-line call: this function's body **became** the shared
    `features.pupil.prepare_screen_pupil` in stage D step 2b, because the answer
    screen does the same work by a different route. Kept as a name because the
    paragraph pipeline reads better for it.
    """
    return prepare_screen_pupil(paragraph_ia, pupil_stats, _PUPIL_RAW_COLS)


def span_visit_counts(
    df: pd.DataFrame,
    nearest_by_trial: dict,
    verbose: bool = True,
) -> pd.DataFrame:
    """Visits to each span, counted the same way the answer screen counts them.

    A thin call since stage D step 2c. Building the resolved span sequence is
    `sequences.build_area_sequences`, shared with the answer screen; what is left
    here is collapsing it to transitions and counting, which is two lines.

    The leading-question cleanup the answer screen applies is deliberately NOT
    applied: there is no question area on the paragraph screen.
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
    """One row per (participant, trial, span) with all eight metrics.

    A thin call since stage D step 2b: the merge loop this used to contain is now
    `area_metrics.build_area_metrics`, shared with the answer screen and
    parameterized by the `Screen`. What stays here is the one genuinely
    paragraph-specific input -- visit counts resolved against the nearest-IA
    queue, because this screen has no simplified label sequence to count from.
    """
    return am.build_area_metrics(
        df,
        PARAGRAPH_SCREEN,
        visit_counts=span_visit_counts(df, nearest_by_trial, verbose=verbose),
        metric_cols=PARAGRAPH_METRIC_COLUMNS,
    )


# ---------------------------------------------------------------------------
# RT / TFD, moved out of the answer pipeline
# ---------------------------------------------------------------------------

def build_paragraph_rt_tfd(
    paragraph_ia: pd.DataFrame,
    fixations_path: Path = FIX_PARAGRAPH_PATH,
    include_run_based_rt: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    """Per-span RT / TFD / TimeSinceOffset, one row per trial.

    A thin call since stage D step 2b. The assembly -- span-based measure, rename
    to `TimeSinceOffset_*`, run-based `RT_*` from the fixation report, join --
    is `reading_times.build_screen_rt_tfd`, shared with the answer screen. The
    only difference between the two screens is the `Screen` passed in.
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
    include_rt_tfd: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    """Raw paragraph reports -> one trial-level paragraph feature table.

    Columns: `<metric>__<span>` for the eight metrics, the RT/TFD families per
    span, and `question_preview`. Keyed on (participant_id, TRIAL_INDEX) so
    anything wanting paragraph features joins them explicitly.
    """
    if verbose:
        print("Scanning the paragraph fixation report...")
    pupil_stats, nearest_by_trial = scan_paragraph_fixations(
        fixations_path, verbose=verbose
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
    """Load the cached paragraph features produced by `save_paragraph_features`.

    Moved here from `answer_RTs/features.py` in stage E step 1. Reading the cache
    pulls in no part of the paragraph pipeline, which is why it was worth keeping
    together with the writer rather than with either caller.
    """
    return pd.read_csv(path)
