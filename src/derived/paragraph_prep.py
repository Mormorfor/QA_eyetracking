# src/derived/paragraph_prep.py
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
# Every metric comes from `derived/area_metrics.py`, the one implementation both
# screens share, differing only in the grouping column (`auxiliary_span_type`
# here, `area_label` there). The RT/TFD definitions come from
# `derived/reading_times.py`, which was already parameterized by area column.
# The pupil baseline follows the project rule -- per person, computed over THIS
# screen's fixations (`todo.md` T3.20).

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from src import constants as C
from src.constants import TRIAL_ID_COLS
from src.data_paths import (
    FIX_PARAGRAPH_PATH,
    IA_PARAGRAPH_PATH,
    PARAGRAPH_SPAN_FEATURES_PATH,
)
from src.derived import area_metrics as am
from src.derived.pupil_norm import scale_pupil_area_to_mm, zscore_pupil_by_participant
from src.derived.reading_times import (
    FIX_DURATION_COL,
    FIX_IA_ID_COL,
    FIX_START_COL,
    PARAGRAPH_AREA_COL,
    PARAGRAPH_REGIONS,
    compute_reading_times,
    compute_run_based_rt_from_fixations,
    load_paragraph_fixations,
)
from src.predictive_modeling.common.feature_builders import build_area_metric_pivot

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


SPAN_COL = PARAGRAPH_AREA_COL
SPANS = PARAGRAPH_REGIONS

QUESTION_PREVIEW_COL = C.QUESTION_PREVIEW_COLUMN

_PUPIL_RAW_COLS = (
    C.IA_MAX_FIX_PUPIL_SIZE,
    C.IA_MIN_FIX_PUPIL_SIZE,
    C.IA_AVERAGE_FIX_PUPIL_SIZE,
)

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

    Coercion happens ONCE here, for every metric, rather than inside each metric
    on the way past -- the same arrangement the answer pipeline is moving toward
    (`todo.md` T3.11), and the reason the two screens can share one set of metric
    functions at all.
    """
    df = am.coerce_ia_columns(paragraph_ia, pupil=True)

    for col in _PUPIL_RAW_COLS:
        df[col] = scale_pupil_area_to_mm(df[col])
        df = zscore_pupil_by_participant(
            df=df,
            pupil_col=col,
            participant_col=C.PARTICIPANT_ID,
            stats=pupil_stats,
            out_col=f"{col}_z",
        )
    return df


def span_visit_counts(
    df: pd.DataFrame,
    nearest_by_trial: dict,
    verbose: bool = True,
) -> pd.DataFrame:
    """Visits to each span, counted the same way the answer screen counts them.

    The trial's fixation sequence is resolved onto interest areas -- filling
    off-area fixations from the nearest-interest-area queue instead of dropping
    them -- mapped to spans, collapsed so consecutive repeats become one visit,
    and counted.

    The leading-question cleanup the answer screen applies is deliberately NOT
    applied: there is no question area on the paragraph screen.
    """
    rows = []
    trials = 0
    matched = 0
    dropped = 0
    for (pid, tid), g in df.groupby(list(TRIAL_ID_COLS), sort=False):
        trials += 1
        if _key(pid, tid) in nearest_by_trial:
            matched += 1
        ia_to_span = dict(zip(g[C.INTEREST_AREA_ID], g[SPAN_COL]))
        known = set(g[C.INTEREST_AREA_ID].unique())

        sequence = am.parse_ia_sequence(
            g[C.INTEREST_AREA_FIXATION_SEQUENCE].iloc[0]
        )
        resolved, n_dropped = am.resolve_fixation_sequence(
            sequence, known, nearest_by_trial.get(_key(pid, tid), [])
        )
        dropped += n_dropped
        spans = [ia_to_span.get(i) for i in resolved]
        collapsed = am.collapse_runs([s for s in spans if s is not None])
        counts = am.visit_counts_from_sequence(collapsed, SPANS)

        for span in SPANS:
            rows.append(
                {
                    C.PARTICIPANT_ID: pid,
                    C.TRIAL_ID: tid,
                    SPAN_COL: span,
                    C.NUM_LABEL_VISITS: counts[span],
                }
            )
    # NOTHING HERE ABORTS A RUN. A fixation we cannot place is dropped, and the
    # drop is counted and reported -- never fatal (Diana, 2026-09-23). The point
    # of the counts is that "the fallback silently became a no-op" and "this data
    # genuinely has no off-area fixations" look identical otherwise, and the first
    # would quietly restore the pre-T1.7 behaviour.
    queued = len(nearest_by_trial)
    unmatched = queued - matched
    if queued and matched == 0:
        print(
            f"  WARNING: the nearest-interest-area queue matched NONE of {trials} "
            f"trials, though it holds {queued} entries. The trial keys from the "
            "fixation report and the IA report are probably not comparable, so "
            "every off-area fixation is being dropped rather than placed. Check "
            "`_key`. Continuing -- dropped fixations are not a reason to stop."
        )
    elif unmatched:
        print(
            f"  WARNING: {unmatched} of {queued} nearest-interest-area queue "
            f"entries match no trial in the IA report. Those trials' off-area "
            "fixations are dropped. Continuing."
        )
    if verbose:
        print(
            f"  off-area fixation queue: {queued:,} trials queued, "
            f"{matched:,} matched ({100 * matched / max(trials, 1):.1f}% of "
            f"{trials:,} trials); {dropped:,} fixation(s) could not be placed "
            "and were dropped"
        )
    return pd.DataFrame(
        rows, columns=list(TRIAL_ID_COLS) + [SPAN_COL, C.NUM_LABEL_VISITS]
    )


def build_span_metrics(
    df: pd.DataFrame,
    nearest_by_trial: dict,
    verbose: bool = True,
) -> pd.DataFrame:
    """One row per (participant, trial, span) with all eight metrics.

    Every builder here is the shared one from `derived/area_metrics.py`; the only
    thing that makes this the paragraph version is `SPAN_COL`.
    """
    group = list(TRIAL_ID_COLS) + [SPAN_COL]
    parts = [
        am.mean_dwell_time(df, SPAN_COL),
        am.mean_fixations_count(df, SPAN_COL),
        am.mean_first_fix_duration(df, SPAN_COL),
        am.skip_rate(df, SPAN_COL),
        am.dwell_proportion(df, SPAN_COL),
        am.mean_pupil_size(df, SPAN_COL, include_raw=False, include_z=True),
        am.first_encounter_pupil_size(
            df, SPAN_COL, include_raw=False, include_z=True
        ),
        span_visit_counts(df, nearest_by_trial, verbose=verbose),
    ]
    out = parts[0]
    for part in parts[1:]:
        out = out.merge(part, on=group, how="outer")
    return out[group + PARAGRAPH_METRIC_COLUMNS]


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

    Lifted verbatim in behaviour from `reading_times.build_rt_and_tfd`'s
    paragraph branch, which the answer pipeline used to run. The definitions
    themselves still come from `reading_times`, so the answer and paragraph sides
    compute reading time the same way: `RT_*` is run-based, and the span-based
    first-to-last-fixation measure is renamed `TimeSinceOffset_*` to keep the two
    apart.
    """
    span_based = compute_reading_times(
        paragraph_ia, area_col=SPAN_COL, regions=SPANS
    )
    if not include_run_based_rt:
        return span_based

    rename = {
        c: c.replace("RT_pure_", "TimeSinceOffset_pure_", 1).replace(
            "RT_normalized_", "TimeSinceOffset_normalized_", 1
        )
        for c in span_based.columns
        if c.startswith("RT_pure_") or c.startswith("RT_normalized_")
    }
    out = span_based.rename(columns=rename)

    if verbose:
        print("Computing paragraph-region RT (run-based)...")
    fixations = load_paragraph_fixations(fixations_path, verbose=verbose)
    run_rt = compute_run_based_rt_from_fixations(
        fixations, paragraph_ia, area_col=SPAN_COL, regions=SPANS
    )
    merged = out.merge(run_rt, on=list(TRIAL_ID_COLS), how="left")
    assert len(merged) == len(out), (
        f"paragraph run-based RT join changed the row count: "
        f"{len(out)} -> {len(merged)}"
    )
    return merged


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
