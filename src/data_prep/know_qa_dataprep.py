"""End-to-end data preparation for the KnowQA experiment.

This is `experiment_builder/data_prep_new_exp.ipynb` turned into a module, plus
the KnowQA participant/trial identity scheme (see below). It takes the raw
DataViewer reports of a run and produces:

    data_raw/<run>/csvs/              converted CSVs (one per raw report)
    data_raw/<run>/csvs/cleaned/      column-standardized CSVs the pipeline reads
    data/<run>_runs/all_participants.csv              IA-level processed data
    data/<run>_runs/Auxiliary/                        pipeline intermediates
    data/<run>_runs/L1_model_ready_all_features.csv   one row per trial

Run it with `prepare_know_qa()`, or from the command line::

    python -m src.data_prep.know_qa_dataprep                    # all four steps
    python -m src.data_prep.know_qa_dataprep --steps convert clean
    python -m src.data_prep.know_qa_dataprep --steps features

Identity: why `participant_id` is not the recording label
---------------------------------------------------------
A KnowQA `RECORDING_SESSION_LABEL` looks like ``4000_21``: ``4XXX`` is the
person, then ``Y`` (1-3) is the batch and ``Z`` (1-18) is the list, with no
separator between the last two because the tracker caps labels at 8 characters.
The label therefore names a **session**, and the same person can sit for several
sessions -- ``4000_11``, ``4000_21``, ``4000_31`` are all person ``4000``.

So this module splits the label apart:

    session_id      4000_21   the recording label, verbatim
    participant_id  4000      the person, stable across runs
    article_batch   2         int
    list_number     1         int
    trial_number    5         the tracker's within-session trial counter, int
    TRIAL_INDEX     b2l01t005 composite trial id, unique within a person

`participant_id` alone no longer identifies a session and
(`participant_id`, `trial_number`) no longer identifies a trial -- person 4000
has three trial 1s. Rather than widen the (participant_id, TRIAL_INDEX) key that
the whole pipeline is built on, the batch and list are folded into `TRIAL_INDEX`
itself, so every existing groupby and merge keeps working untouched.

The composite id is deliberately non-numeric and fixed-width:

* non-numeric so that code expecting a number fails loudly (NaN) instead of
  silently reading a plausible-but-wrong integer;
* fixed-width so that lexicographic order matches numeric order, which keeps
  the order-sensitive `min()`/`<` in
  `button_clicks_processing.truncate_recordings_at_first_malformed_trial`
  correct.

One place does require a numeric TRIAL_INDEX:
`derived.reading_times.load_paragraph_fixations` coerces it to int64 and drops
the un-coercible rows, so a composite id would silently reduce the paragraph
fixation report to nothing. KnowQA has no paragraph report yet and the
paragraph steps are off, so `run_pipeline` refuses `include_paragraph=True`
rather than let that happen quietly.

Pupil normalization follows the session, not the person
-------------------------------------------------------
`participant_id` covering several sessions has one real consequence for the
features: pupil z-scoring is the pipeline's only per-participant step
(`derived.pupil_norm.compute_participant_pupil_stats` groups by
`participant_id`), so a person's sessions would be pooled into a single
mean/SD. Pupil baseline shifts between sittings (setup, lighting, fatigue), so
this module z-scores within the session instead -- `clean_reports` does the mm
scaling and z-scoring itself and the pipeline's own pupil base feature is
skipped. Pass `pupil_norm_unit="participant"` to pool them instead.
"""

from __future__ import annotations

import argparse
import ast
import csv
import re
import sys
from pathlib import Path
from typing import Iterable, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src import constants as Con
from src.data_paths import KNOW_QA_OUT_DIR, KNOW_QA_PATH

# ---------------------------------------------------------------------------
# Runs
# ---------------------------------------------------------------------------

# name -> (raw report folder, processed output folder).
#
# Only runs whose recording labels follow the `4XXX_YZ` scheme belong here. The
# two pilots do not: second_test labels sessions `r1_l1_b1` and testrun_QA uses
# `T1_l1`, so `parse_session_label` rejects them by design. They stay in
# data_prep_new_exp.ipynb, which keys trials on the raw label instead.
#
# A further collection in the KnowQA format just needs a line here (or a `runs`
# mapping passed to `prepare_know_qa`) plus its paths in src/data_paths.py.
RAW_RUNS: dict[str, tuple[Path, Path]] = {
    "KnowQA": (KNOW_QA_PATH, KNOW_QA_OUT_DIR),
}

DEFAULT_RUN = "KnowQA"

# ---------------------------------------------------------------------------
# Reading the raw reports
# ---------------------------------------------------------------------------

# Every export is plain tab-separated text; only the extension, the encoding and
# the file naming differ, so one reader covers them all.
RAW_REPORT_ENCODINGS = {
    ".xls": "utf-16",  # DataViewer's default export
    ".tsv": "utf-8-sig",  # second_test export (UTF-8 with BOM)
    ".csv": "utf-8-sig",
}

RAW_REPORT_PATTERNS = ("*.xls", "*.tsv")

CSV_SUBDIR = "csvs"
CLEANED_SUBDIR = "cleaned"

IA_REPORT_NAME = "IA_answers.csv"
FIX_REPORT_NAME = "fixations_answers.csv"
MSG_REPORT_NAME = "messages_answers.csv"

# The reports the pipeline actually needs cleaned.
CLEANED_REPORT_NAMES = (IA_REPORT_NAME, FIX_REPORT_NAME)


def read_raw_report(path: Path) -> pd.DataFrame:
    """Read one raw DataViewer report (.xls / .tsv / .csv) into a DataFrame.

    Quoting is disabled because the reports carry bare quote characters inside
    the stimulus text. The UTF-16 .xls exports need the python parser; the
    UTF-8 .tsv ones go through the (much faster) C parser.
    """
    encoding = RAW_REPORT_ENCODINGS[path.suffix.lower()]
    kwargs: dict = dict(encoding=encoding, sep="\t", quoting=csv.QUOTE_NONE)
    if encoding == "utf-16":
        kwargs["engine"] = "python"
    else:
        kwargs["low_memory"] = False
    return pd.read_csv(path, **kwargs)


def canonical_report_name(path: Path) -> str:
    """Map a raw report filename onto the canonical CSV name the pipeline reads.

    KnowQA and testrun_QA already use the canonical names
    (`IA_Answers.xls`, ...), while second_test prefixes them with the report
    letter (`A_IA.tsv`, `A_fixations.tsv`); all normalize onto
    `<report>_answers.csv`.
    """
    tokens = re.split(r"[_\-\s]+", path.stem.lower())
    if any(t.startswith("fixation") for t in tokens):
        return FIX_REPORT_NAME
    if any(t.startswith("message") or t == "msg" for t in tokens):
        return MSG_REPORT_NAME
    if "ia" in tokens:
        return IA_REPORT_NAME
    return path.with_suffix(".csv").name


def convert_raw_reports(raw_dir: Path, verbose: bool = True) -> dict[str, Path]:
    """Convert every raw report in `raw_dir` to a canonically-named CSV.

    Returns {canonical name: written path}. Written to `raw_dir/csvs/`.
    """
    raw_dir = Path(raw_dir)
    out_dir = raw_dir / CSV_SUBDIR
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_reports = sorted(
        p for pattern in RAW_REPORT_PATTERNS for p in raw_dir.glob(pattern)
    )
    if not raw_reports:
        raise FileNotFoundError(
            f"no {' / '.join(RAW_REPORT_PATTERNS)} reports found in {raw_dir}"
        )

    written: dict[str, Path] = {}
    for raw_path in raw_reports:
        df = read_raw_report(raw_path)
        name = canonical_report_name(raw_path)
        csv_path = out_dir / name
        df.to_csv(csv_path, index=False, encoding="utf-8")
        written[name] = csv_path
        if verbose:
            print(f"{raw_path.name}: {df.shape} -> {CSV_SUBDIR}/{name}")

    missing = [n for n in CLEANED_REPORT_NAMES if n not in written]
    if missing:
        raise FileNotFoundError(
            f"{raw_dir} produced no {', '.join(missing)} -- got "
            f"{sorted(written)}. Check the raw report filenames."
        )
    return written


# ---------------------------------------------------------------------------
# Participant / trial identity
# ---------------------------------------------------------------------------

# `4XXX_YZ`: person, batch (1 digit, 1-3), list (1-18, no separator). The list
# alternative tries two digits first so `4000_118` reads as batch 1 / list 18
# rather than batch 1 / list 1 with a stray "8".
SESSION_LABEL_RE = re.compile(
    r"^(?P<participant>\d+)_(?P<batch>[1-3])(?P<list>1[0-8]|[1-9])$"
)

BATCH_RANGE = (1, 3)
LIST_RANGE = (1, 18)

# Raw columns the experiment writes alongside the label; used to cross-check the
# parse instead of trusting the label alone.
RAW_BATCH_COLUMN = "article_batch"
RAW_LIST_COLUMN = "list_numeric"

# Written by the experiment; a recalibration consumes a trial index without
# producing an Answers interest period. See `check_trial_index_gaps`.
RECALIBRATIONS_COLUMN = "NUM_RECALIBRATIONS"


def parse_session_label(label: str) -> tuple[str, int, int]:
    """Split a recording label into (participant_id, batch, list_number).

    >>> parse_session_label("4000_21")
    ('4000', 2, 1)
    >>> parse_session_label("4123_118")
    ('4123', 1, 18)
    """
    match = SESSION_LABEL_RE.match(str(label).strip())
    if match is None:
        raise ValueError(
            f"recording label {label!r} does not look like 4XXX_YZ "
            f"(person, batch 1-3, list 1-18, no separator)"
        )
    participant = match.group("participant")
    batch = int(match.group("batch"))
    list_number = int(match.group("list"))
    if not BATCH_RANGE[0] <= batch <= BATCH_RANGE[1]:
        raise ValueError(f"label {label!r}: batch {batch} outside {BATCH_RANGE}")
    if not LIST_RANGE[0] <= list_number <= LIST_RANGE[1]:
        raise ValueError(f"label {label!r}: list {list_number} outside {LIST_RANGE}")
    return participant, batch, list_number


def format_trial_id(batch: int, list_number: int, trial_number: int) -> str:
    """Build the composite trial id, e.g. (2, 1, 5) -> 'b2l01t005'.

    Non-numeric on purpose (so numeric misuse fails loudly rather than
    silently) and fixed-width (so lexicographic order matches numeric order).
    """
    return f"b{int(batch)}l{int(list_number):02d}t{int(trial_number):03d}"


def _check_against_raw(
    df: pd.DataFrame, parsed: pd.DataFrame, raw_col: str, parsed_col: str
) -> list[str]:
    """Compare a parsed identity column against the experiment's own column."""
    if raw_col not in df.columns:
        return []
    raw = pd.to_numeric(df[raw_col], errors="coerce")
    mismatched = raw.notna() & (raw.astype("Int64") != parsed[parsed_col].astype("Int64"))
    if not mismatched.any():
        return []
    sample = (
        df.loc[mismatched, [Con.SESSION_ID, raw_col]]
        .assign(**{f"parsed_{parsed_col}": parsed.loc[mismatched, parsed_col]})
        .drop_duplicates()
        .head(5)
    )
    return [
        f"{int(mismatched.sum())} row(s) where the label's {parsed_col} disagrees "
        f"with the report's {raw_col}:\n{sample.to_string(index=False)}"
    ]


def add_identity_columns(
    df: pd.DataFrame,
    label_col: str = Con.RECORDING_SESSION_LABEL,
    strict: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    """Derive the KnowQA identity columns from the recording label.

    Adds `session_id`, `participant_id`, `article_batch`, `list_number` and
    `trial_number`, and replaces `TRIAL_INDEX` with the composite trial id.
    The parse is cross-checked against the report's own `article_batch` /
    `list_numeric` columns; `strict` raises on a disagreement instead of warning.

    Idempotent: a frame that already carries `session_id` is returned unchanged.
    """
    df = df.copy()

    if Con.SESSION_ID in df.columns:
        return df  # already standardized

    if label_col not in df.columns:
        raise KeyError(
            f"{label_col!r} not found -- cannot derive the KnowQA identity "
            f"columns. Columns present: {list(df.columns)[:10]}..."
        )
    if Con.TRIAL_ID not in df.columns:
        raise KeyError(f"{Con.TRIAL_ID!r} not found -- cannot build the trial id.")

    df[Con.SESSION_ID] = df[label_col].astype(str).str.strip()

    # Parse each distinct label once, then map back onto the rows: the reports
    # are hundreds of thousands of rows but only a handful of sessions.
    labels = df[Con.SESSION_ID].drop_duplicates()
    parsed_by_label = pd.DataFrame(
        [parse_session_label(lab) for lab in labels],
        columns=[Con.PARTICIPANT_ID, Con.BATCH_COLUMN, Con.LIST_COLUMN],
        index=labels,
    )
    parsed = parsed_by_label.reindex(df[Con.SESSION_ID]).reset_index(drop=True)
    parsed.index = df.index

    problems: list[str] = []
    problems += _check_against_raw(df, parsed, RAW_BATCH_COLUMN, Con.BATCH_COLUMN)
    problems += _check_against_raw(df, parsed, RAW_LIST_COLUMN, Con.LIST_COLUMN)
    if problems:
        message = (
            "recording labels disagree with the report's own batch/list columns:\n"
            + "\n".join(problems)
        )
        if strict:
            raise ValueError(message)
        print(f"WARNING: {message}")

    df[Con.PARTICIPANT_ID] = parsed[Con.PARTICIPANT_ID].str.lower()
    df[Con.BATCH_COLUMN] = parsed[Con.BATCH_COLUMN].astype("int64")
    df[Con.LIST_COLUMN] = parsed[Con.LIST_COLUMN].astype("int64")

    trial_number = pd.to_numeric(df[Con.TRIAL_ID], errors="coerce")
    if trial_number.isna().any():
        raise ValueError(
            f"{Con.TRIAL_ID} holds {int(trial_number.isna().sum())} non-numeric "
            f"value(s) -- the reports were already standardized?"
        )
    df[Con.TRIAL_NUMBER] = trial_number.astype("int64")
    df[Con.TRIAL_ID] = [
        format_trial_id(b, l, t)
        for b, l, t in zip(
            df[Con.BATCH_COLUMN], df[Con.LIST_COLUMN], df[Con.TRIAL_NUMBER]
        )
    ]

    if verbose:
        sessions = (
            df[[Con.PARTICIPANT_ID, Con.SESSION_ID]]
            .drop_duplicates()
            .groupby(Con.PARTICIPANT_ID)[Con.SESSION_ID]
            .apply(list)
        )
        for participant, labels_for_participant in sessions.items():
            print(f"  {participant}: {', '.join(sorted(labels_for_participant))}")
        repeats = sessions[sessions.apply(len) > 1]
        if len(repeats):
            # Worth stating plainly: any genuinely per-participant feature now
            # pools these sessions. That is right for a person-level trait
            # (pattern_breaking's dominance_score) and wrong for a measurement
            # baseline, which is why pupil z-scoring is done per session
            # instead -- see the module docstring.
            print(
                f"  NOTE: {len(repeats)} participant(s) sat more than one "
                f"session ({', '.join(repeats.index)}), so per-participant "
                f"features pool across them."
            )

    return df


# ---------------------------------------------------------------------------
# Column standardization
# ---------------------------------------------------------------------------

# Rename map: raw KnowQA column (left) -> pipeline column (right).
# RECORDING_SESSION_LABEL is deliberately absent: participant_id is derived from
# it by add_identity_columns, not renamed from it.
RENAME_MAP: dict[str, str] = {
    # `onestopqa_question_id` is deliberately NOT renamed to
    # `same_critical_span`: they are different orderings of the same three
    # questions. See `add_item_id_columns`.
    "practice": Con.PRACTICE_TRIAL_COLUMN,
    "correct_answer": Con.CORRECT_ANSWER_POSITION_COLUMN,  # 0-3
    "FINAL_ANSWER": Con.SELECTED_ANSWER_POSITION_COLUMN,  # 0-3
    # answers are 0-indexed (answer_0..answer_3); shift up one to the 1-indexed
    # answer_1..answer_4 the pipeline reads (this drops the old answer_0 name).
    f"{Con.ANSWER_PREFIX}0": f"{Con.ANSWER_PREFIX}1",
    f"{Con.ANSWER_PREFIX}1": f"{Con.ANSWER_PREFIX}2",
    f"{Con.ANSWER_PREFIX}2": f"{Con.ANSWER_PREFIX}3",
    f"{Con.ANSWER_PREFIX}3": f"{Con.ANSWER_PREFIX}4",
    # answer_a..answer_d already hold the labelled options; capitalize the
    # letter to the answer_A..answer_D the pipeline uses. Because these are
    # supplied directly, add_answer_text_columns (which would recompute them) is
    # excluded from the pipeline run.
    f"{Con.ANSWER_PREFIX}a": f"{Con.ANSWER_PREFIX}A",
    f"{Con.ANSWER_PREFIX}b": f"{Con.ANSWER_PREFIX}B",
    f"{Con.ANSWER_PREFIX}c": f"{Con.ANSWER_PREFIX}C",
    f"{Con.ANSWER_PREFIX}d": f"{Con.ANSWER_PREFIX}D",
}

# Base features to skip. answer_A..D are supplied directly by the rename above,
# so there is nothing to recompute.
EXCLUDED_BASE_FUNCS = {"add_answer_text_columns"}

# Skipped on top of those when pupil sizes are normalized per session -- the
# clean step then does the mm scaling and z-scoring itself. See
# `add_zscored_pupil_columns_by`.
PUPIL_BASE_FUNC = "add_zscored_pupil_columns"


def answers_order_to_letters(value):
    """Convert the raw numeric `answers_order` to the letter form the pipeline reads.

    build_exp_inputs.py builds answers_order as screen_position -> original
    answer index (0=A, 1=B, 2=C, 3=D), e.g. [2, 1, 0, 3]. The pipeline expects
    the *letter* at each screen position, e.g. ['C', 'B', 'A', 'D']. Values
    already in letter form are left untouched.
    """
    order = ast.literal_eval(value) if isinstance(value, str) else value
    return [
        Con.ANSWER_LABELS[int(x)] if str(x).lstrip("-").isdigit() else x for x in order
    ]


def add_item_id_columns(
    df: pd.DataFrame, strict: bool = True, verbose: bool = True
) -> pd.DataFrame:
    """Derive `same_critical_span` from the report's own `text_id_with_q`.

    The reports carry two different question indices for the same item:

    * `onestopqa_question_id` -- OneStopQA's ordering, which the experiment
      builder selects items by (its Latin square keys off this), and which the
      presented stimuli follow;
    * the last component of `text_id_with_q` -- the *L1* ordering, i.e. L1's
      `same_critical_span`. The experiment inputs inherited this key from L1
      through the text tables (`adv_texts_with_ids.csv`, built by
      `extract_text.ipynb` off L1's processed data).

    Within a paragraph the two are different permutations of the same three
    questions -- never a grouping, so both index items uniquely, but they
    disagree on ~40% of trials. Building `text_id_with_q` out of
    `onestopqa_question_id` therefore mints an L1-shaped key holding a non-L1
    value, so `1_6_Adv_1_0` would name a *different* question here than in L1
    and cross-dataset item matching would silently pair up the wrong items.

    So `same_critical_span` is taken from the report's `text_id_with_q` suffix.
    `add_text_id_with_q` in the pipeline then rebuilds the identical string,
    which keeps the key byte-for-byte L1-compatible, and
    `onestopqa_question_id` is kept under its own name for the OneStopQA
    ordering.
    """
    df = df.copy()

    if Con.TEXT_ID_WITH_Q_COLUMN not in df.columns:
        raise KeyError(
            f"{Con.TEXT_ID_WITH_Q_COLUMN!r} not in the report -- cannot derive "
            f"{Con.SAME_CRITICAL_SPAN_COLUMN!r}. The experiment inputs are "
            f"expected to carry it."
        )

    tid = df[Con.TEXT_ID_WITH_Q_COLUMN].astype(str).str.strip()
    prefix = tid.str.rsplit("_", n=1).str[0]
    suffix = tid.str.rsplit("_", n=1).str[1]

    # The prefix must be text_id (batch_article_difficulty_paragraph), otherwise
    # the pipeline's add_text_id_with_q would not reproduce this same string.
    expected_prefix = (
        df[Con.BATCH_COLUMN].astype(str)
        + "_"
        + df[Con.ARTICLE_COLUMN].astype(str)
        + "_"
        + df[Con.DIFFICULTY_COLUMN].astype(str)
        + "_"
        + df[Con.PARAGRAPH_COLUMN].astype(str)
    )
    mismatched = prefix != expected_prefix
    if mismatched.any():
        sample = (
            df.loc[mismatched, [Con.TEXT_ID_WITH_Q_COLUMN]]
            .assign(expected_prefix=expected_prefix[mismatched])
            .drop_duplicates()
            .head(5)
        )
        message = (
            f"{int(mismatched.sum())} row(s) where {Con.TEXT_ID_WITH_Q_COLUMN} does "
            f"not start with batch_article_difficulty_paragraph, so the pipeline "
            f"would not rebuild it identically:\n{sample.to_string(index=False)}"
        )
        if strict:
            raise ValueError(message)
        print(f"WARNING: {message}")

    bad_suffix = ~suffix.str.fullmatch(r"\d+").fillna(False)
    if bad_suffix.any():
        raise ValueError(
            f"{int(bad_suffix.sum())} row(s) have a non-numeric "
            f"{Con.TEXT_ID_WITH_Q_COLUMN} suffix, e.g. "
            f"{tid[bad_suffix].unique()[:3].tolist()}"
        )

    df[Con.SAME_CRITICAL_SPAN_COLUMN] = suffix.astype("int64")

    if verbose and Con.ONESTOPQA_QUESTION_ID in df.columns:
        trials = df.drop_duplicates([Con.SESSION_ID, Con.TRIAL_NUMBER])
        agree = (
            trials[Con.SAME_CRITICAL_SPAN_COLUMN]
            == pd.to_numeric(trials[Con.ONESTOPQA_QUESTION_ID], errors="coerce")
        )
        print(
            f"  question index: same_critical_span (L1 ordering) agrees with "
            f"onestopqa_question_id on {int(agree.sum())}/{len(trials)} trials; "
            f"both kept"
        )

    return df


def standardize_columns(
    df: pd.DataFrame, strict: bool = True, verbose: bool = True
) -> pd.DataFrame:
    """Rename raw KnowQA columns to the pipeline's names and derive the identity columns.

    Renames are applied in a single pass, so the answer 0->1, 1->2, ... shift
    does not collide; columns not present are left untouched.
    """
    df = df.rename(columns=RENAME_MAP)
    if Con.ANSWERS_ORDER_COLUMN in df.columns:
        df[Con.ANSWERS_ORDER_COLUMN] = df[Con.ANSWERS_ORDER_COLUMN].apply(
            answers_order_to_letters
        )
    df = add_identity_columns(df, strict=strict, verbose=verbose)
    return add_item_id_columns(df, strict=strict, verbose=verbose)


def check_trial_index_gaps(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Account for trial indices that never reach the report.

    A recalibration consumes a trial index without producing an Answers
    interest period, so gaps in TRIAL_INDEX are expected -- and the count of
    gaps in a session should equal that session's NUM_RECALIBRATIONS. Anything
    else is real data loss, so this reports the two side by side instead of
    letting a silently short session pass.

    Returns one row per session with the gap list and the recalibration count.
    """
    trials = df.drop_duplicates([Con.SESSION_ID, Con.TRIAL_NUMBER])
    rows = []
    for session, g in trials.groupby(Con.SESSION_ID):
        present = set(g[Con.TRIAL_NUMBER])
        gaps = sorted(set(range(1, max(present) + 1)) - present)
        recal = pd.to_numeric(g.get(RECALIBRATIONS_COLUMN), errors="coerce")
        n_recal = int(recal.max()) if recal is not None and recal.notna().any() else None
        rows.append({
            Con.SESSION_ID: session,
            "n_trials": len(present),
            "max_trial_number": max(present),
            "n_gaps": len(gaps),
            "n_recalibrations": n_recal,
            "explained": n_recal is not None and len(gaps) == n_recal,
            "gaps": gaps,
        })
    out = pd.DataFrame(rows)

    if verbose:
        for r in rows:
            status = "ok" if r["explained"] else "UNEXPLAINED"
            print(
                f"  {r[Con.SESSION_ID]}: {r['n_trials']} trials, "
                f"{r['n_gaps']} missing index(es) {r['gaps']}, "
                f"{r['n_recalibrations']} recalibration(s) -> {status}"
            )
        unexplained = [r for r in rows if not r["explained"]]
        if unexplained:
            print(
                "  WARNING: missing trial indices exceed the recalibration count in "
                f"{len(unexplained)} session(s) -- that is real data loss, not a "
                "recalibration artifact."
            )
    return out


# Screen-position answer texts, after RENAME_MAP has shifted answer_0..3 up to
# answer_1..4. The interest areas run in this order: the question, then the four
# answers in screen order.
SCREEN_TEXT_COLUMNS = ("question",) + tuple(
    f"{Con.ANSWER_PREFIX}{i}" for i in range(1, 5)
)


def _tokens(value) -> list[str]:
    return str(value).split()


def check_text_alignment(
    ia_df: pd.DataFrame, strict: bool = False, verbose: bool = True
) -> pd.DataFrame:
    """Check that the interest areas line up word-for-word with the stimulus text.

    The interest areas are one per word, in reading order: the question, then
    the four answers in screen order. Rebuilding that sequence from the
    `question` / `answer_1..4` columns and comparing it to `IA_LABEL` catches
    trials where the two have drifted out of step.

    This is not hypothetical. The experiment's stored text sometimes carries a
    displaced double quote (e.g. `word "cool" "?"` for a screen that rendered
    `word "cool"?`), which adds a token the display never showed. It comes from
    the presentation software rather than from anything in this repo: the stimulus
    CSVs hold `"cool"?` as one token and the raw recording already has it split.
    It is predictable -- the 12 stimulus rows whose question ends in a quoted term
    followed by `?` are the whole at-risk set.

    **What it no longer means.** This check used to be the only thing standing
    between that defect and the data, because `add_IA_screen_location` placed words
    by counting stored tokens, so one phantom token shifted every later area
    boundary. It now places words by their on-screen rectangle and reports any
    override (`data_csv_generation._reconcile_area_with_geometry`), so **the area
    labels on these trials are correct** and the flagged trials need no repair.

    What the check is still for: the stored text is a defect worth fixing upstream
    in the experiment materials, and anything derived from that text rather than
    from the interest areas -- a word count, a length -- is off by a token on these
    trials. See `docs/todo.md` T3.18 and
    `notebooks/text_alignment_investigation.ipynb`.

    Returns one row per misaligned trial. `strict` raises instead of warning.
    """
    missing = [c for c in SCREEN_TEXT_COLUMNS if c not in ia_df.columns]
    if missing:
        if verbose:
            print(f"  text alignment: skipped, columns absent: {missing}")
        return pd.DataFrame()
    if Con.IA_LABEL not in ia_df.columns:
        if verbose:
            print(f"  text alignment: skipped, {Con.IA_LABEL!r} absent")
        return pd.DataFrame()

    trial_keys = [Con.SESSION_ID, Con.TRIAL_NUMBER]
    ordered = ia_df.sort_values(trial_keys + [Con.INTEREST_AREA_ID])

    rows = []
    for key, g in ordered.groupby(trial_keys, sort=False):
        on_screen = [str(x) for x in g[Con.IA_LABEL]]
        first = g.iloc[0]
        expected: list[str] = []
        for col in SCREEN_TEXT_COLUMNS:
            expected += _tokens(first[col])
        if on_screen == expected:
            continue
        # locate the first divergence, which is where the boundaries slipped
        slip = next(
            (i for i, (a, b) in enumerate(zip(on_screen, expected)) if a != b),
            min(len(on_screen), len(expected)),
        )
        rows.append({
            Con.SESSION_ID: key[0],
            Con.TRIAL_NUMBER: key[1],
            Con.TEXT_ID_WITH_Q_COLUMN: first.get(Con.TEXT_ID_WITH_Q_COLUMN),
            "n_interest_areas": len(on_screen),
            "n_expected_tokens": len(expected),
            "first_mismatch_at": slip,
            "on_screen": " ".join(on_screen[max(slip - 1, 0):slip + 3]),
            "expected": " ".join(expected[max(slip - 1, 0):slip + 3]),
        })

    out = pd.DataFrame(rows)
    n_trials = ordered.groupby(trial_keys, sort=False).ngroups

    if verbose:
        print(
            f"  text alignment: {n_trials - len(out)}/{n_trials} trials line up "
            f"word-for-word with the stimulus text"
        )
    if len(out):
        message = (
            f"{len(out)} trial(s) whose interest areas do not match the stored "
            "stimulus text. The screen-area labels are NOT affected -- words are "
            "placed by their on-screen rectangle -- so nothing here needs "
            "repairing in the recordings. It is the stored text that is wrong, "
            "which is a defect to fix in the experiment materials, and any measure "
            "taken from that text rather than from the interest areas is off by a "
            "token on these trials:"
            "\n"
            f"{out.to_string(index=False)}"
        )
        if strict:
            raise ValueError(message)
        if verbose:
            print(f"  WARNING: {message}")
    return out


# ---------------------------------------------------------------------------
# Pupil normalization
# ---------------------------------------------------------------------------

# The IA-level pupil columns that get scaled to mm and z-scored.
PUPIL_IA_COLUMNS = (
    Con.IA_MAX_FIX_PUPIL_SIZE,
    Con.IA_MIN_FIX_PUPIL_SIZE,
    Con.IA_AVERAGE_FIX_PUPIL_SIZE,
)

PUPIL_STATS_NAME = "pupil_stats_by_session.csv"

# "session" normalizes each recording separately; "participant" pools all of a
# person's sessions, which is what the pipeline's own base feature does.
PUPIL_NORM_UNITS = ("session", "participant")


def compute_pupil_stats_by(fixations: pd.DataFrame, group_col: str) -> pd.DataFrame:
    """Mean and SD of fixation pupil size (in mm), per `group_col`.

    `derived.pupil_norm.compute_participant_pupil_stats` does the same thing but
    hard-codes `participant_id` as the group; this takes the column so pupil
    size can be normalized per recording session.
    """
    from src.derived.pupil_norm import scale_pupil_area_to_mm

    if Con.CURRENT_FIX_PUPIL_SIZE not in fixations.columns:
        raise KeyError(
            f"{Con.CURRENT_FIX_PUPIL_SIZE!r} not in the fixations report -- "
            f"cannot compute pupil stats."
        )
    pupil_mm = scale_pupil_area_to_mm(fixations[Con.CURRENT_FIX_PUPIL_SIZE])
    return (
        pd.DataFrame({group_col: fixations[group_col].values, "pupil_mm": pupil_mm.values})
        .groupby(group_col)["pupil_mm"]
        .agg(pupil_mean="mean", pupil_sd="std")
        .reset_index()
    )


def add_zscored_pupil_columns_by(
    ia_df: pd.DataFrame,
    fixations_df: pd.DataFrame,
    group_col: str = Con.SESSION_ID,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Scale the IA pupil columns to mm and z-score them within `group_col`.

    The same work as `data_csv_generation.add_zscored_pupil_columns`, except
    that function z-scores within `participant_id`. For KnowQA a person can
    have several recording sessions, and pupil baseline shifts between them
    (setup, lighting, fatigue), so pooling a person's sessions into one mean/SD
    is the wrong baseline -- normalize within the session instead.

    Returns (ia_df with the `_z` columns, the stats used).
    """
    from src.derived.pupil_norm import (
        scale_pupil_area_to_mm,
        zscore_pupil_by_participant,
    )

    stats = compute_pupil_stats_by(fixations_df, group_col)

    out = ia_df.copy()
    for col in PUPIL_IA_COLUMNS:
        if col not in out.columns:
            raise KeyError(f"{col!r} not in the IA report -- cannot z-score pupils.")
        out[col] = scale_pupil_area_to_mm(out[col])
        out = zscore_pupil_by_participant(
            df=out,
            pupil_col=col,
            participant_col=group_col,
            stats=stats,
            out_col=f"{col}_z",
        )
    return out, stats


# ---------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------


def clean_reports(
    raw_dir: Path,
    pupil_norm_unit: str = "session",
    strict: bool = True,
    verbose: bool = True,
) -> dict[str, Path]:
    """Standardize a run's converted CSVs into `raw_dir/csvs/cleaned/`.

    The IA and fixations reports are handled together because the pupil
    normalization spans both: the per-session mean/SD comes from the fixations
    report and is applied to the IA report's pupil columns.

    Returns {canonical name: written path}.
    """
    if pupil_norm_unit not in PUPIL_NORM_UNITS:
        raise ValueError(
            f"pupil_norm_unit must be one of {PUPIL_NORM_UNITS}, got {pupil_norm_unit!r}"
        )

    raw_dir = Path(raw_dir)
    csv_dir = raw_dir / CSV_SUBDIR
    cleaned_dir = csv_dir / CLEANED_SUBDIR
    cleaned_dir.mkdir(parents=True, exist_ok=True)

    frames: dict[str, pd.DataFrame] = {}
    for name in CLEANED_REPORT_NAMES:
        source = csv_dir / name
        if not source.exists():
            raise FileNotFoundError(
                f"{source} not found -- run the convert step for {raw_dir.name} first."
            )
        if verbose:
            print(f"{name}:")
        df = pd.read_csv(source, low_memory=False)
        frames[name] = standardize_columns(df, strict=strict, verbose=verbose)

    if verbose:
        print("trial-index accounting (gaps should equal recalibrations):")
    check_trial_index_gaps(frames[IA_REPORT_NAME], verbose=verbose)

    # Interest areas must stay word-aligned with the stimulus text; a displaced
    # quote in the stored text silently shifts every later area by one word.
    check_text_alignment(frames[IA_REPORT_NAME], strict=False, verbose=verbose)

    written: dict[str, Path] = {}

    if pupil_norm_unit == "session":
        if verbose:
            print(f"pupil z-scoring within {Con.SESSION_ID}:")
        frames[IA_REPORT_NAME], pupil_stats = add_zscored_pupil_columns_by(
            frames[IA_REPORT_NAME], frames[FIX_REPORT_NAME], group_col=Con.SESSION_ID
        )
        stats_path = cleaned_dir / PUPIL_STATS_NAME
        pupil_stats.to_csv(stats_path, index=False)
        written[PUPIL_STATS_NAME] = stats_path
        if verbose:
            print(pupil_stats.to_string(index=False))

    for name, df in frames.items():
        out_path = cleaned_dir / name
        df.to_csv(out_path, index=False)
        written[name] = out_path
        if verbose:
            print(f"{name}: {df.shape} -> {CSV_SUBDIR}/{CLEANED_SUBDIR}/{name}")
    return written


# ---------------------------------------------------------------------------
# The processing pipeline
# ---------------------------------------------------------------------------


def _require_pupil_z_columns(ia_path: Path) -> None:
    """Fail early if a cleaned IA report is missing the per-session `_z` columns.

    Guards against running with `pupil_norm_unit="session"` over a cleaned CSV
    that was written under `"participant"`: the pupil base feature would then be
    skipped with nothing having produced its output, and the pupil group
    features would silently come back empty.
    """
    header = pd.read_csv(ia_path, nrows=0)
    missing = [f"{c}_z" for c in PUPIL_IA_COLUMNS if f"{c}_z" not in header.columns]
    if missing:
        raise ValueError(
            f"{ia_path} is missing {missing} -- it was cleaned without "
            f"per-session pupil normalization. Re-run the clean step with "
            f"pupil_norm_unit='session', or run the pipeline with "
            f"pupil_norm_unit='participant'."
        )


def run_pipeline(
    cleaned_dir: Path,
    out_dir: Path,
    pupil_norm_unit: str = "session",
    include_paragraph: bool = False,
    verbose: bool = True,
) -> Path:
    """Run `data_csv_generation.main()` on a run's cleaned reports.

    Runs base + group features, plus the last-label and RT/TFD steps (fed by
    button clicks extracted from the fixations report itself, which is
    self-contained, so a missing messages report does not matter).

    `pupil_norm_unit` must match what `clean_reports` used: with `"session"`
    the cleaned IA report already carries the mm-scaled and z-scored pupil
    columns, so the pipeline's own (per-participant) pupil base feature is
    skipped.

    Paragraph-region RT/TFD is off: KnowQA has no paragraph-reading report, and
    `derived.reading_times.load_paragraph_fixations` coerces TRIAL_INDEX to
    int64, which the composite trial ids cannot survive -- see the module
    docstring.
    """
    from src.data_prep import data_csv_generation as dcg

    if pupil_norm_unit not in PUPIL_NORM_UNITS:
        raise ValueError(
            f"pupil_norm_unit must be one of {PUPIL_NORM_UNITS}, got {pupil_norm_unit!r}"
        )
    if include_paragraph:
        raise ValueError(
            "include_paragraph=True is unsupported for KnowQA: "
            "derived.reading_times.load_paragraph_fixations coerces TRIAL_INDEX "
            "to int64 and drops what will not convert, so the composite trial "
            "ids would silently reduce the paragraph fixations to nothing. Make "
            "that loader dtype-agnostic before turning this on."
        )

    cleaned_dir = Path(cleaned_dir)
    out_dir = Path(out_dir)
    aux_dir = out_dir / "Auxiliary"
    out_dir.mkdir(parents=True, exist_ok=True)
    aux_dir.mkdir(parents=True, exist_ok=True)

    ia_path = cleaned_dir / IA_REPORT_NAME
    fix_path = cleaned_dir / FIX_REPORT_NAME
    for label, path in [("IA report", ia_path), ("fixations report", fix_path)]:
        if not path.exists():
            raise FileNotFoundError(
                f"{label} not found at {path} -- run the clean step first."
            )

    output_path = out_dir / "all_participants.csv"

    # Run every registered base feature except the excluded ones (built from the
    # registry so it stays correct if base features are added or removed).
    excluded = set(EXCLUDED_BASE_FUNCS)
    if pupil_norm_unit == "session":
        # The clean step already scaled and z-scored the pupil columns per
        # session; the pipeline's version would redo it per participant.
        excluded.add(PUPIL_BASE_FUNC)
        _require_pupil_z_columns(ia_path)
    base_function_names = [
        name
        for name, entry in dcg.FUNCTION_REGISTRY.items()
        if entry["kind"] == "base" and name not in excluded
    ]

    dcg.main(
        ia_answers_path=ia_path,
        output_path=output_path,
        # The single canonical fixations report: feeds both the pupil stats and
        # create_fixation_sequence_tags, so all group features can run.
        fixations_path=fix_path,
        # Both of the next two are INERT on this path, and deliberately kept.
        # KnowQA needs no participant-level pupil baseline: pupil size is already
        # z-scored per `session_id` in Stage 0, which is the right unit here
        # because one participant_id spans several sittings. So
        # `add_zscored_pupil_columns` is excluded above (see PUPIL_BASE_FUNC), and
        # main() gates its whole pupil-stats block on that base feature running --
        # nothing reads or writes a pupil-stats file for KnowQA.
        #
        # They stay because the alternative is worse: `pupil_stats_path` defaults
        # to L1's PARTICIPANT_PUPILS_PATH, so dropping it would mean that anyone
        # re-enabling the base feature for KnowQA would silently baseline against
        # L1's answer screen -- exactly the bug `todo.md` T3.20 is about. Pinned
        # here, that cannot happen.
        #
        # NB: `data/KnowQA_runs/Auxiliary/participant_pupils.csv` on disk is a
        # leftover from a pre-Stage-0 run, NOT output of this path. Do not read it.
        compute_pupil_stats=True,
        pupil_stats_path=aux_dir / "participant_pupils.csv",
        button_clicks_path=aux_dir / "button_clicks_data.csv",
        last_labels_path=aux_dir / "all_participants_last.csv",
        rt_and_tfd_path=aux_dir / "RT_and_TFD.csv",
        save_auxiliary=True,
        # last-label + RT/TFD features, fed by the button clicks built below
        add_last=True,
        add_rts=True,
        include_paragraph=False,
        rebuild_button_clicks=True,
        # Button clicks come from the fixations report itself (it holds both the
        # message-list and fixation columns), not the legacy CSV + TSV pair.
        button_clicks_fix_csv_path=fix_path,
        button_clicks_fix_tsv_path=None,  # single source
        button_clicks_msg_participant_col=Con.PARTICIPANT_ID,
        all_answers_is_cumulative=False,  # ALL_ANSWERS is per-trial here
        base_function_names=base_function_names,
        group_function_names=None,  # None = all registered group features
        # No repeated-reading trials in this experiment (no
        # repeated_reading_trial column), so that filter is off; practice trials
        # are still removed.
        remove_repeats=False,
        remove_practice=True,
        verbose=verbose,
    )

    if verbose:
        print(f"\n[OK] pipeline output -> {output_path}")
    return output_path


def read_prepared_csv(path: Path, **kwargs) -> pd.DataFrame:
    """Read one of this module's CSVs, keeping `participant_id` a string.

    A KnowQA person id is all digits (`4000`), so a plain `read_csv` infers
    int64 for it while every other dataset in the project (L1's `l42_2070`)
    carries strings. Any merge across the two then dies on the dtype mismatch,
    so pin it here at every read.
    """
    dtype = {Con.PARTICIPANT_ID: str, **kwargs.pop("dtype", {})}
    return pd.read_csv(Path(path), dtype=dtype, low_memory=False, **kwargs)


def build_features(
    processed_path: Path, out_dir: Path, verbose: bool = True
) -> tuple[Path, pd.DataFrame]:
    """Turn the processed IA-level data into the one-row-per-trial feature table.

    Written to `out_dir/L1_model_ready_all_features.csv`; read it back with
    `answer_correctness.model_data.load_all_features`.

    This mirrors `model_data.save_all_features` (every include_* flag on) with
    one exception: the paragraph-span block is left out. Those features are
    measured on the paragraph-reading screen, which only the partial-knowledge
    regime has -- a third of the trials -- so they are not usable here, and
    `save_all_features` would otherwise left-join L1's cache and leave three
    all-NaN columns behind.
    """
    from src.predictive_modeling.answer_correctness.model_data import (
        build_trial_level_model_df,
    )

    processed_path = Path(processed_path)
    if not processed_path.exists():
        raise FileNotFoundError(
            f"{processed_path} not found -- run the pipeline step first."
        )

    features_path = Path(out_dir) / "L1_model_ready_all_features.csv"
    processed_ia = read_prepared_csv(processed_path)

    features_df = build_trial_level_model_df(
        df=processed_ia,
        keep_cols=[Con.TEXT_ID_WITH_Q_COLUMN],
        target_col=Con.IS_CORRECT_COLUMN,
        include_area_features=True,
        include_derived_features=True,
        include_last_lbl_before_confirm_features=True,
        include_last_lbl_before_select_features=True,
        include_rt_tfd_features=True,
        include_total_answering_rt=True,
        include_pattern_features=True,
        include_paragraph_features=False,  # no paragraph screen for 2 of 3 regimes
    )

    features_path.parent.mkdir(parents=True, exist_ok=True)
    features_df.to_csv(features_path, index=False)

    if verbose:
        print(
            f"Saved {len(features_df)} trials x {len(features_df.columns)} cols "
            f"to {features_path}"
        )
        print("target (is_correct) distribution:")
        print(features_df[Con.IS_CORRECT_COLUMN].value_counts(dropna=False).to_string())
    return features_path, features_df


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

STEPS = ("convert", "clean", "pipeline", "features")


def prepare_know_qa(
    run: str = DEFAULT_RUN,
    steps: Iterable[str] = STEPS,
    pupil_norm_unit: str = "session",
    strict: bool = True,
    verbose: bool = True,
    runs: Mapping[str, tuple[Path, Path]] = RAW_RUNS,
) -> dict[str, Path]:
    """Run the KnowQA preparation end to end for one raw run.

    `steps` selects which stages run, in the fixed order convert -> clean ->
    pipeline -> features, so a later stage can be re-run on its own once the
    earlier ones have been done. `strict` controls whether a label/report
    batch-list disagreement raises or only warns. `pupil_norm_unit` sets the
    baseline pupil sizes are z-scored against -- see `add_zscored_pupil_columns_by`.

    Returns a {name: path} map of what was produced.
    """
    if run not in runs:
        raise KeyError(f"unknown run {run!r}; known runs: {sorted(runs)}")
    requested = [s for s in STEPS if s in set(steps)]
    unknown = set(steps) - set(STEPS)
    if unknown:
        raise ValueError(f"unknown step(s) {sorted(unknown)}; known steps: {STEPS}")

    raw_dir, out_dir = runs[run]
    cleaned_dir = raw_dir / CSV_SUBDIR / CLEANED_SUBDIR
    processed_path = out_dir / "all_participants.csv"

    if verbose:
        print(f"run:   {run}")
        print(f"raw:   {raw_dir}")
        print(f"out:   {out_dir}")
        print(f"steps: {', '.join(requested)}")
        print(f"pupil: z-scored per {pupil_norm_unit}")

    produced: dict[str, Path] = {}

    if "convert" in requested:
        if verbose:
            print("\n--- convert ---")
        produced.update(convert_raw_reports(raw_dir, verbose=verbose))

    if "clean" in requested:
        if verbose:
            print("\n--- clean ---")
        cleaned = clean_reports(
            raw_dir,
            pupil_norm_unit=pupil_norm_unit,
            strict=strict,
            verbose=verbose,
        )
        produced.update({f"cleaned/{k}": v for k, v in cleaned.items()})

    if "pipeline" in requested:
        if verbose:
            print("\n--- pipeline ---")
        produced["processed"] = run_pipeline(
            cleaned_dir,
            out_dir,
            pupil_norm_unit=pupil_norm_unit,
            verbose=verbose,
        )

    if "features" in requested:
        if verbose:
            print("\n--- features ---")
        features_path, _ = build_features(processed_path, out_dir, verbose=verbose)
        produced["features"] = features_path

    return produced


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare a KnowQA raw run for analysis.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--run",
        default=DEFAULT_RUN,
        choices=sorted(RAW_RUNS),
        help="which raw run to prepare",
    )
    parser.add_argument(
        "--steps",
        nargs="+",
        default=list(STEPS),
        choices=list(STEPS),
        metavar="STEP",
        help="stages to run (convert clean pipeline features)",
    )
    parser.add_argument(
        "--pupil-norm-unit",
        default="session",
        choices=list(PUPIL_NORM_UNITS),
        help="baseline pupil sizes are z-scored against; 'participant' pools a "
        "person's sessions",
    )
    parser.add_argument(
        "--lenient",
        action="store_true",
        help="warn instead of raising when a label's batch/list disagrees with "
        "the report's own columns",
    )
    parser.add_argument("--quiet", action="store_true", help="suppress progress output")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> dict[str, Path]:
    args = _parse_args(argv)
    return prepare_know_qa(
        run=args.run,
        steps=args.steps,
        pupil_norm_unit=args.pupil_norm_unit,
        strict=not args.lenient,
        verbose=not args.quiet,
    )


def _make_stdout_unicode_safe() -> None:
    """Stop a redirected run from dying on the pipeline's non-ASCII progress output.

    `data_csv_generation.main` (and `answers_paragraphs_csv`) finish by printing
    a U+2713 check mark. On Windows that is fine on a console but not when
    stdout is a file or a pipe, where Python falls back to the cp1252 code page
    and the print raises UnicodeEncodeError -- after all the work is done. Only
    the command-line entry point does this, so importing the module and calling
    `prepare_know_qa` from a notebook leaves the streams alone.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except (ValueError, OSError):
            pass  # already detached or not reconfigurable; nothing to do


if __name__ == "__main__":
    _make_stdout_unicode_safe()
    main()
