"""Prepare Study 2 (KnowQA) data, from the tracker's raw exports to the model table.

Four steps, run in this order by `prepare_know_qa()`:

    convert    raw .xls / .tsv reports -> CSVs        data_raw/<run>/csvs/
    clean      rename columns, build the ids          data_raw/<run>/csvs/cleaned/
    pipeline   run ingest/build.py on the cleaned     data/datasets/knowqa/interim/
               reports -> the three tables
    features   the three tables -> one row per        data/datasets/knowqa/features/
               trial, the same way as Study 1

From the command line::

    python -m src.ingest.knowqa                    # all four steps
    python -m src.ingest.knowqa --steps convert clean
    python -m src.ingest.knowqa --steps features

How a Study 2 trial is identified
---------------------------------
A Study 2 recording is named like ``4000_21``: ``4XXX`` is the person, ``Y``
(1-3) the batch and ``Z`` (1-18) the list, with no separator because the
tracker allows only 8 characters. So the name identifies a **session**, and one
person can come back for several: ``4000_11``, ``4000_21`` and ``4000_31`` are
all person ``4000``.

The name is split into:

    session_id      4000_21   the recording name, as is
    participant_id  4000      the person
    article_batch   2
    list_number     1
    trial_number    5         the tracker's trial counter within the session
    TRIAL_INDEX     b2l01t005 batch + list + trial, unique within a person

The whole project identifies a trial by (participant_id, TRIAL_INDEX). Folding
batch and list into TRIAL_INDEX keeps that pair unique for Study 2 too, so all
the shared code works unchanged.

TRIAL_INDEX is text, not a number, on purpose: code that expects a number gets
NaN and fails, instead of quietly reading a wrong one. It is fixed-width, so
sorting it as text gives the same order as sorting by batch, list and trial --
which `clicks.truncate_recordings_at_first_malformed_trial` relies on.

Study 2 has no paragraph data, so `run_pipeline` refuses
`include_paragraph=True`. (The paragraph loader would also need changing first:
it converts TRIAL_INDEX to a number and drops what will not convert, which for
these ids would be every row.)

Pupil baseline: one per person
------------------------------
Each participant's typical pupil size is worked out once, pooling all their
sessions and all three knowledge regimes -- the same as Study 1, so the pupil
measures mean the same thing in both studies. `pupil_norm_unit="session"` (or
``--pupil-norm-unit session``) instead gives one baseline per session; then the
z-scoring is done in the `clean` step and the pipeline's own pupil feature is
skipped.
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

from src.config import columns as Con
from src.config.datasets import KNOW_QA_OUT_DIR, KNOW_QA_PATH, dataset, dataset_for_root

# ---------------------------------------------------------------------------
# Runs
# ---------------------------------------------------------------------------

# name -> (raw report folder, processed output folder).
#
# Only collections whose recordings are named `4XXX_YZ` belong here. The two
# pilots are named differently (`r1_l1_b1`, `T1_l1`), so `parse_session_label`
# rejects them; they are prepared by notebooks/builders/data_prep_new_exp.ipynb.
#
# Another collection in this format needs a line here (or a `runs` mapping
# passed to `prepare_know_qa`) and a dataset record in config/datasets.py.
RAW_RUNS: dict[str, tuple[Path, Path]] = {
    "KnowQA": (KNOW_QA_PATH, KNOW_QA_OUT_DIR),
}

DEFAULT_RUN = "KnowQA"

# ---------------------------------------------------------------------------
# Reading the raw reports
# ---------------------------------------------------------------------------

# Every export is tab-separated text; only the extension, the text encoding and
# the file names differ, so one reader handles them all.
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

# The two reports the pipeline needs: words and fixations.
CLEANED_REPORT_NAMES = (IA_REPORT_NAME, FIX_REPORT_NAME)


def read_raw_report(path: Path) -> pd.DataFrame:
    """Read one raw report (.xls / .tsv / .csv) exported by the tracker software.

    Quote characters are read as ordinary text, because the stimulus text
    contains stray quotes. The UTF-16 .xls files need pandas' slower python
    parser; the others use the fast one.
    """
    encoding = RAW_REPORT_ENCODINGS[path.suffix.lower()]
    kwargs: dict = dict(encoding=encoding, sep="\t", quoting=csv.QUOTE_NONE)
    if encoding == "utf-16":
        kwargs["engine"] = "python"
    else:
        kwargs["low_memory"] = False
    return pd.read_csv(path, **kwargs)


def canonical_report_name(path: Path) -> str:
    """The standard CSV name for a raw report file.

    The exports are named differently (`IA_Answers.xls`, `A_IA.tsv`,
    `A_fixations.tsv`, ...); this maps them all onto `IA_answers.csv`,
    `fixations_answers.csv` or `messages_answers.csv`.
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
    """Convert every raw report in `raw_dir` to a standard-named CSV in `raw_dir/csvs/`.

    Stops with an error if the word or fixation report is missing. Returns
    {CSV name: path written}.
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

# `4XXX_YZ`: person, batch (1 digit, 1-3), list (1-18, no separator). Two-digit
# lists are tried first, so `4000_118` reads as batch 1 / list 18 rather than
# batch 1 / list 1 with a stray "8".
SESSION_LABEL_RE = re.compile(
    r"^(?P<participant>\d+)_(?P<batch>[1-3])(?P<list>1[0-8]|[1-9])$"
)

BATCH_RANGE = (1, 3)
LIST_RANGE = (1, 18)

# The experiment also writes the batch and list as their own columns. The parsed
# recording name is checked against them rather than trusted alone.
RAW_BATCH_COLUMN = "article_batch"
RAW_LIST_COLUMN = "list_numeric"

# How many times the tracker was recalibrated. Each recalibration uses up a
# trial number without producing an answer screen. See `check_trial_index_gaps`.
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
    """The Study 2 trial id, e.g. (2, 1, 5) -> 'b2l01t005'. See the module docstring."""
    return f"b{int(batch)}l{int(list_number):02d}t{int(trial_number):03d}"


def _check_against_raw(
    df: pd.DataFrame, parsed: pd.DataFrame, raw_col: str, parsed_col: str
) -> list[str]:
    """Compare a value parsed from the recording name with the experiment's own
    column. Returns a description of any rows that disagree."""
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
    """Split the recording name into the Study 2 id columns.

    Adds `session_id`, `participant_id`, `article_batch`, `list_number` and
    `trial_number`, and replaces `TRIAL_INDEX` with the Study 2 trial id (see
    the module docstring). The parsed batch and list are checked against the
    report's own columns; with `strict` a disagreement stops the run, otherwise
    it is printed as a warning.

    A frame that already has `session_id` is returned unchanged, so running
    this twice is harmless.
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

    # Parse each distinct name once, then copy back onto the rows: the reports
    # have hundreds of thousands of rows but only a handful of sessions.
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
            # Anything computed per participant -- the pupil baseline, the
            # dominant scan strategy -- pools these sessions together.
            print(
                f"  NOTE: {len(repeats)} participant(s) sat more than one "
                f"session ({', '.join(repeats.index)}), so per-participant "
                f"features pool across them."
            )

    return df


# ---------------------------------------------------------------------------
# Column standardization
# ---------------------------------------------------------------------------

# Raw Study 2 column name (left) -> the name the pipeline uses (right).
# RECORDING_SESSION_LABEL is not here: participant_id is worked out from it by
# add_identity_columns, not simply renamed.
RENAME_MAP: dict[str, str] = {
    # `onestopqa_question_id` is NOT renamed to `same_critical_span`: they number
    # the same three questions in different orders. See `add_item_id_columns`.
    "practice": Con.PRACTICE_TRIAL_COLUMN,
    "correct_answer": Con.CORRECT_ANSWER_POSITION_COLUMN,  # 0-3
    "FINAL_ANSWER": Con.SELECTED_ANSWER_POSITION_COLUMN,  # 0-3
    # The report numbers screen positions from 0 (answer_0..answer_3); the
    # pipeline numbers them from 1 (answer_1..answer_4).
    f"{Con.ANSWER_PREFIX}0": f"{Con.ANSWER_PREFIX}1",
    f"{Con.ANSWER_PREFIX}1": f"{Con.ANSWER_PREFIX}2",
    f"{Con.ANSWER_PREFIX}2": f"{Con.ANSWER_PREFIX}3",
    f"{Con.ANSWER_PREFIX}3": f"{Con.ANSWER_PREFIX}4",
    # answer_a..answer_d already hold the text by letter; only the case changes.
    # Because of this, add_answer_text_columns is skipped for Study 2.
    f"{Con.ANSWER_PREFIX}a": f"{Con.ANSWER_PREFIX}A",
    f"{Con.ANSWER_PREFIX}b": f"{Con.ANSWER_PREFIX}B",
    f"{Con.ANSWER_PREFIX}c": f"{Con.ANSWER_PREFIX}C",
    f"{Con.ANSWER_PREFIX}d": f"{Con.ANSWER_PREFIX}D",
}

# The row features Study 2 skips, as listed on its dataset record in
# config/datasets.py. The pilots' notebook imports this name; `run_pipeline`
# reads the record directly.
EXCLUDED_BASE_FUNCS = set(dataset("knowqa").skip_base_features)

# Also skipped when pupil size is normalized per session, because the clean step
# has then already converted and z-scored it. See `add_zscored_pupil_columns_by`.
PUPIL_BASE_FUNC = "add_zscored_pupil_columns"


def answers_order_to_letters(value):
    """Turn a numeric `answers_order` into the letters the pipeline reads.

    The Study 2 materials (`src/experiment/lists.py`) store, for each screen
    position, the number of the answer shown there (0=A, 1=B, 2=C, 3=D), e.g.
    [2, 1, 0, 3]. The pipeline wants the letters, e.g. ['C', 'B', 'A', 'D'].
    Values that are already letters are left as they are.
    """
    order = ast.literal_eval(value) if isinstance(value, str) else value
    return [
        Con.ANSWER_LABELS[int(x)] if str(x).lstrip("-").isdigit() else x for x in order
    ]


def add_item_id_columns(
    df: pd.DataFrame, strict: bool = True, verbose: bool = True
) -> pd.DataFrame:
    """Add `same_critical_span`, taken from the end of the report's `text_id_with_q`.

    Study 2's reports number each paragraph's three questions in two different
    orders:

    * `onestopqa_question_id` -- OneStopQA's own order, which the Study 2
      materials were built from;
    * the last part of `text_id_with_q` -- Study 1's order, i.e. Study 1's
      `same_critical_span`.

    The two disagree on about a third of trials. Using Study 1's order here means a
    `text_id_with_q` such as `1_6_Adv_1_0` names the same question in both
    studies, so items can be matched across them. `onestopqa_question_id` is
    kept under its own name.

    Checks that `text_id_with_q` starts with the paragraph's own text_id, so the
    pipeline will rebuild exactly the same string; with `strict` a mismatch
    stops the run.
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

    # The start must be text_id (batch_article_difficulty_paragraph), or the
    # pipeline's add_text_id_with_q would build a different string.
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
    """Rename the raw Study 2 columns, convert `answers_order` to letters, and
    add the id columns.

    All renames happen at once, so shifting answer_0 -> answer_1 -> ... cannot
    overwrite a column before it is moved.
    """
    df = df.rename(columns=RENAME_MAP)
    if Con.ANSWERS_ORDER_COLUMN in df.columns:
        df[Con.ANSWERS_ORDER_COLUMN] = df[Con.ANSWERS_ORDER_COLUMN].apply(
            answers_order_to_letters
        )
    df = add_identity_columns(df, strict=strict, verbose=verbose)
    return add_item_id_columns(df, strict=strict, verbose=verbose)


def check_trial_index_gaps(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Check that every missing trial number is explained by a recalibration.

    Each recalibration uses up a trial number without producing an answer
    screen, so some gaps in the trial numbers are expected: a session should
    have exactly as many gaps as recalibrations. More gaps than that means
    trials were genuinely lost, and is printed as a warning.

    Returns one row per session with its gaps and its recalibration count.
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


# The text columns in the order the words appear on screen: the question, then
# the four answers by screen position (answer_1..4, after RENAME_MAP).
SCREEN_TEXT_COLUMNS = ("question",) + tuple(
    f"{Con.ANSWER_PREFIX}{i}" for i in range(1, 5)
)


def _tokens(value) -> list[str]:
    return str(value).split()


def check_text_alignment(
    ia_df: pd.DataFrame, strict: bool = False, verbose: bool = True
) -> pd.DataFrame:
    """Report trials whose stored text does not match the words shown on screen.

    The interest areas are one per word, in screen order: the question, then
    the four answers. Rebuilding that word list from the stored `question` and
    `answer_1..4` text and comparing it with `IA_LABEL` finds the trials where
    they differ.

    It does happen. The stored text sometimes has a quote split off a word --
    `word "cool" "?"` for a screen that showed `word "cool"?` -- which adds a word
    the screen never showed. It comes from the presentation software, which is
    not in this repo, and only questions ending in a quoted word followed by `?`
    are affected.

    The screen-area labels on these trials are still right, because words are
    placed by their position on screen (`geometry.py`), not by counting stored
    words. What is off by one word on these trials is anything taken from the
    stored text itself, such as its word count.

    Returns one row per mismatched trial. `strict` stops the run instead of
    printing a warning.
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
        # where the two word lists first differ
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

# The per-word pupil columns that get converted to mm and z-scored.
PUPIL_IA_COLUMNS = (
    Con.IA_MAX_FIX_PUPIL_SIZE,
    Con.IA_MIN_FIX_PUPIL_SIZE,
    Con.IA_AVERAGE_FIX_PUPIL_SIZE,
)

PUPIL_STATS_NAME = "pupil_stats_by_session.csv"

# "participant" (the default) gives one pupil baseline per person, pooling their
# sessions, as the main pipeline does. "session" gives one per recording.
PUPIL_NORM_UNITS = ("session", "participant")


def compute_pupil_stats_by(fixations: pd.DataFrame, group_col: str) -> pd.DataFrame:
    """Mean and standard deviation of pupil size (in mm) over all fixations,
    per `group_col` (a person or a session).

    Like `features.pupil.compute_participant_pupil_stats`, but the grouping
    column is a choice, so the baseline can be per session.
    """
    from src.features.pupil import scale_pupil_area_to_mm

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
    group_col: str = Con.PARTICIPANT_ID,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Convert the per-word pupil columns to mm and z-score them within `group_col`.

    Like `features.pupil.add_zscored_pupil_columns`, but the grouping column is
    a choice. Only used when `pupil_norm_unit="session"`; by default the main
    pipeline does this per person.

    Careful: the raw pupil columns are overwritten with their mm values, not
    kept beside them.

    Returns (the frame with `_z` columns added, the means and SDs used).
    """
    from src.features.pupil import (
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
    pupil_norm_unit: str = "participant",
    strict: bool = True,
    verbose: bool = True,
) -> dict[str, Path]:
    """Clean a run's converted CSVs into `raw_dir/csvs/cleaned/`.

    For both the word and the fixation report: rename the columns and add the
    id columns (`standardize_columns`). Then print two checks -- missing trial
    numbers against recalibrations, and stored text against the words on
    screen.

    With `pupil_norm_unit="session"`, pupil z-scoring also happens here, using
    per-session averages from the fixation report. With "participant" (the
    default) it is left to the main pipeline.

    Returns {CSV name: path written}.
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

    # Report trials whose stored text does not match the words on screen. This
    # only reports; it does not change or drop anything.
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
    """Stop if a "session" run is about to use a report cleaned for "participant".

    With "session", the pipeline skips its own pupil z-scoring and expects the
    cleaned report to have the `_z` columns already. If they are missing, every
    pupil feature would quietly come out empty.
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


def _refuse_participant_over_session_cleaned(ia_path: Path) -> None:
    """Stop if a "participant" run is about to use a report cleaned for "session".

    The "session" clean step overwrites the raw pupil columns with mm values.
    The "participant" pipeline would then convert them to mm a second time --
    nothing would fail, and every pupil z-score would be quietly wrong. If this
    stops your run, re-run the clean step with `pupil_norm_unit="participant"`.
    """
    header = pd.read_csv(ia_path, nrows=0)
    already = [f"{c}_z" for c in PUPIL_IA_COLUMNS if f"{c}_z" in header.columns]
    if already:
        raise ValueError(
            f"{ia_path} already carries {already}, so it was cleaned with "
            "pupil_norm_unit='session' -- and its raw pupil columns hold "
            "millimetres, not area units. Running the participant path over it "
            "would scale them a second time and silently produce wrong z-scores. "
            "Re-run the clean step with pupil_norm_unit='participant' first."
        )


def run_pipeline(
    cleaned_dir: Path,
    out_dir: Path,
    pupil_norm_unit: str = "participant",
    include_paragraph: bool = False,
    verbose: bool = True,
) -> dict[str, Path]:
    """Run `ingest/build.py::main()` on a run's cleaned reports.

    Writes the three tables (interest_areas, areas_answers, trials) and the
    side files, all to the dataset's `interim/` folder. The button presses come
    from the fixation report itself, which holds the messages too, so no
    separate messages report is needed.

    `pupil_norm_unit` must match what `clean_reports` used; two checks stop the
    run if it does not. With "session" the pipeline's own pupil z-scoring is
    skipped, because the clean step has done it.

    Returns {name: path} for the three tables.
    """
    from src.ingest import registry as registry_mod
    from src.ingest import build as ingest_build

    # Which dataset this is, worked out from `out_dir`. Everything
    # dataset-specific below is read from its record.
    ds = dataset_for_root(out_dir)

    if pupil_norm_unit not in PUPIL_NORM_UNITS:
        raise ValueError(
            f"pupil_norm_unit must be one of {PUPIL_NORM_UNITS}, got {pupil_norm_unit!r}"
        )
    if include_paragraph and not ds.has_paragraph:
        raise ValueError(
            f"include_paragraph=True is unsupported for {ds.key} "
        )

    cleaned_dir = Path(cleaned_dir)
    out_dir = Path(out_dir)
    # Output paths come from the dataset record, so they always match where the
    # rest of the project looks.
    aux_dir = ds.aux
    ds.interim.mkdir(parents=True, exist_ok=True)
    aux_dir.mkdir(parents=True, exist_ok=True)

    ia_path = cleaned_dir / IA_REPORT_NAME
    fix_path = cleaned_dir / FIX_REPORT_NAME
    for label, path in [("IA report", ia_path), ("fixations report", fix_path)]:
        if not path.exists():
            raise FileNotFoundError(
                f"{label} not found at {path} -- run the clean step first."
            )

    # The row features this dataset skips come from its record. Check the names
    # against the registry, so a typo stops the run instead of skipping nothing.
    unknown = set(ds.skip_base_features) - set(registry_mod.FUNCTION_REGISTRY)
    if unknown:
        raise KeyError(
            f"{ds.key}.skip_base_features names {sorted(unknown)}, which "
            f"FUNCTION_REGISTRY does not define."
        )
    excluded = set(ds.skip_base_features)
    if pupil_norm_unit == "session":
        # The clean step has already converted and z-scored pupil size per
        # session, so the pipeline must not do it again.
        excluded.add(PUPIL_BASE_FUNC)
        _require_pupil_z_columns(ia_path)
    else:
        # The pipeline will convert and z-score pupil size per person, which is
        # only right if the cleaned report still has the raw values.
        _refuse_participant_over_session_cleaned(ia_path)
    base_function_names = [
        name
        for name, entry in registry_mod.FUNCTION_REGISTRY.items()
        if entry["kind"] == "row" and name not in excluded
    ]

    ingest_build.main(
        # Saves the three tables. No group subsets: Study 2 has no
        # question_preview column to split on.
        dataset=ds,
        ia_answers_path=ia_path,
        # Where the single wide table would go if `write_broadcast_table=True`
        # were passed. It is not: the features step rebuilds the wide table
        # from the three tables. Named anyway, so that turning it on could
        # never write into Study 1's folder (the default).
        output_path=ds.all_participants,
        # One fixation report feeds the pupil baseline, the fixation sequence
        # and the reading times.
        fixations_path=fix_path,
        # With pupil_norm_unit="participant" (the default), main() works out each
        # person's baseline from THIS dataset's fixations, pooling their
        # sessions and regimes, and saves it here. With "session" the pipeline's
        # pupil step is skipped (see above) and nothing is written here.
        pupil_stats_path=ds.pupil_baseline_answers,
        button_clicks_path=aux_dir / "button_clicks_data.csv",
        last_labels_path=aux_dir / "all_participants_last.csv",
        rt_and_tfd_path=aux_dir / "RT_and_TFD.csv",
        save_auxiliary=True,
        # the area looked at before each press, and the reading times
        add_last=True,
        add_rts=True,
        # The fixation report holds the message log too, so it is the only
        # source for the button presses.
        button_clicks_fix_tsv_path=None,
        button_clicks_msg_participant_col=Con.PARTICIPANT_ID,
        all_answers_is_cumulative=False,  # each trial's ALL_ANSWERS is its own
        base_function_names=base_function_names,
        group_function_names=None,  # None = all registered area/trial features
        # Study 2 has no repeated-reading trials (and no column for them), so
        # that filter is off. Practice trials are still removed.
        remove_repeats=False,
        remove_practice=True,
        verbose=verbose,
    )

    if verbose:
        print(f"\n[OK] pipeline output -> {ds.interim}")
    return {name: getattr(ds, name) for name in ("interest_areas", "areas_answers", "trials")}


def build_features(out_dir: Path, verbose: bool = True) -> tuple[Path, pd.DataFrame]:
    """Build the one-row-per-trial table and save it as the dataset's
    `features/model_ready.csv`.

    Exactly as for Study 1: `load_model_input` joins the three tables the
    `pipeline` step wrote back into one wide word-level table, and
    `save_model_ready` builds the trial table from it and saves it. If the three
    tables are missing, `load_model_input` stops with an error.

    No paragraph features are added, because Study 2 has no paragraph data.
    `out_dir` says which dataset this is. Returns (the path written, the table).
    """
    from src.features.build import load_model_input, save_model_ready

    ds = dataset_for_root(out_dir)
    features_df = save_model_ready(
        load_model_input(ds, verbose=verbose),
        output_path=ds.model_ready,
        dataset=ds.key,
        verbose=verbose,
    )

    if verbose:
        print("target (is_correct) distribution:")
        print(features_df[Con.IS_CORRECT_COLUMN].value_counts(dropna=False).to_string())
    return ds.model_ready, features_df


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

STEPS = ("convert", "clean", "pipeline", "features")


def prepare_know_qa(
    run: str = DEFAULT_RUN,
    steps: Iterable[str] = STEPS,
    pupil_norm_unit: str = "participant",
    strict: bool = True,
    verbose: bool = True,
    runs: Mapping[str, tuple[Path, Path]] = RAW_RUNS,
) -> dict[str, Path]:
    """Prepare one Study 2 collection, from raw exports to the model table.

    `steps` chooses which of convert -> clean -> pipeline -> features to run;
    they always run in that order, so a later one can be re-run alone once the
    earlier ones are done. `features` reads the three tables `pipeline` wrote.

    `strict` stops the run when a recording name disagrees with the report's
    own batch/list columns; otherwise it only warns. `pupil_norm_unit` chooses
    one pupil baseline per person ("participant") or per session.

    Returns {name: path} for everything produced.
    """
    if run not in runs:
        raise KeyError(f"unknown run {run!r}; known runs: {sorted(runs)}")
    requested = [s for s in STEPS if s in set(steps)]
    unknown = set(steps) - set(STEPS)
    if unknown:
        raise ValueError(f"unknown step(s) {sorted(unknown)}; known steps: {STEPS}")

    raw_dir, out_dir = runs[run]
    cleaned_dir = raw_dir / CSV_SUBDIR / CLEANED_SUBDIR

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
        produced.update(run_pipeline(
            cleaned_dir,
            out_dir,
            pupil_norm_unit=pupil_norm_unit,
            verbose=verbose,
        ))

    if "features" in requested:
        if verbose:
            print("\n--- features ---")
        features_path, _ = build_features(out_dir, verbose=verbose)
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
        default="participant",
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
    """Let the command-line run print special characters on Windows.

    The pipeline prints characters such as the check mark that Windows' default
    text encoding cannot write when output goes to a file or a pipe. Without
    this, such a run would crash on its last print, after all the work is done.
    Only the command-line entry point calls it.
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
