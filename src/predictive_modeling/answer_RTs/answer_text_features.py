# answer_text_features.py
#
# Stimulus-level answer/question text features for the answer reading-time
# (answer_RTs) regression.
#
# Motivation: the target is the reading time on one specific answer, and the
# single most obvious driver of that is how much text the answer contains. These
# are properties of the *stimulus*, not of how the participant read the answer
# area, so they are admissible under the module's paragraph-only rule (which
# excludes answer-area eye-tracking measures, not the text itself).
#
# One row per trial with per-label text sizes:
#   answer_len_words_<A..D>, answer_len_chars_<A..D>, question_len_words, ...
# `model_data.make_answer_rt_dataset` turns these into target-aligned columns
# (`answer_len_words` = the length of whichever answer is being predicted).
#
# NOTE on the default target: `RT_normalized_answer_X` is already RT divided by
# the answer's word count (see src/derived/reading_times.py), so length has been
# divided out of it -- these features then test the *residual* length effect
# (do longer answers get read faster per word?). Against `RT_pure_answer_X`
# length is a first-order predictor.

from __future__ import annotations

import ast
from pathlib import Path
from typing import Iterable, List, Optional

import pandas as pd

from src import constants as Con
from src.constants import TRIAL_ID_COLS
from src.data_paths import ANSWER_TEXT_FEATURES_PATH, IA_ANSWERS_PATH

ANSWER_LABELS = tuple(Con.ANSWER_LABELS)

# Positional answer-text columns in the raw answers report; `answers_order` maps
# each screen position to its correctness label.
_POSITIONAL_ANSWER_COLS = [f"{Con.ANSWER_PREFIX}{i}" for i in range(1, 5)]
_QUESTION_COL = "question"

_SOURCE_COLS = (
    list(TRIAL_ID_COLS)
    + [Con.ANSWERS_ORDER_COLUMN, _QUESTION_COL]
    + _POSITIONAL_ANSWER_COLS
)

# Chunk size for streaming the (large) raw answers IA report.
_CHUNK_ROWS = 500_000


def answer_len_col(answer: str, unit: str = "words") -> str:
    """Per-label length column name, e.g. ('A', 'words') -> answer_len_words_A."""
    return f"answer_len_{unit}_{answer}"


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

def _trial_level_answer_text(
    ia_answers_path: Path,
    chunksize: int = _CHUNK_ROWS,
) -> pd.DataFrame:
    """One row per trial with the raw question / answer_1..4 text columns.

    The answers IA report has one row per interest area, with the stimulus text
    repeated on every row; this streams it in chunks and keeps the first row per
    (participant_id, TRIAL_INDEX).
    """
    keys = list(TRIAL_ID_COLS)
    parts: List[pd.DataFrame] = []
    reader = pd.read_csv(
        ia_answers_path,
        usecols=_SOURCE_COLS,
        chunksize=chunksize,
    )
    for chunk in reader:
        parts.append(chunk.drop_duplicates(subset=keys))

    out = pd.concat(parts, ignore_index=True)
    return out.drop_duplicates(subset=keys).reset_index(drop=True)


def _labelled_answer_text(df: pd.DataFrame) -> pd.DataFrame:
    """Map positional answer text to correctness labels via `answers_order`.

    Mirrors `data_prep.data_csv_generation.add_answer_text_columns`: the order
    list gives the label of each screen position, so the text for label L is the
    positional column at `order.index(L) + 1`.
    """
    out = df.copy()

    def parse_order(value) -> Optional[list]:
        if isinstance(value, (list, tuple)):
            return list(value)
        try:
            parsed = ast.literal_eval(value)
        except (ValueError, SyntaxError, TypeError):
            return None
        return list(parsed) if isinstance(parsed, (list, tuple)) else None

    orders = out[Con.ANSWERS_ORDER_COLUMN].map(parse_order)

    for label in ANSWER_LABELS:
        positions = orders.map(
            lambda o, lab=label: (o.index(lab) + 1) if o and lab in o else None
        )
        texts = [
            out[f"{Con.ANSWER_PREFIX}{int(pos)}"].iloc[i] if pd.notna(pos) else None
            for i, pos in enumerate(positions)
        ]
        out[f"answer_{label}"] = texts

    return out


def _add_length_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Word / character counts for each labelled answer and the question.

    Word counts use whitespace tokenization, matching the interest-area
    segmentation that `RT_normalized` divides by.
    """
    out = df.copy()

    def word_len(series: pd.Series) -> pd.Series:
        return series.fillna("").astype(str).str.split().map(len)

    def char_len(series: pd.Series) -> pd.Series:
        return series.fillna("").astype(str).str.len()

    for label in ANSWER_LABELS:
        text = out[f"answer_{label}"]
        out[answer_len_col(label, "words")] = word_len(text)
        out[answer_len_col(label, "chars")] = char_len(text)

    out["question_len_words"] = word_len(out[_QUESTION_COL])
    out["question_len_chars"] = char_len(out[_QUESTION_COL])

    word_cols = [answer_len_col(l, "words") for l in ANSWER_LABELS]
    out["answers_total_len_words"] = out[word_cols].sum(axis=1)
    out["answers_mean_len_words"] = out[word_cols].mean(axis=1)
    # Spread across the four options: a screen with one long and three short
    # answers scans differently from four even ones.
    out["answers_std_len_words"] = out[word_cols].std(axis=1)

    return out


ANSWER_TEXT_FEATURE_COLS: List[str] = (
    [answer_len_col(l, "words") for l in ANSWER_LABELS]
    + [answer_len_col(l, "chars") for l in ANSWER_LABELS]
    + [
        "question_len_words",
        "question_len_chars",
        "answers_total_len_words",
        "answers_mean_len_words",
        "answers_std_len_words",
    ]
)


def build_answer_text_features(
    ia_answers: Optional[pd.DataFrame] = None,
    ia_answers_path: Path = IA_ANSWERS_PATH,
) -> pd.DataFrame:
    """One row per trial with answer/question length features.

    `ia_answers` may be passed in-memory (already trial-level or IA-level);
    otherwise the raw answers report is streamed from `ia_answers_path`.
    """
    if ia_answers is None:
        trial_text = _trial_level_answer_text(Path(ia_answers_path))
    else:
        missing = [c for c in _SOURCE_COLS if c not in ia_answers.columns]
        if missing:
            raise KeyError(f"Missing answer-text source columns: {missing}")
        trial_text = ia_answers[_SOURCE_COLS].drop_duplicates(
            subset=list(TRIAL_ID_COLS)
        )

    labelled = _labelled_answer_text(trial_text)
    with_lengths = _add_length_columns(labelled)
    return with_lengths[list(TRIAL_ID_COLS) + ANSWER_TEXT_FEATURE_COLS]


# ---------------------------------------------------------------------------
# Cached feature CSV
# ---------------------------------------------------------------------------

def save_answer_text_features(
    ia_answers: Optional[pd.DataFrame] = None,
    output_path: Path = ANSWER_TEXT_FEATURES_PATH,
    ia_answers_path: Path = IA_ANSWERS_PATH,
    verbose: bool = True,
) -> pd.DataFrame:
    """Build the trial-level answer-text features and save them to CSV."""
    features = build_answer_text_features(
        ia_answers=ia_answers,
        ia_answers_path=ia_answers_path,
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(output_path, index=False)
    if verbose:
        print(
            f"Saved {len(features)} trials x {len(features.columns)} cols "
            f"to {output_path}"
        )
    return features


def load_answer_text_features(
    path: Path = ANSWER_TEXT_FEATURES_PATH,
    ia_answers_path: Path = IA_ANSWERS_PATH,
    build_if_missing: bool = True,
) -> pd.DataFrame:
    """Load the cached answer-text features, building them on first use."""
    path = Path(path)
    if not path.exists():
        if not build_if_missing:
            raise FileNotFoundError(f"Answer-text features not found: {path}")
        return save_answer_text_features(
            output_path=path,
            ia_answers_path=ia_answers_path,
        )
    return pd.read_csv(path)


if __name__ == "__main__":
    save_answer_text_features()
