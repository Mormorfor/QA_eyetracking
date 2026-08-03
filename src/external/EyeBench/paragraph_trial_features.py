# paragraph_trial_features.py
#
# Runs the lab's trial-level feature extraction (taken from a colleague's
# EyeBench / OneStop pipeline and kept verbatim in
# `utils - paragraph feature extraction.py`, alongside this file) on *our*
# paragraph reading reports.
#
# The external file expects data that has already been through that project's
# preprocessing. Our raw reports differ in a few ways, so everything this module
# does is bridging:
#
#   * it satisfies the `src.configs.*` imports the snippet makes, by aliasing
#     them onto `EyeBench/configs` before loading it (`_alias_config_package`);
#   * the preprocessing steps of their `OneStopProcessor` / `our_processing` are
#     reproduced here function-for-function (dtype conversions, 0-based word
#     indices, `normalized_ID`, `total_skip`, `is_content_word`, merging the IA
#     measures onto the fixations), since our reports are raw where theirs are
#     already processed;
#   * `ptb_pos` there means the *reduced* POS tag (FUNC / NOUN / VERB / ADJ /
#     UNKNOWN); in our reports those values live in `Reduced_POS`, while our
#     `ptb_pos` holds full Penn-Treebank tags, so we swap them as they do.
#
# Known difference from their output, kept deliberately behind a flag: their
# `ptb_pos` features are all zero (see `fix_ptb_pos_double_mapping`).
#
# Entry point: `save_paragraph_trial_level_features()` -> one row per
# (participant_id, TRIAL_INDEX) trial, a few hundred feature columns, written to
# `PARAGRAPH_TRIAL_FEATURES_PATH`.

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Iterable, Optional, Sequence

import pandas as pd

from src import constants as Con
from src.external.EyeBench.configs.constants import (
    REDUCED_POS_TO_NUMBER,
    DataType,
    gsf_features,
    numerical_feature_aggregations,
    numerical_fixation_trial_columns,
    numerical_ia_trial_columns,
)
from src.data_paths import (
    FIX_PARAGRAPH_PATH,
    IA_PARAGRAPH_PATH,
    PARAGRAPH_TRIAL_FEATURES_DIR,
    PARAGRAPH_TRIAL_FEATURES_PATH,
)

# ---------------------------------------------------------------------------
# Loading the external snippet
# ---------------------------------------------------------------------------

EXTERNAL_MODULE_PATH = Path(__file__).with_name(
    "utils - paragraph feature extraction.py"
)


def _alias_config_package() -> None:
    """Make the vendored file's `src.configs.*` imports resolve to `EyeBench/configs`.

    The source project keeps these constants in its own `src.configs` package.
    Registering the aliases here means the vendored file can stay byte-identical
    to the copy we were given rather than having its imports rewritten -- so a
    newer copy from the source project can be dropped in as-is.
    """
    from src.external.EyeBench import configs
    from src.external.EyeBench.configs import constants, models
    from src.external.EyeBench.configs.models import base_model

    sys.modules.setdefault("src.configs", configs)
    sys.modules.setdefault("src.configs.constants", constants)
    sys.modules.setdefault("src.configs.models", models)
    sys.modules.setdefault("src.configs.models.base_model", base_model)


def load_external_module():
    """Import `utils - paragraph feature extraction.py` (its name is not importable)."""
    module_name = "external_paragraph_feature_extraction"
    if module_name in sys.modules:
        return sys.modules[module_name]

    _alias_config_package()
    spec = importlib.util.spec_from_file_location(module_name, EXTERNAL_MODULE_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {EXTERNAL_MODULE_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# Column conventions
# ---------------------------------------------------------------------------

TRIAL_GROUPBY_COLUMNS = list(Con.TRIAL_ID_COLS)  # participant_id, TRIAL_INDEX

# The interest area a fixation landed on; 0 means "not on any word".
FIX_IA_INDEX_COL = "CURRENT_FIX_INTEREST_AREA_INDEX"
NEXT_FIX_IA_INDEX_COL = "NEXT_FIX_INTEREST_AREA_INDEX"
REDUCED_POS_COL = "Reduced_POS"

# Content-word lookup, copied from `text_metrics.utils` (lacclab/text-metrics,
# now `psycholing_metrics/text_processing.py`) -- the library the source
# project's `add_additional_metrics` calls to build `is_content_word`. Tags not
# listed (e.g. SYM) fall through the `.get` default to False, as they do there.
CONTENT_WORDS = {
    "PUNCT": False,
    "PROPN": True,
    "NOUN": True,
    "PRON": False,
    "VERB": True,
    "SCONJ": False,
    "NUM": False,
    "DET": False,
    "CCONJ": False,
    "ADP": False,
    "AUX": False,
    "ADV": True,
    "ADJ": True,
    "INTJ": False,
    "X": False,
    "PART": False,
}


def is_content_word(pos: str) -> bool:
    """As in `text_metrics.utils.is_content_word`."""
    return CONTENT_WORDS.get(pos, False)


# Column lists from the source project's `convert_to_int_features` /
# `convert_to_float_features`. Int columns get '.'/NaN replaced by 0 (so a
# missing saccade duration aggregates as zero, not as a skipped value); float
# columns get '.' replaced by null. Intersected with what our reports have.
_TO_INT_COLUMNS = {
    DataType.IA: [
        "article_batch",
        "article_id",
        "paragraph_id",
        "repeated_reading_trial",
        "practice_trial",
        "IA_DWELL_TIME",
        "IA_FIRST_FIXATION_DURATION",
        "IA_REGRESSION_PATH_DURATION",
        "IA_FIRST_RUN_DWELL_TIME",
        "IA_FIXATION_COUNT",
        "IA_REGRESSION_IN_COUNT",
        "IA_REGRESSION_OUT_FULL_COUNT",
        "IA_RUN_COUNT",
        "IA_FIRST_FIXATION_VISITED_IA_COUNT",
        "IA_FIRST_RUN_FIXATION_COUNT",
        "IA_SKIP",
        "IA_REGRESSION_OUT_COUNT",
        "IA_SELECTIVE_REGRESSION_PATH_DURATION",
        "IA_SPILLOVER",
        "IA_LAST_FIXATION_DURATION",
        "IA_LAST_RUN_DWELL_TIME",
        "IA_LAST_RUN_FIXATION_COUNT",
        "IA_LEFT",
        "IA_TOP",
        "TRIAL_DWELL_TIME",
        "TRIAL_FIXATION_COUNT",
        "TRIAL_IA_COUNT",
        "TRIAL_INDEX",
        "TRIAL_TOTAL_VISITED_IA_COUNT",
        "IA_FIRST_FIX_PROGRESSIVE",
    ],
    DataType.FIXATIONS: [
        "article_batch",
        "article_id",
        "paragraph_id",
        "repeated_reading_trial",
        "practice_trial",
        "CURRENT_FIX_INTEREST_AREA_INDEX",
        "NEXT_FIX_INTEREST_AREA_INDEX",
        "CURRENT_FIX_DURATION",
        "CURRENT_FIX_PUPIL",
        "CURRENT_FIX_X",
        "CURRENT_FIX_Y",
        "CURRENT_FIX_INDEX",
        "NEXT_SAC_DURATION",
    ],
}

_TO_FLOAT_COLUMNS = {
    DataType.IA: [
        "IA_AVERAGE_FIX_PUPIL_SIZE",
        "IA_DWELL_TIME_%",
        "IA_FIXATION_%",
        "IA_FIRST_RUN_FIXATION_%",
        "IA_FIRST_SACCADE_AMPLITUDE",
        "IA_FIRST_SACCADE_ANGLE",
        "IA_LAST_RUN_FIXATION_%",
        "IA_LAST_SACCADE_AMPLITUDE",
        "IA_LAST_SACCADE_ANGLE",
        "IA_FIRST_RUN_LANDING_POSITION",
        "IA_LAST_RUN_LANDING_POSITION",
    ],
    DataType.FIXATIONS: [
        "CURRENT_FIX_INTEREST_AREA_INDEX",
        "NEXT_FIX_INTEREST_AREA_INDEX",
        "NEXT_FIX_ANGLE",
        "PREVIOUS_FIX_ANGLE",
        "NEXT_FIX_DISTANCE",
        "PREVIOUS_FIX_DISTANCE",
        "NEXT_SAC_AMPLITUDE",
        "NEXT_SAC_ANGLE",
        "NEXT_SAC_AVG_VELOCITY",
        "NEXT_SAC_PEAK_VELOCITY",
        "NEXT_SAC_END_X",
        "NEXT_SAC_START_X",
        "NEXT_SAC_END_Y",
        "NEXT_SAC_START_Y",
    ],
}

# The column each mode indexes words by, as in `get_constants_by_mode`.
_IA_FIELD = {DataType.IA: "IA_ID", DataType.FIXATIONS: FIX_IA_INDEX_COL}

# The `gsf_features` entries that `compute_fixation_trial_level_features` builds
# itself, per trial, rather than expecting them in the input.
_CATEGORY_NORMALIZED_GSF = frozenset(
    f"{cluster}_normalized_{measure}"
    for cluster in ("LengthCategory", "universal_pos")
    for measure in ("IA_DWELL_TIME", "IA_FIRST_FIXATION_DURATION")
)

# Word-level (IA) measures the fixation-side features need. The external code
# reads them off the *fixation* rows, so they are merged in per interest area.
# `IA_REGRESSION_IN_COUNT_sum` is excluded: `add_missing_features` derives it.
_IA_MEASURES_FOR_FIXATIONS = [
    col
    for col in dict.fromkeys(numerical_fixation_trial_columns + gsf_features)
    if col.startswith("IA_") and col != "IA_REGRESSION_IN_COUNT_sum"
]

# Columns wanted from each raw report. Both files are several GB, so we only read
# these -- intersected with the actual header, since names that the external
# lists expect may either be derived later (`is_content_word`, the normalized
# counts) or exist only after the source project's preprocessing.
_TRIAL_ATTRIBUTE_COLUMNS = [
    Con.PRACTICE_TRIAL_COLUMN,
    Con.REPEATED_TRIAL_COLUMN,
    Con.QUESTION_PREVIEW_COLUMN,
]

IA_WANTED_COLUMNS = list(
    dict.fromkeys(
        TRIAL_GROUPBY_COLUMNS
        + numerical_ia_trial_columns
        + _IA_MEASURES_FOR_FIXATIONS
        + [
            "IA_ID",
            "TRIAL_IA_COUNT",
            "PARAGRAPH_RT",
            "word_length",
            "universal_pos",
            REDUCED_POS_COL,
            "entity_type",
        ]
        + _TRIAL_ATTRIBUTE_COLUMNS
    )
)

FIX_WANTED_COLUMNS = list(
    dict.fromkeys(
        TRIAL_GROUPBY_COLUMNS
        + numerical_fixation_trial_columns
        + gsf_features
        + [
            FIX_IA_INDEX_COL,
            NEXT_FIX_IA_INDEX_COL,
            "universal_pos",
            "entity_type",
        ]
        + _TRIAL_ATTRIBUTE_COLUMNS
    )
)

# Without these the extraction is either wrong or crashes, so a rename in the
# reports should fail loudly rather than silently drop features.
_REQUIRED_IA_COLUMNS = TRIAL_GROUPBY_COLUMNS + [
    "IA_ID",
    "IA_DWELL_TIME",
    "IA_FIRST_FIXATION_DURATION",
    "IA_SKIP",
    "TRIAL_IA_COUNT",
    "PARAGRAPH_RT",
    "universal_pos",
    REDUCED_POS_COL,
    "entity_type",
    "word_length",
]
_REQUIRED_FIX_COLUMNS = TRIAL_GROUPBY_COLUMNS + [
    FIX_IA_INDEX_COL,
    NEXT_FIX_IA_INDEX_COL,
    "CURRENT_FIX_DURATION",
    "CURRENT_FIX_X",
    "CURRENT_FIX_Y",
    "universal_pos",
    "word_length",
    "gpt2_surprisal",
    "distance_to_head",
    "left_dependents_count",
    "right_dependents_count",
]


def _resolve_usecols(path: Path, wanted: Sequence[str], required: Sequence[str]) -> list[str]:
    """Intersect the wanted columns with the report's header, keeping order."""
    header = set(pd.read_csv(path, nrows=0).columns)
    missing_required = [col for col in required if col not in header]
    if missing_required:
        raise KeyError(f"{Path(path).name} is missing required columns: {missing_required}")
    return [col for col in wanted if col in header]


# ---------------------------------------------------------------------------
# Loading / filtering
# ---------------------------------------------------------------------------


def load_paragraph_reports(
    ia_path: Path = IA_PARAGRAPH_PATH,
    fixations_path: Optional[Path] = FIX_PARAGRAPH_PATH,
    remove_practice: bool = True,
    remove_repeats: bool = True,
    participants: Optional[Sequence[str]] = None,
    verbose: bool = True,
) -> tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    """Read the paragraph IA and fixation reports, keeping only what we need.

    `participants` restricts both reports to a subset of participant ids -- handy
    for a quick trial run, since the full extraction takes a while.
    """
    if verbose:
        print(f"Reading {ia_path.name} ...")
    ia_usecols = _resolve_usecols(ia_path, IA_WANTED_COLUMNS, _REQUIRED_IA_COLUMNS)
    ia = pd.read_csv(ia_path, usecols=ia_usecols, low_memory=False)
    ia = _filter_trials(ia, remove_practice, remove_repeats, participants)
    if verbose:
        print(f"  IA rows: {len(ia):,} | trials: {_n_trials(ia):,}")

    fixations = None
    if fixations_path is not None:
        if verbose:
            print(f"Reading {Path(fixations_path).name} (this one is big) ...")
        fix_usecols = _resolve_usecols(
            fixations_path, FIX_WANTED_COLUMNS, _REQUIRED_FIX_COLUMNS
        )
        fixations = pd.read_csv(fixations_path, usecols=fix_usecols, low_memory=False)
        fixations = _filter_trials(
            fixations, remove_practice, remove_repeats, participants
        )
        if verbose:
            print(
                f"  fixation rows: {len(fixations):,} | trials: {_n_trials(fixations):,}"
            )

    return ia, fixations


def _filter_trials(
    df: pd.DataFrame,
    remove_practice: bool,
    remove_repeats: bool,
    participants: Optional[Sequence[str]],
) -> pd.DataFrame:
    """Apply the project's standard trial filters (see `split_hunters_and_gatherers`)."""
    out = df
    if remove_practice and Con.PRACTICE_TRIAL_COLUMN in out.columns:
        out = out[out[Con.PRACTICE_TRIAL_COLUMN] == False]  # noqa: E712
    if remove_repeats and Con.REPEATED_TRIAL_COLUMN in out.columns:
        out = out[out[Con.REPEATED_TRIAL_COLUMN] == False]  # noqa: E712
    if participants is not None:
        out = out[out[Con.PARTICIPANT_ID].isin(list(participants))]
    return out.copy()


def _n_trials(df: pd.DataFrame) -> int:
    return df.groupby(TRIAL_GROUPBY_COLUMNS).ngroups


def _to_numeric(df: pd.DataFrame, columns: Iterable[str]) -> None:
    """Coerce in place; EyeLink writes '.' for missing values, making columns strings."""
    for col in columns:
        if col not in df.columns:
            continue
        if pd.api.types.is_numeric_dtype(df[col]) or pd.api.types.is_bool_dtype(df[col]):
            continue
        df[col] = pd.to_numeric(df[col], errors="coerce")


# ---------------------------------------------------------------------------
# Preparation: our columns -> the columns the external code expects
#
# The functions below mirror the source project's `OneStopProcessor` and its
# `our_processing` helpers one-for-one, in the same order, so that the frames
# handed to the extraction match what it gets there. Named after their
# counterparts; deviations are commented.
# ---------------------------------------------------------------------------


def _convert_to_int_features(df: pd.DataFrame, mode: DataType) -> pd.DataFrame:
    """As `convert_to_int_features`: '.'/NaN -> 0, then int."""
    columns = [c for c in _TO_INT_COLUMNS[mode] if c in df.columns]
    df[columns] = df[columns].replace({".": 0}).fillna(0).astype(int)
    return df


def _convert_to_float_features(df: pd.DataFrame, mode: DataType) -> pd.DataFrame:
    """As `convert_to_float_features`: '.' -> null, then float."""
    columns = [c for c in _TO_FLOAT_COLUMNS[mode] if c in df.columns]
    df[columns] = df[columns].replace({".": None}).astype(float)
    return df


def _adjust_indexing(df: pd.DataFrame, mode: DataType) -> pd.DataFrame:
    """As `adjust_indexing`: make the word indices 0-based.

    Both tables are shifted by the same amount, so the IA <-> fixation join is
    unaffected; a fixation that landed off any word goes from 0 to -1.
    """
    columns = (
        ["IA_ID"] if mode == DataType.IA else [FIX_IA_INDEX_COL, NEXT_FIX_IA_INDEX_COL]
    )
    columns = [c for c in columns if c in df.columns]
    df[columns] -= 1
    return df


def _drop_missing_fixation_data(df: pd.DataFrame) -> pd.DataFrame:
    """As `drop_missing_fixation_data` (a no-op once '.' has become 0)."""
    return df.dropna(subset=[FIX_IA_INDEX_COL, NEXT_FIX_IA_INDEX_COL])


def _compute_span_level_metrics(
    df: pd.DataFrame, mode: DataType, trial_groupby_columns: list[str]
) -> pd.DataFrame:
    """As `compute_span_level_metrics`: shift each trial's IA_ID to start at 0
    and attach the trial's min/max word index."""
    ia_field = _IA_FIELD[mode]
    grouped = df.groupby(trial_groupby_columns)[ia_field]

    if mode == DataType.IA:
        df[ia_field] = df[ia_field] - grouped.transform("min")
        grouped = df.groupby(trial_groupby_columns)[ia_field]

    df["min_IA_ID"] = grouped.transform("min")
    df["max_IA_ID"] = grouped.transform("max")
    return df


def _compute_normalized_features(df: pd.DataFrame, mode: DataType) -> pd.DataFrame:
    """As `compute_normalized_features`: word position in the trial, 0..1."""
    ia_field = _IA_FIELD[mode]
    df["normalized_ID"] = (df[ia_field] - df["min_IA_ID"]) / (
        df["max_IA_ID"] - df["min_IA_ID"]
    )
    return df


def _use_reduced_pos(df: pd.DataFrame) -> pd.DataFrame:
    """As their `df.drop(columns=['ptb_pos']).rename({'Reduced_POS': 'ptb_pos'})`.

    Their `ptb_pos` is the reduced tag set (FUNC/NOUN/VERB/ADJ/UNKNOWN); the full
    Penn-Treebank tags our reports carry under that name are dropped.
    `Reduced_POS` is kept alongside, since the adapter refers to it later.

    Only our IA report carries `Reduced_POS`; on the fixation side this drops the
    Penn-Treebank column, and `ptb_pos` is set once the merge has brought the
    reduced tags over.
    """
    df = df.drop(columns=["ptb_pos"], errors="ignore")
    if REDUCED_POS_COL in df.columns:
        df["ptb_pos"] = df[REDUCED_POS_COL]
    return df


def _add_additional_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """As `add_additional_metrics` (IA mode only)."""
    df["regression_rate"] = df["IA_REGRESSION_OUT_FULL_COUNT"] / df["IA_RUN_COUNT"]
    df["total_skip"] = df["IA_DWELL_TIME"] == 0
    df["is_content_word"] = df["universal_pos"].apply(is_content_word)
    return df


def prepare_ia_data(
    ia: pd.DataFrame,
    trial_groupby_columns: Sequence[str] = TRIAL_GROUPBY_COLUMNS,
) -> pd.DataFrame:
    """Run the source project's IA-mode preprocessing over our IA report."""
    trial_groupby_columns = list(trial_groupby_columns)
    out = ia.copy()

    out = _convert_to_int_features(out, DataType.IA)
    out = _convert_to_float_features(out, DataType.IA)
    # Pupil columns are ours (EXTRA_IA_TRIAL_COLUMNS) so they are in neither of
    # their conversion lists.
    _to_numeric(out, numerical_ia_trial_columns + ["word_length", "PARAGRAPH_RT"])
    out = _adjust_indexing(out, DataType.IA)
    out = _compute_span_level_metrics(out, DataType.IA, trial_groupby_columns)
    out = _compute_normalized_features(out, DataType.IA)
    out = _use_reduced_pos(out)
    out = _add_additional_metrics(out)

    return out


def prepare_fixation_data(
    fixations: pd.DataFrame,
    trial_groupby_columns: Sequence[str] = TRIAL_GROUPBY_COLUMNS,
) -> pd.DataFrame:
    """Run the source project's fixation-mode preprocessing over our fixation report."""
    trial_groupby_columns = list(trial_groupby_columns)
    out = fixations.copy()

    out = _convert_to_int_features(out, DataType.FIXATIONS)
    out = _convert_to_float_features(out, DataType.FIXATIONS)
    _to_numeric(
        out,
        [c for c in numerical_fixation_trial_columns + gsf_features if not c.startswith("IA_")],
    )
    out = _adjust_indexing(out, DataType.FIXATIONS)
    out = _drop_missing_fixation_data(out)
    out = _compute_span_level_metrics(out, DataType.FIXATIONS, trial_groupby_columns)
    out = _compute_normalized_features(out, DataType.FIXATIONS)
    out = _use_reduced_pos(out)

    return out


def add_ia_report_features_to_fixation_data(
    ia_df: pd.DataFrame,
    fix_df: pd.DataFrame,
    trial_groupby_columns: Sequence[str] = TRIAL_GROUPBY_COLUMNS,
) -> pd.DataFrame:
    """As `OneStopProcessor.add_ia_report_features_to_fixation_data`.

    Merges the word-level IA measures the fixation features read onto each
    fixation, keyed by the interest area it landed on, and attaches the trial's
    word count. Columns already present in the fixation report are kept from
    there, as in the original.
    """
    trial_groupby_columns = list(trial_groupby_columns)

    ia_features = [
        col
        for col in _IA_MEASURES_FOR_FIXATIONS
        + ["is_content_word", REDUCED_POS_COL, "entity_type", "TRIAL_IA_COUNT"]
        if col in ia_df.columns
    ]
    merge_keys = trial_groupby_columns + [FIX_IA_INDEX_COL]

    ia_df = ia_df.rename(columns={"IA_ID": FIX_IA_INDEX_COL})
    ia_df = ia_df[list(dict.fromkeys(merge_keys + ia_features))]
    duplicate_columns = (set(fix_df.columns) & set(ia_df.columns)) - set(merge_keys)
    ia_df = ia_df.drop(columns=list(duplicate_columns))

    enriched = fix_df.merge(ia_df, on=merge_keys, how="left", validate="many_to_one")

    words_per_trial = ia_df.groupby(trial_groupby_columns).apply(
        len, include_groups=False
    )
    words_per_trial.name = "num_of_words_in_trial"
    enriched = enriched.merge(words_per_trial, on=trial_groupby_columns, how="left")

    # The reduced tags only reach the fixation table through this merge, so the
    # POS swap their preprocessing does per-table happens here for fixations.
    return _use_reduced_pos(enriched)


# Columns the source project's `numerical_ia_trial_columns` expects that our
# report calls something else. `IA_DWELL_TIME` is DataViewer's total fixation
# duration on the interest area, and `IA_FIRST_FIX_DURATION` was confirmed
# against our export.
#
# Note that each target is *also* listed separately in
# `numerical_ia_trial_columns`, so filling these produces features that duplicate
# existing ones exactly (`ia_feature_mean_IA_FIRST_FIX_DURATION` ==
# `ia_feature_mean_IA_FIRST_FIXATION_DURATION`).
_ALIASED_IA_COLUMNS = {
    "IA_FIRST_FIX_DURATION": "IA_FIRST_FIXATION_DURATION",
    "IA_TOTAL_FIXATION_DURATION": "IA_DWELL_TIME",
    "landing_position": "IA_FIRST_RUN_LANDING_POSITION",
}

# Per-word saccade measures the source expects on the IA table. Our IA report has
# no saccade columns at all, so they are aggregated from the fixation report:
# every fixation that landed on a word contributes the saccade that left it.
_SACCADE_IA_COLUMNS = {
    "mean_sacc_dur": "NEXT_SAC_DURATION",
    "peak_sacc_velocity": "NEXT_SAC_PEAK_VELOCITY",
}


def reconstruct_source_columns(
    ia: pd.DataFrame,
    fixations: Optional[pd.DataFrame] = None,
    trial_groupby_columns: Sequence[str] = TRIAL_GROUPBY_COLUMNS,
    verbose: bool = True,
) -> pd.DataFrame:
    """Fill IA columns the source project's feature list expects but we lack.

    Ours, not theirs: renames of columns we do have, plus per-word saccade means
    aggregated from the fixation report. See `_ALIASED_IA_COLUMNS` /
    `_SACCADE_IA_COLUMNS` for what each is derived from, and note the aliases
    duplicate features that already exist under our own column names.

    `IA_FIRST_FIX_DWELL_TIME` and `IA_REGRESSION_OUT_TIME` are left alone: the
    first is ambiguous between first-fixation and first-run dwell time, and the
    second has no equivalent anywhere in our export.
    """
    out = ia.copy()
    filled = []

    for target, source in _ALIASED_IA_COLUMNS.items():
        if target not in out.columns and source in out.columns:
            out[target] = out[source]
            filled.append(f"{target} <- {source}")

    if fixations is not None:
        trial_groupby_columns = list(trial_groupby_columns)
        available = {
            target: source
            for target, source in _SACCADE_IA_COLUMNS.items()
            if target not in out.columns and source in fixations.columns
        }
        if available:
            on_word = fixations[fixations[FIX_IA_INDEX_COL] >= 0]
            per_word = (
                on_word.groupby(trial_groupby_columns + [FIX_IA_INDEX_COL])[
                    list(available.values())
                ]
                .mean()
                .rename(columns={source: target for target, source in available.items()})
                .reset_index()
                .rename(columns={FIX_IA_INDEX_COL: "IA_ID"})
            )
            out = out.merge(
                per_word, on=trial_groupby_columns + ["IA_ID"], how="left", validate="one_to_one"
            )
            # A word that was never fixated launched no saccade.
            out[list(available)] = out[list(available)].fillna(0)
            filled += [f"{t} <- mean per-word {s}" for t, s in available.items()]

    if verbose and filled:
        print(f"  reconstructed {len(filled)} source columns: {'; '.join(filled)}")
    return out


def _report_skipped_feature_columns(
    ia: pd.DataFrame, fixations: Optional[pd.DataFrame]
) -> None:
    """Print the feature columns the extraction will silently skip.

    `compute_*_trial_level_features` only aggregates the columns of
    `numerical_*_trial_columns` that are actually present, so a column the source
    project's preprocessing produces but our reports lack just yields no
    features. Worth saying out loud rather than leaving to a column count.
    """
    for label, frame, wanted in (
        ("IA", ia, numerical_ia_trial_columns),
        ("fixation", fixations, numerical_fixation_trial_columns),
    ):
        if frame is None:
            continue
        missing = [col for col in wanted if col not in frame.columns]
        if missing:
            print(
                f"  note: {len(missing)} {label} columns absent from our reports, "
                f"so {len(missing) * len(numerical_feature_aggregations)} features "
                f"are not produced: {', '.join(missing)}"
            )


def _check_gsf_features(fixations: pd.DataFrame) -> None:
    """`gsf_features` is indexed directly, so a missing one is a KeyError deep
    inside the per-trial loop."""
    missing = [
        col
        for col in gsf_features
        if col not in fixations.columns and col not in _CATEGORY_NORMALIZED_GSF
    ]
    if missing:
        raise KeyError(f"fixation data is missing gsf_features columns: {missing}")


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------


def build_paragraph_trial_level_features(
    ia: Optional[pd.DataFrame] = None,
    fixations: Optional[pd.DataFrame] = None,
    ia_path: Path = IA_PARAGRAPH_PATH,
    fixations_path: Optional[Path] = FIX_PARAGRAPH_PATH,
    include_fixation_features: bool = True,
    participants: Optional[Sequence[str]] = None,
    remove_practice: bool = True,
    remove_repeats: bool = True,
    fix_ptb_pos_double_mapping: bool = False,
    reconstruct_missing_columns: bool = True,
    processed_data_path: Path = PARAGRAPH_TRIAL_FEATURES_DIR,
    verbose: bool = True,
) -> pd.DataFrame:
    """Compute trial-level paragraph reading features, one row per trial.

    Mirrors `OneStopProcessor.dataset_specific_processing`: preprocess each
    report, merge the word-level measures onto the fixations, run
    `add_missing_features` over both, then extract.

    Args:
        ia / fixations: already-loaded raw reports; read from disk when omitted.
        include_fixation_features: set False for the (much faster) IA-only half.
        participants: restrict to a subset of participant ids.
        fix_ptb_pos_double_mapping: leave False to reproduce the source
            pipeline, where `ptb_pos` is mapped to numbers by
            `add_missing_features` and then mapped again per trial by
            `compute_fixation_trial_level_features` -- the second mapping yields
            all-NaN, so every `ptb_pos_*` feature comes out 0. True hands the
            fixation side the reduced-POS strings instead, so the 80 `ptb_pos_*`
            columns carry real values (and are named `ptb_pos_<n>.0_*`).
        reconstruct_missing_columns: fill the IA columns their feature list wants
            but our report does not have, from equivalents we do have (see
            `reconstruct_source_columns`). Set False to leave them absent, which
            is what the extraction sees when run on OneStop there.
        processed_data_path: where the external code writes its two
            `*_trial_level_feature_keys.csv` files (feature name -> model family).

    Returns:
        DataFrame indexed by (participant_id, TRIAL_INDEX).
    """
    ext = load_external_module()

    if ia is None or (include_fixation_features and fixations is None):
        ia_loaded, fixations_loaded = load_paragraph_reports(
            ia_path=ia_path,
            fixations_path=fixations_path if include_fixation_features else None,
            remove_practice=remove_practice,
            remove_repeats=remove_repeats,
            participants=participants,
            verbose=verbose,
        )
        ia = ia if ia is not None else ia_loaded
        fixations = fixations if fixations is not None else fixations_loaded

    if verbose:
        print("Preparing IA data ...")
    ia_prepared = prepare_ia_data(ia)

    fixations_prepared = None
    if include_fixation_features and fixations is not None:
        if verbose:
            print("Preparing fixation data ...")
        fixations_prepared = prepare_fixation_data(fixations)

    if reconstruct_missing_columns:
        ia_prepared = reconstruct_source_columns(
            ia_prepared, fixations_prepared, verbose=verbose
        )

    if fixations_prepared is not None:
        fixations_prepared = add_ia_report_features_to_fixation_data(
            ia_prepared, fixations_prepared
        )

    # Their processing loop: `add_missing_features` over both tables. It maps
    # `ptb_pos` to numbers, adds LengthCategory, and (fixations only) the
    # regression / progressive transition counts.
    reduced_pos = (
        None if fixations_prepared is None else fixations_prepared[REDUCED_POS_COL]
    )
    ia_prepared = ext.add_missing_features(
        et_data=ia_prepared,
        trial_groupby_columns=TRIAL_GROUPBY_COLUMNS,
        mode=DataType.IA,
    )
    if fixations_prepared is not None:
        fixations_prepared = ext.add_missing_features(
            et_data=fixations_prepared,
            trial_groupby_columns=TRIAL_GROUPBY_COLUMNS,
            mode=DataType.FIXATIONS,
        )
        if fix_ptb_pos_double_mapping:
            # Undo the first mapping so the per-trial one still finds strings,
            # and make the IA-side categories float, so that trials whose
            # fixations all landed on a word (int categories) and trials with
            # off-word fixations (float, from the unmapped NaN) do not produce
            # two half-empty sets of `ptb_pos_*` columns.
            fixations_prepared["ptb_pos"] = reduced_pos
            ia_prepared["ptb_pos"] = ia_prepared["ptb_pos"].astype(float)
        _check_gsf_features(fixations_prepared)

    processed_data_path = Path(processed_data_path)
    processed_data_path.mkdir(parents=True, exist_ok=True)

    if verbose:
        _report_skipped_feature_columns(ia_prepared, fixations_prepared)
        print(
            f"Computing trial level features for {_n_trials(ia_prepared):,} trials "
            f"({'IA + fixations' if fixations_prepared is not None else 'IA only'}) ..."
        )
    features = ext.compute_trial_level_features(
        raw_fixation_data=fixations_prepared,
        raw_ia_data=ia_prepared,
        trial_groupby_columns=TRIAL_GROUPBY_COLUMNS,
        processed_data_path=processed_data_path,
    )

    features = _add_trial_metadata(features, ia_prepared)
    if verbose:
        print(f"Done: {features.shape[0]:,} trials x {features.shape[1]} columns")
    return features


def _add_trial_metadata(features: pd.DataFrame, ia_prepared: pd.DataFrame) -> pd.DataFrame:
    """Attach the question-preview (hunter/gatherer) flag to the feature table."""
    if Con.QUESTION_PREVIEW_COLUMN not in ia_prepared.columns:
        return features
    preview = (
        ia_prepared.groupby(TRIAL_GROUPBY_COLUMNS)[Con.QUESTION_PREVIEW_COLUMN]
        .first()
        .astype("boolean")
        .astype("Int64")
    )
    return features.join(preview, how="left")


def save_paragraph_trial_level_features(
    output_path: Path = PARAGRAPH_TRIAL_FEATURES_PATH,
    verbose: bool = True,
    **kwargs,
) -> pd.DataFrame:
    """Build the trial-level features and write them to CSV (index kept as columns)."""
    features = build_paragraph_trial_level_features(verbose=verbose, **kwargs)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(output_path, index=True)
    if verbose:
        print(f"Saved {len(features):,} trials x {features.shape[1]} cols to {output_path}")
    return features


def load_paragraph_trial_level_features(
    path: Path = PARAGRAPH_TRIAL_FEATURES_PATH,
) -> pd.DataFrame:
    """Load the cached feature table produced by `save_paragraph_trial_level_features`."""
    return pd.read_csv(Path(path))


if __name__ == "__main__":
    save_paragraph_trial_level_features()
