"""Constants required by `src/external/utils - paragraph feature extraction.py`.

The external snippet expects these names to come from its original project's
config package. The column lists, `gsf_features` and the enums below are copied
verbatim from that project's `src/configs/constants.py`; only the parts the
snippet actually imports are reproduced (its config module also holds model /
trainer / dataset enums we have no use for).

Two notes on how these lists meet *our* raw reports
(`data_raw/full/ia_Paragraph.csv`, `data_raw/full/fixations_Paragraph.csv`):

* `compute_*_trial_level_features` skips any column of the two
  `numerical_*_trial_columns` lists that is missing from the data, so columns
  that only exist after the source project's own preprocessing simply produce no
  features here. Absent on our side: `landing_position`, `IA_FIRST_FIX_DURATION`,
  `IA_FIRST_FIX_DWELL_TIME`, `IA_REGRESSION_OUT_TIME`,
  `IA_TOTAL_FIXATION_DURATION`, `mean_sacc_dur`, `peak_sacc_velocity` -- all but
  the last two are near-duplicates of columns we do have
  (`IA_FIRST_RUN_LANDING_POSITION`, `IA_FIRST_FIXATION_DURATION`,
  `IA_DWELL_TIME`).
* `gsf_features`, by contrast, is indexed directly and every one of its columns
  must be present on the fixation side -- `src/external/paragraph_trial_features.py`
  merges the word-level `IA_*` measures onto each fixation for exactly that
  reason.
"""

from __future__ import annotations

from enum import StrEnum

# ---------------------------------------------------------------------------
# Numerical features used for the sklearn baselines
# ---------------------------------------------------------------------------

numerical_feature_aggregations = [
    'mean',
    'std',
    'median',
    'skew',
    'kurtosis',
    'max',
    'min',
]

numerical_ia_trial_columns = [
    'landing_position',
    'IA_FIRST_FIX_DURATION',
    'IA_FIRST_FIX_DWELL_TIME',
    'IA_REGRESSION_OUT_TIME',
    'IA_DWELL_TIME',
    'IA_TOTAL_FIXATION_DURATION',
    'IA_FIXATION_COUNT',
    'mean_sacc_dur',
    'peak_sacc_velocity',
    'right_dependents_count',
    'IA_LAST_RUN_LANDING_POSITION',
    'IA_FIRST_RUN_LANDING_POSITION',
    'IA_FIRST_FIXATION_DURATION',
    'IA_RUN_COUNT',
    'IA_TOP',
    'IA_REGRESSION_IN_COUNT',
    'IA_LAST_RUN_DWELL_TIME',
    'wordfreq_frequency',
    'IA_LAST_FIXATION_DURATION',
    'IA_REGRESSION_OUT_COUNT',
    'IA_FIRST_FIXATION_VISITED_IA_COUNT',
    'IA_SELECTIVE_REGRESSION_PATH_DURATION',
    'IA_SKIP',
    'PARAGRAPH_RT',
    'left_dependents_count',
    'repeated_reading_trial',
    'word_length',
    'is_content_word',
    'IA_FIRST_RUN_FIXATION_COUNT',
    'gpt2_surprisal',
    'IA_LAST_RUN_FIXATION_COUNT',
    'IA_FIRST_FIX_PROGRESSIVE',
    'IA_REGRESSION_PATH_DURATION',
    'IA_REGRESSION_OUT_FULL_COUNT',
    'IA_FIRST_RUN_DWELL_TIME',
    'IA_LEFT',
]

numerical_fixation_trial_columns = [
    'CURRENT_FIX_DURATION',
    'CURRENT_FIX_INTEREST_AREA_DWELL_TIME',
    'CURRENT_FIX_INTEREST_AREA_FIX_COUNT',
    'CURRENT_FIX_NEAREST_INTEREST_AREA_DISTANCE',
    'CURRENT_FIX_RUN_SIZE',
    'NEXT_SAC_DURATION',
    'NEXT_SAC_PEAK_VELOCITY',
    'NEXT_SAC_AMPLITUDE',
    'TRIAL_FIXATION_TOTAL',
    'CURRENT_FIX_X',
    'CURRENT_FIX_Y',
    'NEXT_FIX_DISTANCE',
    'NEXT_FIX_ANGLE',
    'right_dependents_count',
    'IA_FIRST_FIXATION_DURATION',
    'IA_RUN_COUNT',
    'IA_FIXATION_COUNT',
    'NEXT_SAC_END_Y',
    'IA_REGRESSION_IN_COUNT',
    'IA_LAST_RUN_DWELL_TIME',
    'wordfreq_frequency',
    'IA_LAST_FIXATION_DURATION',
    'normalized_incoming_regression_count',
    'IA_REGRESSION_OUT_COUNT',
    'PREVIOUS_FIX_ANGLE',
    'normalized_outgoing_regression_count',
    'IA_FIRST_FIXATION_VISITED_IA_COUNT',
    'CURRENT_FIX_PUPIL',
    'IA_SELECTIVE_REGRESSION_PATH_DURATION',
    'IA_SKIP',
    'NEXT_SAC_START_Y',
    'NEXT_SAC_ANGLE',
    'word_length',
    'IA_FIRST_RUN_FIXATION_COUNT',
    'gpt2_surprisal',
    'IA_LAST_RUN_FIXATION_COUNT',
    'IA_DWELL_TIME',
    'IA_FIRST_FIX_PROGRESSIVE',
    'NEXT_SAC_START_X',
    'NEXT_SAC_AVG_VELOCITY',
    'IA_REGRESSION_IN_COUNT_sum',
    'IA_REGRESSION_PATH_DURATION',
    'IA_REGRESSION_OUT_FULL_COUNT',
    'IA_FIRST_RUN_DWELL_TIME',
    'NEXT_SAC_END_X',
    'PREVIOUS_FIX_DISTANCE',
]

# Gaze scanpath features averaged within each word category (BEyeLSTM-style):
# each is averaged per category of `is_content_word` / `ptb_pos` / `entity_type`
# / `universal_pos` and flattened into `<category>_<value>_<feature>` columns.
gsf_features = [
    'gpt2_surprisal',
    'word_length',
    'left_dependents_count',
    'right_dependents_count',
    'distance_to_head',
    'IA_FIRST_FIXATION_DURATION',
    'IA_DWELL_TIME',
    'normalized_incoming_regression_count',
    'CURRENT_FIX_X',
    'CURRENT_FIX_Y',
    'normalized_outgoing_regression_count',
    'normalized_outgoing_progressive_count',
    'LengthCategory_normalized_IA_DWELL_TIME',
    'universal_pos_normalized_IA_DWELL_TIME',
    'LengthCategory_normalized_IA_FIRST_FIXATION_DURATION',
    'universal_pos_normalized_IA_FIRST_FIXATION_DURATION',
]

# ---------------------------------------------------------------------------
# Local additions (not in the source project)
# ---------------------------------------------------------------------------

# Pupil size matters elsewhere in this project (see src/derived/pupil_norm.py),
# so it is aggregated per trial too. Set to [] to reproduce the source project's
# feature set exactly.
EXTRA_IA_TRIAL_COLUMNS = [
    'IA_AVERAGE_FIX_PUPIL_SIZE',
    'IA_MAX_FIX_PUPIL_SIZE',
    'IA_MIN_FIX_PUPIL_SIZE',
]
numerical_ia_trial_columns = numerical_ia_trial_columns + EXTRA_IA_TRIAL_COLUMNS

# The external code's `ptb_pos` holds the *reduced* tag set; our reports keep
# those values in `Reduced_POS` (our `ptb_pos` is the full Penn-Treebank tag).
# Same mapping the external code applies internally.
REDUCED_POS_TO_NUMBER = {'FUNC': 0, 'NOUN': 1, 'VERB': 2, 'ADJ': 3, 'UNKNOWN': 4}


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class DataType(StrEnum):
    """Types of eye-tracking data the external code moves around."""

    IA = 'ia'
    FIXATIONS = 'fixations'
    RAW = 'raw'
    TRIAL_LEVEL = 'trial_level'
    METADATA = 'metadata'


class SetNames(StrEnum):
    """Split / evaluation-regime names (only used by `load_fold_data`)."""

    TRAIN = 'train'
    VAL = 'val'
    TEST = 'test'
    SEEN_SUBJECT_UNSEEN_ITEM = 'seen_subject_unseen_item'
    UNSEEN_SUBJECT_SEEN_ITEM = 'unseen_subject_seen_item'
    UNSEEN_SUBJECT_UNSEEN_ITEM = 'unseen_subject_unseen_item'


REGIMES = [
    SetNames.SEEN_SUBJECT_UNSEEN_ITEM,
    SetNames.UNSEEN_SUBJECT_SEEN_ITEM,
    SetNames.UNSEEN_SUBJECT_UNSEEN_ITEM,
]
