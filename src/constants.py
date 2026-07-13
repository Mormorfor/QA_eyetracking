# ---------------------------------------------------------------------------
# Pre-existing column name constants
# ---------------------------------------------------------------------------

ARTICLE_COLUMN = "article_id"
DIFFICULTY_COLUMN = "difficulty_level"
BATCH_COLUMN = "article_batch"
PARAGRAPH_COLUMN = "paragraph_id"

LIST_COLUMN = "list_number"

REPEATED_TRIAL_COLUMN = "repeated_reading_trial"
PRACTICE_TRIAL_COLUMN = "practice_trial"
QUESTION_PREVIEW_COLUMN = "question_preview"

SELECTED_ANSWER_POSITION_COLUMN = "selected_answer_position"
CORRECT_ANSWER_POSITION_COLUMN = "correct_answer_position"
ANSWERS_ORDER_COLUMN = "answers_order"

INTEREST_AREA_ID = "IA_ID"
TRIAL_ID = "TRIAL_INDEX"
PARTICIPANT_ID = "participant_id"

IA_DWELL_TIME = "IA_DWELL_TIME"
IA_FIXATIONS_COUNT = "IA_FIXATION_COUNT"
IA_FIRST_FIXATION_DURATION = "IA_FIRST_FIXATION_DURATION"
IA_LAST_FIXATION_TIME = "IA_LAST_FIXATION_TIME"

INTEREST_AREA_FIXATION_SEQUENCE = "INTEREST_AREA_FIXATION_SEQUENCE"

AUXILIARY_SPAN_TYPE_COLUMN = "auxiliary_span_type"
SAME_CRITICAL_SPAN_COLUMN = "same_critical_span"

DOMINANT_EYE_COLUMN = "EYE_TRACKED"
IA_AVERAGE_FIX_PUPIL_SIZE = "IA_AVERAGE_FIX_PUPIL_SIZE"
IA_MAX_FIX_PUPIL_SIZE = "IA_MAX_FIX_PUPIL_SIZE"
IA_MIN_FIX_PUPIL_SIZE = "IA_MIN_FIX_PUPIL_SIZE"


# ---------------------------------------------------------------------------
# Created column name constants
# ---------------------------------------------------------------------------

TEXT_ID_COLUMN = "text_id"
TEXT_ID_WITH_Q_COLUMN = "text_id_with_q"

IS_CORRECT_COLUMN = "is_correct"

AREA_SCREEN_LOCATION = "area_screen_loc"
AREA_LABEL_COLUMN = "area_label"
SELECTED_ANSWER_LABEL_COLUMN = "selected_answer_label"

AREA_SKIPPED = "area_skipped"
TOTAL_IA_DWELL_TIME = "total_area_dwell_time"
TOTAL_TRIAL_DWELL_TIME = "total_dwell_time"

LAST_VISITED_LABEL = "last_answer_area_visited_lbl"
LAST_VISITED_LOCATION = "last_answer_area_visited_loc"

LAST_LBL_BEFORE_CONFIRM = "last_lbl_before_confirm"
LAST_LBL_BEFORE_SELECT = "last_lbl_before_select"

FIX_SEQUENCE_BY_LABEL = "fix_by_label"
FIX_SEQUENCE_BY_LOCATION = "fix_by_loc"

SIMPLIFIED_FIX_SEQ_BY_LABEL = "simpl_fix_by_label"
SIMPLIFIED_FIX_SEQ_BY_LOCATION = "simpl_fix_by_loc"

STRATEGY_COL = "strategy"

# Pattern-breaking / starting-strategy features
# Participant-level:
STARTING_STRATEGY_COL = "starting_strategy"
DOMINANT_STARTING_STRATEGY = "dominant_starting_strategy"
DOMINANCE_SCORE = "dominance_score"
N_STRATEGY_TRIALS = "n_strategy_trials"

# Per-trial model features, in two variants (question tokens kept vs. dropped
# before computing the starting strategy). "breaks_pattern" flags a trial whose
# starting strategy differs from the participant's dominant one.
BREAKS_PATTERN_WITH_Q = "breaks_pattern_with_q"
BREAKS_PATTERN_NO_Q = "breaks_pattern_no_q"
DOMINANCE_SCORE_WITH_Q = "dominance_score_with_q"
DOMINANCE_SCORE_NO_Q = "dominance_score_no_q"

# Interaction term: breaks_pattern * dominance_score (per q-variant). Equals the
# dominance score on trials that break the pattern, 0 otherwise -- i.e. "how much
# breaking matters, scaled by how dominant the participant's pattern is".
BREAKS_X_DOMINANCE_WITH_Q = "breaks_x_dominance_with_q"
BREAKS_X_DOMINANCE_NO_Q = "breaks_x_dominance_no_q"

# Graded "breaks pattern": token-level Levenshtein distance between the trial's
# starting strategy and the participant's dominant one (0 = identical).
STRATEGY_DISTANCE_WITH_Q = "strategy_distance_with_q"
STRATEGY_DISTANCE_NO_Q = "strategy_distance_no_q"

SELECTED_DWELL_DURATION = "selected_a_dwell_duration"
SELECTED_SCREEN_LOCATION = "selected_a_screen_loc"

SEGMENT_COLUMN = "time_segment"
SEQUENCE_LENGTH_COLUMN = "sequence_length"
SKIPPED_COLUMN = "skipped"

MEAN_AVG_FIX_PUPIL_SIZE = "mean_avg_fix_pupil_size"
MEAN_MAX_FIX_PUPIL_SIZE = "mean_max_fix_pupil_size"
MEAN_MIN_FIX_PUPIL_SIZE = "mean_min_fix_pupil_size"

MEAN_AVG_FIX_PUPIL_SIZE_Z = "mean_avg_fix_pupil_size_z"
MEAN_MAX_FIX_PUPIL_SIZE_Z = "mean_max_fix_pupil_size_z"
MEAN_MIN_FIX_PUPIL_SIZE_Z = "mean_min_fix_pupil_size_z"

FIRST_ENCOUNTER_AVG_PUPIL_SIZE = "first_encounter_avg_pupil_size"
FIRST_ENCOUNTER_AVG_PUPIL_SIZE_Z = "first_encounter_avg_pupil_size_z"

# ---------------------------------------------------------------------------
# Helper Constants
# ---------------------------------------------------------------------------

ANSWER_PREFIX = "answer_"
ANSWER_LABELS = ["A", "B", "C", "D"]

# must mantain consistent ordering + consistency with ANSWER_LABEL_CHOICES

LOC_CHOICES = [
    "question",
    "answer_0(top)",
    "answer_1(left)",
    "answer_2(right)",
    "answer_3(bottom)",
]

# DO NOT CHANGE
LABEL_CHOICES = ["question", "answer_A", "answer_B", "answer_C", "answer_D"]

NUM_LABEL_VISITS = "num_label_visits"
NUM_LOC_VISITS = "num_loc_visits"

TRIAL_ID_COLS = (PARTICIPANT_ID, TRIAL_ID)

# ---------------------------------------------------------------------------
# Derived (contrast) feature suffixes / helpers
# ---------------------------------------------------------------------------

CORRECT_SUFFIX = "correct"
WRONG_MEAN_SUFFIX = "wrong_mean"
CONTRAST_SUFFIX = "contrast"
DISTANCE_FURTHEST_SUFFIX = "distance_furthest"
DISTANCE_CLOSEST_SUFFIX = "distance_closest"

DERIVED_SEP = "__"

# ---------------------------------------------------------------------------
# Existing metrics
# ---------------------------------------------------------------------------

MEAN_DWELL_TIME = "mean_dwell_time"
MEAN_FIXATIONS_COUNT = "mean_fixations_count"
MEAN_FIRST_FIXATION_DURATION = "mean_first_fixation_duration"
SKIP_RATE = "skip_rate"
AREA_DWELL_PROPORTION = "area_dwell_proportion"

NUM_OF_SELECTS = "ANSWER_PRESS_NUMBER"

# Raw trial-level total answering RT column (time from showing answers to the
# confirm click) and the friendlier feature name it is exposed under.
CONFIRM_FINAL_ANSWER_RT = "CONFIRM_FINAL_ANSWER_RT"
TOTAL_ANSWERING_RT = "total_answering_RT"
TOTAL_ANSWERING_RT_NORMALIZED = "total_answering_RT_normalized"



AREA_METRIC_COLUMNS_MODELING = [
    MEAN_DWELL_TIME,
    MEAN_FIXATIONS_COUNT,
    MEAN_FIRST_FIXATION_DURATION,
    SKIP_RATE,
    AREA_DWELL_PROPORTION,
    MEAN_AVG_FIX_PUPIL_SIZE_Z,
    MEAN_MAX_FIX_PUPIL_SIZE_Z,
    MEAN_MIN_FIX_PUPIL_SIZE_Z,
    FIRST_ENCOUNTER_AVG_PUPIL_SIZE_Z,
    NUM_LABEL_VISITS,
]

AREA_METRIC_COLUMNS_VIZES = [
    MEAN_DWELL_TIME,
    MEAN_FIXATIONS_COUNT,
    MEAN_FIRST_FIXATION_DURATION,
    SKIP_RATE,
    AREA_DWELL_PROPORTION,
    MEAN_AVG_FIX_PUPIL_SIZE,
    MEAN_MAX_FIX_PUPIL_SIZE,
    MEAN_MIN_FIX_PUPIL_SIZE,
    FIRST_ENCOUNTER_AVG_PUPIL_SIZE,
    NUM_LABEL_VISITS,
]


PREF_SPECS = [
    (MEAN_DWELL_TIME, "high"),
    (MEAN_FIXATIONS_COUNT, "high"),
    (MEAN_FIRST_FIXATION_DURATION, "high"),
    (SKIP_RATE, "low"),
    (AREA_DWELL_PROPORTION, "high"),
    (MEAN_AVG_FIX_PUPIL_SIZE_Z, "high"),
    (MEAN_MAX_FIX_PUPIL_SIZE_Z, "high"),
    (MEAN_MIN_FIX_PUPIL_SIZE_Z, "low"),
    (FIRST_ENCOUNTER_AVG_PUPIL_SIZE_Z, "high"),
    (NUM_LABEL_VISITS, "high"),
]

# ---------------------------------------------------------------------------
# Fixation Level Constants
# ---------------------------------------------------------------------------

CURRENT_FIX_PUPIL_SIZE = "CURRENT_FIX_PUPIL"
NEAREST_IA = "CURRENT_FIX_NEAREST_INTEREST_AREA"
CURRENT_FIX_LABEL = "CURRENT_FIX_LABEL"  # contains current fix ms!
CURRENT_FIX_INTEREST_AREAS = "CURRENT_FIX_INTEREST_AREAS"


# ---------------------------------------------------------------------------
# Trial-level select-confirm / event reconstruction constants
# ---------------------------------------------------------------------------

RECORDING_SESSION_LABEL = "RECORDING_SESSION_LABEL"
ALL_ANSWERS = "ALL_ANSWERS"

CURRENT_FIX_MSG_LIST_TEXT = "CURRENT_FIX_MSG_LIST_TEXT"
CURRENT_FIX_MSG_LIST_TIME = "CURRENT_FIX_MSG_LIST_TIME"

NEXT_SAC_MSG_LIST_TEXT = "NEXT_SAC_MSG_LIST_TEXT"
NEXT_SAC_MSG_LIST_TIME = "NEXT_SAC_MSG_LIST_TIME"

ALL_ANSWERS_LIST = "ALL_ANSWERS_LIST"
PREV_ALL_ANSWERS_LIST = "PREV_ALL_ANSWERS_LIST"
TRIAL_ANSWERS = "TRIAL_ANSWERS"

IS_MALFORMED = "IS_MALFORMED"
FIRST_MALFORMED_TRIAL = "FIRST_MALFORMED_TRIAL"

MISSING_LIST_MARKER = "."

CHOOSE_ANSWER_KEYWORD = "CHOOSE_ANSWER"
CONFIRM_ANSWER_KEYWORD = "CONFIRM_ANSWER"

MATCH_TIMESTAMPS_1 = "MATCH_TIMESTAMPS_1"
MATCH_TIMESTAMPS_2 = "MATCH_TIMESTAMPS_2"
MATCH_TIMESTAMPS = "MATCH_TIMESTAMPS"
MATCH_TIMESTAMP = "MATCH_TIMESTAMP"

SELECT_ANS_TIMESTAMPS = "SELECT_ANS_TIMESTAMPS"
CONFIRM_TIMESTAMPS = "CONFIRM_TIMESTAMPS"

FIXATION_TIMESTAMP = "FIXATION_TIMESTAMP"
FIXATION_PAIR = "FIXATION_PAIR"
FIXATION_TIMESTAMPS_IA = "FIXATION_TIMESTAMPS_IA"

LAST_FIXATIONS_BEFORE_SELECT = "LAST_FIXATIONS_BEFORE_SELECT"
LAST_FIXATIONS_BEFORE_CONFIRM = "LAST_FIXATIONS_BEFORE_CONFIRM"

NUM_OF_SELECTS_DERIVED = "NUM_OF_SELECTS_derived"
