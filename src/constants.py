
# ---------------------------------------------------------------------------
# Pre-existing column name constants
# ---------------------------------------------------------------------------

ARTICLE_COLUMN = 'article_id'
DIFFICULTY_COLUMN = 'difficulty_level'
BATCH_COLUMN = 'article_batch'
PARAGRAPH_COLUMN = 'paragraph_id'

REPEATED_TRIAL_COLUMN = "repeated_reading_trial"
PRACTICE_TRIAL_COLUMN = "practice_trial"
QUESTION_PREVIEW_COLUMN = "question_preview"

SELECTED_ANSWER_POSITION_COLUMN = "selected_answer_position"
CORRECT_ANSWER_POSITION_COLUMN = "correct_answer_position"
ANSWERS_ORDER_COLUMN = "answers_order"

INTEREST_AREA_ID = "IA_ID"
TRIAL_ID = 'TRIAL_INDEX'
PARTICIPANT_ID = 'participant_id'

IA_DWELL_TIME = "IA_DWELL_TIME"
IA_FIXATIONS_COUNT = "IA_FIXATION_COUNT"
IA_FIRST_FIXATION_DURATION = "IA_FIRST_FIXATION_DURATION"
IA_LAST_FIXATION_TIME = "IA_LAST_FIXATION_TIME"

INTEREST_AREA_FIXATION_SEQUENCE = "INTEREST_AREA_FIXATION_SEQUENCE"

AUXILIARY_SPAN_TYPE_COLUMN = "auxiliary_span_type"
SAME_CRITICAL_SPAN_COLUMN = "same_critical_span"
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

LAST_VISITED_LABEL = "last_area_visited_lbl"
LAST_VISITED_LOCATION = "last_area_visited_loc"

FIX_SEQUENCE_BY_LABEL = "fix_by_label"
FIX_SEQUENCE_BY_LOCATION = "fix_by_loc"

SIMPLIFIED_FIX_SEQ_BY_LABEL = "simpl_fix_by_label"
SIMPLIFIED_FIX_SEQ_BY_LOCATION = "simpl_fix_by_loc"

STRATEGY_COL = 'strategy'

SELECTED_DWELL_DURATION = "selected_a_dwell_duration"
SELECTED_SCREEN_LOCATION = "selected_a_screen_loc"

SEGMENT_COLUMN = "time_segment"
SEQUENCE_LENGTH_COLUMN = "sequence_length"
SKIPPED_COLUMN = "skipped"
# ---------------------------------------------------------------------------
# Helper Constants
# ---------------------------------------------------------------------------

ANSWER_PREFIX = "answer_"
ANSWER_LABELS = ["A", "B", "C", "D"]

#AREA_LABEL_CHOICES = ['question', 'top', 'left', 'right', 'bottom']
AREA_LABEL_CHOICES = ['question', 'answer_0(top)', 'answer_1(left)', 'answer_2(right)', 'answer_3(bottom)']

#DO NOT CHANGE
ANSWER_LABEL_CHOICES = ['question', 'answer_A', 'answer_B', 'answer_C', 'answer_D']

# ---------------------------------------------------------------------------
# Existing metrics
# ---------------------------------------------------------------------------

MEAN_DWELL_TIME = "mean_dwell_time"
MEAN_FIXATIONS_COUNT = "mean_fixations_count"
MEAN_FIRST_FIXATION_DURATION = "mean_first_fixation_duration"
SKIP_RATE = "skip_rate"
AREA_DWELL_PROPORTION = "area_dwell_proportion"

AREA_METRIC_COLUMNS = [
    MEAN_DWELL_TIME,
    MEAN_FIXATIONS_COUNT,
    MEAN_FIRST_FIXATION_DURATION,
    SKIP_RATE,
    AREA_DWELL_PROPORTION,
]
