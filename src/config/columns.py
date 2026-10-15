"""The name of every column the project uses.

Code refers to columns through these constants instead of typing the strings, so
each column has exactly one spelling. Nothing in this file computes anything.

The file has three parts:

1. columns that arrive in the eye tracker's reports,
2. columns the pipeline creates,
3. lists and naming rules built from them.

Two naming patterns cover most feature names you will meet:

    <metric>__<area>        a measurement on one screen area      skip_rate__answer_B
    <metric>__<comparison>  that measurement compared across the  skip_rate__contrast
                            correct and wrong answers

`answer_A` is always the correct answer. The letters are meanings, not screen
positions; where each answer actually sat on screen is in the `*_loc` columns.
"""

# ===========================================================================
# 1. Columns that arrive in the eye tracker's reports
# ===========================================================================

# Which text the trial used.
ARTICLE_COLUMN = "article_id"
DIFFICULTY_COLUMN = "difficulty_level"
BATCH_COLUMN = "article_batch"
PARAGRAPH_COLUMN = "paragraph_id"

# Which counterbalancing list the participant was given.
LIST_COLUMN = "list_number"

# Trials dropped as soon as the data is read: re-readings of a text, and
# practice trials.
REPEATED_TRIAL_COLUMN = "repeated_reading_trial"
PRACTICE_TRIAL_COLUMN = "practice_trial"

# Study 1's manipulation. True = the participant saw the question before
# reading the paragraph ("hunters"); False = they read the paragraph first
# ("gatherers"). Each participant is in exactly one group.
QUESTION_PREVIEW_COLUMN = "question_preview"

# Study 2's manipulation: how much the participant knows when they reach the
# answers -- "full knowledge" (shown the correct answer's content beforehand),
# "partial knowledge" (read the paragraph) or "no knowledge" (nothing).
# Every Study 2 participant does all three, so splitting the data by regime
# splits each person's trials. Not present in Study 1.
REGIME_COLUMN = "regime"

# More Study 2 columns. None of these exist in Study 1 files.
REGIME_LIST_COLUMN = "regime_list"
REGIME_ORDER_COLUMN = "regime_order"
REGIME_ORDER_POSITION_COLUMN = "regime_order_position"
#: The participant's own 1-5 rating of how sure they were, asked after every
#: Study 2 trial. Study 1 has no equivalent.
CONFIDENCE_RATING_COLUMN = "confidence_rating"
CONFIDENCE_RATING_TXT_COLUMN = "confidence_rating_txt"
#: Which of the three counterbalancing article groups the item came from.
ARTICLE_GROUP_COLUMN = "article_group"
#: Full-knowledge regime only: the reworded correct answer the participant is
#: shown before the answer screen.
REPHRASED_ANSWER_A_COLUMN = "rephrased_answer_a"

# Screen positions (0-3) of the chosen answer and of the correct answer, and
# which letter sat in each position. Together these give `is_correct` and
# translate between positions and letters.
SELECTED_ANSWER_POSITION_COLUMN = "selected_answer_position"
CORRECT_ANSWER_POSITION_COLUMN = "correct_answer_position"
ANSWERS_ORDER_COLUMN = "answers_order"

# An interest area is one word on screen; `IA_ID` numbers the words in reading
# order. A trial is identified by participant_id and TRIAL_INDEX together.
INTEREST_AREA_ID = "IA_ID"
TRIAL_ID = "TRIAL_INDEX"
PARTICIPANT_ID = "participant_id"

# Study 2 only. A Study 2 recording is named `4XXX_YZ`: the person (`4XXX`),
# the batch (`Y`, 1-3) and the list (`Z`, 1-18), with no separator because the
# tracker allows only 8 characters. One person can come back for several
# sessions, so `participant_id` holds just the person and `session_id` holds
# the full recording name. See src/ingest/knowqa.py.
SESSION_ID = "session_id"
# The tracker's own trial counter within one session. Kept as a number because
# in Study 2 `TRIAL_INDEX` is a text code made of batch, list and trial
# (e.g. `b2l01t005`).
TRIAL_NUMBER = "trial_number"

# The word an interest area covers. Read in order, these spell out the text
# on screen.
IA_LABEL = "IA_LABEL"

# Per-word measurements from the tracker. A word nobody looked at has dwell
# time 0 and fixation count 0, but its first-fixation duration is "." --
# there was no fixation to time.
IA_DWELL_TIME = "IA_DWELL_TIME"
IA_FIXATIONS_COUNT = "IA_FIXATION_COUNT"
IA_FIRST_FIXATION_DURATION = "IA_FIRST_FIXATION_DURATION"
IA_LAST_FIXATION_TIME = "IA_LAST_FIXATION_TIME"
# When the first look at a word started, and how long the last look lasted.
# Reading-time code uses them to measure an area from the start of the first
# look to the end of the last one.
IA_FIRST_FIXATION_TIME = "IA_FIRST_FIXATION_TIME"
IA_LAST_FIXATION_DURATION = "IA_LAST_FIXATION_DURATION"

# Where the word's box sits on screen, in pixels. This, not the stored text,
# decides which area (the question or which answer) a word belongs to, because
# the stored text does not always match what was displayed.
IA_TOP = "IA_TOP"
IA_LEFT = "IA_LEFT"

#: The report's own A/B/C/D record of the choice. The pipeline does not read
#: it: `selected_answer_label` is worked out from `answers_order` and
#: `selected_answer_position` instead, and the two agree on every Study 1 row.
SELECTED_ANSWER_COLUMN = "selected_answer"

#: The question and answer text, as the report stores it.
#:
#: Careful: `answer_1`..`answer_4` are screen POSITIONS (whatever was shown in
#: slot 1), while `answer_A`..`answer_D` everywhere else are MEANINGS (A is the
#: correct one). `add_answer_text_columns` uses `answers_order` to rewrite the
#: text by letter.
QUESTION_TEXT_COLUMN = "question"
ANSWER_TEXT_COLUMNS = ["answer_1", "answer_2", "answer_3", "answer_4"]

# The tracker's list of which words were looked at, in order. Kept in
# trials.csv as the tracker's own record; nothing in the pipeline reads it. The
# fixation sequences are built from the fixation report (features/sequences.py).
INTEREST_AREA_FIXATION_SEQUENCE = "INTEREST_AREA_FIXATION_SEQUENCE"

# Paragraph screen only: which part of the paragraph a word is in --
# "critical" (holds the answer), "distractor" (supports a wrong answer) or
# "outside" (everything else).
AUXILIARY_SPAN_TYPE_COLUMN = "auxiliary_span_type"

# Each paragraph has three possible questions, and these two columns number
# them in two DIFFERENT orders. `same_critical_span` uses Study 1's order (it is
# the last part of `text_id_with_q`); `onestopqa_question_id` uses OneStopQA's
# own order, which the Study 2 materials were built from. They often disagree,
# so never use one in place of the other.
SAME_CRITICAL_SPAN_COLUMN = "same_critical_span"
ONESTOPQA_QUESTION_ID = "onestopqa_question_id"

# Which eye was recorded (left or right).
DOMINANT_EYE_COLUMN = "EYE_TRACKED"

# Pupil size during the fixations on a word. The report gives it in the
# tracker's own units (an area); the pipeline converts these same columns to a
# diameter in mm (`pupil.add_zscored_pupil_columns`), so saved tables hold mm.
IA_AVERAGE_FIX_PUPIL_SIZE = "IA_AVERAGE_FIX_PUPIL_SIZE"
IA_MAX_FIX_PUPIL_SIZE = "IA_MAX_FIX_PUPIL_SIZE"
IA_MIN_FIX_PUPIL_SIZE = "IA_MIN_FIX_PUPIL_SIZE"


# ===========================================================================
# 2. Columns the pipeline creates
# ===========================================================================

# `text_id` names a paragraph. `text_id_with_q` adds which question was asked
# about it; that pair is one test item.
TEXT_ID_COLUMN = "text_id"
TEXT_ID_WITH_Q_COLUMN = "text_id_with_q"

# 1 if the participant chose the correct answer, else 0. This is what the
# model predicts.
IS_CORRECT_COLUMN = "is_correct"

# Which of the five screen areas a word belongs to, named two ways: by place
# on screen (`area_screen_loc`) or by meaning (`area_label`). See LOC_CHOICES
# and LABEL_CHOICES below.
AREA_SCREEN_LOCATION = "area_screen_loc"
AREA_LABEL_COLUMN = "area_label"
# The letter (A-D) the participant chose.
SELECTED_ANSWER_LABEL_COLUMN = "selected_answer_label"

# Working columns behind the area measurements: whether a word was never looked
# at, total looking time on one area, and total looking time on the whole trial.
AREA_SKIPPED = "area_skipped"
TOTAL_IA_DWELL_TIME = "total_area_dwell_time"
TOTAL_TRIAL_DWELL_TIME = "total_dwell_time"

# The last answer the participant looked at in the trial. If their very last
# look was at the question, this is the last answer before it.
LAST_VISITED_LABEL = "last_answer_area_visited_lbl"
LAST_VISITED_LOCATION = "last_answer_area_visited_loc"

# The area looked at just before the participant confirmed their answer, and
# just before they selected it. Timed from the button presses.
LAST_LBL_BEFORE_CONFIRM = "last_lbl_before_confirm"
LAST_LBL_BEFORE_SELECT = "last_lbl_before_select"

# The same two looks as saved in the side file all_participants_last.csv
# (features/last_visited.py): the word and its area, before the last selection
# press and before the confirming press. The two area columns enter the trial
# table under the names above.
LAST_IA_BEFORE_SELECT = "last_ia_before_last_select"
AREA_LABEL_BEFORE_SELECT = "area_label_before_last_select"
LAST_IA_BEFORE_CONFIRM = "last_ia_before_confirm"
AREA_LABEL_BEFORE_CONFIRM = "area_label_before_confirm"

# The order in which areas were looked at: one entry per run of fixations on
# one word, named by meaning (`fix_by_label`) or by place (`fix_by_loc`).
FIX_SEQUENCE_BY_LABEL = "fix_by_label"
FIX_SEQUENCE_BY_LOCATION = "fix_by_loc"

# The same sequence with repeats merged, so A A B A becomes A B A. Each entry
# is a move into a new area.
SIMPLIFIED_FIX_SEQ_BY_LABEL = "simpl_fix_by_label"
SIMPLIFIED_FIX_SEQ_BY_LOCATION = "simpl_fix_by_loc"

# --- The opening scan ("starting strategy") ---------------------------------
#
# A trial's starting strategy is the order in which the participant first
# visited the four answer positions, e.g. (top, right, bottom, left).

# The name the scan-strategy analysis uses for a trial's opening scan.
STRATEGY_COL = "strategy"

# One trial's opening scan; the participant's most common one; the share of
# their trials that use it; and how many trials that share is out of.
STARTING_STRATEGY_COL = "starting_strategy"
DOMINANT_STARTING_STRATEGY = "dominant_starting_strategy"
DOMINANCE_SCORE = "dominance_score"
N_STRATEGY_TRIALS = "n_strategy_trials"

# Per-trial features, each in two versions: with visits to the question kept in
# the sequence (`_with_q`) or removed first (`_no_q`).
#
# breaks_pattern = 1 if this trial's opening differs from the participant's
# usual one.
BREAKS_PATTERN_WITH_Q = "breaks_pattern_with_q"
BREAKS_PATTERN_NO_Q = "breaks_pattern_no_q"
DOMINANCE_SCORE_WITH_Q = "dominance_score_with_q"
DOMINANCE_SCORE_NO_Q = "dominance_score_no_q"

# breaks_pattern x dominance_score: the dominance score on trials that break
# the pattern, 0 on trials that don't. Breaking a strong habit counts for more
# than breaking a weak one.
BREAKS_X_DOMINANCE_WITH_Q = "breaks_x_dominance_with_q"
BREAKS_X_DOMINANCE_NO_Q = "breaks_x_dominance_no_q"

# A graded version of breaks_pattern: how many single-step edits turn this
# trial's opening into the usual one (0 = identical).
STRATEGY_DISTANCE_WITH_Q = "strategy_distance_with_q"
STRATEGY_DISTANCE_NO_Q = "strategy_distance_no_q"

# How many trials the participant's usual opening was worked out from. Kept on
# every row because a share computed from 12 trials is much less certain than
# one computed from 50, and nothing else in the table would show that.
N_STRATEGY_TRIALS_WITH_Q = "n_strategy_trials_with_q"
N_STRATEGY_TRIALS_NO_Q = "n_strategy_trials_no_q"

# Not used by any code.
SELECTED_DWELL_DURATION = "selected_a_dwell_duration"
SELECTED_SCREEN_LOCATION = "selected_a_screen_loc"

# Used only by the time-course analysis, which splits each trial into before,
# during and after the first look at the answer the participant went on to
# choose: the segment, how many word rows fall in it, and whether a word was
# never looked at.
SEGMENT_COLUMN = "time_segment"
SEGMENT_IA_ROW_COUNT = "segment_ia_row_count"
SKIPPED_COLUMN = "skipped"

# Pupil size averaged over an area's words: the mean of each word's average,
# largest and smallest pupil size during its fixations.
MEAN_AVG_FIX_PUPIL_SIZE = "mean_avg_fix_pupil_size"
MEAN_MAX_FIX_PUPIL_SIZE = "mean_max_fix_pupil_size"
MEAN_MIN_FIX_PUPIL_SIZE = "mean_min_fix_pupil_size"

# The same, z-scored: expressed relative to the participant's own typical pupil
# size, so that people with naturally larger pupils do not dominate.
MEAN_AVG_FIX_PUPIL_SIZE_Z = "mean_avg_fix_pupil_size_z"
MEAN_MAX_FIX_PUPIL_SIZE_Z = "mean_max_fix_pupil_size_z"
MEAN_MIN_FIX_PUPIL_SIZE_Z = "mean_min_fix_pupil_size_z"

# Pupil size on the area's first word BY POSITION that was looked at (averaged
# over that word's fixations), raw and z-scored. Despite the name, this is not
# necessarily the first word looked at in time.
FIRST_ENCOUNTER_AVG_PUPIL_SIZE = "first_encounter_avg_pupil_size"
FIRST_ENCOUNTER_AVG_PUPIL_SIZE_Z = "first_encounter_avg_pupil_size_z"


# ===========================================================================
# 3. Lists and naming rules
# ===========================================================================

ANSWER_PREFIX = "answer_"
ANSWER_LABELS = ["A", "B", "C", "D"]

# The five screen areas, named two ways: by place (LOC_CHOICES) and by meaning
# (LABEL_CHOICES). These are NOT one list in two spellings. The answers are
# shuffled across positions on every trial, so the top answer is not answer_A.
#
# Do not reorder either list: code relies on the order. An answer's position
# number is its index in LOC_CHOICES minus 1, and that number is used to look
# it up in `answers_order`.

LOC_CHOICES = [
    "question",
    "answer_0(top)",
    "answer_1(left)",
    "answer_2(right)",
    "answer_3(bottom)",
]

# DO NOT CHANGE
LABEL_CHOICES = ["question", "answer_A", "answer_B", "answer_C", "answer_D"]

# The question's area ("question" under both namings), and the four answers by
# meaning. answer_A is always the correct answer and B to D the wrong ones.
QUESTION_AREA = LABEL_CHOICES[0]
ANSWER_AREAS = LABEL_CHOICES[1:]
CORRECT_ANSWER_AREA = ANSWER_AREAS[0]
WRONG_ANSWER_AREAS = ANSWER_AREAS[1:]

# How many separate times the participant entered an area, counted by meaning
# or by place.
NUM_LABEL_VISITS = "num_label_visits"
NUM_LOC_VISITS = "num_loc_visits"

# The two columns that together identify one trial. Almost every join uses them.
TRIAL_ID_COLS = (PARTICIPANT_ID, TRIAL_ID)

# --- Comparing the correct answer with the wrong ones -----------------------
#
# How a per-answer measurement becomes one number per trial. For a metric such
# as skip_rate:
#
#   skip_rate__correct            its value on the correct answer (answer_A)
#   skip_rate__wrong_mean         its average over the three wrong answers
#   skip_rate__contrast           correct minus wrong_mean
#   skip_rate__distance_furthest  the largest gap between the correct answer
#                                 and any one wrong answer
#   skip_rate__distance_closest   the smallest such gap
#
# The two distances are absolute values: they say how different the correct
# answer is, not in which direction.

CORRECT_SUFFIX = "correct"
WRONG_MEAN_SUFFIX = "wrong_mean"
CONTRAST_SUFFIX = "contrast"
DISTANCE_FURTHEST_SUFFIX = "distance_furthest"
DISTANCE_CLOSEST_SUFFIX = "distance_closest"

# The separator between a metric and its area or comparison: skip_rate__contrast.
DERIVED_SEP = "__"

# --- The core per-area measurements ------------------------------------------
#
# Each is computed separately for each of the five areas:
#
#   mean_dwell_time               average looking time per word; words never
#                                 looked at count as 0
#   mean_fixations_count          average number of fixations per word; words
#                                 never looked at count as 0
#   mean_first_fixation_duration  average length of the first fixation, over
#                                 the words that were looked at
#   skip_rate                     share of the area's words never looked at
#   area_dwell_proportion         share of the trial's looking time spent on
#                                 this area

MEAN_DWELL_TIME = "mean_dwell_time"
MEAN_FIXATIONS_COUNT = "mean_fixations_count"
MEAN_FIRST_FIXATION_DURATION = "mean_first_fixation_duration"
SKIP_RATE = "skip_rate"
AREA_DWELL_PROPORTION = "area_dwell_proportion"

# How many times the participant pressed to select an answer before confirming.
NUM_OF_SELECTS = "ANSWER_PRESS_NUMBER"

# Time from the answers appearing to the confirming press, in ms.
# `total_answering_RT` is the same number under a clearer name;
# `total_answering_RT_normalized` divides it by the number of words on screen.
CONFIRM_FINAL_ANSWER_RT = "CONFIRM_FINAL_ANSWER_RT"
TOTAL_ANSWERING_RT = "total_answering_RT"
TOTAL_ANSWERING_RT_NORMALIZED = "total_answering_RT_normalized"

# --- Reading times per area (made by features/reading_times.py) --------------
#
# Three measures, each in two versions. Each is one column per area, named
# f"{prefix}_{area}" -- e.g. RT_normalized_answer_A. Note the single "_" before
# the area, where the per-area measurements above use "__".
#
#   RT_*               time spent on visits to the area
#   TFD_*              total fixation duration: the looking time on the area's
#                      words, added up
#   TimeSinceOffset_*  from the start of the first look at the area to the end
#                      of the last look
#   *_pure             in ms
#   *_normalized       divided by the number of words in the area

RT_PURE = "RT_pure"
RT_NORMALIZED = "RT_normalized"
TFD_PURE = "TFD_pure"
TFD_NORMALIZED = "TFD_normalized"
TIME_SINCE_OFFSET_PURE = "TimeSinceOffset_pure"
TIME_SINCE_OFFSET_NORMALIZED = "TimeSinceOffset_normalized"

RT_TFD_METRICS = (
    RT_PURE,
    RT_NORMALIZED,
    TFD_PURE,
    TFD_NORMALIZED,
    TIME_SINCE_OFFSET_PURE,
    TIME_SINCE_OFFSET_NORMALIZED,
)


# The ten per-area measurements, as two lists. The model's list uses the
# z-scored pupil columns; the plots' list uses pupil size in mm, not z-scored.
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


# For each measurement, whether a high or a low value means the participant
# favoured an answer. Not used by any code.
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

# --- Columns of the fixation report (one row per fixation) -----------------
#
# Read through `features/fixations.py`. The start time and duration are in ms.
# The fixation's word is CURRENT_FIX_INTEREST_AREA_ID ("." if it landed on no
# word); NEAREST_IA is the word it landed nearest.

CURRENT_FIX_START = "CURRENT_FIX_START"
CURRENT_FIX_DURATION = "CURRENT_FIX_DURATION"
CURRENT_FIX_INTEREST_AREA_ID = "CURRENT_FIX_INTEREST_AREA_ID"
CURRENT_FIX_PUPIL_SIZE = "CURRENT_FIX_PUPIL"
NEAREST_IA = "CURRENT_FIX_NEAREST_INTEREST_AREA"


# --- Rebuilding the button presses -------------------------------------------
#
# Columns used to work out when the participant selected and confirmed an
# answer, from the tracker's message log. See src/ingest/clicks.py.

RECORDING_SESSION_LABEL = "RECORDING_SESSION_LABEL"
ALL_ANSWERS = "ALL_ANSWERS"

CURRENT_FIX_MSG_LIST_TEXT = "CURRENT_FIX_MSG_LIST_TEXT"
CURRENT_FIX_MSG_LIST_TIME = "CURRENT_FIX_MSG_LIST_TIME"

NEXT_SAC_MSG_LIST_TEXT = "NEXT_SAC_MSG_LIST_TEXT"
NEXT_SAC_MSG_LIST_TIME = "NEXT_SAC_MSG_LIST_TIME"

ALL_ANSWERS_LIST = "ALL_ANSWERS_LIST"
PREV_ALL_ANSWERS_LIST = "PREV_ALL_ANSWERS_LIST"
TRIAL_ANSWERS = "TRIAL_ANSWERS"

# Trials whose answer log cannot be read.
IS_MALFORMED = "IS_MALFORMED"
FIRST_MALFORMED_TRIAL = "FIRST_MALFORMED_TRIAL"

# How the report writes an empty value.
MISSING_LIST_MARKER = "."

# The messages the experiment logs when an answer is selected and confirmed.
CHOOSE_ANSWER_KEYWORD = "CHOOSE_ANSWER"
CONFIRM_ANSWER_KEYWORD = "CONFIRM_ANSWER"

MATCH_TIMESTAMPS_1 = "MATCH_TIMESTAMPS_1"
MATCH_TIMESTAMPS_2 = "MATCH_TIMESTAMPS_2"
MATCH_TIMESTAMPS = "MATCH_TIMESTAMPS"
MATCH_TIMESTAMP = "MATCH_TIMESTAMP"

# Each trial's press times, in ms on the tracker's clock: every selection
# press, and the confirming press.
SELECT_ANS_TIMESTAMPS = "SELECT_ANS_TIMESTAMPS"
CONFIRM_TIMESTAMPS = "CONFIRM_TIMESTAMPS"

# The number of selection presses, counted from the message log.
NUM_OF_SELECTS_DERIVED = "NUM_OF_SELECTS_derived"
