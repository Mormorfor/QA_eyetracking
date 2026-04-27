import sys

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Raw Data — full CSV reports (data_raw/full)
# ---------------------------------------------------------------------------

RAW_FULL_DIR = PROJECT_ROOT / "data_raw" / "full"

IA_ANSWERS_PATH = RAW_FULL_DIR / "ia_Answers.csv"
IA_PARAGRAPH_PATH = RAW_FULL_DIR / "ia_Paragraph.csv"
IA_QA_PATH = RAW_FULL_DIR / "ia_QA.csv"
IA_FEEDBACK_PATH = RAW_FULL_DIR / "ia_Feedback.csv"
IA_TITLE_PATH = RAW_FULL_DIR / "ia_Title.csv"
IA_QUESTION_PREVIEW_PATH = RAW_FULL_DIR / "ia_Question_Preview.csv"
IA_QUESTIONS_PATH = RAW_FULL_DIR / "ia_Questions.csv"

FIX_ANSWERS_PATH = RAW_FULL_DIR / "fixations_Answers.csv"
FIX_PARAGRAPH_PATH = RAW_FULL_DIR / "fixations_Paragraph.csv"
FIX_QA_PATH = RAW_FULL_DIR / "fixations_QA.csv"
FIX_FEEDBACK_PATH = RAW_FULL_DIR / "fixations_Feedback.csv"
FIX_TITLE_PATH = RAW_FULL_DIR / "fixations_Title.csv"
FIX_QUESTION_PREVIEW_PATH = RAW_FULL_DIR / "fixations_Question_Preview.csv"
FIX_QUESTIONS_PATH = RAW_FULL_DIR / "fixations_Questions.csv"

# ---------------------------------------------------------------------------
# Raw Data — TSV reports (data_raw/tsv)
# ---------------------------------------------------------------------------

RAW_TSV_DIR = PROJECT_ROOT / "data_raw" / "tsv"
IA_TSV_DIR = RAW_TSV_DIR / "IA reports"
FIX_TSV_DIR = RAW_TSV_DIR / "Fixations reports"

IA_A_TSV_PATH = IA_TSV_DIR / "ia_A.tsv"
IA_F_TSV_PATH = IA_TSV_DIR / "ia_F.tsv"
IA_P_TSV_PATH = IA_TSV_DIR / "ia_P.tsv"
IA_Q_TSV_PATH = IA_TSV_DIR / "ia_Q.tsv"
IA_QA_TSV_PATH = IA_TSV_DIR / "ia_QA.tsv"
IA_Q_PREVIEW_TSV_PATH = IA_TSV_DIR / "ia_Q_preview.tsv"
IA_T_TSV_PATH = IA_TSV_DIR / "ia_T.tsv"

FIX_A_TSV_PATH = FIX_TSV_DIR / "fixations_A.tsv"
FIX_F_TSV_PATH = FIX_TSV_DIR / "fixations_F.tsv"
FIX_P_TSV_PATH = FIX_TSV_DIR / "fixations_P.tsv"
FIX_Q_TSV_PATH = FIX_TSV_DIR / "fixations_Q.tsv"
FIX_QA_TSV_PATH = FIX_TSV_DIR / "fixations_QA.tsv"
FIX_Q_PREVIEW_TSV_PATH = FIX_TSV_DIR / "fixations_Q_preview.tsv"
FIX_T_TSV_PATH = FIX_TSV_DIR / "fixations_T.tsv"

# ---------------------------------------------------------------------------
# Processed Data (data/)
# ---------------------------------------------------------------------------

DATA_DIR = PROJECT_ROOT / "data"

HUNTERS_PROCESSED_PATH = DATA_DIR / "hunters.csv"
GATHERERS_PROCESSED_PATH = DATA_DIR / "gatherers.csv"

HUNTERS_LAST_PATH = DATA_DIR / "hunters_last.csv"
GATHERERS_LAST_PATH = DATA_DIR / "gatherers_last.csv"

HUNT_PARAGRAPH_AND_ANSWERS = DATA_DIR / "hunters_paragraph_answer_merge.csv"
GATH_PARAGRAPH_AND_ANSWERS = DATA_DIR / "gatherers_paragraph_answer_merge.csv"

PARTICIPANT_PUPILS_PATH = DATA_DIR / "participant_pupils.csv"
BUTTON_CLICKS_PATH = DATA_DIR / "button_clicks_data.csv"
STRANGE_TRIALS_PATH = DATA_DIR / "strange_trials.csv"

# ---------------------------------------------------------------------------
# Cross-validation folds (data/*Folds)
# ---------------------------------------------------------------------------

HUNTERS_FOLDS_DIR = DATA_DIR / "HuntersFolds"
GATHERERS_FOLDS_DIR = DATA_DIR / "GatherersFolds"
GATHERERS_REFOLDED_DIR = DATA_DIR / "GatherersRefolded"

HUNTING_IS_CORRECT_FOLDS_DIR = DATA_DIR / "HuntingIsCorrectFolds"
HUNTING_IS_CORRECT_ALL_FOLDS_PATH = (
    HUNTING_IS_CORRECT_FOLDS_DIR / "all_folds_subjects_items.csv"
)
HUNTING_IS_CORRECT_ITEMS_DIR = HUNTING_IS_CORRECT_FOLDS_DIR / "items"
HUNTING_IS_CORRECT_SUBJECTS_DIR = HUNTING_IS_CORRECT_FOLDS_DIR / "subjects"

FOLD_TRIAL_IDS_FILENAME_TEMPLATE = "fold_{fold_idx}_trial_ids_by_regime.csv"


# ---------------------------------------------------------------------------
# Feature-columns generation
# ---------------------------------------------------------------------------

COL_SAVE_PATH = PROJECT_ROOT / "reports" / "report_data" / "answer_correctness" / "feature_columns"
