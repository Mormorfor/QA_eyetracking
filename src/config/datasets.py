"""Where every dataset lives on disk.

Locations only -- no logic, no output destinations. Report destinations are in
`config/outputs.py`, because mixing the two is what let `reports/` paths get
hardcoded at call sites (`todo.md` T5.3).

This module used to insert PROJECT_ROOT into `sys.path` as an import side
effect. It no longer does: the repo root is put on the path by `.env` (or the
`sys.path` cell at the top of a driver notebook), and a module of path constants
should not be able to reconfigure where Python looks for everything else.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Literal, Mapping

# src/config/datasets.py -> repo root is three levels up.
PROJECT_ROOT = Path(__file__).resolve().parents[2]


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


TEST_RUN_PATH = PROJECT_ROOT / "data_raw" / "testrun_QA"

# ---------------------------------------------------------------------------
# New experiment (testrun_QA) — converted/cleaned raw inputs
# ---------------------------------------------------------------------------

# CSVs converted from the .xls reports (via data_prep_new_exp.ipynb).
NEW_EXP_CSV_DIR = TEST_RUN_PATH / "csvs"
# Column-standardized CSVs the pipeline reads (renamed cols, letter answers_order).
NEW_EXP_CLEANED_DIR = NEW_EXP_CSV_DIR / "cleaned"

NEW_EXP_IA_ANSWERS_PATH = NEW_EXP_CLEANED_DIR / "IA_answers.csv"
NEW_EXP_FIX_ANSWERS_PATH = NEW_EXP_CLEANED_DIR / "fixations_answers.csv"
NEW_EXP_MESSAGES_PATH = NEW_EXP_CSV_DIR / "messages_answers.csv"

# ---------------------------------------------------------------------------
# New experiment, second run (second_test) — raw .tsv reports
# ---------------------------------------------------------------------------

# Same report layout as testrun_QA, but exported as UTF-8 .tsv instead of
# UTF-16 .xls, under report-prefixed names (A_IA.tsv / A_fixations.tsv), and
# with no messages report. data_prep_new_exp.ipynb converts them to the same
# csvs/ + csvs/cleaned/ layout as the testrun_QA data.
SECOND_TEST_PATH = PROJECT_ROOT / "data_raw" / "second_test"

SECOND_TEST_CSV_DIR = SECOND_TEST_PATH / "csvs"
SECOND_TEST_CLEANED_DIR = SECOND_TEST_CSV_DIR / "cleaned"

SECOND_TEST_IA_ANSWERS_PATH = SECOND_TEST_CLEANED_DIR / "IA_answers.csv"
SECOND_TEST_FIX_ANSWERS_PATH = SECOND_TEST_CLEANED_DIR / "fixations_answers.csv"

# ---------------------------------------------------------------------------
# KnowQA — first real data collection (data_raw/KnowQA)
# ---------------------------------------------------------------------------

# The first non-test run of the experiment. Same report columns as the two test
# runs, exported like testrun_QA (UTF-16 .xls, tab separated) under canonical
# names (IA_Answers.xls / fixations_Answers.xls) and with no messages report.
# data_prep_new_exp.ipynb converts them to the same csvs/ + csvs/cleaned/ layout.
KNOW_QA_PATH = PROJECT_ROOT / "data_raw" / "KnowQA"

KNOW_QA_CSV_DIR = KNOW_QA_PATH / "csvs"
KNOW_QA_CLEANED_DIR = KNOW_QA_CSV_DIR / "cleaned"

KNOW_QA_IA_ANSWERS_PATH = KNOW_QA_CLEANED_DIR / "IA_answers.csv"
KNOW_QA_FIX_ANSWERS_PATH = KNOW_QA_CLEANED_DIR / "fixations_answers.csv"

# ---------------------------------------------------------------------------
# Processed Data (data/)
# ---------------------------------------------------------------------------

DATA_DIR = PROJECT_ROOT / "data"

# ---------------------------------------------------------------------------
# The dataset registry
#
# One record per dataset, and every per-dataset path derived from it. Before
# this, the same five concepts appeared four times under four prefixes -- none
# for L1, `NEW_EXP_`, `SECOND_TEST_`, `KNOW_QA_` -- so adding a study meant
# writing a dozen constants and threading them through call sites, and a
# hand-maintained constant could go stale without anything noticing.
#
# **Adding a dataset is now one Dataset(...) entry.**
#
# **The layout is now the target in `restructure-map.md` section 8** (stage G,
# 2026-10-07). The moves happened, and they cost exactly what this record promised:
# four `root=` lines and three properties, instead of ninety constants. 132 files
# and 10.7 GB moved with the file count and byte total verified identical before
# and after.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Dataset:
    """Where one dataset lives, and what shape it is.

    `kind` separates the two real collections from the two pilots, so code that
    loops over datasets can say which it means rather than treating all four
    alike -- the distinction a folder name alone cannot carry.
    """

    key: str
    label: str
    kind: Literal["study", "pilot"]
    study: Literal[1, 2]
    raw_dir: Path
    root: Path
    has_paragraph: bool
    #: Fixation report the per-participant pupil baseline is computed from.
    #: Explicit per dataset: defaulting it is what made paragraph-span pupil
    #: z-scores baseline against L1's answer screen regardless of dataset
    #: (`todo.md` T3.20).
    pupil_baseline_fixations: Path
    #: Base features this dataset does NOT run, name -> why.
    #:
    #: The default is to run every registered base feature, and a dataset
    #: declares only its *difference* from that. Declaring the whole list instead
    #: would mean editing all four entries every time a feature is added to
    #: FUNCTION_REGISTRY -- the per-dataset repetition this record exists to
    #: remove.
    #:
    #: Names, not functions, because `config/` must not import `features/`. They
    #: are checked against the registry where they are used, so a typo fails
    #: loudly at run time rather than silently skipping nothing.
    #:
    #: Note what does NOT belong here: `add_zscored_pupil_columns` is skipped by
    #: KnowQA only when `pupil_norm_unit="session"`, which is a property of how
    #: you chose to run, not of the dataset. That one stays conditional in the
    #: runner.
    skip_base_features: Mapping[str, str] = field(default_factory=dict)

    # -- processed outputs, all derived ------------------------------------
    @property
    def interim(self) -> Path:
        """Everything between the raw reports and the model-ready table."""
        return self.root / "interim"

    @property
    def features(self) -> Path:
        """Built feature tables -- what modelling and the analyses read."""
        return self.root / "features"

    @property
    def aux(self) -> Path:
        """Intermediates: button clicks, pupil stats, RT/TFD, last-area labels."""
        return self.interim / "aux"

    @property
    def all_participants(self) -> Path:
        """IA-level table -- one row per word, per area, per trial."""
        return self.interim / "all_participants.csv"

    @property
    def model_ready(self) -> Path:
        """Trial-level table the models consume.

        Was `L1_model_ready_all_features.csv` in every dataset, so three of the
        four claimed to be L1 data -- `restructure-map.md` section 8 calls that the
        move that mattered most, a correctness hazard rather than tidiness. The
        file now says what it is and the folder says whose it is.
        """
        return self.features / "model_ready.csv"

    @property
    def participant_pupils(self) -> Path:
        return self.aux / "participant_pupils.csv"

    @property
    def button_clicks(self) -> Path:
        return self.aux / "button_clicks_data.csv"


L1 = Dataset(
    key="l1",
    label="OneStop L1 (native speakers)",
    kind="study",
    study=1,
    raw_dir=RAW_FULL_DIR,
    root=DATA_DIR / "datasets" / "l1_onestop",
    has_paragraph=True,
    pupil_baseline_fixations=FIX_ANSWERS_PATH,
)

KNOWQA = Dataset(
    key="knowqa",
    label="KnowQA (knowledge regimes)",
    kind="study",
    study=2,
    raw_dir=KNOW_QA_PATH,
    root=DATA_DIR / "datasets" / "knowqa",
    # Only the answer screen is exported, and only a third of trials show a
    # paragraph at all -- `pitfalls.md` section 5.
    has_paragraph=False,
    pupil_baseline_fixations=KNOW_QA_FIX_ANSWERS_PATH,
    skip_base_features={
        "add_answer_text_columns": (
            "answer_A..D are supplied directly by the Stage 0 column rename, "
            "so there is nothing to recompute from the stimulus text."
        )
    },
)

TESTRUN_QA = Dataset(
    key="testrun_qa",
    label="KnowQA pilot 1 (testrun_QA)",
    kind="pilot",
    study=2,
    raw_dir=TEST_RUN_PATH,
    # The folder is named new_exp_try_runs; it is the testrun_QA data
    # (confirmed 2026-09-05). Renaming it is move 5 in section 8.
    root=DATA_DIR / "datasets" / "pilots" / "testrun_qa",
    has_paragraph=False,
    pupil_baseline_fixations=NEW_EXP_FIX_ANSWERS_PATH,
    skip_base_features={
        "add_answer_text_columns": (
            "answer_A..D are supplied directly by the Stage 0 column rename, "
            "so there is nothing to recompute from the stimulus text."
        )
    },
)

SECOND_TEST = Dataset(
    key="second_test",
    label="KnowQA pilot 2 (second_test)",
    kind="pilot",
    study=2,
    raw_dir=SECOND_TEST_PATH,
    root=DATA_DIR / "datasets" / "pilots" / "second_test",
    has_paragraph=False,
    pupil_baseline_fixations=SECOND_TEST_FIX_ANSWERS_PATH,
    skip_base_features={
        "add_answer_text_columns": (
            "answer_A..D are supplied directly by the Stage 0 column rename, "
            "so there is nothing to recompute from the stimulus text."
        )
    },
)

DATASETS: Dict[str, Dataset] = {d.key: d for d in (L1, KNOWQA, TESTRUN_QA, SECOND_TEST)}


def dataset(key: str) -> Dataset:
    """Look a dataset up by key, failing loudly on a typo."""
    try:
        return DATASETS[key]
    except KeyError:
        raise KeyError(
            f"unknown dataset {key!r}; expected one of {sorted(DATASETS)}"
        ) from None


def dataset_for_root(root) -> Dataset:
    """The dataset whose processed outputs live at `root`.

    Lets a pipeline step that was handed an output directory recover which
    dataset it is working on, and so read that dataset's configuration instead
    of having it passed down or hardcoded.
    """
    from pathlib import Path as _Path

    root = _Path(root).resolve()
    for d in DATASETS.values():
        if d.root.resolve() == root:
            return d
    raise KeyError(
        f"no dataset registered with root {root}; known roots: "
        f"{ {d.key: str(d.root) for d in DATASETS.values()} }"
    )


def studies() -> List[Dataset]:
    """The two real collections -- the default for anything that loops."""
    return [d for d in DATASETS.values() if d.kind == "study"]


def pilots() -> List[Dataset]:
    """The two pilots. Runnable, but you have to ask for them."""
    return [d for d in DATASETS.values() if d.kind == "pilot"]


# Final IA-level data produced by the data_csv_generation pipeline.
# The flat names below are DERIVED from the registry above -- one source of
# truth, so a dataset move changes the Dataset entry and nothing else. The
# names are kept because ~27 modules import them; migrating those call sites
# to `dataset("l1").model_ready` is left for the stage that moves the files.
L1_BASED_DATA_DIR = L1.root
# Intermediate artifacts generated on the way by that pipeline (participant
# pupil stats, button clicks, RT/TFD, last-area labels, ...).
AUXILIARY_DATA_DIR = L1.aux

HUNTERS_PROCESSED_PATH = L1.interim / "hunters.csv"
GATHERERS_PROCESSED_PATH = L1.interim / "gatherers.csv"
ALL_PARTICIPANTS_PROCESSED_PATH = L1.all_participants

ALL_PARTICIPANTS_LAST_PATH = AUXILIARY_DATA_DIR / "all_participants_last.csv"
HUNTERS_LAST_PATH = AUXILIARY_DATA_DIR / "hunters_last.csv"
GATHERERS_LAST_PATH = AUXILIARY_DATA_DIR / "gatherers_last.csv"

HUNT_PARAGRAPH_AND_ANSWERS = L1.interim / "hunters_paragraph_answer_merge.csv"
GATH_PARAGRAPH_AND_ANSWERS = L1.interim / "gatherers_paragraph_answer_merge.csv"

PARTICIPANT_PUPILS_PATH = L1.participant_pupils
BUTTON_CLICKS_PATH = L1.button_clicks
STRANGE_TRIALS_PATH = L1.interim / "strange_trials.csv"
RT_AND_TFD_PATH = AUXILIARY_DATA_DIR / "RT_and_TFD.csv"



READY_ALL_FEATURES_PATH = L1.model_ready

# Paragraph-span (critical / distractor / outside) area features, mirroring the
# per-answer area features but computed over the paragraph-reading screen.
# Used as predictors for the answer reading-time regression (answer_RTs).
PARAGRAPH_SPAN_FEATURES_PATH = L1.features / "paragraph_spans.csv"

# Trial-level answer/question text sizes (word and character counts per answer
# label). Stimulus properties, not eye-tracking measures; used as predictors for
# the answer reading-time regression alongside the paragraph features.
ANSWER_TEXT_FEATURES_PATH = L1.features / "answer_text.csv"

# Trial-level paragraph reading features produced by the lab's external feature
# extraction code (src/vendor/eyebench/paragraph_trial_features.py): aggregated IA and
# fixation measures, gaze entropy, and per-word-category scanpath features, one
# row per (participant, trial). The folder also holds the two feature-key CSVs
# that map each feature name to the model family it came from.
# Named for the source pipeline, to keep these apart from our own paragraph-span
# features above.
PARAGRAPH_TRIAL_FEATURES_DIR = L1.features / "eyebench"
PARAGRAPH_TRIAL_FEATURES_PATH = (
    PARAGRAPH_TRIAL_FEATURES_DIR / "L1_paragraph_trial_level_features.csv"
)

# ---------------------------------------------------------------------------
# New experiment (testrun_QA) — processed outputs (data/new_exp_try_runs)
# ---------------------------------------------------------------------------

NEW_EXP_OUT_DIR = TESTRUN_QA.root
NEW_EXP_OUT_PATH = TESTRUN_QA.all_participants
NEW_EXP_AUX_DIR = TESTRUN_QA.aux
NEW_EXP_FEATURES_PATH = TESTRUN_QA.model_ready

# ---------------------------------------------------------------------------
# New experiment, second run (second_test) — processed outputs
# ---------------------------------------------------------------------------

SECOND_TEST_OUT_DIR = SECOND_TEST.root
SECOND_TEST_OUT_PATH = SECOND_TEST.all_participants
SECOND_TEST_AUX_DIR = SECOND_TEST.aux
SECOND_TEST_FEATURES_PATH = SECOND_TEST.model_ready

# ---------------------------------------------------------------------------
# KnowQA — processed outputs (data/KnowQA_runs)
# ---------------------------------------------------------------------------

KNOW_QA_OUT_DIR = KNOWQA.root
KNOW_QA_OUT_PATH = KNOWQA.all_participants
KNOW_QA_AUX_DIR = KNOWQA.aux
KNOW_QA_FEATURES_PATH = KNOWQA.model_ready

# ---------------------------------------------------------------------------
# Experiment data (data/Experiment)
# ---------------------------------------------------------------------------

EXPERIMENT_DIR = DATA_DIR / "stimuli"

N1_BASE_PATH = EXPERIMENT_DIR / "onestop_list_bases" / "n1_base.csv"
N2_BASE_PATH = EXPERIMENT_DIR / "onestop_list_bases" / "n2_base.csv"
N3_BASE_PATH = EXPERIMENT_DIR / "onestop_list_bases" / "n3_base.csv"

EXPERIMENT_TEXT_COMPLETED_PATH = EXPERIMENT_DIR / "experiment_text_completed.zip"
TEXTS_MANUAL_REPHRASINGS_PATH = EXPERIMENT_DIR / "texts_manual_rephrasings.csv"
TEXTS_NO_REPHRASALS_PATH = EXPERIMENT_DIR / "texts_no_rephrasals.csv"

# ---------------------------------------------------------------------------
# Cross-validation folds (data/cv_folds/*Folds)
# ---------------------------------------------------------------------------

CV_FOLDS_DIR = DATA_DIR / "cv_folds"

HUNTERS_FOLDS_DIR = CV_FOLDS_DIR / "HuntersFolds"
GATHERERS_FOLDS_DIR = CV_FOLDS_DIR / "GatherersFolds"
GATHERERS_REFOLDED_DIR = CV_FOLDS_DIR / "GatherersRefolded"

HUNTING_IS_CORRECT_FOLDS_DIR = CV_FOLDS_DIR / "HuntingIsCorrectFolds"
HUNTING_IS_CORRECT_ALL_FOLDS_PATH = (
    HUNTING_IS_CORRECT_FOLDS_DIR / "all_folds_subjects_items.csv"
)
HUNTING_IS_CORRECT_ITEMS_DIR = HUNTING_IS_CORRECT_FOLDS_DIR / "items"
HUNTING_IS_CORRECT_SUBJECTS_DIR = HUNTING_IS_CORRECT_FOLDS_DIR / "subjects"

FOLD_TRIAL_IDS_FILENAME_TEMPLATE = "fold_{fold_idx}_trial_ids_by_regime.csv"
