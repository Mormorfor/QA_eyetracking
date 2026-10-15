"""Where every dataset's files are on disk.

There are four datasets, each described once by a `Dataset` record further
down: Study 1 (OneStop L1), Study 2 (KnowQA), and two Study 2 pilots. Every
per-dataset path is built from its record, so moving a dataset means changing
one `root=` line.

This file holds input and data locations only. Where results are written is in
`config/outputs.py`.

The raw reports at the top are fixed paths into `data_raw/`. `data_raw/full`
and `data_raw/tsv` are links to the OneStop corpus, which is stored outside the
repo because it is not ours to redistribute.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Literal, Mapping

# src/config/datasets.py -> repo root is three levels up.
PROJECT_ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# Study 1 raw reports, CSV (data_raw/full)
#
# The eye tracker exports two kinds of report per screen: an interest-area
# report (one row per word) and a fixation report (one row per fixation).
# Only the Answers, Paragraph and QA screens are read by the code.
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
# Study 1 raw reports, TSV (data_raw/tsv)
#
# The same reports in another export format. Only `fixations_A.tsv` is read:
# it carries the message log that the button-press reconstruction needs.
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
# Study 2, first pilot (data_raw/testrun_QA)
#
# The tracker's .xls exports are first converted to CSV (csvs/), then cleaned
# into the column names the pipeline expects (csvs/cleaned/). For both pilots
# that is done by notebooks/builders/data_prep_new_exp.ipynb.
# ---------------------------------------------------------------------------

NEW_EXP_CSV_DIR = TEST_RUN_PATH / "csvs"
NEW_EXP_CLEANED_DIR = NEW_EXP_CSV_DIR / "cleaned"

NEW_EXP_IA_ANSWERS_PATH = NEW_EXP_CLEANED_DIR / "IA_answers.csv"
NEW_EXP_FIX_ANSWERS_PATH = NEW_EXP_CLEANED_DIR / "fixations_answers.csv"
NEW_EXP_MESSAGES_PATH = NEW_EXP_CSV_DIR / "messages_answers.csv"

# ---------------------------------------------------------------------------
# Study 2, second pilot (data_raw/second_test)
#
# Same reports as the first pilot, exported as UTF-8 .tsv files named
# A_IA.tsv / A_fixations.tsv, and with no messages report.
# ---------------------------------------------------------------------------

SECOND_TEST_PATH = PROJECT_ROOT / "data_raw" / "second_test"

SECOND_TEST_CSV_DIR = SECOND_TEST_PATH / "csvs"
SECOND_TEST_CLEANED_DIR = SECOND_TEST_CSV_DIR / "cleaned"

SECOND_TEST_IA_ANSWERS_PATH = SECOND_TEST_CLEANED_DIR / "IA_answers.csv"
SECOND_TEST_FIX_ANSWERS_PATH = SECOND_TEST_CLEANED_DIR / "fixations_answers.csv"

# ---------------------------------------------------------------------------
# Study 2, the real collection (data_raw/KnowQA)
#
# Exported like the first pilot (UTF-16 .xls), with no messages report.
# Converted and cleaned by src/ingest/knowqa.py.
# ---------------------------------------------------------------------------

KNOW_QA_PATH = PROJECT_ROOT / "data_raw" / "KnowQA"

KNOW_QA_CSV_DIR = KNOW_QA_PATH / "csvs"
KNOW_QA_CLEANED_DIR = KNOW_QA_CSV_DIR / "cleaned"

KNOW_QA_IA_ANSWERS_PATH = KNOW_QA_CLEANED_DIR / "IA_answers.csv"
KNOW_QA_FIX_ANSWERS_PATH = KNOW_QA_CLEANED_DIR / "fixations_answers.csv"

# ---------------------------------------------------------------------------
# Processed data (data/)
# ---------------------------------------------------------------------------

DATA_DIR = PROJECT_ROOT / "data"

# ---------------------------------------------------------------------------
# The dataset records
#
# One `Dataset` per dataset. Every processed-data path for a dataset is worked
# out from its `root`, so to add a dataset you add one record, and to move one
# you change its `root`.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Dataset:
    """Where one dataset lives and how it differs from the others.

    key            short name used in code, e.g. "l1"
    label          readable name for printed output
    kind           "study" for the two real collections, "pilot" for the pilots,
                   so code that loops over datasets can choose which it means
    study          1 or 2
    raw_dir        the raw reports
    root           where its processed tables go
    has_paragraph  whether its trials include a paragraph-reading screen
    skip_base_features
                   per-word features this dataset does not compute, and why
    """

    key: str
    label: str
    kind: Literal["study", "pilot"]
    study: Literal[1, 2]
    raw_dir: Path
    root: Path
    has_paragraph: bool
    #: Per-word features this dataset does NOT compute: name -> reason.
    #:
    #: Every registered per-word feature runs by default, so a dataset only
    #: lists its exceptions. Listing everything instead would mean editing all
    #: four records whenever a feature is added.
    #:
    #: These are names rather than functions because `config/` must not import
    #: `features/`. They are checked against the registry when the pipeline
    #: runs, so a misspelt name stops the run instead of silently skipping
    #: nothing.
    skip_base_features: Mapping[str, str] = field(default_factory=dict)

    # -- processed outputs, all derived from `root` --------------------------
    @property
    def interim(self) -> Path:
        """Tables between the raw reports and the final model table."""
        return self.root / "interim"

    @property
    def features(self) -> Path:
        """Finished feature tables -- what the models and analyses read."""
        return self.root / "features"

    @property
    def aux(self) -> Path:
        """Side products of a pipeline run, rebuilt every time it runs.

        Button presses, pupil baselines, reading times, last-looked-at labels.
        """
        return self.interim / "aux"

    @property
    def all_participants(self) -> Path:
        """One wide table: one row per word, with every area and trial value
        copied onto each word.

        Only written when `ingest/build.py::main` is called with
        `write_broadcast_table=True`. Normally the pipeline writes the three
        separate tables below instead.
        """
        return self.interim / "all_participants.csv"

    @property
    def model_ready(self) -> Path:
        """The final table: one row per trial, with every feature a model can use."""
        return self.features / "model_ready.csv"

    # -- the three tables, one per level of detail ------------------------
    #
    # The pipeline writes its results at three levels: one row per word, one
    # row per screen area, one row per trial. Each file is named after its
    # level. A group's subset adds `__<group>` to the name, e.g.
    # `trials__hunters.csv`.

    @property
    def interest_areas(self) -> Path:
        """One row per word, per trial. Holds only what changes from word to word."""
        return self.interim / "interest_areas.csv"

    @property
    def areas_answers(self) -> Path:
        """One row per trial per answer-screen area: the per-area measurements."""
        return self.interim / "areas_answers.csv"

    @property
    def areas_paragraph(self) -> Path:
        """One row per trial per paragraph part (critical, distractor, outside).

        Declared here, but no code writes this file: the paragraph screen's
        results are saved one row per trial, in `features/paragraph_spans.csv`
        (`PARAGRAPH_SPAN_FEATURES_PATH` below).
        """
        return self.interim / "areas_paragraph.csv"

    @property
    def trials(self) -> Path:
        """One row per trial: fixation sequences, timings, identity."""
        return self.interim / "trials.csv"

    def model_ready_split(self, facet: str) -> Path:
        """The model-ready table for one group, e.g. `model_ready_hunters.csv`."""
        base = self.model_ready
        return base.with_name(f"{base.stem}_{facet}{base.suffix}")

    def grain_split(self, grain: str, facet: str) -> Path:
        """One group's subset of a table, e.g. `trials__hunters.csv`.

        The groups come from the dataset's split column: hunters and gatherers
        for Study 1. Study 2 has no such split. Its regimes are not groups of
        people -- every participant does all three -- so splitting by regime
        would cut each person's trials into thirds rather than separate people.
        """
        base = getattr(self, grain)
        return base.with_name(f"{base.stem}__{facet}{base.suffix}")

    # Each person's typical pupil size, worked out separately for the answer
    # screen and the paragraph screen. The two screens differ in brightness and
    # in task, so the same person's baseline differs between them. Each is
    # computed per participant from this dataset's own fixations on that screen.
    @property
    def pupil_baseline_answers(self) -> Path:
        return self.aux / "pupil_baseline__answers.csv"

    @property
    def pupil_baseline_paragraph(self) -> Path:
        return self.aux / "pupil_baseline__paragraph.csv"

    @property
    def button_clicks(self) -> Path:
        """When each answer was selected and confirmed, per trial."""
        return self.aux / "button_clicks_data.csv"


L1 = Dataset(
    key="l1",
    label="OneStop L1 (native speakers)",
    kind="study",
    study=1,
    raw_dir=RAW_FULL_DIR,
    root=DATA_DIR / "datasets" / "l1_onestop",
    has_paragraph=True,
)

KNOWQA = Dataset(
    key="knowqa",
    label="KnowQA (knowledge regimes)",
    kind="study",
    study=2,
    raw_dir=KNOW_QA_PATH,
    root=DATA_DIR / "datasets" / "knowqa",
    # Only the answer screen was exported for Study 2, and only a third of its
    # trials show a paragraph at all.
    has_paragraph=False,
    skip_base_features={
        "add_answer_text_columns": (
            "The cleaned reports already hold the answer text by letter "
            "(answer_A..answer_D), so there is nothing to rebuild."
        )
    },
)

TESTRUN_QA = Dataset(
    key="testrun_qa",
    label="KnowQA pilot 1 (testrun_QA)",
    kind="pilot",
    study=2,
    raw_dir=TEST_RUN_PATH,
    root=DATA_DIR / "datasets" / "pilots" / "testrun_qa",
    has_paragraph=False,
    skip_base_features={
        "add_answer_text_columns": (
            "The cleaned reports already hold the answer text by letter "
            "(answer_A..answer_D), so there is nothing to rebuild."
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
    skip_base_features={
        "add_answer_text_columns": (
            "The cleaned reports already hold the answer text by letter "
            "(answer_A..answer_D), so there is nothing to rebuild."
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
    """The dataset whose processed files live at `root`.

    Lets code that was handed an output folder find out which dataset it is
    working on, and read that dataset's settings.
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
    """The two real collections. The default for anything that loops."""
    return [d for d in DATASETS.values() if d.kind == "study"]


def pilots() -> List[Dataset]:
    """The two pilots. They run like any other dataset, but you have to ask."""
    return [d for d in DATASETS.values() if d.kind == "pilot"]


# ---------------------------------------------------------------------------
# Fixed paths, for code that names a file directly
#
# All built from the dataset records above, so they move with them.
#
# `hunters.csv`, `gatherers.csv` and `all_participants.csv` are only written on
# request (see `Dataset.all_participants`), so on a normal build these three
# paths point at files that do not exist.
# ---------------------------------------------------------------------------

L1_BASED_DATA_DIR = L1.root
AUXILIARY_DATA_DIR = L1.aux

HUNTERS_PROCESSED_PATH = L1.interim / "hunters.csv"
GATHERERS_PROCESSED_PATH = L1.interim / "gatherers.csv"
ALL_PARTICIPANTS_PROCESSED_PATH = L1.all_participants

ALL_PARTICIPANTS_LAST_PATH = AUXILIARY_DATA_DIR / "all_participants_last.csv"
HUNTERS_LAST_PATH = AUXILIARY_DATA_DIR / "hunters_last.csv"
GATHERERS_LAST_PATH = AUXILIARY_DATA_DIR / "gatherers_last.csv"

HUNT_PARAGRAPH_AND_ANSWERS = L1.interim / "hunters_paragraph_answer_merge.csv"
GATH_PARAGRAPH_AND_ANSWERS = L1.interim / "gatherers_paragraph_answer_merge.csv"

PUPIL_BASELINE_ANSWERS_PATH = L1.pupil_baseline_answers
PUPIL_BASELINE_PARAGRAPH_PATH = L1.pupil_baseline_paragraph
BUTTON_CLICKS_PATH = L1.button_clicks
STRANGE_TRIALS_PATH = L1.interim / "strange_trials.csv"
RT_AND_TFD_PATH = AUXILIARY_DATA_DIR / "RT_and_TFD.csv"



READY_ALL_FEATURES_PATH = L1.model_ready

# Per-trial measurements over the three parts of the paragraph (critical,
# distractor, outside): the paragraph-screen version of the per-answer
# measurements.
PARAGRAPH_SPAN_FEATURES_PATH = L1.features / "paragraph_spans.csv"

# Per-trial word and character counts of the question and each answer.
# Properties of the text, not eye-movement measures.
ANSWER_TEXT_FEATURES_PATH = L1.features / "answer_text.csv"

# Per-trial paragraph-reading features made by the lab's EyeBench code
# (src/vendor/eyebench/), one row per trial. The folder also holds two key files
# that say which model family each feature comes from. Kept in their own folder
# so they are never confused with our own paragraph measurements above.
PARAGRAPH_TRIAL_FEATURES_DIR = L1.features / "eyebench"
PARAGRAPH_TRIAL_FEATURES_PATH = (
    PARAGRAPH_TRIAL_FEATURES_DIR / "L1_paragraph_trial_level_features.csv"
)

# ---------------------------------------------------------------------------
# Study 2, first pilot -- processed (data/datasets/pilots/testrun_qa)
# ---------------------------------------------------------------------------

NEW_EXP_OUT_DIR = TESTRUN_QA.root
NEW_EXP_OUT_PATH = TESTRUN_QA.all_participants
NEW_EXP_AUX_DIR = TESTRUN_QA.aux
NEW_EXP_FEATURES_PATH = TESTRUN_QA.model_ready

# ---------------------------------------------------------------------------
# Study 2, second pilot -- processed (data/datasets/pilots/second_test)
# ---------------------------------------------------------------------------

SECOND_TEST_OUT_DIR = SECOND_TEST.root
SECOND_TEST_OUT_PATH = SECOND_TEST.all_participants
SECOND_TEST_AUX_DIR = SECOND_TEST.aux
SECOND_TEST_FEATURES_PATH = SECOND_TEST.model_ready

# ---------------------------------------------------------------------------
# Study 2, the real collection -- processed (data/datasets/knowqa)
# ---------------------------------------------------------------------------

KNOW_QA_OUT_DIR = KNOWQA.root
KNOW_QA_OUT_PATH = KNOWQA.all_participants
KNOW_QA_AUX_DIR = KNOWQA.aux
KNOW_QA_FEATURES_PATH = KNOWQA.model_ready

# ---------------------------------------------------------------------------
# Study 2 stimulus materials (data/stimuli)
# ---------------------------------------------------------------------------

EXPERIMENT_DIR = DATA_DIR / "stimuli"

N1_BASE_PATH = EXPERIMENT_DIR / "onestop_list_bases" / "n1_base.csv"
N2_BASE_PATH = EXPERIMENT_DIR / "onestop_list_bases" / "n2_base.csv"
N3_BASE_PATH = EXPERIMENT_DIR / "onestop_list_bases" / "n3_base.csv"

EXPERIMENT_TEXT_COMPLETED_PATH = EXPERIMENT_DIR / "experiment_text_completed.zip"
TEXTS_MANUAL_REPHRASINGS_PATH = EXPERIMENT_DIR / "texts_manual_rephrasings.csv"
TEXTS_NO_REPHRASALS_PATH = EXPERIMENT_DIR / "texts_no_rephrasals.csv"

# ---------------------------------------------------------------------------
# Cross-validation folds (data/cv_folds)
#
# Precomputed splits of the trials into training and test sets.
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


# ---------------------------------------------------------------------------
# Checking the records against the disk
#
# If a record points at the wrong place, every path built from it is wrong too,
# and the first sign is usually a "file not found" from deep inside pandas.
# These two functions check up front instead.
#
# They are not run automatically on import. A dataset that has not been built
# yet is a normal state, and failing on import would make the package unusable
# on a fresh copy of the repo.
# ---------------------------------------------------------------------------


def check_dataset_paths(keys=None, built_only: bool = True) -> dict:
    """List, per dataset, the expected files that are missing.

    `built_only=True` skips datasets whose folder does not exist at all: not
    built yet is fine, built in the wrong place is not.

    Returns `{dataset_key: [missing names]}`. An empty result means every built
    dataset has all its files where its record says.
    """
    # `all_participants` is left out on purpose: it is only written on request
    # (see `Dataset.all_participants`), so its absence is normal.
    artifacts = (
        "interim",
        "features",
        "aux",
        "interest_areas",
        "areas_answers",
        "trials",
        "model_ready",
    )
    out = {}
    for key in keys or DATASETS:
        ds = DATASETS[key]
        if built_only and not ds.root.exists():
            continue
        missing = [name for name in artifacts if not getattr(ds, name).exists()]
        if missing:
            out[key] = missing
    return out


def assert_dataset_paths(keys=None, built_only: bool = True) -> None:
    """Like `check_dataset_paths`, but stop with an error naming what is missing."""
    bad = check_dataset_paths(keys, built_only=built_only)
    if bad:
        lines = [
            f"  {key}: {', '.join(names)}  (root: {DATASETS[key].root})"
            for key, names in bad.items()
        ]
        raise FileNotFoundError(
            "dataset records disagree with the disk -- a built dataset declares "
            "artifacts that are not there:\n" + "\n".join(lines)
        )
