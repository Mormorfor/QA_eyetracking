# Restructure map

**Status: agreed 2026-09-05. This is the plan of record.** Every open question in it was
answered; Diana signed it off as "tested and agreed upon for now". It stays **revisable** —
expect implementation to argue back, particularly at the three friction points named in §13 —
but a change to it is a decision to record here, not a preference to act on silently.

**Nothing has been moved yet.** Agreeing the map is not permission to execute it. Stages run in
the order set out in §11, and each one is proposed before it starts, per the hard rules in
`CLAUDE.md`.

Written 2026-09-05 against Diana's four answers:

| | |
|---|---|
| **Top-level axis** | one shared data-prep area, then **a folder per functionality**, subfolders where things are related. **Not** split by study. |
| **Optimize for** | clear structure, easy to follow, **open to adding new things by default**. Extra folders are not a cost. |
| **Notebooks** | thin drivers **plus scripted entry points** — the paper regenerates without opening a notebook. |
| **Scope** | everything except `archive/`. Data moves proposed individually before anything happens. |

---

## 1. What is actually wrong with the current shape

Not "it's messy" — four specific structural faults, each of which has already produced a bug.

**1.1 — Visualisation is split from the analysis it visualises.** `viz/` holds 14 modules
named after *plots*; the analyses that produce the numbers live in `derived/`,
`statistics/` and `predictive_modeling/`. So a single question ("what does the dominance
threshold figure show?") is spread across `viz/visualisations_strategies.py` and
`derived/pattern_breaking.py` — which is exactly how those two ended up with **two different
implementations of the same starting-strategy concept** and **two different threshold
operators** (T1.1, T1.6). Distance between related code is what let them drift.

**1.2 — There is no layer for generic machinery, so generic things get copied.** Concretely,
today there are **three separate plot-saving paths**:

| path | where |
|---|---|
| `save_plot` / `save_fig` / `save_df_csv` | `viz/plot_output.py` |
| `save_plot_and_report` | `viz/viz_helpers.py:149` |
| `maybe_save_plot` | `predictive_modeling/common/viz_utils.py:12` |

and **two `wilson_ci`s** (`derived/correctness_measures.py:52`, and a nested `_wilson_ci` at
`viz/visualisations_correctness_measures.py:608` — *in a file that already imports the real
one*). Diana's own example — "wilcoxon CIs could maybe go to some general util folder" — is
the correct instinct: there is no such folder, so the primitives live wherever they were first
needed and get re-written when needed again.

**1.3 — Some files carry several unrelated jobs.** The five largest are all mixed-purpose:

| file | size | jobs it currently holds |
|---|---|---|
| `data_prep/data_csv_generation.py` | 56 KB | report loading · area metrics · sequences · pupil columns · RT orchestration · the pipeline `main()` |
| `answer_correctness/answer_correctness_viz.py` | 55 KB | every figure family for the model, in one module |
| `data_prep/know_qa_dataprep.py` | 48 KB | KnowQA reading · trial-id construction · text alignment · session handling · its own pipeline |
| `answer_correctness/cross_validation.py` | 39 KB | generic CV machinery **and** answer-correctness specifics **and** CV result plots |
| `viz/visualisations_correctness_measures.py` | 36 KB | figures · a duplicated statistic · summary tables |

`viz/viz_helpers.py` is the same fault in miniature and is the clearest illustration: in 150
lines it holds `split_participant_groups` (domain — hunters/gatherers), `p_to_stars`
(generic), `add_wilson_errorbars_and_ns` and `add_significance_bracket` (generic plotting),
`ensure_dir` (generic io) and `save_plot_and_report` (a duplicate save path). Four different
layers in one file.

**1.4 — Everything dataset-shaped is hand-repeated per dataset.** `data_paths.py` is ~90 flat
module-level constants in which the same five concepts appear four times under four different
prefixes — none for L1, `NEW_EXP_`, `SECOND_TEST_`, `KNOW_QA_`. Two consequences:

- **Adding a study means writing another dozen constants and threading them through call
  sites.** That is the opposite of "open to adding new things by default".
- **Hand-maintained constants go stale silently.** Six currently point at files that are not
  there ⚠️ *(read from directory listings; verify with a shell before acting)*:

  | constant | points at | actually |
  |---|---|---|
  | `HUNTERS_LAST_PATH` | `Auxiliary/hunters_last.csv` | not present |
  | `GATHERERS_LAST_PATH` | `Auxiliary/gatherers_last.csv` | not present |
  | `N1_BASE_PATH` `N2_` `N3_` | `Experiment/n{1,2,3}_base.csv` | live in `Experiment/onestop_list_bases/` |
  | `EXPERIMENT_TEXT_COMPLETED_PATH` | `…_completed.csv` | the file is `.zip` |

  And the branding leaks into the data itself: **KnowQA's model-ready table is named
  `L1_model_ready_all_features.csv`**, as are the two pilots'. Three files claiming to be L1.

**The restructure is mostly finishing a migration you already started.** `RT_correlations/`
(10 modules: `data`, `columns`, `correlations`, `comparisons`, `proportions`, `bootstrap`,
`plots`, `report`, `_utils`) and `person_variance/` (7 modules including its own
`plot_style.py`) are both already **one folder per analysis, owning its own plots**. They are
the newest code in the repo. The flat `viz/` + flat `statistics/` layout is the oldest. The
proposal below is: make the newest pattern the rule.

---

## 2. Principles

1. **Layers, and imports only ever point downwards.** `lib` → `config` → `ingest` →
   `features` → `modeling` → `analyses`. Nothing imports `analyses`. `lib` imports nothing
   from the project at all. This is checkable and worth checking in CI.
2. **A folder per functionality, and it owns everything about that functionality** — the
   computation, the statistics, the figures, the tables. If two analyses need the same thing,
   that thing moves *down* a layer; it is never copied sideways.
3. **Study is a parameter, never a folder.** One pipeline, one set of feature builders, one
   model harness. L1 and KnowQA differ by configuration and by a small number of clearly named
   dataset-specific modules — never by a parallel tree.
4. **Live and parked are visibly separate.** `analyses/` is paper code. `explorations/` is
   everything kept for future directions, with the same internal shape. You can tell which is
   which from the path.
5. **Generated things mirror the code that generated them.** `reports/scan_strategies/` comes
   from `analyses/scan_strategies/`. "Which code made this figure?" is answerable from the
   path alone.
6. **One name per concept.** One save path, one `wilson_ci`, one starting-strategy
   implementation, one per-area metric implementation.

---

## 3. The target tree

Package name is a placeholder — **`qa_eyetrack` is a suggestion, rename freely** (`eyeqa`,
`qa_eyetracking`, …). It is the one thing here that is purely yours.

*No `todo.md` item numbers in the tree — which fix lands where is in §11, in one table.*

```
QA_eyetracking_workspace/
├── pyproject.toml        package metadata + deps; pip install -e .
├── environment.yml       conda env, Python 3.11 pinned
├── README.md             what it is, how to run it, the build order
├── CLAUDE.md
├── docs/                 the seven docs + this map + decisions/
│
├── src/qa_eyetrack/
│   │
│   ├── config/           vocabulary and locations. No logic.
│   │   ├── columns.py        was constants.py
│   │   ├── datasets.py       the dataset registry — see §7
│   │   └── outputs.py        reports/ and papers/ roots, mirroring
│   │
│   ├── lib/              generic. Imports nothing from qa_eyetrack.
│   │   ├── stats/
│   │   │   ├── proportions.py    wilson_ci (the only one), p_to_stars
│   │   │   ├── resampling.py     bootstrap, cluster bootstrap
│   │   │   └── sequences.py      levenshtein, windows, seq parsing
│   │   ├── plotting/
│   │   │   ├── output.py         the one save path, + papers mirroring
│   │   │   ├── annotate.py       error bars, brackets, stars
│   │   │   └── style.py          palette, rcParams
│   │   ├── io.py                 read/write helpers, ensure_dir
│   │   └── checks.py             assert_full_coverage, invariants
│   │
│   ├── ingest/           raw vendor reports → tidy IA-level tables
│   │   ├── readers.py        report loading, dtypes, the "." sentinel
│   │   ├── trials.py         trial / participant / session identity
│   │   ├── clicks.py         button clicks → selection & confirm events
│   │   ├── alignment.py      the text-alignment check
│   │   ├── l1.py             OneStop specifics
│   │   ├── knowqa.py         composite TRIAL_INDEX, regimes, sessions
│   │   └── build.py          orchestrator: dataset → interim tables
│   │
│   ├── features/         IA tables → trial-level features
│   │   ├── area_metrics.py   the eight per-area metrics, ONE impl
│   │   ├── reading_times.py  RT / TFD / TimeSinceOffset
│   │   ├── pupil.py          scaling, per-dataset baseline, z-scores
│   │   ├── sequences.py      simplified sequences, visit counts
│   │   ├── strategies.py     starting strategy, dominance, breaking
│   │   ├── last_visited.py   the three last-* variants
│   │   ├── preference.py     preference matching
│   │   ├── scope.py          within_group / global machinery
│   │   ├── paragraph/        its own subpackage
│   │   │   ├── spans.py          critical / distractor / outside
│   │   │   ├── text_features.py  answer & question text sizes
│   │   │   └── eyebench.py       wrapper around the vendored extractor
│   │   └── build.py          assemble the model-ready trial table
│   │
│   ├── modeling/         shared prediction machinery
│   │   ├── models/           logreg · glmer_r · julia · dummy · gbm
│   │   ├── folds.py          predefined folds, the seven regimes
│   │   ├── crossval.py       the CV loop
│   │   ├── evaluate.py       metrics, fold aggregation
│   │   ├── inference.py      coef CIs: wald, bootstrap, clustered
│   │   └── feature_sets.py   named feature-column sets
│   │
│   ├── analyses/         paper code. One folder per functionality.
│   │   ├── attention_allocation/     compute · stats · plots · report
│   │   ├── scan_strategies/          + dominant_eye.py
│   │   ├── answer_rt_comparison/     correct vs distractor asymmetry
│   │   ├── last_visitation/
│   │   ├── time_course/
│   │   ├── correctness_associations/ thresholds + Fisher
│   │   ├── correctness_prediction/
│   │   │   ├── run.py · report.py
│   │   │   ├── plots/                the 55 KB viz module, split up
│   │   │   ├── person_variance/
│   │   │   └── knowledge_regimes/
│   │   └── text_qa_relationship/     was statistics/RT_correlations/
│   │
│   ├── explorations/     parked. Same shape; not paper code.
│   │   ├── answer_reading_times/     was answer_RTs/
│   │   ├── answer_location/          was answer_loc/
│   │   ├── participant_clustering/   was answer_correctness/clusters/
│   │   ├── feature_search/           column options, feature selection
│   │   ├── text_answer_effects/      was mixed_text_answer_effects
│   │   └── unlikely_analysis/
│   │
│   ├── experiment/       study-2 material generation
│   │   ├── texts.py · lists.py · batches.py
│   │
│   └── vendor/
│       └── eyebench/     untouched upstream — replace wholesale
│
├── scripts/              reproducible entry points
│   ├── build_dataset.py      --dataset l1 | knowqa | testrun_qa
│   ├── run_analysis.py       --name scan_strategies
│   └── make_paper_figures.py every figure in the paper
│
├── notebooks/
│   ├── drivers/          thin: load, call, display. No definitions.
│   └── exploration/      free-form. Never a source of a reported number.
│
├── data/  data_raw/      §8
├── reports/              §9 — mirrors analyses/
├── papers/               untouched, Overleaf-synced
└── archive/              untouched this round, by your instruction
```

---

## 4. Why these layer boundaries

**`config/` vs `lib/`.** `lib` is code you could lift into another project unchanged. `config`
is this project's vocabulary. Keeping them apart is what makes the "does this belong in lib?"
question answerable: *if it mentions an eye-tracking column name, it is not lib.*

**`ingest/` vs `features/`.** Ingest is **dataset-shaped**: it knows what an EyeLink report
looks like, how KnowQA's trial ids are built, that `"."` is a sentinel. Features are
**question-shaped**: they know what a dwell proportion is. The boundary is the tidy IA-level
table. This is precisely the boundary **T6.1** is about — today `reading_times.py` reaches
back across it to read paragraph reports during QA prep, and the paragraph feature builders
live inside a *prediction* module. Making the boundary structural is what stops that
recurring.

**`modeling/` separate from `analyses/correctness_prediction/`.** Today `cross_validation.py`
lives inside `answer_correctness/` but nothing in it is specific to answer correctness — it is
generic "cross-validate on predefined folds across regimes" machinery. Same for
`evaluation_core.py`, the model wrappers and the coefficient-CI code. Pulling them out means
the RT regression, any future model and the correctness model share one CV implementation and
one inference implementation — so a fix like **T3.3** (clustered bootstrap CIs) lands once.

**`analyses/` vs `explorations/`.** Your rule was that parked work should end up in the right
folder without spending effort fixing it. A sibling directory does that: parked code moves in
as-is, broken or not, and its brokenness stops being ambiguous. It also makes the release
honest — `explorations/` ships, clearly labelled as not backing the paper.

---

## 5. Where every current file goes

### 5.1 Straight moves

| today | destination |
|---|---|
| `src/constants.py` | `config/columns.py` |
| `src/data_paths.py` | `config/datasets.py` + `config/outputs.py` (split — §7) |
| `derived/pupil_norm.py` | `features/pupil.py` |
| `derived/reading_times.py` | `features/reading_times.py` |
| `derived/select_confirm_last.py` | `features/last_visited.py` |
| `derived/preference_matching.py` | `features/preference.py` |
| `derived/pattern_breaking.py` | `features/strategies.py` (+ merge, §6.2) |
| `derived/correctness_measures.py` | `analyses/correctness_associations/compute.py` |
| `statistics/correctness_measures_tests.py` | `analyses/correctness_associations/stats.py` |
| `statistics/mixed_area_comparisons.py` | `analyses/attention_allocation/stats.py` |
| `statistics/preference_correctness_tests.py` | `analyses/attention_allocation/stats.py` |
| `statistics/RT_correlations/**` | `analyses/text_qa_relationship/**` (shape already correct) |
| `answer_correctness/person_variance/**` | `analyses/correctness_prediction/person_variance/**` |
| `answer_correctness/knowledge_regimes_analysis/**` | `analyses/correctness_prediction/knowledge_regimes/**` |
| `answer_correctness/models/**` | `modeling/models/**` |
| `answer_RTs/models/**` | `modeling/models/**` (gbm, linreg join the same registry) |
| `answer_correctness/evaluation_core.py` | `modeling/evaluate.py` |
| `answer_correctness/feature_groups.py` + `common/feature_specs.py` | `modeling/feature_sets.py` |
| `common/data_utils.py` | split: CIs → `modeling/inference.py`, splits → `modeling/folds.py` |
| `common/feature_builders.py` | `features/build.py` |
| `common/prepared_dataset.py` | `modeling/evaluate.py` |
| `viz/visualisations_area_bars.py`, `_area_matrices.py`, `_area_significance_heatmaps.py`, `_preference_correctness.py` | `analyses/attention_allocation/plots.py` |
| `viz/visualisations_strategies.py`, `sequence_visualisations.py`, `visualisations_simplified_visits.py` | `analyses/scan_strategies/plots.py` |
| `viz/visualisations_dominant_eye.py` | `analyses/scan_strategies/dominant_eye.py` |
| `viz/visualisations_last_label.py` | `analyses/last_visitation/plots.py` |
| `viz/visualisations_time_segments.py` | `analyses/time_course/plots.py` |
| `viz/visualisations_correctness_measures.py` | `analyses/correctness_associations/plots.py` |
| `external/EyeBench/**` | `vendor/eyebench/**` |
| `derived/external/EyeBench/runner.py` | `features/paragraph/eyebench.py` |
| `experiment_builder/*.ipynb` | logic → `experiment/`, drivers → `notebooks/drivers/` |

Two of those deserve a note:

- **`viz/visualisations.py` disappears.** It is a pure re-export façade — ten `from
  src.viz.X import …` lines and nothing else. Its job (one import point for notebooks) is done
  better by each analysis package's `__init__.py`.
- **`derived/external/EyeBench/runner.py` is not a duplicate copy.** It imports from
  `src.external.EyeBench.paragraph_trial_features`, so it is *our wrapper* around the vendored
  code, sitting in a confusingly-named folder. Wrapper and vendor go to different places.

### 5.2 The `viz_helpers.py` split — the pattern for every mixed file

| function | goes to | why |
|---|---|---|
| `p_to_stars` | `lib/stats/proportions.py` | generic |
| `add_wilson_errorbars_and_ns` | `lib/plotting/annotate.py` | generic |
| `add_significance_bracket` | `lib/plotting/annotate.py` | generic |
| `barplot_accuracy` | `lib/plotting/annotate.py` | generic |
| `ensure_dir` | `lib/io.py` | generic |
| `save_plot_and_report` | **deleted** | duplicate save path |
| `split_participant_groups` | `features/scope.py` | domain — knows about hunters/gatherers |

The last row matters: `split_participant_groups` is the function whose `include_all=False`
default hides the all-participants figure, and it is a **grouping** concept, not a plotting
one. Putting it next to the T3.21 scope machinery is where it belongs.

### 5.3 Per-area reading times: feature below, comparison above

Diana's ruling, 2026-09-05, and it is worth stating as a general rule because it will come up
again for other measures:

> **A per-area quantity is a feature. A contrast between per-area quantities is an analysis.**

So the per-area RTs join every other reading-time measure in `features/reading_times.py`,
computed once, for every area, with no opinion about which comparison matters. The
correct-vs-distractor asymmetry — currently defined in `presentation_prep.ipynb` cells and
existing nowhere else — becomes `analyses/answer_rt_comparison/`, which selects the two areas,
contrasts them, tests the difference and draws the figure.

**How this stays distinct from `attention_allocation/`**, since both compare things across
answer areas and the boundary would otherwise blur within a month:

| | `attention_allocation/` | `answer_rt_comparison/` |
|---|---|---|
| built from | the eight per-area IA metrics — dwell, skip rate, fixation count, first fixation, pupil, dwell proportion, visits | the RT / TFD family, which is run-based and derived from click timestamps, not from IA aggregation |
| the question | *where does attention go across the five areas, and does it track the selected answer?* | *is time on the correct answer asymmetric with time on the distractor?* |
| grain | all five areas at once | one designed contrast between two of them |

If a future contrast is over IA metrics rather than RT, it belongs in `attention_allocation/`;
if it is over RT, it belongs here. That is the test.

---

## 6. Files that get split, and how

### 6.1 `data_csv_generation.py` (56 KB) → six modules

| new home | what moves |
|---|---|
| `ingest/readers.py` | `load_raw_*`, dtype handling, `"."` coercions |
| `features/area_metrics.py` | every `create_*` per-area metric function |
| `features/sequences.py` | simplified-sequence construction, visit counts |
| `features/pupil.py` | the pupil column block, `create_first_encounter_pupil_size` |
| `features/build.py` | the group-function registry and assembly |
| `ingest/build.py` | `main()` / `run_pipeline` orchestration |

This split is also the fix for **T3.11**: the five in-place mutating functions create an
undocumented ordering dependency (`create_first_encounter_pupil_size` only works because
`create_mean_first_fix_duration` already coerced a column to int). Once they are separate
modules with explicit inputs and outputs, the dependency has to be written down or it breaks
loudly.

### 6.2 The two starting-strategy implementations → one

`viz/visualisations_strategies.py::build_strategy_dataframe` (descriptive, with prefix
completion) and `derived/pattern_breaking.py::build_starting_strategies` (model features, no
completion) become one function in `features/strategies.py` with `window_len`,
`drop_question`, `complete: bool` and `scope`. **T1.6**, and the reason it must wait for
**T3.21**: the scope flag lives in exactly this function.

### 6.3 `cross_validation.py` (39 KB) → `modeling/{folds,crossval,evaluate}.py`

with the CV-result figures moving out to `analyses/correctness_prediction/plots/`. The generic
half is what other analyses will reuse.

### 6.4 `answer_correctness_viz.py` (55 KB) → `analyses/correctness_prediction/plots/`

split by figure family — coefficients, CV comparison, confusion, probability distributions.
One module per figure family means a figure's code is findable by name.

### 6.5 `know_qa_dataprep.py` (48 KB) → `ingest/knowqa.py` + shared stages

The parts that are genuinely KnowQA-specific (composite `TRIAL_INDEX`, regimes, per-session
handling, alignment checking) stay; everything it currently reimplements from the L1 path
moves to the shared `ingest/` modules. This is where "study is a parameter" gets earned.

---

## 7. The dataset registry — replacing 90 flat constants

The single highest-leverage change for "open to adding new things by default".

**Today:** `KNOW_QA_FEATURES_PATH`, `NEW_EXP_FEATURES_PATH`, `SECOND_TEST_FEATURES_PATH`,
`READY_ALL_FEATURES_PATH` — four constants for one concept, and every call site picks one by
name.

**Proposed:** one `Dataset` description per dataset, and everything else derived from a common
shape.

```python
# config/datasets.py  (sketch — the shape, not final API)

@dataclass(frozen=True)
class Dataset:
    key:            str          # "l1" | "knowqa" | "testrun_qa" | "second_test"
    label:          str          # "OneStop L1"
    kind:           Literal["study", "pilot"]   # pilots stay runnable, but are labelled
    raw_dir:        Path
    root:           Path         # data/datasets/<...>/<key>/
    has_paragraph:  bool         # False for KnowQA — replaces the include_paragraph flag
    pupil_baseline: PupilBaseline | None   # explicit; no silent L1 default

    @property
    def interim(self)  -> Path: return self.root / "interim"
    @property
    def features(self) -> Path: return self.root / "features"
    @property
    def model_ready(self) -> Path: return self.features / "model_ready.csv"

DATASETS = {d.key: d for d in (L1, KNOWQA, TESTRUN_QA, SECOND_TEST)}

def dataset(key: str) -> Dataset: ...
def studies() -> list[Dataset]: ...   # kind == "study" — the default for loops
def pilots()  -> list[Dataset]: ...   # asked for explicitly
```

What this buys, concretely:

- **A new study is one `Dataset(...)` entry**, not a dozen constants plus call-site edits.
- **`has_paragraph` replaces `include_paragraph`**, which is currently threaded by hand through
  four call layers (`data_csv_generation:1278, 1319, 1408, 1575` → `reading_times:492`) purely
  because KnowQA has no paragraph screen. It becomes a property of the dataset, asked once.
- **`pupil_baseline` has no default**, which is the structural form of **T3.20**: a dataset
  either declares its baseline or the pipeline refuses to compute pupil z-scores. No silent
  fallback to L1's answer screen.
- **The stale-constant class of bug disappears**, because paths are derived from one root and
  can be validated in one loop at import or in a test.
- **Filenames stop being dataset-branded.** `model_ready.csv` under `datasets/knowqa/features/`
  says what it is by location. No more `L1_model_ready_all_features.csv` in KnowQA's folder.

Output paths (`COL_SAVE_PATH`, `CROSS_VALIDATION_RUNS_DIR`, `PER_PERSON_LOO_RESULTS_DIR`)
move to `config/outputs.py` — they are report destinations, not data sources, and mixing them
into the same module is why `reports/` paths are currently hardcoded in ~40 places (**T5.3**).

---

## 8. `data/` layout

⚠️ **Every move below is a proposal to approve individually.** Nothing moves without the how
and the why agreed first, per your standing rule.

```
data/
├── datasets/
│   ├── l1_onestop/
│   │   ├── interim/        all_participants.csv · hunters.csv · gatherers.csv
│   │   │   └── aux/        button_clicks · RT_and_TFD · participant_pupils · last
│   │   └── features/       model_ready.csv · paragraph_spans.csv · answer_text.csv
│   │       └── eyebench/   paragraph_trial_level_features.csv + the two key files
│   ├── knowqa/             interim/ features/     (same shape)
│   └── pilots/             the same shape, one level down and labelled
│       ├── testrun_qa/     interim/ features/     (was new_exp_try_runs)
│       └── second_test/    interim/ features/     (was second_test_runs)
├── cv_folds/
│   ├── l1_hunters/ · l1_gatherers/ · l1_gatherers_refolded/ · l1_hunting_is_correct/
└── stimuli/                was data/Experiment
    ├── onestop_list_bases/ · batched_lists/ · texts/
```

**On the pilots.** They keep the identical internal shape and stay fully runnable (T5.8) —
`pilots/` is a label, not a downgrade. Two things make the labelling real rather than cosmetic:
the folder itself, and a `kind` field on the dataset record (§7) so anything that lists or
loops over datasets can say *pilot* or skip them by default. Without that field the grouping
only helps a human reading a directory listing, and code would still treat all four alike.

The individual moves, in the order I'd propose them:

| # | move | why |
|---|---|---|
| 1 | `CV folds/` → `cv_folds/` | a space in a path that appears in ~10 constants |
| 2 | `*/Auxiliary/` → `*/interim/aux/` | same shape for every dataset |
| 3 | `L1_based_data/` → `datasets/l1_onestop/` | consistent naming |
| 4 | `KnowQA_runs/` → `datasets/knowqa/` | " |
| 5 | `new_exp_try_runs/` → `datasets/pilots/testrun_qa/` | ✅ confirmed 2026-09-05: this *is* `testrun_QA`, so the folder finally matches its `data_raw/` counterpart |
| 6 | `second_test_runs/` → `datasets/pilots/second_test/` | matches `data_raw/second_test` |
| 7 | **`*/L1_model_ready_all_features.csv` → `*/features/model_ready.csv`** | three non-L1 files currently claim to be L1 |
| 8 | `data/{hunters,gatherers}_paragraph_answer_merge.csv` → `l1_onestop/interim/` | dataset files at the `data/` root |
| 9 | `data/Experiment/` → `data/stimuli/` | stimulus material is not processed eye-tracking output |
| 10 | `data/Experiment/fake results/` → `data_raw/pilots/` | these are **raw recordings**, in `data/` |
| 11 | `data/strange_trials.csv` → `l1_onestop/interim/` | ⚠️ pending what this file should become (`glossary.md`) |

Move **7** is the one that matters most and the one I'd do first among the renames — it is a
correctness hazard, not tidiness. `all_participants_with_practice.csv` (3.9 GB) stays, per
T5.8.

---

## 9. `reports/` layout

Today: `reports/{plots,report_data}/<topic>/`, with **9 of 15 plot topics having no
`report_data` counterpart** and three of the six that do having a *different name*
(`area_significance_heatmaps` ↔ `area_mixed_models`, `texts_to_answers` ↔ `slopes`).

**Proposed: invert the nesting so it mirrors `analyses/`.**

```
reports/
  <analysis_name>/
    figures/
    tables/
```

This makes **T4.0**'s acceptance test structural rather than a convention someone has to
remember: a figure with no `tables/` sibling is visible at a glance, in the same folder, rather
than requiring a comparison of two parallel trees. It also makes provenance a path lookup —
`reports/scan_strategies/` came from `analyses/scan_strategies/`, necessarily.

Current topic → new home:

| `reports/plots/<topic>` | → |
|---|---|
| `strategies`, `simpl_visit_matrices`, `dominant_eye` | `scan_strategies/` |
| `basic_stats_barcharts`, `basic_stats_heatmaps`, `area_significance_heatmaps` | `attention_allocation/` |
| `correctness_measures` (+ the five `correctness_by_*` data folders), `matching_correctness`, `total_answering_RT_normalized` | `correctness_associations/` |
| `answer_correctness` | `correctness_prediction/` |
| `last_label_before_confirm` | `last_visitation/` |
| `time_segments` | `time_course/` |
| `texts_to_answers` | `explorations/text_answer_effects/` |
| `feature_selection` | `explorations/feature_search/` |
| `participant_similarity` | `explorations/participant_clustering/` |
| *(nothing today)* | `text_qa_relationship/` — currently writes nothing at all, **T4.1** |

**Both of the ones I flagged are now settled, and both landed in `correctness_associations`:**

- **`matching_correctness`** — your call, 2026-09-05. It asks whether preference matching
  predicts correctness, which is a correctness association, not a description of attention.
- **`total_answering_RT_normalized`** — settled by reading the code rather than by asking.
  Its figure comes from `plot_correctness_by_total_answering_rt_continuous`
  (`visualisations_correctness_measures.py:808`), so it is literally another
  `correctness_by_*` plot and belongs with its five siblings. My original question ("time
  course, or a descriptives folder?") was based on the folder name alone and was the wrong
  question — see the note below.

That leaves **no `descriptives/` folder**, which is the right outcome: every "basic" figure
turned out to be a figure *about something*, so each has a real home. Had one been a genuine
context-free distribution, a `descriptives/` folder would have been the honest answer.

The `papers/` mirroring mechanism is unaffected — `save_plot(paper_dirs=[...])` keeps working,
just with the new relative paths. It also fixes the **two conflicting `paper_dirs`
conventions** noted in T1.5, since the relative path now comes from one place.

---

## 10. Notebooks and scripts

You chose thin drivers **plus** scripted entry points, which is the more demanding option and
the one that makes the release reproducible.

```
scripts/
  build_dataset.py      --dataset l1          raw → interim → features
  run_analysis.py       --name scan_strategies    one analysis, figures + tables
  make_paper_figures.py                       every figure in the paper, in order
notebooks/
  drivers/              one thin notebook per analysis: load, call, display
  exploration/          free-form; explicitly not the source of any reported number
```

**What has to move out of notebooks first.** Two analyses are currently *defined* in notebook
cells and have no home in `src/`:

- the **correct-vs-distractor RT asymmetry** (`presentation_prep.ipynb`) — **settled
  2026-09-05, and it splits in two**: the per-area reading times are *features*, built in
  `features/reading_times.py` alongside every other RT; the **comparison between them is its
  own analysis**, `analyses/answer_rt_comparison/`. See §5.3 for why that boundary is the
  right one and how it differs from `attention_allocation/`.
- **`longest_alternating_run`** (the XYXY counter-evidence) → `features/sequences.py`, with the
  analysis in `analyses/scan_strategies/`

`text_associations.ipynb` is the current authoritative text↔QA analysis and holds its results
only as cell outputs in a 1.2 MB notebook (**T4.1**) — routing it through the standard save
path is what turns it into a driver.

`experiment_builder/` is five notebooks and **no `.py` at all**. Its logic (text preparation,
list building, batching) moves to `qa_eyetrack/experiment/`, leaving thin drivers. Note this is
distinct from `ingest/knowqa.py`: `experiment/` *generates the materials*, `ingest/` *reads the
recordings that came back*.

---

## 11. Staging

Ordered so that each stage is independently verifiable and the riskiest work happens after the
scaffolding that makes it checkable. Every stage ends with a run that must reproduce or
deliberately change a known number.

| Stage | What | Numbers move? | T-items landing |
|---|---|---|---|
| **0** | **Quick wins, no restructuring.** Cheap, independent, high value. | **yes** (T3.1) | T3.1, T1.1, T1.4, verify T2.4 |
| **A** | **Make it a package.** `pyproject.toml`, `__init__.py` throughout, one import convention, delete the `sys.path` hacks (including the one inside `data_paths.py:6`), `environment.yml`. **Nothing moves.** | no | T5.1, T5.2, T5.4 |
| **B** | **`config/` + `lib/`.** Lift generic primitives; kill the two duplicate `wilson_ci`s and the two extra save paths; split `viz_helpers.py`; build the dataset registry; fix the six stale constants; migrate the ~40 hardcoded `"../reports/..."` literals. | no | T1.5, T5.3, part of T3.20 |
| **C** | **`ingest/` + `features/`.** The big correctness stage: separate paragraph from QA prep, unify the two per-area metric implementations, unify the two starting-strategy implementations, land the scope flag, assert every join. | **yes** | T6.1, T1.7, T1.6, T3.6, T3.14, T3.20, T3.21, T3.17, T3.7, T3.5, T1.8 |
| **D** | **`modeling/`.** Extract CV, evaluation, model wrappers and inference. Clustered bootstrap CIs become reachable and default. | **yes** | T3.3, T3.9, T3.10, T5.11, T2.4 |
| **E** | **`analyses/` + `explorations/` + `reports/` inversion.** `viz/` dissolves into per-analysis `plots.py`; one saving framework everywhere; re-run what needs re-running. | figures only | T1.3, T4.0, T4.1, T4.2, T3.2, T3.18, T3.19 |
| **F** | **Entry points.** `scripts/`, thin notebooks, notebook-only logic lifted into the package. | no | T5.6, T5.10 |
| **G** | **Data + release.** The `data/` moves (individually approved), README, run-from-scratch verification. | no | T5.5, T5.7, T5.8, T5.9, §8 |

**Why this order.** Stage A costs almost nothing and makes every later stage's imports
mechanical instead of fragile. Stage B is pure consolidation with no behaviour change, so it
can be verified by "every number identical" — which is exactly the check you *cannot* run
during C and D, where numbers are supposed to move. Doing B first means that when a number
moves in C, the restructure is not a suspect.

**Stage C is the one to be careful with.** It is where the integrity work concentrates, and it
is the only stage where a mistake would be hard to distinguish from an intended change. Worth
running the headline model before and after and logging both, per `findings.md`'s change log.

---

## 12. Adding things afterwards

The test of whether this worked:

**A new analysis** — create `analyses/<name>/` with `compute.py`, `stats.py`, `plots.py`,
`report.py`; add a driver notebook; outputs appear at `reports/<name>/`. Nothing else is
touched, and nothing else can break.

**A new dataset or study** — add one `Dataset(...)` entry to `config/datasets.py`; add an
`ingest/<name>.py` only if its raw reports differ in shape. Every feature builder and every
analysis works on it unchanged. **No new top-level folder, ever.** This is the concrete payoff
of not splitting by study.

**A new feature** — one module in `features/`, registered in `features/build.py`, named in
`modeling/feature_sets.py` if a model should see it.

**A new statistic** — if it mentions a column name it belongs to an analysis; otherwise it
belongs in `lib/stats/`. That single question is the whole rule.

---

## 13. Open questions

**All six of the original questions were answered on 2026-09-05.** Nothing in this map is now
waiting on a decision.

| | Answer |
|---|---|
| Package name | **`qa_eyetrack`** |
| `matching_correctness` | `correctness_associations/` (§9) |
| `total_answering_RT_normalized` | `correctness_associations/` — settled by reading the code, not by asking (§9) |
| correct-vs-distractor RT asymmetry | feature in `features/reading_times.py`; the contrast is its own analysis, `analyses/answer_rt_comparison/` (§5.3) |
| `new_exp_try_runs` = `testrun_QA`? | yes (§8) |
| pilots | grouped under `datasets/pilots/`, still fully runnable, with a `kind` field so code can tell (§7, §8) |

**Still deferred, by instruction rather than by uncertainty:**

- **`archive/`** — out of scope this round. Worth its own pass eventually: 2.5 GB of superseded
  CSVs, three generations of plot folders (`plots` → `new_plots` → `third_plots`), and eight
  old notebooks.

**Decisions the map assumes but that only implementation can confirm.** These are not
questions for you now; they are the places where I expect the design to meet friction, and
they are worth re-reading when the relevant stage starts:

1. **That `ingest/` really can be shared between L1 and KnowQA** with only `l1.py` and
   `knowqa.py` differing. `know_qa_dataprep.py` is 48 KB, and how much of that is genuine
   difference versus reimplementation is a question the merge will answer, not this document.
2. **That `modeling/` comes out cleanly** — i.e. that nothing in `cross_validation.py` secretly
   depends on answer-correctness specifics. Read from the code, not tested.
3. **That the reports inversion is a pure move.** ~40 hardcoded `"../reports/..."` literals
   have to route through one place first (Stage B) or the inversion will scatter them further.
