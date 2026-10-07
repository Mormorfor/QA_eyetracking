# Restructure map

**Status: agreed 2026-09-05. This is the plan of record.** Every open question in it was
answered; Diana signed it off as "tested and agreed upon for now". It stays **revisable** —
expect implementation to argue back, particularly at the three friction points named in §13 —
but a change to it is a decision to record here, not a preference to act on silently.

**No file has been moved yet.** Agreeing the map is not permission to execute it. Stages run in
the order set out in §11, and each one is proposed before it starts, per the hard rules in
`CLAUDE.md`.

---

## Read this before starting — what the plan no longer has to do

*Updated 2026-10-06. The map was written on 2026-09-05; a month of fixes happened before the
restructure began, and several of them did the plan's work early. Re-verified against the code
on 2026-10-06.*

**The headline: the stages that were supposed to move numbers no longer do.** §11 calls C and
D the dangerous ones *because results change there*. All of that work has already landed as
standalone fixes, each with a `findings.md` change-log entry. **What remains in every
remaining stage is file movement**, so "every number identical" — the check §11 claims only
for stage B — now applies throughout. That is a materially easier job than the one described
below.

| Already true, so the plan can skip it | Where the map still describes it as a problem |
|---|---|
| **One save path.** `save_plot_and_report` and `maybe_save_plot` are gone; `save_output` is the only writer, and it *requires* the numbers behind each figure | §1.2's three-paths table |
| **One `wilson_ci`.** All nested copies deleted | §1.2 |
| **One starting-strategy implementation**, one threshold operator (`≥`) | §1.1's worked example, §6.2 |
| **One per-area metric implementation** — `derived/area_metrics.py` already holds the shared formulas | §6.1's `features/area_metrics.py` row |
| **Paragraph prep already split out** — `derived/paragraph_prep.py` exists; the QA pipeline opens no paragraph report | §6.1, §6.5, T6.1 |
| **`reports/` already inverted** to `<analysis>/{figures,tables}/` | §9 (marked implemented) |
| **No hardcoded `"../reports/…"` literals** anywhere in code or notebooks | §11 stage B's T5.3 |
| **Join assertions exist** — `src/checks.py::assert_full_coverage`, 13 call sites | §11 stage C |
| **Scope is an explicit argument** on participant-level aggregates | §11 stage C's T3.21 |

**What is genuinely untouched**, and therefore what stages A → G still have to do:

- **A** — 🟨 **mostly done 2026-10-06**, minus packaging: 22 `__init__.py`, one import
  convention, `environment.yml` pinned. `pyproject.toml` is deliberately deferred, so the repo
  still reaches its code via `PYTHONPATH` rather than an install. **One thing this uncovered
  that the later stages inherit:** `src/statistics/` shadowed Python's standard-library
  `statistics` module the moment `src/` was on `sys.path` and the folder became a real package
  — seaborn's `from statistics import NormalDist` resolved to our code, and 21 modules stopped
  importing. Fixed twice over: `src/` came off the path, and the folder was **renamed
  `src/statistics/` → `src/stats/`** (Diana, 2026-10-06). Every other directory and module name
  under `src/` was checked against the standard library and the installed third-party names —
  that was the only collision, and the names §3 introduces (`analyses`, `modeling`, `features`,
  `ingest`, `lib`, `config`) are all clear. Full account in `pitfalls.md` §6.
- **B** — the `config/` + `lib/` lift itself, and the dataset registry. Nine stale constants,
  not six (§1.4) — and **`COL_SAVE_PATH` is the one that blocks work**, because the
  cross-validation cannot be re-run until those five feature-set JSONs have a home.
- **C, D, E, F, G** — the moves, exactly as written, minus the integrity work above.

**One new file the map does not mention:** `knowledge_regimes_analysis/descriptives.py` is now
**the largest module in the repo at 81 KB** — bigger than anything in §1.3's table. It is
Study 2 work that grew after the map was written and has no row in §5. It wants the same
treatment as the other oversized modules; where it lands is a stage-E question.

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

> ✅ **That particular drift was repaired 2026-09-20** — there is now one implementation and one
> operator. A **third** copy turned up during the merge, in `visualisations_dominant_eye.py`,
> with an unstable tie-break that moved one participant: three copies of a concept nobody
> intended to write even twice.
>
> **The fault itself is untouched.** `src/viz/` still holds **14 modules named after plots**,
> still separated from the analyses that compute their numbers. The repair was manual and
> does not prevent the next drift — which is the whole argument for stage E.

**1.2 — There is no layer for generic machinery, so generic things get copied.**
✅ **The symptoms are fixed (2026-09-20/27); the missing layer is not.**

There used to be **three separate plot-saving paths** (`save_plot`/`save_fig`/`save_df_csv` in
`viz/plot_output.py`, `save_plot_and_report` in `viz/viz_helpers.py`, `maybe_save_plot` in
`predictive_modeling/common/viz_utils.py`) and **two `wilson_ci`s** — plus, found later, three
*nested* copies of `_wilson_ci` inside a single file that already imported the real one.

Today there is **one** writer (`save_output`) and **one** `wilson_ci`. Verified 2026-10-06.

**But the underlying fault is unchanged, and it is the reason for stage B:** there is still no
`lib/` for generic primitives, so the next shared helper has nowhere to live and will be
written twice again. Diana's own example — "wilcoxon CIs could maybe go to some general util
folder" — is the correct instinct; the folder still does not exist. The duplicates were
removed one at a time, which does not stop the next one.

**1.3 — Some files carry several unrelated jobs.** ⚠️ **Table re-measured 2026-10-06 — the
ranking changed, and it got worse, not better.**

| file | size (was) | jobs it currently holds |
|---|---|---|
| **`knowledge_regimes_analysis/descriptives.py`** | **81 KB** *(new)* | **Not in the 2026-09-05 table at all** — Study 2 descriptive work that grew after the map was written. Now the **largest module in the repo**, and §5 has no row for it |
| `data_prep/data_csv_generation.py` | **66 KB** (was 56) | report loading · base features · the `create_*` metric wrappers · sequences · pupil columns · the pipeline `main()`. **Grew** — T3.18's screen-geometry labelling and T3.17's assertions landed in it |
| `answer_correctness/answer_correctness_viz.py` | 55 KB | every figure family for the model, in one module |
| `data_prep/know_qa_dataprep.py` | **52 KB** (was 48) | KnowQA reading · trial-id construction · text alignment · session handling · its own pipeline |
| `answer_correctness/cross_validation.py` | **48 KB** (was 39) | generic CV machinery **and** answer-correctness specifics **and** CV result plots. **Grew** — T3.9 and T3.10's regime and weighting work |
| `viz/visualisations_correctness_measures.py` | **33 KB** (was 36) | figures · summary tables. The duplicated statistic is gone |

> **The point this makes is stronger than it was.** Three of these files grew while the cleanup
> was happening, because correctness fixes had nowhere to go *except* the big file they were
> fixing. That is the fault describing itself: without the layers, every improvement lands in
> whichever module already does everything.

`viz/viz_helpers.py` is the same fault in miniature — though it is now **165 lines holding six
functions across three layers**, not four: `split_participant_groups` (domain —
hunters/gatherers), `p_to_stars` (generic stats), `add_wilson_errorbars_and_ns`,
`add_significance_bracket` and `barplot_accuracy` (generic plotting), plus `correctness_tables`
(domain). ✅ `ensure_dir` and `save_plot_and_report` have gone, so the io layer and the
duplicate save path are out of it — §5.2's split table is two rows shorter than it was.

**1.4 — Everything dataset-shaped is hand-repeated per dataset.** `data_paths.py` is ~90 flat
module-level constants in which the same five concepts appear four times under four different
prefixes — none for L1, `NEW_EXP_`, `SECOND_TEST_`, `KNOW_QA_`. Two consequences:

- **Adding a study means writing another dozen constants and threading them through call
  sites.** That is the opposite of "open to adding new things by default".
- **Hand-maintained constants go stale silently.** **Nine** currently point at files or
  directories that are not there — ✅ re-measured 2026-09-27 by importing `data_paths` and
  testing `.exists()`, so these are no longer read off a directory listing:

  | constant | points at | actually |
  |---|---|---|
  | `HUNTERS_LAST_PATH` | `Auxiliary/hunters_last.csv` | not present |
  | `GATHERERS_LAST_PATH` | `Auxiliary/gatherers_last.csv` | not present |
  | `N1_BASE_PATH` `N2_` `N3_` | `Experiment/n{1,2,3}_base.csv` | live in `Experiment/onestop_list_bases/` |
  | `EXPERIMENT_TEXT_COMPLETED_PATH` | `…_completed.csv` | the file is `.zip` |
  | **`COL_SAVE_PATH`** | `report_data/answer_correctness/feature_columns` | **whole tree deleted** in `496f8d0` |
  | **`CROSS_VALIDATION_RUNS_DIR`** | `report_data/answer_correctness/cross_validation_runs` | **deleted** in `496f8d0` |
  | **`PER_PERSON_LOO_RESULTS_DIR`** | `report_data/per_person_corr_loo_results` | **deleted** in `496f8d0` |

  The last three are new, and they are a different kind from the first six: they are not typos
  that drifted, they are **live destinations orphaned by the `reports/` inversion**. All three
  are recoverable from `496f8d0^` (the 109 feature-column JSONs, 3 CV summaries and the 66.5 MB
  LOO pickle are intact in git), but the artefacts there are **pre-rebuild**, so they need
  *regenerating* rather than restoring. **Stage B has to decide where each one points**, and
  `COL_SAVE_PATH` is the one that blocks work: `answer_corr_prediction.ipynb` cell 21 globs it
  for the five feature-set JSONs that define the paper's model comparison, so the CV cannot be
  re-run until it has a home. Whether those five JSONs are *reports* or *configuration* is the
  open question — they are hand-specified feature sets consumed by the headline run, and the
  same cell already defines three more inline.

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
│   │   └── text_qa_relationship/     was stats/RT_correlations/
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

> **One accepted exception, ruled 2026-10-07.** `lib/plotting/output.py` holds `ANALYSES` (the
> 17 analysis names) and `ABBREVIATIONS` (36 shortenings of this project's column names). By the
> test above those are vocabulary and belong in `config/outputs.py` — but `output.py` *uses* them
> to build every filename, and `lib` may not import `config`, so honouring the rule means
> rewiring the module to receive them as injected config and re-pointing all 35 call sites.
>
> **Diana's ruling: leave it** — *"separation is supposed to make things easier, not harder."*
> So `lib/` here means "generic-ish helpers", not "portable without modification". The cost is
> narrow and worth naming: `lib/` could not be lifted into another project as-is. Nothing else
> is affected, and no number depends on it. Revisit only if that portability ever becomes a
> real requirement.

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
| **`src/checks.py`** *(new, T3.17)* | `lib/checks.py` — it is already exactly what §3 planned to create: `assert_full_coverage`, 13 call sites |
| **`derived/area_metrics.py`** *(new, T1.7)* | `features/area_metrics.py` — **already the single shared implementation** §6.1 asks for; this row is a rename, not a merge |
| **`derived/paragraph_prep.py`** *(new, T6.1)* | `features/paragraph/` — paragraph IA prep, span metrics and RT/TFD, already separated from both the QA pipeline and `answer_RTs/` |
| `derived/pupil_norm.py` | `features/pupil.py` |
| `derived/reading_times.py` | `features/reading_times.py` |
| `derived/select_confirm_last.py` | `features/last_visited.py` |
| `derived/preference_matching.py` | `features/preference.py` |
| `derived/pattern_breaking.py` | `features/strategies.py` (+ merge, §6.2) |
| `derived/correctness_measures.py` | `analyses/correctness_associations/compute.py` |
| `stats/correctness_measures_tests.py` | `analyses/correctness_associations/stats.py` |
| `stats/mixed_area_comparisons.py` | `analyses/attention_allocation/stats.py` |
| `stats/preference_correctness_tests.py` | `analyses/attention_allocation/stats.py` |
| `stats/RT_correlations/**` | `analyses/text_qa_relationship/**` (shape already correct) |
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
| `external/EyeBench/**` | `vendor/eyebench/**` — ✅ **done 2026-10-06** |
| ~~`derived/external/EyeBench/runner.py`~~ → ~~`features/paragraph/eyebench.py`~~ | **superseded by Diana's ruling (2026-10-06): `vendor/eyebench/runner.py`.** Everything EyeBench in one place — see the note below |
| `experiment_builder/*.ipynb` | logic → `experiment/`, drivers → `notebooks/drivers/` |

Two of those deserve a note:

- **`viz/visualisations.py` disappears.** It is a pure re-export façade — ten `from
  src.viz.X import …` lines and nothing else. Its job (one import point for notebooks) is done
  better by each analysis package's `__init__.py`.
- **`derived/external/EyeBench/runner.py` was not a duplicate copy** — it was *our wrapper*
  around the vendored code, sitting in a confusingly-named folder. This row used to send wrapper
  and vendor to different places.

  > ✅ **Overruled by Diana, 2026-10-06:** *"EyeBench stuff is unimportant… keep it all in one
  > place, try to avoid IT breaking MY code."* So all four pieces — the file as received, its
  > trimmed `configs/`, our adapter and the driver — live in **`src/vendor/eyebench/`**.
  >
  > **What replaces the separation is a containment rule**, which is stronger and mechanically
  > checkable: *nothing on the live path may import `src/vendor/` at module scope.* The one live
  > consumer (`answer_RTs/model_data.py`) already imported lazily, inside the function. Verified
  > 2026-10-06 — importing all 96 live modules leaves **no `src.vendor.*` entry in
  > `sys.modules`**. A vendored breakage therefore cannot stop the project importing, which is
  > what makes "replace wholesale" affordable.
  >
  > Also verified, because it was the thing most likely to break silently: the `importlib`
  > by-path load of `utils - paragraph feature extraction.py` still resolves, and the
  > `src.configs` alias still points at our trimmed copy. **Note `src.configs` (theirs, a fake
  > package that exists only in `sys.modules` during extraction) is one character from
  > `src.config` (ours).** Commented at the alias site.

- **`src/external/scasim_wrapper.ipynb` was deleted, not moved.** An rpy2 wrapper around the
  `tmalsburg/scanpath` R package; `archive/scasim_wrapper.ipynb` is **byte-identical** to it
  (md5 `ee9ff8f8…`, both 8,681 bytes, both dated 2026-01-30). Removing the `src/` copy loses
  nothing — the archived one is the copy to go back to.

### 5.2 The `viz_helpers.py` split — the pattern for every mixed file

| function | goes to | why |
|---|---|---|
| `p_to_stars` | `lib/stats/proportions.py` | generic |
| `add_wilson_errorbars_and_ns` | `lib/plotting/annotate.py` | generic |
| `add_significance_bracket` | `lib/plotting/annotate.py` | generic |
| `barplot_accuracy` | `lib/plotting/annotate.py` | generic |
| ~~`ensure_dir`~~ | ~~`lib/io.py`~~ | ✅ **already gone** (2026-09-27) — had zero callers |
| ~~`save_plot_and_report`~~ | ~~deleted~~ | ✅ **already deleted** (2026-09-20) with the save-path consolidation |
| `correctness_tables` *(not in the original table)* | `analyses/correctness_associations/` | domain — builds the correctness summary tables |
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

### 5.4 The layering exception — ✅ **CLOSED 2026-10-06**

Stage C left one import crossing the one-way boundary:

```
src/features/build.py:22  from src.ingest.geometry import _reconcile_area_with_geometry
```

**Fixed by option B** (Diana, 2026-10-06: *"Fix the `_reconcile_area_with_geometry` on the
way"*). `features/build.py` is gone, split in two and moved into `ingest/`:

| new file | holds | why there |
|---|---|---|
| `ingest/base_features.py` | the eight row-level `add_*` builders | they produce the **structure** of the tidy table — `area_screen_loc`, `n_interest_areas`, the `*_len` columns, the ids, the target — from the stimulus and the screen geometry, not from behaviour. That is ingest by the definition in both package docstrings |
| `ingest/registry.py` | `FUNCTION_REGISTRY` + the four runners/accessors | the recipe names functions from **both** layers, so it cannot live in `features/` without that package importing `ingest/` |

**Registry order is byte-identical** — 20 entries, same order, same `default_kwargs`. Verified:
`all_participants.csv` rebuilds bit-identical (KnowQA 33,830 × 420, max diff 0.0), 93/97 modules
import, and `grep` for `src.ingest` under `src/features/` returns nothing.

Two things this surfaced, both recorded rather than silently settled:

- **`add_total_answering_RT_normalized` is arguably a measurement**, not structure — it divides a
  reading time by a word count. It moved with the other seven because it is row-level and reads
  `n_interest_areas` from `add_IA_screen_location` directly above it. Flagged in the module
  docstring and in the stage D proposal's "least sure" list.
- **`ingest/` is becoming the pipeline layer, and the name will stop fitting.** Once stage D adds
  the `trial` kind, the registry describes the whole generator rather than its IA half. That is a
  naming question for stage D, deliberately not pre-empted here.

---

## 6. Files that get split, and how

### 6.1 `data_csv_generation.py` (**66 KB**, was 56) → six modules — ✅ **DONE 2026-10-06**

> **It split nine ways, not six.** The 35 functions went to `ingest/readers.py` (2),
> `ingest/build.py` (6), `ingest/geometry.py` (2), `features/build.py` (12),
> `features/area_metrics.py` (5), `features/pupil.py` (3), `features/sequences.py` (3),
> `features/last_visited.py` (1), `features/scope.py` (1) — every function in exactly one place.
> `ingest/geometry.py` is the extra one Diana approved on 2026-10-06: the two functions that
> encode the **screen layout** rather than the file format. The `state` column below is the
> pre-stage picture, kept because the reasoning in the notes after it still applies.
>
> **T3.11's ordering dependency is written down but still not broken.** The modules are separate
> now, so the dependency crosses a module boundary and is visible — but `FUNCTION_REGISTRY` still
> encodes it as list order, which Diana ruled stays (**T3.24 declined**, 2026-10-06): *"there
> should in the end be a default runner, and it should run things in order they are now."*

**Partly done already.** T1.7 pulled the metric *formulas* out into
`derived/area_metrics.py` so the answer and paragraph paths share one implementation, and T6.1
pulled the paragraph work out into `derived/paragraph_prep.py`. What stayed behind is the
pipeline: 35 functions — 11 `create_*` wrappers that call the shared formulas, 10 `add_*` base
features, 2 loaders and `main()`. The file **grew anyway**, because T3.18 and T3.17 had
nowhere else to land.

| new home | what moves | state |
|---|---|---|
| `ingest/readers.py` | `load_raw_*`, dtype handling, `"."` coercions | to do |
| `features/area_metrics.py` | every `create_*` per-area metric function | 🟨 **formulas already extracted** to `derived/area_metrics.py`; the 11 `create_*` wrappers still sit in the big file. This becomes a **move + rename**, not a merge |
| `features/sequences.py` | simplified-sequence construction, visit counts | to do |
| `features/pupil.py` | the pupil column block, `create_first_encounter_pupil_size` | to do |
| `features/build.py` | the group-function registry and assembly | to do |
| `ingest/build.py` | `main()` / `run_pipeline` orchestration | to do |
| `features/paragraph/` | paragraph IA prep, span metrics, paragraph RT/TFD | ✅ **done** — `derived/paragraph_prep.py` (T6.1). The QA pipeline opens no paragraph report at all now |

This split is also the fix for **T3.11**: the five in-place mutating functions create an
undocumented ordering dependency (`create_first_encounter_pupil_size` only works because
`create_mean_first_fix_duration` already coerced a column). Once they are separate
modules with explicit inputs and outputs, the dependency has to be written down or it breaks
loudly.

> **Note added 2026-09-23, after T3.6 landed.** That ordering dependency is now *commented* at
> both ends, but it is **not** fixed — and the split is what has to fix it. Three things this
> stage must get right, learned by doing the T3.6 change:
>
> 1. **The `"."` coercion is currently a side effect of a metric function**, not an ingest
>    step. `create_mean_first_fix_duration` coerces `IA_FIRST_FIXATION_DURATION` in place and
>    then takes the mean; `create_first_encounter_pupil_size` silently depends on that having
>    happened (it filters `> 0`). Under the split the coercion moves to `ingest/readers.py` and
>    **both** consumers receive an already-numeric column, so `features/area_metrics.py` and
>    `features/pupil.py` become pure. Until then, running either group function alone via
>    `group_function_names=[...]` compares `str > int` and raises.
> 2. **The include/exclude convention must move with the metrics, per metric.** It is not a
>    global policy: dwell time and fixation count keep unread words as `0`, first-fixation
>    duration and pupil exclude them. The unified T1.7 functions are parameterized by the
>    grouping column — they must **also** carry the missing-value convention per metric, or the
>    merge will silently pick one and flatten the distinction. `pitfalls.md` §2 is the
>    statement of record; the asymmetry is deliberate.
> 3. **The coercion leaks into the saved table**, so `all_participants.csv` now stores
>    `IA_FIRST_FIXATION_DURATION` as float-with-NaN rather than int. Anything that reads the
>    saved IA-level file and expects an int column needs to know. Making ingest own the
>    coercion makes this a stated output schema instead of a side effect.

> **Update 2026-09-23 — part of this stage has already landed, outside the restructure.**
> T1.7 and T6.1 were done early, because T3.6 made the divergence between the two metric
> implementations concrete enough to act on. Two modules now exist in the *current* tree,
> named and shaped so the eventual move is a rename rather than a rewrite:
>
> | now | destination in this map |
> |---|---|
> | `src/derived/area_metrics.py` — the eight metrics, parameterized by grouping column | `features/area_metrics.py` (§6.1's row, already done) |
> | `src/derived/paragraph_prep.py` — the whole paragraph screen: reports → one feature table | `features/paragraph/` (§3) |
>
> What that settles in advance of Stage C:
>
> * **the answer pipeline no longer reads paragraph input**, so the `ingest`/`features`
>   boundary this map describes at §4 is already real for that half;
> * **`include_paragraph` is gone** — the flag §7 wanted to replace with a `has_paragraph`
>   dataset property no longer exists to replace. A dataset simply does or does not run the
>   paragraph pipeline;
> * **the `how="inner"` paragraph merge is gone**, so one of the silent exclusions the
>   restructure was meant to catch is already caught;
> * `answer_RTs/features.py` is now a compatibility surface, which resolves the map's open
>   question about what happens to that file when `answer_RTs/` is parked — §5.1's table had
>   no row for it.
>
> Still outstanding from §6.1: the `"."` coercion is centralized *within* each pipeline but
> still lives in `features`-layer code rather than `ingest/readers.py`, and the QA metric
> wrappers still mutate the caller's frame to preserve `all_participants.csv`'s schema. T3.11
> is what finishes that.

### 6.2 The two starting-strategy implementations → one — ✅ **DONE 2026-09-20/25**

This split no longer has to happen; only the **move** does.

`viz/visualisations_strategies.py::build_strategy_dataframe` is gone. There is one
implementation, in `derived/pattern_breaking.py`, and `viz/` only plots:
`build_starting_strategies`, `dominant_strategy_by_participant`, `has_dominant_strategy` (`≥`,
the only operator left) and `build_prefix_completion_map` / `add_completed_strategy_column`.
The scope parameters landed with **T3.21** (`scope_df`, `scope_by`), which is what the old
wording meant by "the scope flag lives in exactly this function".

**Stage C's job here is one rename:** `derived/pattern_breaking.py` →
`features/strategies.py`. Two things found while merging are worth not re-learning: a
**third** copy existed in `visualisations_dominant_eye.py` with an order-dependent tie-break,
and the tie-breaking between the original two never actually diverged — the map's earlier
claim that it did was wrong.

### 6.3 `cross_validation.py` (**48 KB**, was 39) → `modeling/{folds,crossval,evaluate}.py`

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

> ### ✅ Implemented 2026-10-06, with one deliberate deviation
>
> `src/config/datasets.py` now holds a frozen `Dataset` record per dataset plus `DATASETS`,
> `dataset(key)`, `studies()` and `pilots()`. **Adding a dataset is one entry.**
>
> **The deviation: it describes the layout on disk today, not the target in §8.** No data file
> has been moved — that needs agreeing move by move (`CLAUDE.md` hard rule 4). The record is
> what makes those moves cheap when they happen: `root` changes in one place instead of a dozen
> constants changing in four.
>
> **The flat constants are kept and *derived* from the record** rather than deleted. Eighteen of
> them — the four-times-repeated `root` / `all_participants` / `aux` / `model_ready` — are now
> one-line properties of a `Dataset`, so there is a single source of truth, while the ~27
> modules that import them by name keep working. Migrating those call sites to
> `dataset("l1").model_ready` is left for the stage that moves the files; doing it now would be
> a second sweep over the same call sites.
>
> **Proved path-identical:** all **96** path constants were snapshotted before and compared
> after — 0 changed, 0 missing, 0 added. Then 89/93 modules import and 45/45 tables come back
> byte-identical.
>
> **Two notes on the sketch below, now that it is real:**
> - `model_ready` is a property, and it is the one place that knows every dataset's
>   trial-level table is called `L1_model_ready_all_features.csv` — so three of the four claim
>   to be L1 data. That rename is move 7 in §8 and still needs approval.
> - **`has_paragraph` no longer has the job the sketch gives it.** T6.1 already deleted the
>   `include_paragraph` flag threaded through `data_csv_generation`; what survives is
>   `include_paragraph_features` on the trial-level builder (`model_data.py`, 3 sites) plus
>   KnowQA's refusal guard. The field is correct and present; wiring it to replace that
>   remaining flag is a small follow-up, not something this change did.

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

> ### ✅ Implemented 2026-09-20, ahead of its stage
>
> This section landed early, with **T1.3**, because T1.3 had to decide where `save_output`
> writes and inventing a second interim layout would have meant migrating 81 call sites twice.
> **This is a deviation from §11**, which schedules the inversion in Stage E — recorded here
> rather than done silently, per the standing rule about changes to this map.
>
> What exists now: `reports/<analysis>/figures/` and `reports/<analysis>/tables/`, with the
> analysis names from the table below, plus a `manifest.json` per analysis recording which
> function produced each figure. Filenames are key-value tagged
> (`correctness_by_seq_len_threshold__group-hunters__threshold-4`), so a figure is identifiable
> without its folder. `save_output` validates the analysis name against a registry, so a typo
> cannot quietly mint a new top-level folder.
>
> ~~What has *not* happened: the old `reports/plots/` and `reports/report_data/` trees are still
> in place (1,812 files, tracked in git) pending a decision on removing them.~~
> **Superseded 2026-09-27:** both old trees were **deleted in `496f8d0`**, and with them all 329
> zero-byte PNGs. `reports/` now holds only the seven `<analysis>/` folders — 710 figures, zero
> zero-byte, every one with a `tables/` sibling. They remain readable at `496f8d0^`.
> Stage E's other work — `viz/` dissolving into per-analysis `plots.py` — is untouched.
>
> ### ✅ Resolved 2026-09-27 by abbreviation — max checkout root 60 → 101 characters
>
> Diana's call: **keep the names readable words, just shorten them.** So the fix is an
> abbreviation table, not a structural change — the other three remodels below (giving `subdir`
> one meaning, sweeps as tidy tables, a checked path budget) are **not done** and stay on the
> shelf until a need shows up.
>
> `plot_output.ABBREVIATIONS` is the single place a shortening is defined — 36 rows, e.g.
> `all_participants → all_P`, `first_encounter_avg_pupil_size → first_enc_pupil`,
> `correctness_by_trial_mean_dwell_continuous → corr_by_dwell_cont`. It is applied to **every
> path component** — facet values, plot names, table names *and* the `subdir` — which is what
> makes one table sufficient: a subdir is always either a plot name or a facet value.
>
> | | before | after |
> |---|---|---|
> | longest relative path | 199 | **158** |
> | longest absolute path (root = 59) | 259 / 260 | **218** |
> | paths over 240 characters | 136 | **0** |
> | **supportable checkout root** | **60** | **101** |
>
> Three properties the table is built to keep:
> - **Facet keys are never abbreviated** — `group-hunters`, not `g-hunters`. The keys are what
>   make the name readable and they are short already.
> - **`manifest.json` keeps the full, unabbreviated facets**, so the index stays queryable on
>   real column names; only the on-disk label shortens.
> - **Two checks run at import** — the table must be injective (two concepts must never
>   abbreviate to one string) and must not map onto its own keys (which would chain).
>
> Lookup is deliberately **case-insensitive**, which fixed a live inconsistency: the
> all-participants group reached `save_output` as `all participants`, `All participants` *and*
> `all_participants` from three different plot families. Those are now one name. Worth fixing at
> the call sites too — normalising here stops it splitting a group across two filenames, but it
> is papering over a caller disagreement.
>
> 1,435 of 1,608 existing files were renamed to match, and all 713 manifest entries repointed.
> The rename was validated before it ran: for every manifest entry the textual transform had to
> reproduce `build_stem()` exactly — 710/710, zero mismatches — and re-running
> `text_qa_relationship` afterwards overwrote the renamed files rather than creating duplicates.
>
> ---
>
> ### ⚠️ The original measurement, kept as the reasoning (2026-09-27)
>
> The longest absolute path under `reports/` is **259 characters**, against Windows' 260-char
> limit, and **136 paths already exceed 240**. The offender is the pattern, not one bad name:
>
> ```
> reports\attention_allocation\tables\first_encounter_avg_pupil_size\
>   area_label_by_loc_heatmap__group-all_participants__metric-first_encounter_avg_pupil_size__
>   selected-D__questions-included__matrix.csv
> ```
>
> The repo root is 59 characters here. **Anyone who clones to a path even one character deeper
> cannot check the repo out on Windows** without long-path support switched on — which makes
> this a **release blocker (Stage G)**, not a tidiness question, since the public repo will be
> cloned to arbitrary locations. It already bites in practice: reading these files needed the
> `\\?\` extended-length prefix twice while auditing.
>
> Two things make it worse than the raw number suggests: the facet is repeated inside the
> filename *and* in the `subdir/` above it (`first_encounter_avg_pupil_size` appears twice in
> the path above), and §9's own design note calls that redundancy deliberate.
>
> **Not turned into a T-item — this is Diana's call**, and the options differ a lot in cost:
> shorten facet values, drop the `subdir/` level now the filename is self-describing, hash long
> stems, or simply document "enable long paths" as a prerequisite.
>
> **Decided: shorten the values.** The three unused options are recorded here because they are
> the fallbacks if 101 characters ever stops being enough:
>
> | remodel | gain | why it was not needed |
> |---|---|---|
> | give `subdir` one meaning (it is currently a facet value 1,104×, a plot name 210×, something else 195× — and always repeated in the filename) | +39 | abbreviation got there without changing the layout |
> | sweeps as tidy tables — `attention_allocation` is 480 figures and 600 tables from 3 families × 4 groups × 14 metrics × 4 selected × 2 question-settings, the tables averaging 506 bytes at shape 4×4 | +4 to the worst path, but −600 files | still worth doing on its own merits for T4.0: one long table per family is queryable, and `findings.md` would regenerate by a groupby instead of globbing 600 files |
> | a checked path budget in `save_output` | — | would turn this from a convention into an invariant that fails on Diana's machine rather than at a reader's clone |

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

The `papers/` mirroring mechanism survives the inversion, with one change made 2026-09-20:
it now writes under a single root, `papers/correctness_prediction/reports/<analysis>/{figures,tables}/`,
mirroring the local layout exactly instead of adding `figures/` and `report_data/` beside the
drafts. The two conflicting `paper_dirs` conventions from T1.5 are gone with `paper_dirs`
itself — the relative path now comes from one place.

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

> ### Status, 2026-09-27 — the number-moving work is already done
>
> **Stage 0 is complete, and every T-item once scheduled for stages C, D and E has landed
> ahead of its stage** — done as standalone fixes between 2026-09-20 and 2026-09-27, each with
> a `findings.md` change-log entry. Two stages also landed structurally ahead of time: **§9's
> `reports/` inversion shipped with T1.3**, and the old `reports/{plots,report_data}/` trees
> were deleted in `496f8d0`.
>
> **This changes the risk profile of the whole plan.** C and D were called the dangerous stages
> *because numbers move there*. They no longer do: what remains in C, D and E is **file
> movement**, which means the "every number identical" check the map only claimed for stage B
> now applies to every remaining stage. Stage C's warning below is kept for the record but no
> longer describes the work.
>
> What is genuinely untouched: **A** (no `pyproject.toml`, no `environment.yml`, 21 bare-import
> sites, 4 `__init__.py` and all of them vendored), the `config/` + `lib/` *lift* in **B**, and
> **F** and **G** entirely.

| Stage | What | Numbers move? | T-items landing | Status |
|---|---|---|---|---|
| **0** | **Quick wins, no restructuring.** Cheap, independent, high value. | **yes** (T3.1) | T3.1, T1.1, T1.4, verify T2.4 | ✅ **complete** |
| **A** | **Make it a package.** `pyproject.toml`, `__init__.py` throughout, one import convention, delete the `sys.path` hacks (including the one inside `data_paths.py:6`), `environment.yml`. **Nothing moves.** | no | T5.1, T5.2, T5.4 | 🟨 **mostly done 2026-10-06.** ✅ **T5.1** 18 `__init__.py` added · ✅ **T5.2** one import convention, 21 lines converted, `src/` no longer on the path · ✅ **T5.4** `environment.yml`. **Deliberately not done:** `pyproject.toml` / `pip install -e .` (Diana, 2026-10-06 — not yet), so the remaining `sys.path` entries stay until it is. Verified by re-running `text_qa_relationship`: **45/45 tables byte-identical** |
| **B** | **`config/` + `lib/`.** Lift generic primitives; kill the two duplicate `wilson_ci`s and the two extra save paths; split `viz_helpers.py`; build the dataset registry; fix the stale constants; migrate the hardcoded `"../reports/..."` literals. | no | T1.5, T5.3, part of T3.20 | ✅ **DONE 2026-10-06.** `src/config/` (columns · datasets · outputs) and `src/lib/` (stats/proportions · plotting/annotate) exist; the dataset registry is in (§7); the three dead output constants resolve; the five feature-set JSONs moved to a new top-level `configs/` (inputs, not results). **96/96 path constants proved unchanged, 89/93 modules import, 45/45 tables byte-identical.** Not done, by decision: migrating ~27 call sites off the flat constant names, and the §8 data moves |
| **C** | **`ingest/` + `features/`.** The big correctness stage: separate paragraph from QA prep, unify the two per-area metric implementations, unify the two starting-strategy implementations, land the scope flag, assert every join. | ~~**yes**~~ **no longer** | T6.1, T1.7, T1.6, T3.6, T3.14, T3.20, T3.21, T3.17, T3.7, T3.5 | ✅ **DONE 2026-10-06** (proposal: `docs/decisions/2026-10-06-stage-c-proposal.md`). Every listed T-item was already done going in, so this was the moves only. `data_prep/` and `derived/external/` are **gone**; `src/ingest/` (readers · geometry · build · clicks · knowqa), `src/features/` (10 modules + `paragraph/`), `src/vendor/eyebench/` and `src/explorations/` exist. The `Dataset` record now carries `skip_base_features` and `has_paragraph`. **Verified: 92/96 modules import (same 4 parked), KnowQA pipeline bit-identical (33,830×420, max diff 0.0), 45/45 `text_qa` tables byte-identical to HEAD, no `src.vendor` in `sys.modules` from any live import.** **T3.24 declined** (Diana, 2026-10-06). The one layering exception it left was **closed 2026-10-06** as step 1 of stage D — see §5.4. |
| **D** | **Unify feature generation, then `modeling/`.** One registry for every feature at every grain (`row` · `group` · `trial`), the produced-column map **observed** rather than declared, `needs` on each entry so the raw-column whitelist and a registry order-check both fall out of one declaration, the 8 `include_*` booleans become `Dataset.skip_features`, and the two overlapping feature-set modules merge. Then extract CV, evaluation, model wrappers and inference. | ~~**yes**~~ **no longer** | T3.3, T3.9, T3.10, T5.11, T2.4 | ✅ **DONE 2026-10-07.** T3.3 ✅ T3.9 ✅ T3.10 ✅ T2.4 ✅. **Steps 1, 2b and 6 done 2026-10-06.** Step 1 = the §5.4 layering fix. Step 6 = one fixation-to-area rule (`coalesce(exact, nearest)`, both screens) and run-based RT off the button-clicks table — **run out of order at Diana's instruction, and it moved numbers**: L1 answer +0.82%, L1 paragraph +1.77%, KnowQA 10 of 155 columns. It uncovered **two bugs**, one of them (`.agg("last")` skipping the NaN that marks a trial's final run) present in the paragraph RT since T6.1. Logged in `findings.md`. Step 2b = the `Screen` record: the paragraph and answer screens now share one metric-merge, one pupil prep, one RT assembly and one `RT_*`→`TimeSinceOffset_*` rename, **numbers bit-identical on all three artifacts**. **Proposal: `docs/decisions/2026-10-06-stage-d-proposal.md`.** ✅ **ALL STEPS DONE 2026-10-07** — 2, 2c, 3 and 5 on 10-07; step 4 audited and **dropped** (its premise did not hold); step 7 last. L1 prep was rerun on 10-07, so step 6's RT change is propagated and the `text_qa_relationship` RT tables were regenerated (19 files, reading-times only — the dwell-proportion maps are untouched, which is the predicted blast radius). **Step 7 moved no numbers and was verified so:** all 39 shared feature-set constants and all six `get_*_feature_cols` predicates return byte-identical results on the real L1 frame (19,436 × 219), all 20 cross-validation functions relocated with none missing, and the fold chain runs end to end on real data. `modeling/` imports nothing from `viz/` or `predictive_modeling/`; `features/` imports nothing from `ingest/`. 130/131 modules import (the one failure is parked `pymer4`). **The three open placements were ruled on 2026-10-07** and are in the proposal's table: the cache pair went to `features/build.py` renamed `save_model_ready` / `load_model_ready` (so `model_data.py` is deleted); the three never-called `make_*_dataset` builders were deleted outright; and `plot_confusion_heatmap` is agreed for `lib/plotting/` but **parked in `viz/` until `plot_output.py` moves there**, because `lib/` may import nothing outside `lib/`. `plot_output.py` moved to `lib/plotting/output.py` the same day, verified byte-identical on a `text_qa_relationship` regeneration, which let `plot_confusion_heatmap` land in `lib/plotting/confusion.py` as agreed. **Still open:** T5.11 and new **T3.22** — nothing else. (`ANALYSES` / `ABBREVIATIONS` staying in `lib/` is a ruled exception, §4.) |
| **E** | **`analyses/` + `explorations/` + `reports/` inversion.** `viz/` dissolves into per-analysis `plots.py`; one saving framework everywhere; re-run what needs re-running. | figures only | T1.3, T4.0, T4.1, T4.2, T3.18, T3.19 | ✅ **DONE 2026-10-07** (proposal: `docs/decisions/2026-10-07-stage-e-proposal.md`). `viz/`, `stats/`, `derived/` and `predictive_modeling/` are **gone**; `analyses/` holds 7 folders / 48 modules and `explorations/` 8 / 30. The 1,717-line `answer_correctness_viz.py` split into 5 figure-family modules plus `explorations/mixed_models/plots.py`. **Verified: 132/133 modules import** (the one failure is parked `pymer4`), **`text_qa_relationship` regenerated byte-identical** — every table and figure, with only `manifest.json`'s `produced_by` changing, which is the field that records the module path and so is the proof the move happened. **Six of seven layers clean.** ⚠️ **One known violation:** `explorations/{feature_search,mixed_models}` import `analyses/correctness_prediction/run.py` (6 imports) — they want its *run* half and get its *plot* half with it. The fix is to split run from plot, a signature change on the paper's main run path, flagged rather than done. **Still pushed:** T4.0 and T4.2 (Diana's manual pass), T3.19 |
| **F** | **Entry points.** `scripts/`, thin notebooks, notebook-only logic lifted into the package. | no | T5.6, T5.10 | ✅ **DONE 2026-10-07** (proposal: `docs/decisions/2026-10-07-stage-f-proposal.md`). **`scripts/`** holds `build_dataset.py` (the build order is now code, not documentation — T5.6), `run_analysis.py` (validated against `ANALYSES`, 5 analyses × 15 runners) and `make_paper_figures.py` (**a skeleton by decision** — 5 of 11 figure groups wired, the other 6 listed as TODO with their reasons rather than silently skipped). **`notebooks/`** split four ways — `drivers/` 9 · `builders/` 6 · `exploration/` 10 · `experiment/` 3 — each with a README saying what the folder is for; nothing left loose at the top. **`src/experiment/`** is new and is the answer to *where is the experiment source*: 13 functions lifted out of `experiment_builder/`, which is gone. Its three notebooks turned out to be three different things — generation, pilot ingest, and a `knowledge_regimes` driver — and were filed accordingly. **`analyses/answer_rt_comparison/`** lifts the correct-vs-distractor RT asymmetry out of a presentation notebook and **persists its numbers for the first time**; that was the last `ANALYSES` key without a folder bar the one investigation. **T5.10 was already done.** All 14 notebooks that resolved the repo root by `Path.cwd().parent` now walk up to the folder containing `src/`, so they work from any depth — several were already broken by the earlier subfolders. **Verified: 137/138 import** (the one failure is parked `pymer4`), 28/28 notebooks valid JSON, and `run_analysis.py --name last_visitation` reproduced its tables **byte-identical** to the notebook path |
| **G** | **Data + release.** The `data/` moves (individually approved), README, run-from-scratch verification. | no | T5.5, T5.7, T5.8, T5.9, §8 | ✅ **DONE 2026-10-07.** All 11 §8 moves executed with `os.rename` (atomic, same filesystem — no copying, so no partial-write risk), **verified file-count and byte-total identical before and after: 132 files, 10,691,871,278 bytes, size multiset unchanged.** `data/` is gitignored, so there was no git safety net; the manifest was the safety net. **The three `data_raw` symlinks were never touched** and still resolve (14 + 7 + 7 entries). Move 7 — the one the map calls the correctness hazard — landed: `model_ready.csv` under each dataset, so three files stopped claiming to be L1. The registry cost exactly what it promised: **four `root=` lines and three properties**, plus one hardcoded filename in `ingest/knowqa.py` that would otherwise have kept writing the old name. **Nine stale constants resolved**; the five paths that still do not exist are output destinations not yet written, not dead pointers. **T5.9 closed** — the EyeBench cache now gets a `.meta.json` recording `fix_ptb_pos_double_mapping`, and the existing cache was back-labelled after determining empirically (68 of 70 `ptb_pos_*` columns non-zero) that it was built with the fix on. ⚠️ **T5.5 and T5.7 are NOT done and the stage is green without them.** A `README.md` was written covering the `L1` = native-speaker trap, the OneStop placement instructions and the build order, then **deleted the same day** — Diana, 2026-10-07: *"we have not yet begun documenting things."* Writing a reader-facing release document before deciding what the release says was premature, so the two items go back to open and belong to whatever documentation pass comes after the paper settles. **Verified: 137/138 import, all six layering invariants clean, and `run_analysis.py --name last_visitation` byte-identical after the move** |

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
