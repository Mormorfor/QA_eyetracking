# Stage E — `analyses/` + `explorations/`

**Status: proposed 2026-10-07. Not started.** Stage D closed the same day; this is the last
structural stage. The plan of record is `restructure-map.md` §11; this fills in the detail for
one row of its table, the way the stage C and stage D proposals did.

**What is already done inside this stage:** the `reports/` inversion (map §9), **T1.3** (one
`save_output`, `tables=` required), **T4.1** (`RT_correlations` persists its numbers) and
**T3.18**. What remains is the code tree.

---

## What this stage is for

Three things currently share a folder with things they are unrelated to:

1. **`viz/` is named after plots, not after questions.** 15 modules, 4,540 lines, each named
   `visualisations_<something>` — and the analysis that computes the numbers behind each one
   lives in `derived/`, `stats/` or `predictive_modeling/`. This is fault 1.1 in the map, and it
   is the distance that let the two starting-strategy implementations drift apart.
2. **`predictive_modeling/` holds paper code and parked code side by side.** 30 files: the
   headline model and its person-variance analysis sit next to `answer_loc/`, which does not
   import, and `clusters/`, which was superseded.
3. **Nothing tells a reader which is which.** The release ships both. Today the only way to know
   that `answer_loc/` is not a result is to read `docs/`.

After this stage the path says it: `analyses/` is paper code, `explorations/` is kept-but-parked,
and each folder owns the whole of one question — compute, stats, plots, report.

---

## The diagnosis, measured

### 1. The destinations are already decided, and already in use

This is the thing that makes stage E much more mechanical than its size suggests.
`lib/plotting/output.py::ANALYSES` already names **17 analyses**, including the
`explorations/…` ones, and `reports/` has been laid out that way since September. So the target
folder names are not a design question — they are a lookup.

### 2. Most modules already declare where they belong

Scanning every `save_output(analysis=…)` and `collect_tables(…)` call: **27 modules name their
own destination in code.** `visualisations_correctness_measures.py` makes 28 — it declares
`ANALYSIS = "correctness_associations"` as a module constant, which the call-site scan misses.

That leaves **four** modules in `viz/` with no declared destination, and each has an answer:

| module | lines | evidence | destination |
|---|---|---|---|
| `visualisations_area_significance_heatmaps.py` | 266 | imported only by `stats/mixed_area_comparisons.py`, which writes `attention_allocation` | `analyses/attention_allocation/plots.py` |
| `visualisations.py` | 62 | a pure re-export façade. **Correction, 2026-10-07:** an earlier draft of this table said "imported by nothing" — wrong, the grep pattern missed `from src.viz import visualisations as V`. It has **three notebook importers** in `vizes for purpouse/`, 16 `V.*` calls between them | **delete** (map §5.1), with those 16 calls rewritten as direct imports — see the order of work |
| `viz_helpers.py` | 48 | one function (`correctness_tables`) **plus a re-export façade** for `lib.plotting.annotate` and `lib.stats.proportions`, whose own comment says it stands only "while stage B is in flight" | see §"Files that split" |
| `sequence_visualisations.py` | 511 | **no `src` importer**; only `notebooks/vizes for purpouse/sequence_visualisations.ipynb` | **open question 1** |

### 3. `answer_RTs/` has no live dependency left — this was not true a month ago

`research-context.md` parks `answer_RTs/` but `todo.md` T6.1(a) warned that
`answer_RTs/features.py` was load-bearing for two live consumers. **That is no longer the case.**
T6.1 landed on 2026-09-23 and `features.py` is now a **45-line compatibility shim** whose own
header says "THIS MODULE NO LONGER BUILDS ANYTHING" — the extraction is in
`features/paragraph/spans.py`.

So the only live thread is `stats/RT_correlations/proportions.py:109`, which imports the
re-exports *through* the shim. Repoint it at `features/paragraph/spans.py` and **all four
`answer_RTs` files become parked with nothing depending on them.**

One loose end: `load_paragraph_features` is *defined* in the shim rather than re-exported, and
both callers want it. It is a three-line cache reader and belongs beside `save_paragraph_features`
in `features/paragraph/spans.py`.

### 4. One analysis is still split across three layers

`correctness_associations` is computed in `derived/correctness_measures.py`, tested in
`stats/correctness_measures_tests.py` and plotted in `viz/visualisations_correctness_measures.py`
(1,025 lines). After this stage those are `compute.py`, `stats.py` and `plots.py` in one folder.
It is the clearest illustration of what the stage is for.

---

## The proposal — where every file goes

### Into `analyses/`

| today | → |
|---|---|
| `viz/visualisations_area_bars.py`, `_area_matrices.py`, `_area_significance_heatmaps.py` | `analyses/attention_allocation/plots.py` |
| `stats/mixed_area_comparisons.py` | `analyses/attention_allocation/stats.py` |
| `stats/preference_correctness_tests.py` | `analyses/attention_allocation/stats.py` |
| `viz/visualisations_strategies.py`, `_simplified_visits.py` | `analyses/scan_strategies/plots.py` |
| `viz/visualisations_dominant_eye.py` | `analyses/scan_strategies/dominant_eye.py` |
| `viz/visualisations_last_label.py` | `analyses/last_visitation/plots.py` |
| `viz/visualisations_time_segments.py` | `analyses/time_course/plots.py` |
| `derived/correctness_measures.py` | `analyses/correctness_associations/compute.py` |
| `stats/correctness_measures_tests.py` | `analyses/correctness_associations/stats.py` |
| `viz/visualisations_correctness_measures.py` (1,025) | `analyses/correctness_associations/plots.py` |
| `viz/visualisations_preference_correctness.py` | `analyses/correctness_associations/plots.py` |
| `stats/RT_correlations/**` (10 modules) | `analyses/text_qa_relationship/**` — shape already correct, a rename |
| `answer_correctness/run_model_bundles.py`, `run_report_analysis.py` | `analyses/correctness_prediction/run.py` · `report.py` |
| `answer_correctness/participant_level.py` | `analyses/correctness_prediction/participant_level.py` |
| `answer_correctness/answer_correctness_viz.py` (1,717) | `analyses/correctness_prediction/plots/` — **split, see below** |
| `viz/visualisations_cross_validation.py` | `analyses/correctness_prediction/plots/cv_comparison.py` |
| `answer_correctness/person_variance/**` (7) | `analyses/correctness_prediction/person_variance/**` |
| `answer_correctness/knowledge_regimes_analysis/**` (3) | `analyses/correctness_prediction/knowledge_regimes/**` |

### Into `explorations/`

| today | → | state it arrives in |
|---|---|---|
| `answer_RTs/**` (4) | `explorations/answer_reading_times/` | runs; `run.py` has no caller |
| `answer_loc/**` (5) | `explorations/answer_location/` | **does not import** (T2.2) — filed broken, labelled |
| `answer_correctness/clusters/**` (5) | `explorations/participant_clustering/` | one module is dead code |
| `answer_correctness/generate_column_options.py` (1,003) | `explorations/feature_search/` | runs |
| `answer_correctness/unlikely_analysis.py` | `explorations/unlikely_analysis/` | stale, per `research-context.md` §3.4 |
| *(already there)* | `explorations/text_answer_effects/` | **does not import** (T2.6) |

**T2.2 and T2.6 are filed as-is, not repaired.** That is the standing decision
(`todo_after_restructure.md` §1): the restructure's job is to make "this does not back the paper"
visible from the path; fixing them is work for whenever those strands are picked up.

---

## Files that split

### `answer_correctness_viz.py`, 1,717 lines → `analyses/correctness_prediction/plots/`

Map §6.4 asks for a split "by figure family". Reading the module, the families are:

| new module | functions | lines |
|---|---|---|
| `results.py` | `show_correctness_model_results`, `correctness_results_to_summary_df` | ~170 |
| `coefficients.py` | `plot_coef_summary_barh`, `build_coef_comparison_table` | ~215 |
| `probabilities.py` | `plot_predicted_probability_hist` | ~130 |
| `feature_ranks.py` | `plot_top_abs_coef_feature_frequency_across_participants`, `compute_feature_avg_rank_across_participants`, `plot_top_features_by_best_avg_rank`, `plot_feature_correlation_heatmap` | ~420 |
| `run_comparison.py` | `_infer_model_family`, `collect_correctness_run_reports`, `plot_correctness_run_comparison`, `_format_comparison_labels`, `plot_cv_model_comparison_staged` | ~630 |
| `cv_comparison.py` | the three from `viz/visualisations_cross_validation.py` | ~190 |

**One family does not belong here:** `plot_random_effects_barh`,
`plot_random_effects_distribution` and `summarize_random_effects` (~120 lines) only make sense
for the Julia/R mixed-model backends, which are future directions. They go to
`explorations/` — **open question 2**.

### `viz/viz_helpers.py` → nothing

Map §5.2's split table for this file is **stale**: the generic half (`p_to_stars`,
`add_wilson_errorbars_and_ns`, `add_significance_bracket`, `barplot_accuracy`) already left
during T1.3 and stage B. Two things remain, and they are different jobs:

* **A re-export façade** for the four names that left, kept so the plotting modules could keep
  importing from one place. Its own comment says it stands only "while stage B is in flight;
  the imports get repointed when `viz/` dissolves in stage E" — i.e. this stage is the one that
  was supposed to remove it. **Repoint its three importers at `lib/` directly and delete it.**
* **`correctness_tables`**, 8 lines, which assembles the `tables=` payload (`summary`, and
  `fisher` when a test ran) shared by the correctness-association plots and by
  `knowledge_regimes/descriptives.py`. Two analyses use it, so it goes **down**, not sideways:
  `lib/plotting/tables.py`. It also carries an unused `from src.config import columns as C` —
  drop that on the way.

---

## Decided 2026-10-07

| | ruling |
|---|---|
| **1. `sequence_visualisations.py`** | **`explorations/`.** Diana: *"exploration is good. Im pretty sure there is a notebook running it, but its still fine"* — and there is: `notebooks/vizes for purpouse/sequence_visualisations.ipynb`. A notebook driver is not a live-path caller, which is the same evidence that parks everything else here |
| **2. mixed-effects → its own folder** | **`explorations/mixed_models/`**, holding *everything to do with fitting a mixed-effects model*. Diana: *"there should probably be some mixed effects modeling folder somewhere with everything that has to do with that, **not to be mistaken for mixed effects difference statistics**."* That distinction is the whole point of the folder — see below |
| **3. package `__init__` re-exports** | **Direct imports.** No new façades; the same reason `viz/visualisations.py` is being deleted |
| **4. `correctness_prediction` shape** | **Keep it together.** Diana: *"Lets keep it all together for now, can separate later if needed"* — so `person_variance/` and `knowledge_regimes/` stay children, and my "is Study 2 a sibling?" worry is answered: not now |

### The mixed-effects line, because the names collide

Two unrelated things in this project are called "mixed effects", and only one of them is parked.

| | what it is | where it goes |
|---|---|---|
| **mixed-effects *modeling*** | fitting the correctness outcome with a mixed model instead of the logistic regression — the Julia and R backends, their random-effects output, and the figures that read it | **`explorations/mixed_models/`** — a future direction (`research-context.md` §6: *"mixed effects have their own problems and were judged not important enough to pursue for this paper"*) |
| **mixed-effects *difference statistics*** | `stats/mixed_area_comparisons.py` — statsmodels `MixedLM` testing whether the five screen areas differ, participant as a random effect | **`analyses/attention_allocation/stats.py`** — **paper code**, settled 2026-09-05; it backs the *Attention allocation* Results subsection |

**Do not file the second with the first.** They share a method name and nothing else.

What moves into `explorations/mixed_models/`:

| from | → |
|---|---|
| `modeling/models/julia_model.py`, `models/glmer_r_model.py` | `explorations/mixed_models/models.py` |
| `run_model_bundles.py:646` `run_full_features_correctness_julia_glmer_bundle` and `:794` `..._fit_all` | `explorations/mixed_models/run.py` |
| `modeling/evaluate.py:265` `fit_julia_mixed_model_on_prepared_full_data` | `explorations/mixed_models/run.py` — it is Julia-only by construction (T2.3) |
| `answer_correctness_viz.py` — `plot_random_effects_barh`, `plot_random_effects_distribution`, `summarize_random_effects` (~120 lines) | `explorations/mixed_models/plots.py` |

> **This fixes a documented wart.** `run_model_bundles.py` opens with: *"Importing this module
> pulls in the Julia backend, so it needs a working juliacall toolchain even when only the
> logistic regression is wanted."* That is a top-level `from …models.julia_model import …` at
> `:36`. Once the Julia bundles live in `explorations/`, the paper's run path imports no Julia at
> all. `glmer_r_model` is already commented out at `:32`, so only one import actually goes.
>
> `explorations/mixed_models` is **not** in `lib/plotting/output.py::ANALYSES` yet and has to be
> added, or its `save_output` calls will fail the registry check — which is the registry working.

> **`explorations/text_answer_effects/` stays separate.** It also uses a mixed model (`Lmer`),
> but its identity is the *question* — text spans against answer dwell — and it is parked because
> `RT_correlations` superseded it, not because of the method. Folding it into `mixed_models/`
> would lose that.

## Order of work

Each step ends with the acceptance check and is not finished until it passes.

| # | step | why this order |
|---|---|---|
| 1 | **Cut the `answer_RTs` thread.** Move `load_paragraph_features` to `features/paragraph/spans.py`; repoint `RT_correlations/proportions.py`; delete the shim | makes `answer_RTs/` parked with nothing depending on it, so step 5 is a pure move |
| 2 | **Dissolve `viz_helpers.py`:** repoint its three `src` importers at `lib/` directly, move `correctness_tables` to `lib/plotting/tables.py`, delete the file | it is already a half-dissolved façade; clearing it first means nothing moves *through* it later |
| 3 | **`analyses/` for the five single-question folders** — `attention_allocation`, `scan_strategies`, `last_visitation`, `time_course`, `correctness_associations` | the simple ones; each is 2–4 files and a rename |
| 4 | **`analyses/text_qa_relationship/`** — `stats/RT_correlations/` moves whole | shape is already right; it is a directory rename plus imports |
| 5 | **`explorations/`** — all six parked strands | no live code depends on any of them after step 1 |
| 6 | **`analyses/correctness_prediction/`** — including the 1,717-line split | largest and most judgment; do it when everything around it has settled |
| 7 | **One notebook pass.** Repoint every notebook import at the new locations, delete `viz/visualisations.py`, and rewrite its 16 `V.*` calls as direct imports | doing this once at the end rather than per-step; a notebook touched six times is six chances to mangle JSON. The façade stays alive until here precisely so the notebooks keep working while the modules move |
| 8 | **Delete the empty shells** — `viz/`, `stats/`, `derived/`, `predictive_modeling/` | they should be empty by here; if one is not, something was missed |

---

## Acceptance test

Stage E moves no numbers, so the bar is "prove nothing moved":

1. **Imports:** 127/128, with the *same* failures — `explorations/text_answer_effects`
   (`pymer4`) and, once filed, `explorations/answer_location` (T2.2). Any third failure is a
   regression.
2. **Layering, now with `analyses/` and `explorations/` in it:** nothing imports `analyses/`;
   `explorations/` imports nothing from `analyses/`; both may import `modeling/`, `features/`,
   `lib/`, `config/`. `lib/` still imports nothing outside `lib/`.
3. **`reports/` is byte-identical after regeneration.** Use `git diff` on `reports/`, not a
   scratch snapshot — those tables are tracked, and a filename-keyed snapshot silently reports
   0/N when an output gets renamed. That has already happened once. The per-analysis check:
   regenerate `text_qa_relationship` (53 files) and `attention_allocation`, and diff.
4. **`manifest.json` `produced_by` changes, and that is expected** — it records the module path,
   which is the thing this stage changes. It is the one field allowed to differ, and it is also
   the proof the move happened.
5. **Every one of the 17 `ANALYSES` keys has a folder** under `analyses/` or `explorations/`,
   and every folder under those has a key. Today nothing checks this; after stage E it is a
   one-line test.
6. **No shells left:** `viz/`, `stats/`, `derived/`, `predictive_modeling/` do not exist.

---

## Explicitly not in this stage

- **No repair of parked code.** T2.2 and T2.6 arrive broken and labelled.
- **No figure regeneration beyond the acceptance check.** T4.2 (re-running everything) and T4.0
  (rebuilding `findings.md` from saved CSVs) are pushed to Diana's manual pass.
- **No T3.19.** The `last_answer_area_visited_lbl` investigation gets *easier* once the three
  last-visitation variants sit together, but it is an investigation, not a move.
- **No `pyproject.toml`** — deferred by decision, and stage F/G material.

---

## What I am least sure about

1. **That `correctness_prediction` really is one analysis.** It is the only folder with
   subfolders (`person_variance/`, `knowledge_regimes/`), and after the split it will hold ~20
   files against 2–4 for every other analysis. That may be honest — it is the paper's core — or
   it may mean `knowledge_regimes` (Study 2) deserves to be a sibling rather than a child. The
   `ANALYSES` registry already commits to child; I am flagging that it was committed to before
   the folder got this big.
2. **That `attention_allocation` wants two stats modules merged.** `mixed_area_comparisons.py`
   and `preference_correctness_tests.py` both land in `stats.py`. They are different tests on the
   same areas; whether that is one module or two is a judgment I would rather make with the files
   open than now.
3. ~~**The `viz_helpers` → `lib/plotting/tables.py` call.**~~ **Checked 2026-10-07, and it
   holds.** `correctness_tables` names only `"summary"` and `"fisher"` — output-table keys, not
   eye-tracking columns — so it passes the §4 test. Its one project import (`config.columns`) is
   unused. Left here as a record that it was verified rather than assumed.
