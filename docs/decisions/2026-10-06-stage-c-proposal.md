# Stage C proposal — `ingest/` + `features/`

**Status: ✅ executed 2026-10-06.** Proposed, approved, and carried out the same day. The body
below is the plan as agreed; **what actually happened, and the three places it diverged, is at
the end under "Outcome".** Everything was measured against the code on 2026-10-06.

---

## What this stage is for, in one paragraph

Two folders today are named after *when* code runs rather than *what it knows*: `data_prep/`
and `derived/`. Stage C replaces them with two named after what they know:

- **`ingest/`** — knows what an EyeLink export looks like. Report formats, the `"."` sentinel,
  trial and session identity, button-click reconstruction, KnowQA's composite trial ids.
- **`features/`** — knows what a measurement is. Dwell proportion, skip rate, reading time,
  scanning strategy, pupil z-score.

**The boundary is the tidy word-level table.** Before it: get data out of the tracker's format.
After it: turn it into quantities. Making that line structural is what stops a feature builder
reaching back to open a raw report — which is how paragraph extraction ended up living inside a
*prediction* module for a year.

**It also makes a true thing visible.** Diana's goal — *"for L1 you run prep with this
configuration, for KnowQA with that"* — is **already how the code works**, and you cannot tell.
`know_qa_dataprep.py` calls the shared `data_csv_generation.main()` (:1003) and
`save_all_features` (:1072); what is genuinely KnowQA-specific is the *front end* only. Buried
in a 51 KB file in a folder called `data_prep`, it reads like two pipelines. It is one pipeline
with two front ends, and the folder names should say so.

---

## The risk profile is not what the map says

`restructure-map.md` §11 calls C "the big correctness stage" and lists ten T-items landing in
it: T6.1, T1.7, T1.6, T3.6, T3.14, T3.20, T3.21, T3.17, T3.7, T3.5.

**All ten are already done.** `derived/area_metrics.py`, `derived/paragraph_prep.py` and
`src/checks.py` exist. So nothing in this stage is supposed to change a number, and the
acceptance test is the same one Stages A and B passed: **every number identical**.

---

## Where every file goes

### `data_prep/` → mostly `ingest/`

| today | size | → | note |
|---|---|---|---|
| `data_csv_generation.py` | 64 KB, 35 defs | **split six ways** — see below | the main job |
| `know_qa_dataprep.py` | 51 KB, 25 defs | `ingest/knowqa.py` | front end stays, shared calls stay shared |
| `button_clicks_processing.py` | 17 KB | `ingest/clicks.py` | ✅ Diana 2026-10-06 |
| `answers_paragraphs_csv.py` | 8 KB | `explorations/text_answer_effects/` | ✅ Diana 2026-10-06 — see "retirements" |
| `pupil_size_standrtization.py` | <1 KB | `archive/` | ✅ Diana 2026-10-06 — dead, nothing imports it |

### `derived/` → `features/`

| today | → |
|---|---|
| `area_metrics.py` | `features/area_metrics.py` *(rename only — already the shared implementation)* |
| `reading_times.py` | `features/reading_times.py` |
| `pupil_norm.py` | `features/pupil.py` |
| `pattern_breaking.py` | `features/strategies.py` |
| `select_confirm_last.py` | `features/last_visited.py` |
| `preference_matching.py` | `features/preference.py` |
| `paragraph_prep.py` | `features/paragraph/spans.py` |
| `external/EyeBench/runner.py` | `features/paragraph/eyebench.py` |
| `correctness_measures.py` | **split — see the correction below** |

### Vendored code — ✅ **Diana 2026-10-06: one place, and it must not break her code**

> *"EyeBench stuff is unimportant. What I wanted from it didn't end up working for me anyway.
> If it breaks — it breaks, keep it all in one place, try to avoid IT breaking MY code, and if
> I ever decide to use it again I'll get to it."*

**This overrides map §5.1**, which splits the vendored extractor from our wrapper and sends
them to two different folders (`vendor/eyebench/` and `features/paragraph/eyebench.py`).
Everything EyeBench goes to **`src/vendor/eyebench/`** instead — the vendored files *and*
`runner.py`, in one place.

**The isolation already exists, and the job is to keep it.** Measured 2026-10-06: the only
live consumer is `answer_RTs/model_data.py:319`, and it imports inside a function, not at
module scope. So a vendored breakage cannot take down anything else today. Two rules follow:

- the import stays **lazy** — nothing on the live path may import `vendor/` at module scope;
- the move does not try to keep the vendored code *working*, only *contained*. If the
  `importlib` by-path load or the `src.configs` alias breaks, that is acceptable per the ruling
  above — but it must fail inside `vendor/`, not leak out.

A check for the acceptance test: import every live module and assert `vendor` never appears in
`sys.modules`.

### How `data_csv_generation.py` splits

Measured, 35 functions — the buckets below sum to exactly 35, with no function in two places:

| → | functions | n |
|---|---|---|
| `ingest/readers.py` | `load_raw_answers_data`, `load_raw_paragraphs_data` | 2 |
| `ingest/build.py` | `main`, `_process`, `_save`, `_save_splits`, the two `_attach_*` | 6 |
| `ingest/geometry.py` | `assign_area_by_geometry`, `_reconcile_area_with_geometry` | 2 |
| `features/build.py` | nine `add_*` base features + `add_base_features` + `generate_new_row_features` + the two `resolve_*` | 12 |
| `features/area_metrics.py` | the five `create_*` metric wrappers | 5 |
| `features/pupil.py` | `add_zscored_pupil_columns`, `create_mean_pupil_size_metrics`, `create_first_encounter_pupil_size` | 3 |
| `features/sequences.py` | `create_fixation_sequence_tags`, `create_simplified_fixation_tags`, `create_simplified_visit_counts` | 3 |
| `features/last_visited.py` | `create_last_area_and_location_visited` | 1 |
| `features/scope.py` | `split_hunters_and_gatherers` | 1 |

---

## Two corrections to the map, found while scoping

**1. `correctness_measures.py` is a mixed file, and §5.1 sends it to the wrong place.**

The map routes it wholesale to `analyses/correctness_associations/compute.py`. It cannot go
wholesale, because `model_data.py` imports four of its functions as *features*:

| half | functions | used by | belongs in |
|---|---|---|---|
| **feature** | `has_back_and_forth_xyx`, `has_back_and_forth_xyxy`, `longest_alternating_answer_run`, `compute_trial_mean_dwell_per_word` (+ the two sequence parsers) | `model_data.py` | `features/sequences.py` |
| **analysis** | `summarize_binary_by_group`, three `build_trial_df_for_*`, three `compute_*_summary` | viz + stats | `analyses/correctness_associations/` at **stage E** |

Same shape as `viz_helpers.py`. **Proposal: split it in C**, feature half into
`features/sequences.py`, analysis half left where it is until stage E has somewhere to put it.

**2. `has_paragraph` had no flag left to replace — so it gets a bigger job.** §7's sketch says
it replaces `include_paragraph`, threaded through four call layers; T6.1 already deleted that.
What survives is `include_paragraph_features` (`model_data.py`, 3 sites) plus KnowQA's refusal
guard.

✅ **Diana 2026-10-06: do it all — the dataset record should carry the pipeline configuration.**
*"Maybe the dataset should have a function list as default, that you can edit if you want to,
but should tend to stick to as is."* So the `Dataset` gains two things:

| field | what it declares |
|---|---|
| `has_paragraph` | wired to `include_paragraph_features`, so a dataset with no paragraph screen cannot be asked for paragraph features by accident |
| `skip_base_features` | `{name: why}` — the base features this dataset does **not** run |

**It declares the *difference*, not the whole list.** Three reasons. A full list means adding a
registry function requires editing all four dataset entries, which is the per-dataset
repetition the registry was built to remove. The default — run everything — is the behaviour to
"stick to as is". And `config/` must not import `features/`, so the record holds *names*, not
functions; the names are validated against the registry at run time, where a typo fails loudly.

Today KnowQA builds this by filtering the registry inline (`know_qa_dataprep.py:997-1001`).
Two exclusions, and **they are not the same kind**:

| exclusion | kind | goes where |
|---|---|---|
| `add_answer_text_columns` | **dataset-intrinsic** — answer_A..D come from the Stage 0 rename, so there is nothing to recompute | `Dataset.skip_base_features` |
| `add_zscored_pupil_columns` | **run-option dependent** — only skipped when `pupil_norm_unit="session"`, because the clean step already z-scored | stays conditional in the runner |

Collapsing the second into the record would be wrong: it is a property of how you chose to run,
not of the dataset. Keeping them apart is the point of the field.

---

## Order of work

Each step ends with the same two checks, and a step is not finished until both pass.

| # | step | why this order |
|---|---|---|
| 1 | `features/` — move the eight clean `derived/` modules | pure renames, biggest confidence gain per unit of risk |
| 2 | split `correctness_measures.py`; move `split_hunters_and_gatherers` to `features/scope.py` | the two mixed files, done while the tree is still simple |
| 3 | `ingest/` — `readers`, `clicks`, `geometry`, `build` out of `data_csv_generation.py` | the big split; `features/` already stable underneath it |
| 4 | `ingest/knowqa.py` — move `know_qa_dataprep.py`, keep its shared calls shared | last, because it depends on everything above having landed |
| 5 | `vendor/eyebench/` + `features/paragraph/eyebench.py` | self-contained; the `importlib` path must follow |
| 6 | retirements: `answers_paragraphs_csv.py` → `explorations/`, `pupil_size_standrtization.py` → `archive/` | after the live tree is settled |

**Blast radius:** 3 python modules + 3 notebooks import `src.data_prep`; 14 + 3 import
`src.derived`. Small, because most of the tree reaches these through `model_data`.

---

## Acceptance test

1. **Imports:** 89 of 93 modules import, with the same four known-parked failures
   (`answer_loc` ×3, `mixed_text_answer_effects`). No new failure.
2. **Numbers:** re-run `text_qa_relationship` → 45/45 tables byte-identical. Check the exit
   code and the file timestamps, not just the comparison — a run that crashes writes nothing
   and then "identical" means nothing. *(This has already caught one false pass.)*
3. **Name collisions:** re-run the stdlib / third-party scan. `features`, `ingest` and `vendor`
   are clear today; re-check after creating them.
4. **The layering rule:** `features/` must not import from `ingest/`. Checkable with a grep,
   and worth checking, because it is the whole point of the stage.

---

## Explicitly not in this stage

- **No data files move.** `restructure-map.md` §8's moves need agreeing one at a time
  (`CLAUDE.md` hard rule 4). `features/paragraph/` reads exactly what it reads today.
- **The `Dataset` record gains fields** (`skip_base_features`, and `has_paragraph` wired up) —
  this is the one piece of redesign in an otherwise mechanical stage, by decision.
- **No call-site migration off the flat path constants.** Stage B derived them from the
  registry; repointing ~27 modules belongs with the data moves, not here.
- **`FUNCTION_REGISTRY` keeps its ordering dependency.** ✅ Diana 2026-10-06: a default runner
  executes the functions in their current order, and the exposure — someone calling them out of
  order in future — is accepted. **T3.24 is declined for now**, not outstanding.
- **`analyses/` is not created.** The analysis half of `correctness_measures.py` and all of
  `viz/` wait for stage E.
- **No numbers are expected to move.** If one does, stop: it means the move changed behaviour,
  which is the one thing this stage must not do.

---

## What I am least sure about

Honest list, so these get attention rather than being discovered.

1. **The vendored `importlib` load path.** `paragraph_trial_features.py` loads
   `utils - paragraph feature extraction.py` by path and aliases `src.configs` into
   `sys.modules`. Moving the folder means both follow correctly, and the failure mode is an
   import that silently resolves to the wrong module. Worth a targeted test rather than trusting
   the sweep.
2. **`know_qa_dataprep.py`'s registry filtering.** It picks its base features by *filtering*
   the shared `FUNCTION_REGISTRY` (:997-1003) rather than declaring a configuration. That works
   and I do not propose changing it here — but it is the mechanism that should eventually become
   the `Dataset` record's job, and moving the file is when someone will notice it.
3. ~~**Whether `ingest/geometry.py` deserves to exist.**~~ ✅ **Diana 2026-10-06: create it.**
   The two functions are T3.18's screen-rectangle area assignment — the one piece of ingest that
   encodes the *screen layout* rather than the file format.


---

## Outcome — what was actually done, 2026-10-06

All six steps ran. After each one: the import sweep, and the KnowQA Stage-1 pipeline compared
against a saved baseline, checking **exit code and file timestamps** as well as the comparison.

### The acceptance test, final run

| check | result |
|---|---|
| live modules import | **92 / 96**, the same four parked failures as before the stage (`answer_loc` ×3, `mixed_text_answer_effects`) |
| KnowQA Stage-1 pipeline | **bit-identical** — 33,830 × 420, columns identical, numeric max abs diff **0.0**, **0** non-numeric cells differing |
| `text_qa_relationship` tables | **45 / 45 byte-identical to HEAD** (`git diff` on `reports/text_qa_relationship/tables/` is empty) |
| vendored containment | importing all 96 live modules leaves **no `src.vendor.*` in `sys.modules`** |
| `features/` ⇸ `ingest/` | **one import left**, see divergence 3 |
| `config/` ⇸ `features/` | clean |

Only one tracked file under `reports/` changed, and it is provenance rather than numbers:
`manifest.json`'s `produced_by` went from `src.statistics.…` to `src.stats.…` (the stage A
rename), plus `saved_at` timestamps. **No number moved anywhere, so there is no `findings.md`
change-log entry to make.**

### Where it diverged from the plan

**1. The split is nine ways, not six.** The proposal's own table already summed to nine buckets;
the "six modules" phrasing was inherited from `restructure-map.md` §6.1 and was stale. Every one
of the 35 functions landed in exactly one place.

**2. The moves left ~57 dead imports behind**, which the proposal did not anticipate. Rebuilding
module headers mechanically from the original file's import block copies imports the moved
functions no longer use. Found with an AST scan, cross-checked against every other module in the
repo so that a **re-exported** name was never removed, then deleted: 24 from `features/build.py`,
26 from `ingest/build.py`, 7 elsewhere. This mattered beyond tidiness — four of them were
`features/build.py` → `ingest/*` imports that made the layering violation look four times worse
than it is.

**3. The layering rule does not fully hold, and the fix is a design decision.**
`features/build.py` still imports `_reconcile_area_with_geometry` from `ingest/geometry.py`,
because `add_IA_screen_location` needs it. **Written up in `restructure-map.md` §5.4 with three
options and a recommendation; left for Diana.**

### Smaller things worth knowing

- **`src/external/scasim_wrapper.ipynb` was deleted, not moved** — `archive/scasim_wrapper.ipynb`
  is byte-identical to it (same md5, same size, same date). Nothing lost.
- **`answers_paragraphs_csv.py` lost its `sys.path.insert` header.** Two levels deeper, its
  `parents[2]` resolved to `src/` instead of the repo root, which would have put `src/` back on
  the import path — the stdlib-shadowing hazard stage A removed. Nothing replaces it; the project
  runs from the repo root. **Seven other modules still carry the same (harmless, redundant)
  header** — a separate cleanup, not done here.
- **`mixed_text_answer_effects.py` moved with the data builder it depends on**, out of
  `src/stats/` into `src/explorations/text_answer_effects/`. It still does not import (`pymer4`
  0.9 dropped `Lmer`) — now a parked failure in a folder named "parked".
- **`Dataset.skip_base_features`** declares the *difference* from the registry, not a whole list,
  and holds **names** validated against `FUNCTION_REGISTRY` at run time — `config/` must not
  import `features/`. `add_zscored_pupil_columns` deliberately stays a runner-level conditional:
  it is a property of how you chose to run, not of the dataset.
- **The paragraph guard is now dataset-driven** (`if include_paragraph and not ds.has_paragraph`)
  and its message names the dataset instead of hardcoding "KnowQA".
