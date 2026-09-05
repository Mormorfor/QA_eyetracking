# Data pipeline

How raw eye-tracker reports become the trial-level table the models consume.
Drafted from `src/data_paths.py`, `src/data_prep/`, `src/derived/`, and
`src/predictive_modeling/*/model_data.py`.

**Status: reviewed with Diana 2026-09-05.** The four datasets, the KnowQA differences and
the symlink question are settled; see the resolved list at the end. Remaining `?` marks are
my inference, not established fact.

---

## 1. The four datasets

The same pipeline runs over four datasets. They differ in their raw report format and
their identity scheme, not in their feature definitions.

| Dataset | Raw location | Processed output | What it is |
|---|---|---|---|
| **L1 / OneStop** | `data_raw/full/` (CSV), `data_raw/tsv/` (TSV) | `data/L1_based_data/` | Study 1. 360 participants. The paper's main dataset. |
| **testrun_QA** | `data_raw/testrun_QA/` | `data/new_exp_try_runs/` | Study 2, first pilot. UTF-16 `.xls`, has a messages report. |
| **second_test** | `data_raw/second_test/` | `data/second_test_runs/` | Study 2, second pilot. UTF-8 `.tsv`, report-prefixed names, no messages report. |
| **KnowQA** | `data_raw/KnowQA/` | `data/KnowQA_runs/` | Study 2, first real collection. UTF-16 `.xls` under canonical names, no messages report. |

Each raw dataset has two report families:

- **IA reports** — one row per interest area (word). Aggregated measures per word.
- **Fixation reports** — one row per fixation. Needed for pupil stats, sequences, and click/timestamp reconstruction.

**Which screens exist differs by study.**

- **L1 / OneStop** exports both families per screen type: `Answers`, `Paragraph`, `QA`,
  `Questions`, `Question_Preview`, `Title`, `Feedback`.
- **KnowQA** has a different, **regime-dependent** screen sequence — what a participant sees
  before the answer screen depends on whether they got full, partial or no knowledge. The
  project currently has access to **the Answers screen only**, which is the one screen that is
  identical across regimes *and* identical to L1's. That is why the pipeline works unchanged
  across both studies.

Other KnowQA screens can be exported and added later if a question needs them; nothing in the
current code depends on them.

**`data_raw/full` and both `data_raw/tsv` subfolders are symlinks** to the OneStop data held
outside the repo. Deliberate — the corpus is not ours to redistribute. **The public release
will ship instructions telling users where to place their own OneStop download** rather than
the data or the links (`todo.md` T5.7).

Practical consequence while working: anything reached through those links is invisible to a
session that only has the repo folder connected. Processed outputs under `data/` are reachable;
raw reports are not.

---

## 2. Stage 0 — raw report conversion and cleaning (Study 2 only)

L1 arrives already in the schema the pipeline expects. Study 2 datasets do not, so they
get a conversion step first.

**Canonical module: `src/data_prep/know_qa_dataprep.py`.**

```
prepare_know_qa()   # or: python -m src.data_prep.know_qa_dataprep
  ├── convert_raw_reports()    .xls (UTF-16) / .tsv (UTF-8-BOM) → canonical CSVs
  ├── clean_reports()          column renames, letter answers_order, identity columns,
  │                            per-SESSION pupil z-scoring
  ├── run_pipeline()           → calls data_csv_generation.main()   [Stage 1]
  └── build_features()         → trial-level feature table          [Stage 2]
```

Writes to `data_raw/<run>/csvs/` and `data_raw/<run>/csvs/cleaned/`.

What this stage adds beyond format conversion:

- **`add_identity_columns`** — splits the `4XXX_YZ` recording label into `participant_id`,
  `session_id`, `article_batch`, `list_number`, `trial_number`, and builds the composite
  string `TRIAL_INDEX`. See glossary §2 for why.
- **`add_item_id_columns`** — derives `same_critical_span` from `text_id_with_q`, *never*
  from `onestopqa_question_id` (they disagree on ~40% of trials).
- **`check_trial_index_gaps`** — trial-counter gaps must equal the number of recalibrations.
- **`check_text_alignment`** — reconstructs on-screen text from `IA_LABEL` and is *intended*
  to catch a displaced-quote bug in the stored stimulus text that would shift every
  interest-area boundary after it by one word. ⚠️ **Whether that bug is real is unconfirmed**
  (`todo.md` T3.18) — the check's diagnosis has not been validated against L1 interest-area
  counts for the same items.

> ⚠️ **`experiment_builder/data_prep_new_exp.ipynb` is a near-verbatim earlier copy of
> this module** and writes the *same* `csvs/cleaned/` files and the *same*
> `KnowQA_runs/L1_model_ready_all_features.csv`, but with the older identity scheme
> (session label as `participant_id`, integer `TRIAL_INDEX`) and **without** the two
> integrity checks or the session-level pupil normalization. Running it with
> `RUN_NAME = "KnowQA"` overwrites the module's output. The notebook says so itself in a
> markdown cell. Prefer the module; the pilots' outputs were built by the notebook.

---

## 3. Stage 1 — IA-level feature generation

**Entry point: `src/data_prep/data_csv_generation.py::main()`** (also runnable as a script).

Input: an IA report + a fixation report.
Output: `all_participants.csv` — **IA-level** (see glossary §1) — plus auxiliary tables.

```
main()
 ├─ 1. button clicks           → Auxiliary/button_clicks_data.csv
 │      (only if add_last or add_rts; rebuilt only when rebuild_button_clicks=True
 │       OR the file is missing)
 ├─ 2. load IA report, filter  repeated_reading_trial == False, practice_trial == False
 ├─ 3. resolve function lists  from FUNCTION_REGISTRY
 ├─ 4. load fixation report ONCE, share it downstream
 ├─ 5. participant pupil stats → Auxiliary/participant_pupils.csv
 ├─ 6. _process()
 │      ├─ add_base_features()            9 functions, in registry order
 │      ├─ generate_new_row_features()    11 group functions, each left-merged back
 │      ├─ _attach_last_label_features()  → Auxiliary/all_participants_last.csv
 │      └─ _attach_rt_and_tfd_features()  → Auxiliary/RT_and_TFD.csv
 └─ 7. _save() + optional _save_splits(split_column=question_preview)
                                          → hunters.csv / gatherers.csv
```

**Base features** (order matters — see the gotcha below): `add_text_id`,
`add_text_id_with_q`, `add_is_correct`, `add_answer_text_columns`,
`add_IA_screen_location`, `add_IA_answer_label`, `add_selected_answer_label`,
`add_zscored_pupil_columns`, `add_total_answering_RT_normalized`.

**Group features:** `create_mean_area_dwell_time`, `create_mean_area_fix_count`,
`create_mean_first_fix_duration`, `create_skip_rate`, `create_dwell_proportions`,
`create_mean_pupil_size_metrics`, `create_first_encounter_pupil_size`,
`create_last_area_and_location_visited`, `create_fixation_sequence_tags`,
`create_simplified_fixation_tags`, `create_simplified_visit_counts`.

Adding a feature means registering it in `FUNCTION_REGISTRY` with its `join_columns`.

### Auxiliary outputs

| File | Grain | Contents |
|---|---|---|
| `Auxiliary/button_clicks_data.csv` | trial | selection/confirm timestamps, `(ts, IA_id)` fixation pairs, last fixations before select/confirm, press counts |
| `Auxiliary/participant_pupils.csv` | participant | pupil mean and SD in mm |
| `Auxiliary/all_participants_last.csv` | trial | last IA and area label before last select / before confirm |
| `Auxiliary/RT_and_TFD.csv` | trial | `RT_pure_*`, `RT_normalized_*`, `TFD_*`, `TimeSinceOffset_*` per region |

### How the KnowQA path differs at this stage

| | L1 | KnowQA |
|---|---|---|
| pupil z-scoring | `add_zscored_pupil_columns` runs here, z-scoring per `participant_id` | already done in Stage 0, per `session_id`. That base function is therefore **removed from the registry list** passed to `main()` (`know_qa_dataprep.py:415`) — if it ran, it would z-score a second time per participant and overwrite the per-session values with ones pooled across a person's sittings |
| `add_answer_text_columns` | runs | excluded |
| repeated-reading filter | on | off (no such column) |
| paragraph RT/TFD | supported | **not applicable — KnowQA has no paragraph data.** Only the answers screen is exported (`data_paths.py` defines no paragraph path for any Study 2 run), and only a third of KnowQA trials show a paragraph at all. `run_pipeline:884` refuses `include_paragraph=True`; its error cites a secondary technical blocker (`load_paragraph_fixations` coerces `TRIAL_INDEX` to int64, which the composite ids would fail), but the real reason is simply that there is nothing to read |
| button clicks source | legacy CSV + separate UTF-16 TSV, cumulative `ALL_ANSWERS` | single fixations CSV, non-cumulative |

---

## 4. Stage 2 — trial-level model table

**Entry point: `src/predictive_modeling/answer_correctness/model_data.py::save_all_features()`**
→ `L1_model_ready_all_features.csv` (one row per trial).

`build_trial_level_model_df()` starts from the IA-level frame and left-merges eight blocks:

1. per-area metric pivot `<metric>__<area>` plus the correct/wrong contrasts (glossary §7)
2. sequence-derived: `seq_len`, `has_xyx`, `has_xyxy`, `longest_alt_answer_run`, `trial_mean_dwell`
3. pattern-breaking / dominance (participant-level)
4. paragraph-span dwell proportions — **read from a cache written by `answer_RTs`**
5. last-label-before-confirm one-hots
6. last-label-before-select one-hots
7. RT / TFD / TimeSinceOffset per region, plus contrasts
8. `ANSWER_PRESS_NUMBER` and `total_answering_RT`

`load_all_features()` reads it back with `participant_id` pinned to `str` — necessary
because KnowQA ids are all digits and would otherwise be inferred as int, breaking merges.

> All eight merges are `how="left"`, so a trial missing from any block would silently gain
> NaN columns, which the model then fills with `0.0`. **A trial should never be missing from
> one block and present in another** — they all derive from the same trial set — so this is an
> invariant to assert rather than a case to absorb (`todo.md` T3.17).

---

## 5. Stage 3 — side caches

Two feature families live outside the main pipeline and are cached to disk.

### Paragraph-span features — `answer_RTs/features.py::save_paragraph_features()`
→ `L1_paragraph_span_features.csv`. Per-area metrics computed over the *paragraph* screen,
split by `critical` / `distractor` / `outside` span. Mirrors the answer-screen metric
definitions.

### EyeBench paragraph-trial features — `derived/external/EyeBench/runner.py`
→ `EyeBenchExtracted/L1_paragraph_trial_level_features.csv` plus two feature-key CSVs
mapping each feature to its source model family.

The extraction code is vendored in `src/external/EyeBench/`:

- `utils - paragraph feature extraction.py` — received from the lab's EyeBench/OneStop
  project, kept as-is so a newer copy can replace it wholesale. **Not importable by name**
  (spaces and a dash), so it is loaded by path via `importlib`, after `configs/` is aliased
  into `sys.modules` as `src.configs`.
- `paragraph_trial_features.py` — the adapter that reproduces the source project's
  preprocessing over our raw reports.
- `configs/` — the constants the vendored file imports.

`src/derived/external/EyeBench/runner.py` holds **only** the orchestration (pick
participants, cache-or-rebuild, read the key files). The split is deliberate: vendored code
stays replaceable, our glue lives elsewhere.

> ⚠️ Two entry points write the same cache with **different feature definitions**:
> `runner.py` defaults `fix_ptb_pos_double_mapping=True`, while
> `paragraph_trial_features.py`'s `__main__` defaults it to `False` (reproducing an
> upstream bug where every `ptb_pos_*` feature comes out 0). The cache file cannot tell
> you which one produced it.

---

## 6. Build order

There is a **circular dependency between the two modeling packages**, resolved only by
build order:

```
Stage 0  know_qa_dataprep (Study 2 only)
            ↓
Stage 1  data_csv_generation.main()
            → all_participants.csv + Auxiliary/*
            ↓
Stage 3a answer_RTs/features.py::save_paragraph_features()
            → L1_paragraph_span_features.csv
            ↓
Stage 2  answer_correctness/model_data.py::save_all_features()
            → L1_model_ready_all_features.csv        ← reads 3a's cache
            ↓
Stage 3b answer_RTs/model_data.py                    ← reads Stage 2's cache
            (targets and span RT predictors)
```

`model_data.py` raises a `FileNotFoundError` telling you to run the paragraph features
first, which is the only place this order is enforced. It is otherwise documented in
docstrings only.

Other builders that must run before certain analyses:

| Builder | Produces | Needed by |
|---|---|---|
| `notebooks/tests and data builders/create_merge_folds_csv.ipynb` | `data/CV folds/*Folds/fold_*_trial_ids_by_regime.csv` | all cross-validation |
| `notebooks/tests and data builders/clicks_selections_confirms.ipynb` | button clicks / last-area labels | last-fixation features |
| `notebooks/tests and data builders/reading_times.ipynb` | RT/TFD regions | RT features |
| `experiment_builder/lists_builder.ipynb` | `data/Experiment/new_batched_lists/` | running Study 2 (hand-off to the presentation software) |

---

## 7. Derived feature modules

`src/derived/` holds the feature definitions that Stage 2 assembles.

| Module | Computes | Invoked by |
|---|---|---|
| `pupil_norm.py` | area→mm conversion, per-participant pupil stats, z-scoring | `data_csv_generation`, `know_qa_dataprep`, `answer_RTs/features` |
| `reading_times.py` | four RT definitions + TFD; assembles `RT_and_TFD.csv` | `data_csv_generation._attach_rt_and_tfd_features` |
| `select_confirm_last.py` | last IA / area label before select and before confirm | `data_csv_generation._attach_last_label_features` |
| `correctness_measures.py` | accuracy-by-group summaries + Wilson CIs; XYX/XYXY detection; trial mean dwell | `viz/visualisations_correctness_measures`, `model_data` |
| `pattern_breaking.py` | starting strategies, dominance, breaks-pattern, Levenshtein distance | `model_data`. **Paper-critical as of draft2** — backs the First-scan behavior Results subsection. Two gaps vs. the paper: no clockwise/counter-clockwise classifier, and no interrupted-scan completion. See glossary §8. |
| `preference_matching.py` | whether the selected answer is the gaze-"preferred" one | `model_data`, `viz/visualisations_preference_correctness` |

### The four RT definitions — they are not the same measure

| Function | Definition |
|---|---|
| `compute_reading_times` | **span-based**: `max_last_fixation − min_first_fixation + last_fixation_duration`. Counts every excursion away and back as part of the region's time. Renamed to `TimeSinceOffset_*` on output. |
| `compute_run_based_rt` | **run-based**, from the `(ts, IA)` sequence in `button_clicks_data.csv` |
| `compute_run_based_rt_from_fixations` | same definition, from the paragraph fixation report |
| `build_rt_and_tfd` | assembles them; guarantees `RT_*` is always run-based |

---

## 8. Known gotchas

Things that will bite someone re-running this. Not yet fixed — listed so they are at
least visible.

1. **`data_csv_generation` group functions mutate the caller's frame in place.** Five of
   them do, which creates an undocumented ordering dependency: `create_first_encounter_pupil_size`
   filters on `IA_FIRST_FIXATION_DURATION > 0` and only works because
   `create_mean_first_fix_duration` already coerced that column to int earlier in registry
   order. Running a subset via `group_function_names=[...]` can therefore compare `str > int`.
   It also means `area_skipped` and the `"." → 0` coercions leak into the saved output.
   → **`todo.md` T3.11**

2. **The paragraph path reimplements the answer path's metrics, and one of them diverged.**
   `answer_RTs/features.py` rewrites eight `create_*` functions from `data_csv_generation`,
   grouped by span instead of area. Only **`mean_first_fixation_duration`** actually differs
   (answer zero-fills `"."`, paragraph drops it) — `skip_rate`, `mean_dwell_time` and
   `mean_fixations_count` are identical, because their source columns carry a real `0` and no
   `"."` to coerce. Fixes: **T3.6** (the coercion) and **T1.7** (the duplication that let them
   drift). *An earlier version of this file claimed `skip_rate` diverged; measurement showed
   it does not.*

3. **Run-based RT fails silently to all-zeros** on a `(participant_id, TRIAL_INDEX)` key
   mismatch — the row is still written with every region at 0, indistinguishable from a real
   zero. Compounded by the fact that `button_clicks_data.csv` is only rebuilt when explicitly
   asked or when missing, so a stale click table from another dataset is reused quietly.
   → **`todo.md` T3.7**

4. **`get_participant_pupil_stats` defaults to a hardcoded L1 path.** `answer_RTs/features.py`
   calls it with no path, so paragraph-span pupil z-scores are baselined against L1's answer
   screen regardless of the dataset. → **`todo.md` T3.20** — the requirement being that each
   dataset writes its own pupil-stats file and every consumer reaches for the right one
   (fix alongside T1.7 and T6.1)

5. **Participant-level features are computed over whatever trials the frame holds.**
   `dominance_score` / `breaks_pattern` accumulate over a participant's trials, and the scope
   is set by whichever frame is passed in. The hunters/gatherers split is safe —
   `question_preview` is between-participant, so a participant's whole trial set is in one
   group file. Trial-level filtering that keeps the participant is not (CV regime rebuilds,
   correct-only subsets, and the KnowQA knowledge regimes, which are within-participant). See
   `docs/pitfalls.md` §3. → **`todo.md` T3.21**, which enumerates every affected quantity and
   makes the scope an explicit flag rather than a property of the caller

6. **Unfixated regions are filled with `0`, not NaN**, throughout the RT/TFD family — and
   that `0` is real data: every area exists on every trial, so it means "never fixated".
   → **no action needed** — confirmed correct 2026-09-04.

7. **Trials with no recorded selection are scored `is_correct = 0`**, not excluded.
   → **`todo.md` T3.5**

8. **`add_base_features` leaves a spurious `index` column** on the L1 path (double
   `reset_index`) but not on the KnowQA path, so the two datasets' `all_participants.csv`
   differ by one column. → **`todo.md` T1.8**

9. `data/L1_based_data/all_participants_with_practice.csv` (~3.9 GB) and
   `Auxiliary/paragraph_RT_run_based.csv` are not registered in `data_paths.py`. The first is
   read only by `experiment_builder/extract_text.ipynb`; the second appears unreferenced. `?`
   → **`todo.md` T5.8**

10. **The `L1_` prefix is a sample restriction, not a study number.** `L1` means *native
    language*: these paths hold data derived from OneStop's **L1 (native English speaker)**
    portion. OneStop also has an L2 portion, unused here. So any future non-native data
    needs its own namespace rather than reusing `L1_based_data/` — and the prefix should be
    explained in the README, since a public reader will otherwise read it as "level 1" or
    "study 1".

---

## 9. Where paths are defined

`src/data_paths.py` is the single registry of dataset locations, anchored on
`PROJECT_ROOT = Path(__file__).resolve().parents[1]`. It also inserts the project root into
`sys.path` on import.

**Convention (agreed 2026-09-04):** all paths resolve through `PROJECT_ROOT`; code is run
from the repo root. The `"../reports/..."` string literals scattered through `src/viz/` and
`generate_column_options.py` predate this and are being migrated out.

---

## Open questions for Diana

1. `Auxiliary/paragraph_RT_run_based.csv` — does anything still read this? To be answered
   during the restructure rather than now (`todo.md` T5.8).

**Resolved 2026-09-04/05:**

- The `data_raw` symlinks are deliberate; the release ships placement instructions for the
  user's own OneStop download (`todo.md` T5.7).
- **KnowQA has no paragraph data at all**, so the paragraph refusal is correct rather than a
  limitation to lift.
- `skip_rate`, `mean_dwell_time` and `mean_fixations_count` are **not** divergent between the
  answer and paragraph paths — only `mean_first_fixation_duration` was (gotcha 2).
- **KnowQA exports only the Answers screen**, which is identical across regimes and to L1 (§1).
- `all_participants_with_practice.csv` (3.9 GB) is a **leftover from `extract_text.ipynb`, and
  is being kept** — not a deletion candidate (`todo.md` T5.8).
- **Both pilots stay runnable**, not archive-only — they may need naming brought into line
  first (`todo.md` T5.8).
- A trial **should never** be present in one feature block and absent from another, and a
  trial **can never** lack a confirmed selection. Both are invariants to assert rather than
  handle (`todo.md` T3.17).
