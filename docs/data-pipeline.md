# Data pipeline

How the eye tracker's raw reports become the table the models read: one row per trial, one
column per feature.

---

## 1. The four datasets

| Key | What it is | Raw reports | Output folder | Built by |
|---|---|---|---|---|
| `l1` | Study 1: OneStop, native English speakers. 360 participants, 19,436 trials | `data_raw/full/` (CSV) and `data_raw/tsv/` (TSV) | `data/datasets/l1_onestop/` | `scripts/build_dataset.py --dataset l1` |
| `knowqa` | Study 2 | `data_raw/KnowQA/` (UTF-16 `.xls`) | `data/datasets/knowqa/` | `scripts/build_dataset.py --dataset knowqa` |
| `testrun_qa` | Study 2, first pilot | `data_raw/testrun_QA/` (UTF-16 `.xls`, plus a messages report) | `data/datasets/pilots/testrun_qa/` | `scripts/build_pilot.py --pilot testrun_qa` |
| `second_test` | Study 2, second pilot | `data_raw/second_test/` (UTF-8 `.tsv`) | `data/datasets/pilots/second_test/` | `scripts/build_pilot.py --pilot second_test` |

Each dataset is described once, by a `Dataset` record in `src/config/datasets.py`, and every
output path is built from that record.

The eye tracker writes two kinds of report:

- an **interest-area report**: one row per word on the screen, with totals for that word
  (looking time, number of fixations, ...);
- a **fixation report**: one row per fixation, in time order. Needed for the order of looks,
  pupil size, the button presses and the visit-based reading time.

**Screens.** Study 1 has reports for every screen of a trial; the features use two of them, the
paragraph screen and the answer screen. Study 2 has the answer screen only. It is the same
screen in both studies, which is why the same code handles both.

**The OneStop data is not in the repo.** `data_raw/full` and the two folders inside
`data_raw/tsv` are links to a copy stored elsewhere, because the corpus is not ours to
redistribute. Anyone running the code needs their own OneStop download in those places.

**`L1` means native language**, not "level 1" or "study 1": Study 1 uses the
native-English-speaker part of OneStop.

---

## 2. Building a dataset

```
python scripts/build_dataset.py --dataset l1
python scripts/build_dataset.py --dataset knowqa
```

**Study 1** runs three steps:

| Step | Code | Writes |
|---|---|---|
| 1. Answer screen (§4) | `ingest/build.py::main` | `interim/interest_areas.csv`, `interim/areas_answers.csv`, `interim/trials.csv`, and side files in `interim/aux/` |
| 2. Paragraph screen (§5) | `features/paragraph/spans.py::save_paragraph_features` | `features/paragraph_spans.csv` |
| 3. Model table (§6) | `features/build.py::save_model_ready` | `features/model_ready.csv` |

Steps 1 and 2 do not depend on each other. Step 3 needs both.

By default it also writes the hunters' and gatherers' subsets of each table
(`trials__hunters.csv`, ..., `model_ready_hunters.csv`, `model_ready_gatherers.csv`).
`--no-splits` leaves them out.

**Study 2** runs `ingest/knowqa.py::prepare_know_qa`: two preparation steps (§3), then the same
answer-screen pipeline as Study 1, then the model table. There is no paragraph step.

**The pilots** are built by their own script, `python scripts/build_pilot.py --pilot <key>`,
because they use an older identity scheme — see §9 item 3.

---

## 3. Study 2's preparation — `ingest/knowqa.py`

Study 1's reports already have the column names the pipeline expects. Study 2's do not, so they
are prepared first:

```
convert    raw .xls / .tsv reports -> CSVs        data_raw/KnowQA/csvs/
clean      rename columns, build the ids          data_raw/KnowQA/csvs/cleaned/
pipeline   ingest/build.py::main                  -> the three tables
features   the three tables, as for Study 1       -> features/model_ready.csv
```

One step at a time: `python -m src.ingest.knowqa --steps clean`.

Cleaning does more than rename:

- **Identity.** A recording is named like `4000_21`: person `4000`, batch `2`, list `1`. One
  person can come back for several sessions, so the name is split into `participant_id`,
  `session_id`, `article_batch`, `list_number` and `trial_number`, and `TRIAL_INDEX` becomes a
  text code like `b2l01t005` (batch, list, trial). That way a trial is still identified by
  `(participant_id, TRIAL_INDEX)`, as everywhere else. Details in `glossary.md` §2.
- **`same_critical_span`** is taken from `text_id_with_q`, never from `onestopqa_question_id`.
  The two number the questions in different orders and disagree on about a third of trials.
- **Two checks.** `check_trial_index_gaps`: a session should have exactly as many gaps in its
  trial numbers as it had recalibrations; more gaps means lost trials, and is printed as a
  warning. `check_text_alignment`: lists trials whose stored question and answer text does not
  match the words on screen. This does happen (the presentation software sometimes splits a
  quote off a word), but it does not change which area a word is in, because words are placed
  by their position on screen.

---

## 4. The answer screen — `ingest/build.py::main`

In order, it:

1. reads the interest-area report and drops repeated-reading and practice trials;
2. reads the fixation report, once (`features/fixations.py`). Everything below that needs
   single fixations uses this one table;
3. rebuilds when each answer was selected and confirmed, from the tracker's message log
   (`ingest/clicks.py`);
4. works out each participant's pupil baseline (§8);
5. runs the features listed in `ingest/registry.py`: first the per-word ones, then the per-area
   and per-trial ones;
6. adds the three reading-time measures (§7) and the area looked at before each button press;
7. splits the columns into the three tables and saves them.

Every side file is rebuilt on every run; nothing is reused from an earlier one.

**Placing a fixation on a word** (`features/fixations.py::place_fixations`). A fixation belongs
to the word it landed on (`CURRENT_FIX_INTEREST_AREA_ID`), or, if it landed between words, to
the nearest word (`CURRENT_FIX_NEAREST_INTEREST_AREA`); the word then gives it its area. The
fixation sequence, the visit-based reading time and the area looked at before each press all
place fixations with this one function, so they cannot disagree about where a fixation was. On
Study 1's answer screen, 76,401 of the 718,207 fixations are placed by the nearest word. A
fixation in an analysed trial that points at a word the trial does not have stops the run.

**The registry** (`ingest/registry.py`) is the list of features. Each entry names the function,
the columns it reads, and whether it works per word, per area or per trial. A second list,
`_PRODUCES`, says which columns each feature adds and at which level, and that decides which of
the three tables each column goes into. To add a feature, add an entry and its `_PRODUCES` line.

**Side files**, in `interim/aux/`:

| File | One row per | Holds |
|---|---|---|
| `button_clicks_data.csv` | trial | the time of every selection press and of the confirming press; the answers clicked; the number of presses |
| `pupil_baseline__answers.csv` | participant | average and SD of pupil size (mm) on the answer screen |
| `all_participants_last.csv` | trial | the area looked at before the last selection press, and before the confirming press |
| `RT_and_TFD.csv` | trial | the three reading-time measures for each area |

**Where Study 2 differs at this step:**

| | Study 1 | Study 2 |
|---|---|---|
| answer text by letter (`add_answer_text_columns`) | built here | skipped: the cleaned reports already have `answer_A`..`answer_D` |
| repeated-reading trials | dropped | there are none |
| button presses | from a separate TSV export of the fixation report, which holds the message log | from the fixation report itself, which holds the messages |

---

## 5. The paragraph screen (Study 1 only) — `features/paragraph/spans.py`

Each paragraph is split into three parts: **critical** (holds the answer), **distractor**
(supports a wrong answer) and **outside** (the rest). Each part gets the same measurements as an
answer-screen area, from the same code, saved one row per trial in
`features/paragraph_spans.csv`.

The paragraph reports are several GB each, so only the columns needed are read: 14 of the
interest-area report, 5 of the fixation report. The fixation report is read once, in chunks, by
the same function as the answer screen's, and that one table gives the paragraph pupil baseline
(saved to `interim/aux/pupil_baseline__paragraph.csv`), the visit counts and the reading times.

---

## 6. The model table — `features/build.py`

`load_model_input` joins the three tables back into one wide table, one row per word.
`build_trial_level_model_df` turns that into one row per trial, from these blocks:

1. the per-area measurements, one column per measurement and area (`mean_dwell_time__answer_B`),
   plus five comparisons of the correct answer with the wrong ones (`glossary.md` §7)
2. the five sequence statistics: `seq_len`, `has_xyx`, `has_xyxy`, `longest_alt_answer_run`,
   `trial_mean_dwell`
3. the opening-scan features, which compare each trial with the participant's usual opening
   scan: `breaks_pattern_*`, `dominance_score_*`, ...
4. the paragraph features (Study 1 only), from `features/paragraph_spans.csv`
5. the area looked at before the confirming press, as 0/1 columns
6. the same, before the last selection press
7. the three reading-time measures for each answer area, plus comparisons
8. `ANSWER_PRESS_NUMBER` and `total_answering_RT` (and its per-word version)

Every block is joined on `(participant_id, TRIAL_INDEX)`. Before each join,
`checks.assert_full_coverage` stops the run unless every trial has exactly one row in the
block. Without it, a missing trial would get NaN, the model would fill that with 0, and the
trial would look like an average one. The paragraph block is the exception: it comes from a
separate file, so it is only checked to add no rows and to match at least one trial.

`load_model_ready` reads the table back with `participant_id` as text. Study 2's ids are all
digits, and read as numbers they would not join with the other tables.

**EyeBench paragraph features** are a separate, extra table: per-trial paragraph-reading
features made by the lab's EyeBench code, built by
`src/vendor/eyebench/runner.py::get_paragraph_trial_features` into `features/eyebench/`. They
are not in the model table; only exploration code (`explorations/answer_reading_times/`) uses
them. Build them
with the runner: the vendored file's own `__main__` gives every `ptb_pos_*` feature the value 0
(a bug in the original code). The runner avoids that, and records how it built the table in a
`.meta.json` file beside it.

---

## 7. The three reading-time measures — `features/reading_times.py`

| Columns | What it measures | From |
|---|---|---|
| `TFD_*` | total fixation duration: the looking time on the area's words, added up | the word table |
| `RT_*` | time spent on visits to the area. A visit is a run of fixations in a row on the area; it counts from its first fixation's start to the start of the next fixation anywhere | the fixation report |
| `TimeSinceOffset_*` | from the start of the first look at the area to the end of the last look, so time spent elsewhere in between counts too | the word table |

Each comes as `_pure` (ms) and `_normalized` (divided by the number of words in the area). The
six column prefixes (`RT_pure`, ..., `TimeSinceOffset_normalized`) are defined once, in
`config/columns.py` (`RT_TFD_METRICS`), and the areas in `config/screens.py`. The build uses
both wherever it makes or collects these columns (`reading_times.py`, `features/build.py`,
`ingest/registry.py`).

`compute_reading_times` makes `TFD_*` and the span measure, `compute_run_based_rt_from_fixations`
makes `RT_*`, and `build_screen_rt_tfd` puts them together for one screen. The span measure
comes out of `compute_reading_times` named `RT_*` and is renamed to `TimeSinceOffset_*`, so the
name `RT_*` always means the visit-based measure.

An area nobody looked at gets 0, never NaN. Every area is on screen in every trial, so 0 can
only mean "not looked at".

---

## 8. Pupil baselines

A pupil z-score is `(pupil size − mean) / SD`, using the participant's own mean and SD: their
**baseline**.

- **One baseline per person**, in every dataset. In Study 2, a person's sessions and all three
  knowledge regimes are pooled into one, so "this person's typical pupil size" means the same
  thing in both studies. (`python -m src.ingest.knowqa --pupil-norm-unit session` gives one
  per session instead.)
- **One per screen.** The answer screen and the paragraph screen differ in brightness and in
  task, so the same person's pupil differs between them. An answer-screen pupil z-score and a
  paragraph one are therefore on different scales; do not compare them as one measure.
- **Each dataset uses its own fixations.** There is no default: `get_participant_pupil_stats`
  stops with an error unless it is told which fixations to use.

| Dataset | Baseline from | Saved to |
|---|---|---|
| Study 1, answer screen | `fixations_Answers.csv` | `l1_onestop/interim/aux/pupil_baseline__answers.csv` |
| Study 1, paragraph screen | `fixations_Paragraph.csv` | `l1_onestop/interim/aux/pupil_baseline__paragraph.csv` |
| Study 2 | its own answer-screen fixation report | `knowqa/interim/aux/pupil_baseline__answers.csv` |
| pilots | their own answer-screen fixation reports | their own `interim/aux/` folder |

Pupil size arrives in the tracker's units (an area) and is converted to a diameter in mm first.

---

## 9. Things to know

1. **A trial missing from the fixation report gets 0 reading time and an empty sequence.** If a
   trial in the word table has no fixations in the fixation report — for example because its
   ids are written differently in the two — every `RT_*` column for it is 0, the same as "not
   looked at", and its fixation sequence is empty. Nothing checks for this. The opposite case
   is checked: a fixation that points at a word its trial does not have stops the run.
2. **Per-participant features depend on which trials they are given.** The opening-scan
   features summarise a person over the trials in the table they are handed. `pitfalls.md` §3
   explains when that matters.
3. **The pilots' tables on disk are older than the current fixation sequences.** Rebuilt with
   `scripts/build_pilot.py`, both pilots come out with the same trials and columns, but the
   features built from the fixation sequence (visit counts, `seq_len`, back-and-forth and
   opening-scan features) differ on a few trials: 7 of 197 columns for `testrun_qa`, 20 for
   `second_test`. Rerun the script to bring them up to date.
4. **`interest_areas.csv` has an `index` column.** It holds row numbers (0 to 760,627 on
   Study 1), left over from resetting the table's index while the per-word features run.
   Nothing reads it.
5. **Two files in `l1_onestop/interim/` are not in `config/datasets.py`:**
   `all_participants_with_practice.csv` (3.9 GB, read only by
   `notebooks/experiment/extract_text.ipynb`) and `aux/paragraph_RT_run_based.csv` (read by
   nothing).

---

## 10. Where paths are defined

Every data location is in `src/config/datasets.py`, built from `PROJECT_ROOT`, the repo root.
Where results are written is in `src/config/outputs.py`. Run everything from the repo root.
