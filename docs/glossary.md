# Glossary

What the project's words mean: the tables, the columns, and the terms that mean something
specific here. Column names are defined in `src/config/columns.py`; the constant is what the code
uses, the column is what appears in the files.

---

## 1. Grain: what one row is

The same data is stored at three levels, and mixing them up is the main way a number goes wrong.

| Level | One row per | File |
|---|---|---|
| **word** (interest area) | one word on screen, in one trial, for one participant | `interim/interest_areas.csv` |
| **area** | one of the five screen areas, in one trial | `interim/areas_answers.csv` |
| **trial** | one participant answering one question | `interim/trials.csv`, and `features/model_ready.csv`, which holds every feature |

A trial has 30–60 words on screen, about 39 on average. When the three files are joined back into
one wide table (`features/build.py::load_model_input`), every trial-level value is copied onto
all of its trial's word rows, so counting rows there makes every n about 39 times too big. To get
back to one row per trial: `groupby(["participant_id", "TRIAL_INDEX"]).first()`.

> Any statistical test about trials must run on a table with one row per trial.

---

## 2. Identity: who, and which trial

A trial is identified by two columns together, `participant_id` and `TRIAL_INDEX`
(`TRIAL_ID_COLS`). Almost every join in the project uses them. The two studies fill them
differently:

| Constant | Column | Study 1 | Study 2 |
|---|---|---|---|
| `PARTICIPANT_ID` | `participant_id` | the participant | the person only (`4000`) |
| `SESSION_ID` | `session_id` | — | the full recording name (`4000_21`) |
| `TRIAL_ID` | `TRIAL_INDEX` | a whole number counting the trials | a text code, e.g. `b2l01t005` |
| `TRIAL_NUMBER` | `trial_number` | — | the tracker's trial counter within one session |

**Why Study 2 is different.** A Study 2 recording is named `4XXX_YZ`: the person (`4XXX`), the
batch (`Y`, 1–3) and the list (`Z`, 1–18), with no separator because the tracker allows only 8
characters. One person can come back for several sessions, so the person alone does not pick out
a session, and the trial counter alone does not pick out a trial. Rather than add a third
column to every join, batch and list are folded into `TRIAL_INDEX` itself: `b2l01t005` is
batch 2, list 1, trial 5.

**What that means for per-person features.** In Study 2, anything computed per person pools all
of their sessions. For the pupil baseline that is on purpose. For the opening-scan features it is
the default, and `scope_by=["session_id"]` computes them per session instead.

---

## 3. Items: which text, which question

| Constant | Column | Meaning |
|---|---|---|
| `ARTICLE_COLUMN` | `article_id` | which article |
| `BATCH_COLUMN` | `article_batch` | which batch of articles (1–3) |
| `PARAGRAPH_COLUMN` | `paragraph_id` | which paragraph of the article |
| `DIFFICULTY_COLUMN` | `difficulty_level` | `Adv` or `Ele`: OneStop's two versions of each text. Study 1 has both; Study 2 uses `Adv` only |
| `LIST_COLUMN` | `list_number` | which counterbalancing list the participant got |
| `TEXT_ID_COLUMN` | `text_id` | one paragraph in one version: `batch_article_difficulty_paragraph` |
| `TEXT_ID_WITH_Q_COLUMN` | `text_id_with_q` | `text_id` plus which of its questions was asked: one test item |

### The two question numberings — never swap them

Each paragraph has three questions, and two columns number them in different orders:

| Constant | Column | Order |
|---|---|---|
| `SAME_CRITICAL_SPAN_COLUMN` | `same_critical_span` | Study 1's order; the last part of `text_id_with_q` |
| `ONESTOPQA_QUESTION_ID` | `onestopqa_question_id` | OneStopQA's own order, which the Study 2 materials were built from |

They disagree on about a third of Study 2's trials, so using one in place of the other silently
pairs up the wrong items. Study 2's cleaning takes `same_critical_span` from `text_id_with_q`,
never from `onestopqa_question_id`.

---

## 4. The answer screen and its five areas

The answer screen shows the question at the top and the four answers in a diamond — one above,
two side by side, one below — matching the gamepad's buttons.

An area can be named two ways:

| | Constant | Values |
|---|---|---|
| **by meaning** | `LABEL_CHOICES` | `question`, `answer_A`, `answer_B`, `answer_C`, `answer_D` |
| **by place** | `LOC_CHOICES` | `question`, `answer_0(top)`, `answer_1(left)`, `answer_2(right)`, `answer_3(bottom)` |

**`answer_A` is always the correct answer.** The letters are meanings, not positions: the answers
are shuffled across the four positions on every trial, and the position is in the `*_loc`
columns. Do not reorder either list; code relies on the order.

### The letters are ranked, not arbitrary

OneStopQA builds each answer from a set relationship to the paragraph:

| Letter | Built from | Role |
|---|---|---|
| `answer_A` | the critical span | correct |
| `answer_B` | the critical span | a misreading of the right part of the text |
| `answer_C` | the distractor span | based on a different part of the text |
| `answer_D` | nothing in the text | not supported by the text at all |

So plausibility runs A > B > C > D, and the choices follow it: on Study 1, 84.1% of choices are
A, 9.0% B, 4.9% C and 2.1% D. The code does not use this ranking — `wrong_mean` averages B, C and
D as equals. See `research-context.md` §2.

| Constant | Column | Meaning |
|---|---|---|
| `AREA_LABEL_COLUMN` | `area_label` | which of the five areas a word is in, by meaning |
| `AREA_SCREEN_LOCATION` | `area_screen_loc` | the same, by place |
| `SELECTED_ANSWER_LABEL_COLUMN` | `selected_answer_label` | the letter the participant chose |
| `SELECTED_ANSWER_POSITION_COLUMN` | `selected_answer_position` | the position they chose |
| `CORRECT_ANSWER_POSITION_COLUMN` | `correct_answer_position` | the position of the correct answer |
| `IS_CORRECT_COLUMN` | `is_correct` | 1 if they chose the correct answer, else 0: what the model predicts. A trial with no recorded choice stops the run rather than counting as wrong |

---

## 5. The two experiments' conditions

### Study 1 — question preview

| Constant | Column | Meaning |
|---|---|---|
| `QUESTION_PREVIEW_COLUMN` | `question_preview` | whether the participant saw the question *before* reading the paragraph |

- **hunters** (`question_preview == True`): saw the question first, so read hunting for the answer
- **gatherers** (`question_preview == False`): read the paragraph first, then saw the question

**Each participant is in one group only:** 360 participants, 180 per group, nobody in both. So the
two groups can be analysed separately without changing anything computed per person. This is the
split behind the `hunters`, `gatherers` and `all_participants` groups in the analyses.

Two kinds of trial are dropped during preparation: repeated readings (`repeated_reading_trial`)
and practice trials (`practice_trial`).

### Study 2 — knowledge regime

Instead of question preview, Study 2 varies **prior knowledge**, in three regimes:

- **full knowledge**: the participant is first shown the correct answer's content, reworded
- **partial knowledge**: the participant reads the paragraph — the closest match to Study 1
- **no knowledge**: no context at all, so a correct answer is a guess

> **Every participant does all three regimes** — the opposite of Study 1. So splitting Study 2 by
> regime splits each person's trials, and anything computed per person would then use about a
> third of their data. See `pitfalls.md` §3.

Study 2 also asks for a confidence rating on every trial, on a 1–5 scale. The lists are
counterbalanced with a double Latin square (regime × article group, then question order); the
design is in `src/experiment/lists.py`.

---

## 6. Per-area measurements

Computed separately for each of the five areas, from the word rows (`features/area_metrics.py`).

**Words nobody looked at are handled two ways, on purpose.** They count as 0 for dwell time and
fixation count: a word nobody read really did get 0 ms and 0 fixations. They are left out of
first-fixation duration and pupil size, because those describe a fixation, and there was none.
Do not make the two consistent; `pitfalls.md` §2 explains why.

| Constant | Column | Meaning |
|---|---|---|
| `MEAN_DWELL_TIME` | `mean_dwell_time` | average looking time per word |
| `MEAN_FIXATIONS_COUNT` | `mean_fixations_count` | average number of fixations per word |
| `MEAN_FIRST_FIXATION_DURATION` | `mean_first_fixation_duration` | average length of the first fixation, over the words that were looked at |
| `SKIP_RATE` | `skip_rate` | share of the area's words never looked at |
| `AREA_DWELL_PROPORTION` | `area_dwell_proportion` | the area's share of the trial's looking time; sums to 1 over the five areas |
| `NUM_LABEL_VISITS` | `num_label_visits` | how many separate times the participant entered the area |
| `NUM_LOC_VISITS` | `num_loc_visits` | the same, counted by place |

**Pupil size**, in mm and z-scored. The tracker reports pupil size as an area in its own units;
the pipeline converts it to a diameter in mm, and the `_z` version is z-scored against the
participant's own baseline (`data-pipeline.md` §8).

| Column (each also has a `_z` version) | Meaning |
|---|---|
| `mean_avg_fix_pupil_size`, `mean_max_fix_pupil_size`, `mean_min_fix_pupil_size` | each looked-at word's average / largest / smallest pupil size during its fixations, averaged over the area |
| `first_encounter_avg_pupil_size` | pupil size on the area's first word *by position* that was looked at. Despite the name, not necessarily the first word looked at in time: on Study 1's answer screen the two are different words in about a third of areas |

Two lists of these measurements, ten each:

- `AREA_METRIC_COLUMNS_MODELING` — for building model columns; uses the z-scored pupil values
- `AREA_METRIC_COLUMNS_VIZES` — for plots; uses pupil size in mm

---

## 7. Comparing the correct answer with the wrong ones

Each per-area measurement on the four answers becomes five numbers per trial, named
`<measurement>__<comparison>` (`DERIVED_SEP = "__"`):

| Constant | Suffix | Meaning |
|---|---|---|
| `CORRECT_SUFFIX` | `correct` | the value on the correct answer |
| `WRONG_MEAN_SUFFIX` | `wrong_mean` | the average over the three wrong answers |
| `CONTRAST_SUFFIX` | `contrast` | `correct` minus `wrong_mean` |
| `DISTANCE_FURTHEST_SUFFIX` | `distance_furthest` | the largest gap between the correct answer and any one wrong answer |
| `DISTANCE_CLOSEST_SUFFIX` | `distance_closest` | the smallest such gap |

The two distances are absolute values: they say how different the correct answer is, not in which
direction. Examples: `mean_dwell_time__contrast`, `skip_rate__correct`.

**Missing values.** A wrong answer with no value is skipped, so a comparison is missing only when
the correct answer has no value or none of the wrong answers does. That happens only for
first-fixation duration and pupil size, the measurements that leave unread words out. The model
fills every missing value with 0. For a `contrast`, 0 means "looked at like the others"; for a
z-scored pupil value, it means "this person's own average". `conventions.md` records this choice
and how many values it affects, under accepted deviations.

---

## 8. Fixation sequences

A trial's fixations, in time order, written as a sequence of areas (`features/sequences.py`).

| Constant | Column | Meaning |
|---|---|---|
| `FIX_SEQUENCE_BY_LABEL` | `fix_by_label` | one entry per fixation: which area, by meaning |
| `FIX_SEQUENCE_BY_LOCATION` | `fix_by_loc` | the same, by place |
| `SIMPLIFIED_FIX_SEQ_BY_LABEL` | `simpl_fix_by_label` | repeats merged, so one entry per visit: `A A B A` → `A B A` |
| `SIMPLIFIED_FIX_SEQ_BY_LOCATION` | `simpl_fix_by_loc` | the same, by place |

Two clean-up steps come first: a fixation that landed on no word is given the nearest word, and a
single fixation on the question at the very start of the trial is dropped, as left over from the
previous screen.

`IA_LABEL` is the word each interest area covers; read in order, they spell out the text on
screen. Study 2's `check_text_alignment` uses it to find trials whose stored text does not match
the screen (`data-pipeline.md` §3).

### The five sequence statistics

| Column | Meaning |
|---|---|
| `seq_len` | how many visits: the length of the merged sequence. Visits to the question count |
| `has_xyx` | 1 if the merged sequence *ends* in x, y, x (e.g. A B A) |
| `has_xyxy` | 1 if it *ends* in x, y, x, y (e.g. A B A B) |
| `longest_alt_answer_run` | the longest stretch anywhere that goes back and forth between two answers |
| `trial_mean_dwell` | average looking time per word over the whole screen |

`has_xyx` and `has_xyxy` look only at the end of the sequence, and any area counts, including the
question: "answer_A, question, answer_A" at the end counts as x-y-x. `longest_alt_answer_run`
leaves the question out.

**Which sequence each one reads.** These statistics read the sequence by meaning
(`simpl_fix_by_label`); the opening-scan features below read it by place (`simpl_fix_by_loc`). For
the statistics it makes no difference, because within one trial the two sequences are the same up
to renaming. For the opening scan it matters: the clockwise finding is about screen positions, and
would disappear if scans were written by letter.

### The opening scan (starting strategy)

The opening scan is the order in which a participant first visits the four answer positions: the
first four entries of the merged *place* sequence, e.g. (top, right, bottom, left). It comes in
two versions, with visits to the question kept (`_with_q`) or removed first (`_no_q`). A scan
that reached fewer than four answers stays shorter. Everything is in `features/strategies.py`.

Terms used in the paper:

| Term | Meaning |
|---|---|
| **first scan** | the opening pass over the four answers |
| **clockwise** | the most common first scan: top, right, bottom, left (`answer_0(top)` → `answer_2(right)` → `answer_3(bottom)` → `answer_1(left)`) |
| **counter-clockwise** | the second most common: from the top, round the other way |
| **dominant strategy** | a participant has one if their most common first scan covers at least half of their trials |
| **interrupted-scan completion** | a repair for first scans shorter than four: each is completed the way that start is most often completed in the dataset's full-length scans |

Columns:

| Constant | Column | Meaning |
|---|---|---|
| `STRATEGY_COL` | `strategy` | a trial's scan strategy |
| `STARTING_STRATEGY_COL` | `starting_strategy` | its first four entries |
| `DOMINANT_STARTING_STRATEGY` | `dominant_starting_strategy` | the participant's most common starting strategy |
| `DOMINANCE_SCORE` | `dominance_score` | the share of their trials that use it |
| `BREAKS_PATTERN_WITH_Q` / `_NO_Q` | `breaks_pattern_with_q` / `_no_q` | 1 if this trial's starting strategy differs from the participant's dominant one |
| `BREAKS_X_DOMINANCE_*` | `breaks_x_dominance_*` | `breaks_pattern` × `dominance_score`: breaking the pattern counts for more in a consistent person |
| `STRATEGY_DISTANCE_*` | `strategy_distance_*` | how far this trial's starting strategy is from the dominant one: the number of entries to change (edit distance) |
| `N_STRATEGY_TRIALS_*` | `n_strategy_trials_*` | how many trials the dominant strategy was worked out from. **Read this before reading a dominance score**: the score is a proportion, and looks the same from 145 trials as from 46 |
| `REGIME_COLUMN` | `regime` | Study 2's knowledge regime (§5), kept in the trial table so results can be grouped by it. Not in Study 1 |

Functions:

| Function | Does |
|---|---|
| `build_starting_strategies` | the per-trial starting strategy |
| `dominant_strategy_by_participant` | each participant's most common one, with `dominance_score` and `n_strategy_trials`. A tie goes to the strategy that sorts first |
| `has_dominant_strategy` | the "at least half" test (`>=`, `DEFAULT_DOMINANCE_THRESHOLD = 0.5`) — the only place the threshold is applied |
| `build_prefix_completion_map`, `add_completed_strategy_column` | interrupted-scan completion: learn from the full-length scans how each start is most often completed, then fill in the short ones |
| `dominance_gap_by_participant`, `strategy_variety_by_participant`, `dominant_strategy_counts`, `summarize_completion_effect` | descriptive summaries for the scan-strategy analysis |
| `build_trial_level_pattern_features` | the model's per-trial columns above |

Things to know:

- **They depend on which trials they are given.** They summarise a person over the trials in the
  table passed in, so a person needs all of their trials there. The hunters/gatherers split is
  safe, since each person is in one group. Dropping some of a person's trials is not:
  cross-validation folds, correct-only subsets, a single Study 2 regime. See `pitfalls.md` §3.
- **Completion is learned once per dataset**, over all of it: not per group, so the repair does
  not depend on the grouping, and never across datasets. The paper's descriptive figures show
  scans both with and without completion; the model features use them without.
- **Clockwise and counter-clockwise are not categories in the code.** Strategies are raw tuples
  of positions; ranked by how common they are, the top two turn out to be the two circuits.
- In the paper, the opening scan *not* predicting correctness is itself a result.

---

## 9. The last look

| Constant | Column | Meaning |
|---|---|---|
| `LAST_VISITED_LABEL` | `last_answer_area_visited_lbl` | the last answer looked at in the trial; if the very last look was at the question, the last answer before it |
| `LAST_LBL_BEFORE_SELECT` | `last_lbl_before_select` | the area looked at just before the final selection press |
| `LAST_LBL_BEFORE_CONFIRM` | `last_lbl_before_confirm` | the area looked at just before the confirming press |

The last two are timed from the button presses, and go into the model as 0/1 columns, one per
area. `last_answer_area_visited_lbl` is not in the model table.

Careful with `last_answer_area_visited_lbl`: in a trial where no answer word was looked at, it
still names an answer, picked arbitrarily. That happens on 28 Study 1 trials and 1 Study 2 trial.

In the paper, the last look is read as checking a decision already made ("self-validation"),
while the attention measures reflect the decision itself.

---

## 10. Response times and reading times

The participant selects an answer, can change it, and then confirms with a second press.

| Constant | Column | Meaning |
|---|---|---|
| `CONFIRM_FINAL_ANSWER_RT` | `CONFIRM_FINAL_ANSWER_RT` | from the answers appearing to the confirming press, in ms, as the tracker reports it |
| `TOTAL_ANSWERING_RT` | `total_answering_RT` | the same number, under a clearer name |
| `TOTAL_ANSWERING_RT_NORMALIZED` | `total_answering_RT_normalized` | divided by the number of words on screen |
| `NUM_OF_SELECTS` | `ANSWER_PRESS_NUMBER` | how many selection presses before confirming |

**Reading time per area**, for `question` and `answer_A`..`answer_D` (and on the paragraph screen,
`critical`, `distractor` and `outside`):

| Columns | Meaning |
|---|---|
| `TFD_pure_*` / `TFD_normalized_*` | total fixation duration: the looking time on the area's words, added up |
| `RT_pure_*` / `RT_normalized_*` | time spent on visits to the area; a visit is a run of fixations in a row on it |
| `TimeSinceOffset_pure_*` / `TimeSinceOffset_normalized_*` | from the start of the first look to the end of the last look, so time spent elsewhere in between counts too |

`_normalized` divides by the area's number of words. An area nobody looked at gets 0.
`data-pipeline.md` §7 has the details.

### The paragraph's three parts

| Constant | Values |
|---|---|
| `AUXILIARY_SPAN_TYPE_COLUMN` (`auxiliary_span_type`) | `critical` (holds the answer), `distractor` (supports a wrong answer), `outside` (the rest) |

---

## 11. Cross-validation regimes

Folds are worked out in advance and read from `data/cv_folds/`, one file per fold
(`fold_{i}_trial_ids_by_regime.csv`). Each participant × paragraph pair is given one of seven
regimes, crossing whether the participant and the item were seen in training:

| Regime | Participant | Item |
|---|---|---|
| `train_train` | seen | seen |
| `val_seen_subject_unseen_item` / `test_seen_subject_unseen_item` | seen | **new** |
| `val_unseen_subject_seen_item` / `test_unseen_subject_seen_item` | **new** | seen |
| `val_unseen_subject_unseen_item` / `test_unseen_subject_unseen_item` | **new** | **new** |

The paper reports three, named in figures and folders `new_item`, `new_subject` and `both`.
`both` (new participant *and* new item) is the strictest test of generalising.

Nothing in the model is tuned, so the `val_*` regimes do not choose anything: they are a second
held-out sample, like `test_*`. `modeling/crossval.py` reads each regime name as a `split` (`val`
or `test`) and a `novelty` (`new_item`, `new_subject` or `both`). `eval_split` picks which splits
to evaluate — both by default — and `summary_by_novelty_df` gives one result per novelty: the
paper's numbers. The three novelties ask different questions and differ a lot in size, so
`summary_overall_df`, which averages over them, is only a quick check that a run worked.

---

## 12. Error analysis and guessing

Terms used in the paper's error-analysis section.

| Term | Meaning |
|---|---|
| **FN / FP asymmetry** | false negatives occur ~**6×** more often than false positives. No recall/precision preference is applied in training, so this is a property of the data, not of the objective |
| **theoretical relabeling** | restating the task as *knows the answer* / *doesn't know*, rather than *answered correctly* / *incorrectly*. The two label sets correlate strongly but not symmetrically |
| **the asymmetry itself** | someone who knows will almost certainly choose correctly; someone who doesn't still has at least a 1-in-4 chance of choosing correctly by luck. So under the confidence reading, FN and FP are not equally serious errors |
| **empirical guess rate** | the share of correct answers in Study 2's **no-knowledge** regime, where the participant could not have known |
| **guessing models** | three assumptions the errors are restated under: guessing among **all four** answers (1/4), having **ruled one out** (1/3), or down to **two** (1/2) |
| **unlikely trial** | a trial whose predicted probability strongly disagrees with the outcome: a probably-guessed correct answer, or a wrong answer where the correct one was seriously considered. Code: `src/explorations/unlikely_analysis/`, which does not back the paper; the paper's error analysis comes from `notebooks/drivers/general_model_confusion.ipynb` |
| **model-friendly / model-unfriendly participant** | someone whose features the model predicts well / poorly. The model struggles with participants who attend heavily to the question, and with those who skim the answers, the correct one included |

---

## 13. Other names

| Term | Meaning |
|---|---|
| **OneStop / OneStopQA** | the source corpus: paragraphs in two versions (`Adv`, `Ele`), each paragraph with three reading-comprehension questions |
| **L1** | native language. Study 1 uses OneStop's **L1** part, the native English speakers, so `l1` and `l1_onestop` mean "the native-speaker sample" — not "level 1" or "study 1". Non-native data would need its own name |
| **L2** | a second, learned language. OneStop also has an L2 part, which is not used here. Study 2's participants have mixed language backgrounds, a stated limitation: the two studies are not matched on it |
| **KnowQA** | Study 2: the knowledge-regime experiment |
| **testrun_QA / second_test** | the two pilots run before KnowQA |
| **EyeBench** | the lab's paragraph-feature code, kept as received in `src/vendor/eyebench/` |
| **matching / preference matching** | whether the chosen answer is the one the participant's gaze favoured on a given measurement (`features/preference.py`) |
| **zero-fixation trials** | trials with no fixation on any word. They stay in the model table, with every reading time 0. Study 1 has 9, from 6 participants, listed in `l1_onestop/interim/strange_trials.csv`. Study 2 has 1, the second pilot 10 (one participant, ten trials in a row) and the first pilot none; no file lists those. Whether they belong in the model table is an open question, and the code does not decide it |
