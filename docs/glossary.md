# Glossary

Vocabulary of the project: the columns, the units of analysis, and the words that
mean something specific here. Drafted from `src/constants.py` and `src/data_paths.py`.

**Status: partially reviewed.** The `L1`/`L2` naming, the A > B > C > D answer construction,
the strategy vocabulary and the regime names are confirmed. The rest is still my reading of
`constants.py` and `data_paths.py` — items marked `?` are inference, not established fact.

---

## 1. Grain — the single most important distinction

Three different "one row per" conventions exist in this project, and mixing them up
is the main way a number goes wrong.

| Grain | One row per | Where it lives |
|---|---|---|
| **IA-level** (interest-area) | one word, in one area, in one trial, for one participant | `all_participants.csv`, `hunters.csv`, `gatherers.csv` — the output of `data_csv_generation` |
| **area-level** | one of the 5 screen areas, in one trial | intermediate; what `mixed_area_comparisons` dedups to |
| **trial-level** | one (participant, trial) | `L1_model_ready_all_features.csv` — the output of `model_data.save_all_features`, and what all modeling consumes |

An IA-level frame has roughly 30–60 rows per trial (one per word on screen). Trial-level
features are *broadcast* onto every IA row when they appear in an IA-level frame, so
`groupby([TRIAL_ID, PARTICIPANT_ID]).first()` is the correct way to recover them —
counting IA rows inflates every n by ~39×.

> Any statistical test on a trial-level property must run on a trial-level frame.

---

## 2. Identity: who and which trial

The L1 (OneStop) and KnowQA datasets identify trials **differently**, and the
KnowQA scheme is deliberately designed to keep the L1 code working unchanged.

| Constant | Column | L1 / OneStop | KnowQA |
|---|---|---|---|
| `PARTICIPANT_ID` | `participant_id` | the recording label; one label = one person | the **person** only (`4000`) |
| `SESSION_ID` | `session_id` | — | the full recording label (`4000_21`) |
| `TRIAL_ID` | `TRIAL_INDEX` | integer trial counter | composite **string** (`b2l01t005`) |
| `TRIAL_NUMBER` | `trial_number` | — | the tracker's within-session integer counter |

**Why.** A KnowQA recording label is `4XXX_YZ`: person `4XXX`, batch `Y` (1–3), list
`Z` (1–18), with no separator because the tracker caps labels at 8 characters. One
person can sit for several (batch, list) sessions, so `participant_id` alone no longer
identifies a session and `(participant_id, trial_number)` no longer identifies a trial.
Rather than widen the `(participant_id, TRIAL_INDEX)` key the whole pipeline is built
on, batch and list are folded into `TRIAL_INDEX` itself.

**Consequence to remember:** for KnowQA, per-participant features (pupil baselines,
dominant strategy, dominance score) pool across a person's several sittings unless
computed per `session_id`. Pupil z-scoring is already done per session; the
pattern-breaking features are not.

`TRIAL_ID_COLS = (PARTICIPANT_ID, TRIAL_ID)` is the join key used by essentially every
merge in the project.

---

## 3. Items: which text, which question

| Constant | Column | Meaning |
|---|---|---|
| `ARTICLE_COLUMN` | `article_id` | which article |
| `BATCH_COLUMN` | `article_batch` | article batch (1–3) |
| `PARAGRAPH_COLUMN` | `paragraph_id` | which paragraph within the article |
| `DIFFICULTY_COLUMN` | `difficulty_level` | Adv / Ele (OneStopQA's two versions of each text) |
| `LIST_COLUMN` | `list_number` | which counterbalancing list the participant got |
| `TEXT_ID_COLUMN` | `text_id` | identifies the paragraph |
| `TEXT_ID_WITH_Q_COLUMN` | `text_id_with_q` | paragraph **+ question** — the item identifier |

### The two question orderings — do not conflate

| Constant | Column | Ordering |
|---|---|---|
| `SAME_CRITICAL_SPAN_COLUMN` | `same_critical_span` | question index within the paragraph **in the L1 experiment's ordering**; the last component of `text_id_with_q` |
| `ONESTOPQA_QUESTION_ID` | `onestopqa_question_id` | question index **in OneStopQA's own ordering**, as recorded by the KnowQA experiment builder |

These are different permutations of the same three questions and **disagree on ~40% of
trials**. Conflating them silently pairs up the wrong items. `know_qa_dataprep` derives
`same_critical_span` from the report's `text_id_with_q` suffix, never from
`onestopqa_question_id`.

---

## 4. The screen and its five areas

The trial screen analysed in this project (OneStop's page 4) shows the question plus
four answer options. Answer options are arranged to match the gamepad's diamond D-pad.

Two orthogonal ways to name an area:

| | Constant | Values |
|---|---|---|
| **By label** (semantic) | `LABEL_CHOICES` | `question`, `answer_A`, `answer_B`, `answer_C`, `answer_D` |
| **By location** (spatial) | `LOC_CHOICES` | `question`, `answer_0(top)`, `answer_1(left)`, `answer_2(right)`, `answer_3(bottom)` |

`ANSWER_LABELS = ["A", "B", "C", "D"]`. **`answer_A` is always the correct option** —
labels are semantic, not positional. Screen position is randomized per trial and lives
in the `*_loc` columns. `LABEL_CHOICES` is marked `# DO NOT CHANGE` in `constants.py`
because ordering consistency is assumed elsewhere.

### The labels are ordinal, not arbitrary

OneStopQA builds each option from a defined relationship to the paragraph:

| Label | Source | Role |
|---|---|---|
| `answer_A` | **critical span** | correct |
| `answer_B` | **critical span** | a misinterpretation of the correctly relevant text |
| `answer_C` | **distractor span** | built from a different real segment of the text |
| `answer_D` | — | not present in the text at all |

So plausibility runs **A > B > C > D**, and the data follows it (selection 84.07 / 8.97 /
4.85 / 2.11%). The code does not encode this ordering — `ANSWER_LABELS` is a flat list and
`wrong_mean` averages B, C and D as equals. That is accepted. See `research-context.md` §2.

| Constant | Column | Meaning |
|---|---|---|
| `AREA_LABEL_COLUMN` | `area_label` | which of the five areas this IA belongs to, by label |
| `AREA_SCREEN_LOCATION` | `area_screen_loc` | same, by screen position |
| `SELECTED_ANSWER_LABEL_COLUMN` | `selected_answer_label` | which label the participant chose |
| `SELECTED_ANSWER_POSITION_COLUMN` | `selected_answer_position` | which position they chose |
| `CORRECT_ANSWER_POSITION_COLUMN` | `correct_answer_position` | where the correct answer sat |
| `IS_CORRECT_COLUMN` | `is_correct` | 1 if selected == correct. A trial can never lack a confirmed selection, so the NaN-compares-unequal branch is unreachable — it should be an assertion (`todo.md` T3.17) |

---

## 5. Experimental conditions

### Study 1 (L1 / OneStop) — question preview

| Constant | Column | Meaning |
|---|---|---|
| `QUESTION_PREVIEW_COLUMN` | `question_preview` | whether the participant saw the question *before* reading the paragraph |

- **hunters** = `question_preview == True` — saw the question first, read to hunt for the answer
- **gatherers** = `question_preview == False` — read the paragraph first, then saw the question

**This is a BETWEEN-participant manipulation.** Every subject is either a hunter or a
gatherer, never both — verified: 360 participants, 180 per group, no overlap. That is why
`hunters.csv` / `gatherers.csv` can be analysed separately without disturbing anything
computed per participant.

This is the split behind `hunters.csv` / `gatherers.csv` and behind every
`hunters` / `gatherers` / `all_participants` output folder.

Also excluded during prep: `REPEATED_TRIAL_COLUMN` (`repeated_reading_trial`) and
`PRACTICE_TRIAL_COLUMN` (`practice_trial`).

### Study 2 (KnowQA) — knowledge regime

The manipulation is **prior knowledge**, in three regimes, replacing question-preview:

- **full knowledge** — the participant is shown the correct answer's content beforehand (rephrased)
- **partial knowledge** — the paragraph is presented to read (the closest analogue to Study 1)
- **no knowledge** — no context at all; any correct answer is a guess

> ⚠️ **This is a WITHIN-participant manipulation — the opposite of Study 1.** Each KnowQA
> participant does **all three regimes in every run**. So, unlike the hunters/gatherers split,
> **splitting KnowQA by regime splits a participant's trials**: anything computed per
> participant (dominant strategy, dominance score, pupil baselines) would then be computed on
> roughly a third of that person's data. See `pitfalls.md` §3.

Plus a per-trial **confidence self-report on a 1–5 scale**, which Study 1 lacked.
Counterbalanced by a double Latin square (regime × article group, then question ordering)
across 27 lists, times 6 regime orderings = 54 ordered lists.

---

## 6. Per-area eye-movement metrics

Computed separately for each of the five areas, from per-word (IA) data. Unfixated words
count as zero for dwell/fixation measures and are excluded from pupil measures.

| Constant | Column | Meaning |
|---|---|---|
| `MEAN_DWELL_TIME` | `mean_dwell_time` | mean per-word dwell time in the area |
| `MEAN_FIXATIONS_COUNT` | `mean_fixations_count` | mean fixations per word |
| `MEAN_FIRST_FIXATION_DURATION` | `mean_first_fixation_duration` | mean duration of each word's first fixation |
| `SKIP_RATE` | `skip_rate` | proportion of words in the area never fixated |
| `AREA_DWELL_PROPORTION` | `area_dwell_proportion` | share of the trial's total dwell time in this area; sums to 1 across the five areas |
| `NUM_LABEL_VISITS` | `num_label_visits` | number of times the participant enters the area (by label) |
| `NUM_LOC_VISITS` | `num_loc_visits` | same, by location |

Pupil measures, raw and z-scored (`_z` suffix = z-scored within participant, or within
session for KnowQA):

`MEAN_AVG_FIX_PUPIL_SIZE`, `MEAN_MAX_FIX_PUPIL_SIZE`, `MEAN_MIN_FIX_PUPIL_SIZE`,
`FIRST_ENCOUNTER_AVG_PUPIL_SIZE` — each with a `_z` counterpart.

Pupil sizes are converted from the tracker's area units to mm diameter before z-scoring.

Two curated lists:
- `AREA_METRIC_COLUMNS_MODELING` — the `_z` pupil variants; what models consume
- `AREA_METRIC_COLUMNS_VIZES` — the raw pupil variants; what plots consume

---

## 7. Derived (contrast) features — how per-area becomes per-trial

Each per-area metric over the four answer options collapses into five trial-level
features. Naming: `{metric}{DERIVED_SEP}{suffix}` where `DERIVED_SEP = "__"`.

| Suffix constant | Suffix | Meaning |
|---|---|---|
| `CORRECT_SUFFIX` | `correct` | the metric's value on the correct answer |
| `WRONG_MEAN_SUFFIX` | `wrong_mean` | mean of the metric across the three wrong answers |
| `CONTRAST_SUFFIX` | `contrast` | correct − wrong_mean |
| `DISTANCE_FURTHEST_SUFFIX` | `distance_furthest` | max absolute distance between correct and any wrong answer |
| `DISTANCE_CLOSEST_SUFFIX` | `distance_closest` | min absolute distance between correct and any wrong answer |

e.g. `mean_dwell_time__contrast`, `skip_rate__correct`.

**Note on imputation:** missing values in these columns are filled with `0.0` before
standardization. For a `contrast` feature, 0 means "read exactly like the others"; for a
z-scored pupil feature, 0 means "this participant's own mean". Worth stating in Methods.

---

## 8. Fixation sequences

A trial's fixations are converted into a sequence of area symbols, in temporal order.

| Constant | Column | Meaning |
|---|---|---|
| `FIX_SEQUENCE_BY_LABEL` | `fix_by_label` | fixation sequence, areas named by label |
| `FIX_SEQUENCE_BY_LOCATION` | `fix_by_loc` | same, by screen location |
| `SIMPLIFIED_FIX_SEQ_BY_LABEL` | `simpl_fix_by_label` | **collapsed**: consecutive identical entries merged into one |
| `SIMPLIFIED_FIX_SEQ_BY_LOCATION` | `simpl_fix_by_loc` | same, by location |
| `SEQUENCE_LENGTH_COLUMN` | `sequence_length` | number of entries in the collapsed sequence = number of area transitions |

Two prep steps applied: fixations outside any interest area are assigned to the nearest
word; and an isolated leading fixation on the question (a single fixation immediately
followed by a move elsewhere) is removed as spillover from the question screen.

`IA_LABEL` (`IA_LABEL`) holds the word an interest area covers — one IA per word in
reading order, so it reconstructs the on-screen text. `know_qa_dataprep.check_text_alignment`
uses this to look for a displaced-quote bug that would shift every downstream area boundary by
one word — though **whether that bug actually occurs is unconfirmed** (`todo.md` T3.18).

### Hesitation patterns

- **XYX** — the collapsed sequence ends in a back-and-forth between two options (`[A, B, A]`)
- **XYXY** — the extended version (`[A, B, A, B]`)
- `longest_alternating_answer_run` — the graded version

### Strategies / first-scan behaviour / pattern breaking

Paper-critical as of draft2 — the opening scan is a Results subsection in its own right.

Paper-side vocabulary:

| Term | Meaning |
|---|---|
| **first scan** | the participant's opening pass over the four answer options |
| **clockwise** | the most common first scan: from the top, then round — by *location*, so `answer_0(top)` → `answer_2(right)` → `answer_3(bottom)` → `answer_1(left)` |
| **counter-clockwise** | the second most common: from the top, the other way round |
| **dominant strategy** | a participant has one if their most common first-scan strategy accounts for **at least half** of their trials |
| **interrupted-scan completion** | a repair step applied to scan passes **shorter than four items**, which raises the measured prevalence of dominant strategies. Implemented as `build_prefix_completion_map_from_series` + `add_completed_sequence_column` in `viz/visualisations_strategies.py` |

Code-side constants:

| Constant | Meaning |
|---|---|
| `STRATEGY_COL` (`strategy`) | the trial's scanning strategy |
| `STARTING_STRATEGY_COL` | the first `window_len=4` tokens of the collapsed **location** sequence (`simpl_fix_by_loc`) |
| `DOMINANT_STARTING_STRATEGY` | the participant's modal starting strategy |
| `DOMINANCE_SCORE` | how consistently they use it |
| `BREAKS_PATTERN_WITH_Q` / `_NO_Q` | this trial's starting strategy differs from the participant's dominant one; two variants, question tokens kept vs. dropped |
| `BREAKS_X_DOMINANCE_*` | interaction: `breaks_pattern × dominance_score` — "how much breaking matters, scaled by how consistent the person is" |
| `STRATEGY_DISTANCE_*` | graded version: token-level Levenshtein distance from the dominant strategy |

**Caveat 1 — frame dependence.** These are computed over whatever trials are in the frame
passed in, so a participant needs their **full trial set** present.

The hunters/gatherers split is **safe**: `question_preview` is between-participant (360
participants, 180 per group, none appearing in both), so each participant's whole trial set is
in one group file and the values come out identical either way. What is *not* safe is any
filtering that keeps a participant but drops some of their trials — CV regime rebuilds,
correct-trials-only subsets. For KnowQA there is a separate issue: one `participant_id` spans
several sessions within the same file, so these features pool across sittings.
See `docs/pitfalls.md` §3.

**Caveat 2 — two implementations, one with completion and one without.** *(Duplication to be
removed — `todo.md` T1.6. The completion-on/off difference is a real parameter; the two code
paths are not.)*
- `viz/visualisations_strategies.py::build_strategy_dataframe` — descriptive side. Followed
  by `build_prefix_completion_map_from_series` + `add_completed_sequence_column`, which **is**
  the interrupted-scan completion: from strategies observed at full length 4, learn how each
  prefix is most often completed, then fill the short ones. The paper's prevalence figures
  come from here, reported both raw and completed.
- `derived/pattern_breaking.py::build_starting_strategies` — model-feature side. **No**
  completion; two variants (question tokens kept / dropped).

They duplicate `_parse_seq` and the windowing, and tie-break differently (`idxmax` vs an
explicit deterministic rule), so they can pick different dominant strategies on ties.

**Caveat 3 — the threshold operator is inconsistent.** `proportion_with_dominant_strategy`
uses strict `prop > threshold`; `summarize_before_after` and
`plot_dominant_strategy_counts_above_threshold` use `prop >= threshold`. Same data, two
numbers (hunters: 46.1% vs 48.9%). Draft2's "at least half" implies `≥`.

**Caveat 4 — no all-participants figure.** `run_all_strategy_plots` calls
`split_participant_groups(..., include_all=False)`, so only hunters and gatherers are
produced.

**Note — clockwise and counter-clockwise are not code categories.** Strategies are raw
location tuples; `plot_dominant_strategy_counts_above_threshold` ranks them and the top two
turn out to *be* the clockwise and counter-clockwise circuits. No classifier exists or is
needed, though naming them would improve the paper figure. See `docs/findings.md` §1.1.

**Note on interpretation:** the first-scan strategy is *not* predictive of correctness, and
draft2 treats that as the finding — evidence that the opening scan is a necessary
scanning behaviour independent of knowledge state. An earlier commit message calls the
feature "a bust"; in draft2 it is a null result doing real work.

---

## 9. Last visitation — the "self-validation" baseline

Three perspectives on the participant's final fixation:

| Constant | Column | Meaning |
|---|---|---|
| `LAST_VISITED_LABEL` | `last_answer_area_visited_lbl` | last *answer* area fixated (steps back if the last fixation was on the question) |
| `LAST_LBL_BEFORE_SELECT` | `last_lbl_before_select` | last area fixated before the final answer selection |
| `LAST_LBL_BEFORE_CONFIRM` | `last_lbl_before_confirm` | last area fixated before the confirmation press |

~80% of participants look at the answer they ultimately select. Used both as a strong
predictor and as an interpretive claim: the attention features reflect the decision, the
last fixation reflects self-validation after it.

---

## 10. Response times

Participants select an answer, then confirm with a second button press; selection can
change before confirmation, and the unconfirmed presses are recorded.

| Constant | Column | Meaning |
|---|---|---|
| `CONFIRM_FINAL_ANSWER_RT` | `CONFIRM_FINAL_ANSWER_RT` | raw: answers-shown → confirm click |
| `TOTAL_ANSWERING_RT` | `total_answering_RT` | the same, exposed under a friendlier name |
| `TOTAL_ANSWERING_RT_NORMALIZED` | `total_answering_RT_normalized` | normalized version |
| `NUM_OF_SELECTS` | `ANSWER_PRESS_NUMBER` | how many selection presses before confirming |

Per-region RT/TFD families, one per area (`question`, `answer_A`…`answer_D`, and for
paragraph runs `outside` / `distractor` / `critical`):

| Prefix | Meaning |
|---|---|
| `RT_pure_*` / `RT_normalized_*` | **run-based** reading time on the region; `_normalized` divides by the region's word count |
| `TFD_pure_*` / `TFD_normalized_*` | total fixation duration — the sum of dwell times |
| `TimeSinceOffset_*` | **span-based**: last-fixation-end minus first-fixation-start, so it counts every excursion away and back as part of the region's time |

`RT_*` is always run-based; the span measure is deliberately renamed to
`TimeSinceOffset_*` to keep the two apart. Unfixated regions are filled with `0`, not
NaN — so "looked for 0 ms" and "no data" are currently indistinguishable.

### Paragraph spans

For paragraph-reading screens, the text is divided by relevance to the question:

| Constant | Values |
|---|---|
| `AUXILIARY_SPAN_TYPE_COLUMN` (`auxiliary_span_type`) | `critical` (the span containing the answer), `distractor` (the span supporting a wrong answer), `outside` (the rest) |

---

## 11. Cross-validation regimes

Folds are pre-computed and read from `fold_{i}_trial_ids_by_regime.csv`. Each
(participant × paragraph) cell is labelled with one of seven regimes, crossing subject
and item novelty:

| Regime | Subject | Item |
|---|---|---|
| `train_train` | seen | seen |
| `val_seen_subject_unseen_item` / `test_seen_subject_unseen_item` | seen | **new** |
| `val_unseen_subject_seen_item` / `test_unseen_subject_seen_item` | **new** | seen |
| `val_unseen_subject_unseen_item` / `test_unseen_subject_unseen_item` | **new** | **new** |

The paper reports three test regimes; in figure and folder names these appear as
`new_item`, `new_subject`, and `both`. `both` = `test_unseen_subject_unseen_item` is the
strict generalization regime.

Since the live model has no tuned hyperparameters, the `val_*` regimes currently serve no
selection purpose — they are effectively a second test set. `?` Worth deciding whether to
report or drop them.

---

## 12. Error analysis and guessing

Vocabulary for the error-analysis Results subsection (expanded substantially in draft2).

| Term | Meaning |
|---|---|
| **FN / FP asymmetry** | false negatives occur ~**6×** more often than false positives. No recall/precision preference is applied at training, so this is a property of the data, not of the objective. |
| **theoretical relabeling** | restating the task as *knows the answer* / *doesn't know*, rather than *answered correctly* / *incorrectly*. The two label sets correlate strongly but asymmetrically. |
| **the asymmetry itself** | someone who knows will almost certainly select correctly; someone who doesn't still has **≥ 1-in-4** chance of selecting correctly by luck. This is why FN and FP are not symmetric errors under the confidence interpretation. |
| **empirical guess rate** | the observed proportion of correct responses in Study 2's **no-knowledge** regime — trials where we know the participant could not have known. Fills in the paper's `x%`. |
| **guessing models** | three assumptions the errors are restated under: guessing among **all four** options (1/4), having **eliminated one** (1/3), or narrowed to a **fifty-fifty** (1/2). |
| **unlikely trial** | a trial whose predicted probability disagrees strongly with the outcome — a likely-guessed correct answer, or a wrong answer where the correct option was seriously considered. Code: `unlikely_analysis.py` — **stale, and probably not part of this paper** (2026-09-05). The error-analysis subsection is served by `general_model_confusion.ipynb` instead |
| **model-friendly / model-unfriendly participant** | someone whose feature profile the model predicts well / poorly. The model struggles on participants who attend heavily to the question, and on those who skim the answers including the correct one. |

---

## 13. Other names worth knowing

| Term | Meaning |
|---|---|
| **OneStop / OneStopQA** | the source corpus: paragraphs at two difficulty levels, each with three comprehension questions |
| **L1** | **native language.** In this project's filenames it means the **L1 OneStop dataset** — the native-English-speaker participants. So `L1_based_data/`, `L1_model_ready_all_features.csv`, `L1_paragraph_span_features.csv` are all derived from the native-speaker sample. The prefix encodes a **sample restriction**, not a study number: any future non-native data needs its own namespace rather than reusing these paths. |
| **L2** | a second, learned language. OneStop also has an L2 (non-native English) portion, which this project does not use. **Study 2 recruits participants of mixed linguistic backgrounds**, which is a stated limitation of that study — the two studies are not matched on language background. |
| **KnowQA** | Study 2: the new knowledge-regime experiment, and the first real collection under it |
| **testrun_QA / second_test** | the two pilots that preceded KnowQA |
| **EyeBench** | the lab's external paragraph-feature extraction pipeline, vendored into `src/external/EyeBench/` |
| **matching / preference matching** | whether the answer the participant selected is the one their gaze "preferred" on a given metric |
| **strange trials** | `data/strange_trials.csv` — old trials that looked suspicious, chiefly **zero fixations**. Historical; may not even hold against the current data, and nothing important depends on it. Not a live exclusion mechanism |

---

## Open questions for Diana

*(None currently — the `val_*` question was answered 2026-09-05, see below.)*

**Resolved 2026-09-04/05:**

- `val_*` regimes — **kept and reported**, not dropped. Because the logreg tunes no
  hyperparameters they function as a second test set rather than a validation set, which is
  fine as long as they are shown separately. What does change: `summary_overall_df` should
  stop averaging all six regimes into a single `mean_balanced_accuracy`, since that number
  mixes val and test (`todo.md` T3.9).

- `L1`/`L2` mean native / second language (§13).
- **"Partial knowledge"** is the canonical name for Study 2's middle regime (§5).
- The **A > B > C > D** answer construction is documented in §4.
- **Interrupted-scan completion exists** (`build_prefix_completion_map_from_series`), and
  **no clockwise/counter-clockwise classifier is needed** — those are observations about
  which strategy tuples came out on top, not a category the code assigns (§8).
- **`strange_trials.csv`** is a historical list of zero-fixation trials; not a live filter (§13).
- **`unlikely_analysis.py`** is stale and probably out of this paper (§12).
- **Study 1's group manipulation is between-participant; Study 2's regime manipulation is
  within-participant** (§5) — they are not analogous splits.
