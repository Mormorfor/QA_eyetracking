# Research context

The science behind the code. Drafted from
`papers/correctness_prediction/Nature Comunications/draft2.tex` (the live paper outline)
and the code itself.

**Status: reviewed with Diana 2026-09-04/05.** §§1–6 have been corrected by her; the
decisions are listed at the end of §7. Still my reading of her paper draft rather than an
authoritative statement of the research — where this and the draft disagree, the draft wins.

> `papers/` is read-only for Claude. This file exists so the code's purpose is legible from
> inside the repo; it is not a substitute for the paper and should not be treated as the
> canonical statement of the claims.
>
> **Tracking:** `draft2.tex` supersedes `draft1.tex` (results-first restructure, three new
> Results subsections). `abstract.tex` and `paper.tex` in the parent folder are the older
> ACM-era drafts and describe a mixed-effects model that was never used — treat them as
> retired.

---

## 1. The question

When someone answers a multiple-choice comprehension question, the only thing normally
recorded is which option they picked. The reasoning that produced it — confident knowledge,
elimination, or a guess — is invisible. Prior work predicting comprehension from eye
movements has looked at eye movements over the **text**; this project looks at eye movements
over the **question and answer options**, after reading is already finished.

Three claims, in the order the paper makes them:

1. **Correctness is predictable** from gaze over the answer options — the outcome is already
   visible at the decision-making stage.
2. **There is legible strategic behaviour** at both ends of the trial: a consistent,
   person-specific opening scan, and a distinct closing validation fixation.
3. **The prediction is graded, not binary.** The model's probability tracks the person's
   strength of belief in their own answer — usable as a substitute for (unreliable)
   confidence self-reports, and as a way to separate genuine knowledge from guessing.

Claim 3 is what Study 2 exists to validate.

---

## 2. Study 1 — OneStop

### Dataset

The OneStop Eye Movements corpus (Berzak et al., 2025). 360 participants, each reading 10
articles, 54 paragraphs total. Each paragraph is followed by one multiple-choice
comprehension question, selected from three possible questions for that paragraph.

**The sample is native English speakers.** `L1` means *native language*, and this project
uses the **L1 portion of OneStop** — hence the `L1_` prefix on every processed output
(`data/L1_based_data/`, `L1_model_ready_all_features.csv`). OneStop also has an L2
(non-native) portion, which is not used here.

This matters for the paper in two places: Study 1's sample description, and Study 2's
limitations — **Study 2 recruits participants of mixed linguistic backgrounds**, so the two
studies are not matched on language background. Draft2's `\section{Limitations}` is currently
empty; this belongs in it.

It also matters for the code: the `L1_` prefix marks a **sample restriction**, not a study
number. Any future non-native data needs its own namespace rather than reusing these paths.

### Design

Participants move through a fixed sequence of screens with no way back:

| Screen | Content |
|---|---|
| 1 | question preview — **only for half the participants** |
| 2 | the paragraph (the other half start here) |
| 3 | the question |
| 4 | four answer options, question still on screen |

**This project analyses screen 4 only** (with the exception of the Text Associations
analysis, which relates screen 2 behaviour to screen 4 behaviour).

The preview manipulation gives the two participant groups the project refers to throughout:

- **hunters** — saw the question first, read the paragraph hunting for the answer
- **gatherers** — read the paragraph first, then saw the question

The draft states there are no noticeable differences between the groups except where
specified — and two places do specify: dominant-strategy prevalence (§3.1) and the
critical-span effect (§3.6).

Responses use a Logitech Gamepad F310 whose diamond D-pad matches the on-screen answer
layout. Participants **select** an option, then **confirm** with a second press; selection
can change before confirmation, and the unconfirmed presses are recorded. Each item has one
correct option (always labelled `answer_A` internally) and three distractors; screen
position is randomized per trial, so label and location are independent (glossary §4).

### How the four answers are constructed — the key to the Text Associations findings

OneStopQA builds each option from a specific relationship to the paragraph:

| Option | Source | Meaning |
|---|---|---|
| **A** | the **critical span** | the correct answer |
| **B** | the **critical span** | a *misinterpretation* of the correctly relevant text |
| **C** | the **distractor span** | built from a different, real segment of the text |
| **D** | — | not present in the text at all |

**So the options are ordered by plausibility: A > B > C > D.** Note that A and B come from the
*same* span — B is the trap for a reader who found the right sentence and misread it, while C
is the trap for a reader who anchored on the wrong sentence.

This ordering is not encoded anywhere in the code (`ANSWER_LABELS` is a flat list), but the
data follows it on two independent measures:

| | A | B | C | D |
|---|---|---|---|---|
| selection rate | 84.07% | 8.97% | 4.85% | 2.11% |
| mean dwell (trials where A chosen) | 245.6 | 177.3 | 150.6 | 146.9 |
| mean fixation count | 1.231 | 0.928 | 0.803 | 0.793 |
| skip rate | 0.356 | 0.452 | 0.492 | 0.484 |

**Selection follows the full ranking; attention flattens at C ≈ D.** The mixed models say the
same: A–B, B–C and B–D are all significant, while **C–D is not** (`findings.md` §3.1). That is
exactly what the construction predicts — B is critical-span material and gets treated as a
live contender, whereas C and D are both "not it" visually even though readers still choose
between them at a 2:1 rate.

The code does not use this ordering — `wrong_mean` averages B, C and D as equals. That is
accepted, not an oversight; using the ordinal structure was considered and deliberately set
aside (2026-09-05).

### Data preparation

Excluded: repeated-reading trials and practice trials.

Because screen 4 is five disjoint text regions rather than continuous prose, every gaze
measure is computed **per region** (question + four answers). Each word is one interest
area; fixations landing outside any interest area are assigned to the nearest word.
Fixations become a temporal sequence of region symbols, then get collapsed so that
consecutive repeats become one symbol — turning the sequence into a record of *transitions
between regions*.

One targeted cleanup: an isolated leading fixation on the question — a single fixation
immediately followed by a move elsewhere — is removed as spillover from screen 3.

### Features

Designed to describe four behaviours:

1. attention to the **correct** answer
2. attention to the **wrong** answers
3. attention to the **question**
4. **shifts** of attention between the five regions

The motivating result from prior work: people attend more to options they subjectively
prefer, and the option a participant believes correct is a preferred one.

Per-region metrics collapse into trial-level features via five contrasts — `correct`,
`wrong_mean`, `contrast`, `distance_furthest`, `distance_closest` (glossary §7). Trial-level
sequence features add hesitation patterns (XYX, XYXY), sequence length, and mean dwell.

**Last fixation** is treated separately, as a different kind of signal — see §3.2.

### Model

Logistic regression with weighted classes, evaluated on **balanced accuracy** (base rate
≈ 0.8 correct). Interpretability is a deliberate constraint, not a convenience: which
behaviours the model finds predictive is itself a research question.

**Framing that matters:** the model knows which option is correct; it does not know what the
participant selected.

### Cross-validation design

Folds cross **subject** and **item** novelty simultaneously, which is stronger than the
usual single-axis split. Seven regimes exist in the code; the paper reports three test
regimes, named `new_item`, `new_subject` and `both` in the figures and folders. `both`
(unseen subject *and* unseen item) is the strict generalization case. See glossary §11.

---

## 3. The findings, as draft2 organizes them

Draft2 is **results-first**: a narrative Results section with six thematic subsections, and
all the "how" moved into Methods. This section follows that order.

Headline: **~0.83 balanced accuracy** on the answer-selection screens.

> Note: the Results opener says "about eighty percent" while Models comparison says 0.83.
> **Not an inconsistency to fix** — the figure varies a little between testing regimes and is
> a placeholder until the final numbers land.

### 3.1 First-scan behavior

**This is new in draft2** and it promotes what was previously a parked experiment into a
headline finding.

- Most participants open with a **clockwise first scan of the answer options, starting from
  the top**. Second most common: **counter-clockwise from the top**.
- This opening scan shows **no predictive relationship with correctness** — read as a
  *necessary* scanning behaviour, independent of knowledge state. (A null result used as
  evidence, not a failure.)
- The scan is **person-static**: X% of participants use their most common first-scan
  strategy on at least half their trials — defined here as having a **dominant strategy**.
- Prevalence differs by group: **Y%** of no-preview (gatherer) participants have a dominant
  strategy vs **Z%** of preview (hunter) participants — i.e. the effect is stronger without
  a question preview.
- All three percentages rise when **interrupted-scan completion** is applied to scan passes
  shorter than four items.

**`X`, `Y` and `Z` have already been computed** — they are in the stored output of
`notebooks/vizes for purpouse/visualisations.ipynb` cell 6. See `docs/findings.md` §1.2 for
the full table; in brief, using the `≥ half of trials` convention that matches draft2's
wording: **hunters 48.9% raw / 53.3% after completion; gatherers 58.3% / 61.1%.** The
group difference runs in the direction the draft claims.

The clockwise / counter-clockwise claim is also already established: the dominant strategies
are literal location tuples, and the top two across both groups are exactly
`(top, right, bottom, left)` and `(top, left, bottom, right)`. Counts in `findings.md` §1.1.

**Code:** `viz/visualisations_strategies.py` (descriptive — `run_all_strategy_plots`, and
the prefix-completion pair `build_prefix_completion_map_from_series` /
`add_completed_sequence_column`, which *is* the interrupted-scan completion),
`viz/visualisations_dominant_eye.py` (`run_dominant_strategy_eye_analysis`),
`viz/visualisations_simplified_visits.py` (the first-four-visits heatmap), and
`derived/pattern_breaking.py` (the per-trial model features derived from the same concept).

> **Two implementations of "starting strategy" currently coexist** — descriptive (with prefix
> completion, in `viz/visualisations_strategies.py`) and model-feature (without, in
> `derived/pattern_breaking.py`). **To be unified: `todo.md` T1.6.** The completion on/off
> difference is a real parameter; two code paths for it is not.
>
> Which variant the *feature* ends up using matters little here: the point of the first-scan
> analysis is that the opening scan is **not** predictive of correctness, and that holds
> either way. Current expectation is the uncompleted version for the feature, completed for
> the descriptive figures.

> ⚠️ Three smaller things, all in `findings.md`: the all-participants `X%` was never
> produced (`run_all_strategy_plots` forces `include_all=False`); the threshold is applied as
> strict `>` in one function and `≥` in two others, which is why the printed 46.1% and the
> summary's 48.9% differ; and "dominant" means *modal*, not consistent — hunters use 6–28
> distinct opening sequences each, modal ≈13. That last point belongs next to the headline.

### 3.2 End of trial behavior

The **final fixation before answer confirmation** falls overwhelmingly on the option the
participant goes on to select (~80%), and is strongly predictive of correctness.

Interpreted as a **validation step before commitment**, not the moment of selection itself.
This is why the attention features and the last-fixation features are treated as different
kinds of evidence rather than pooled.

**Code:** `derived/select_confirm_last.py`, `viz/visualisations_last_label.py`, and the
`LAST_*` feature groups in `answer_correctness/feature_groups.py`.

### 3.3 Attention allocation between answer options

Consistent with prior work on attention to subjectively preferred options: clear preference
for the answer ultimately selected — **lower skip rates, higher mean dwell durations**, etc.

**Code:** `derived/preference_matching.py`, `viz/visualisations_area_bars.py`,
`viz/visualisations_preference_correctness.py` — plus
`statistics/mixed_area_comparisons.py`, which produced the pairwise significance actually
reported in `findings.md` §3.1.

> **Status (2026-09-05): paper code.** Settled — `mixed_area_comparisons.py` is classified with
> the live code, not with the parked mixed-model strands, and its pairwise area comparisons
> belong to this subsection. Practical consequence: its 108 output figures are among the
> zero-byte ones (`todo.md` T4.2) and **need re-running** — the backing CSVs
> (`report_data/area_mixed_models/`, 216 files) survived, so it is a rerun, not a recovery.

> ⚠️ If this subsection uses the pairwise area comparisons, note that
> `reports/plots/area_significance_heatmaps/` contains **108 zero-byte PNGs** — every figure
> from that analysis is an empty file. The backing CSVs in
> `reports/report_data/area_mixed_models/` survived, so it is a regeneration job, not lost
> work. See `docs/status.md`.

### 3.4 Error analysis

Draft2 expands this considerably and gives it a theoretical spine.

- Confusion matrix of the final model on the Study 1 test subset.
- **False negatives are ~6× more frequent than false positives.** No recall/precision
  preference was applied during training, so the model has no built-in reason to favour one
  error type — the asymmetry is a property of the data.
- **The reframing:** if the signal really is confidence, the "true" label should be *the
  participant knows the answer* and "false" *they don't*. Those labels correlate strongly
  with correct/incorrect — but not perfectly, and asymmetrically: someone who knows will
  almost certainly select correctly, whereas someone who doesn't still has **at least a
  1-in-4 chance** of picking correctly by luck.
- Study 2 supplies the empirical version of that guess rate: **x%** of responses in the
  no-knowledge regime are correct.
- Errors are then restated under three guessing models: guessing among **all four** options,
  having **eliminated one**, or narrowed to a **fifty-fifty** between two.

**Code:** `notebooks/general_model_confusion.ipynb` (which already contains a
guessing-correction sensitivity analysis across the three test regimes) and
`person_variance/mistake_types.py`. **`unlikely_analysis.py` is stale and probably not part of
this paper** (2026-09-05).

### 3.5 Participant-level model coefficients

Which features are associated with the model performing well vs. poorly for a given person,
which features help on which trial type (correct vs. wrong), and how consistent that is
across participants — in magnitude, or at least in sign.

The best model is retrained **per participant**, leaving one trial out at a time. Participants
with very few incorrect trials score worst (few negative examples) and are excluded above a
threshold.

**Code:** `answer_correctness/person_variance/` (`coef_consistency.py`,
`univariate_consistency.py`, `accuracy_characterization.py`, `person_accuracy.py`),
`participant_level.py`, `notebooks/per_person_runs.ipynb`.

> Note: draft1's sentences about population-level significance are **commented out** in
> draft2, alongside your `\todo` that "'significance' / CIs are calculated very differently
> here". That instinct is correct — see §5.

### 3.6 Text Associations

Now carries concrete findings. Relates **proportion of time on each paragraph span** to
**proportion of time on each answer option and on the question** (excluding the first
read-through of the question text).

Spans: **critical** (contains the answer), **distractor** (supports a wrong answer),
**outside** (everything else).

- Significant effect of **distractor span → answer C** — answer C is the option built from
  distractor-span content.
- Significant effect of **critical span → answer A** (the correct option) **in
  question-preview participants only**, not in the no-preview group — which fits: only
  preview readers know what to look for while reading.
- Significant **negative** effects on **answers C and D**, which are not built from the
  critical span.

Requires explaining OneStopQA's answer-construction logic, or pointing readers to it.

**Code:** `statistics/RT_correlations/`, driven by `notebooks/text_associations.ipynb`.
**This is the current and authoritative text–QA relationship analysis** (confirmed
2026-09-04). Proportions of time, participant-level Fisher-z, FDR-corrected.

> **`statistics/mixed_text_answer_effects.py` is an older, half-abandoned attempt** at the
> same question and is **not** the source of these findings — it is future-directions code
> with some directions worth revisiting. Its stored output (`findings.md` §7) reaches
> materially different conclusions: `outside` (generic reading effort on the rest of the
> paragraph) as the strongest and only universally significant predictor, `distractor`
> essentially never significant, and `critical`→A significant in *both* groups rather than
> preview-only. Do not mix its numbers into this subsection, and do not treat the
> disagreement as a contradiction to resolve — different method, superseded strand.
>
> The promising part worth keeping: it models item and participant as random effects, which
> `RT_correlations` handles by aggregating to participant level instead.

> ⚠️ **Neither candidate has regenerable figures.** `text_associations.ipynb` writes
> **nothing** to `reports/` — `RT_correlations/plots.py` bypasses `plot_output.py` and takes
> a caller-supplied path the notebook never provides, so those findings exist only as cell
> outputs inside a 1.2 MB notebook. And `reports/plots/texts_to_answers/` — the
> `mixed_text_answer_effects` output — is **165 zero-byte PNGs**.

One byproduct worth keeping: `text_id_with_q` random-effect variance is ~4.8× the
`participant_id` variance. **Item identity matters far more than participant identity**,
which is a good justification for the item-crossed fold design.

---

## 4. Study 2 — KnowQA

### Purpose

Study 1 establishes that correctness is predictable. Study 2 tests the *interpretation* —
that what the model captures is confidence — by **controlling how much the participant
actually knows** at the moment of selection, and by collecting self-reported confidence to
compare against.

### Design

Built on OneStop's design and materials, with one manipulation replacing question-preview.
Each participant meets three **knowledge regimes**, one third of trials each:

| Regime | What the participant gets before the answer screen |
|---|---|
| full knowledge | the correct answer's content, in wording different from the on-screen option |
| **partial knowledge** | the paragraph to read — the closest analogue to Study 1 |
| no knowledge | no context at all; a correct answer here is a guess |

**"Partial knowledge" is the canonical name** (2026-09-04), matching the code. Draft2
currently calls it the paragraph-reading condition; same regime.

Counterbalanced on both the order of regimes and which questions appear in which regime.
Plus a **1–5 confidence self-report after every trial**, which Study 1 lacked.

The counterbalancing is a double Latin square (regime × article group, then question
ordering) giving 27 lists, times 6 regime orderings = **54 ordered lists**, generated by
`experiment_builder/lists_builder.ipynb`. Note: the repo contains **no presentation code** —
the 54 list CSVs are the hand-off to the experiment software, which lives elsewhere.

Draft2 adds a **Participant Demographics** subsection: first languages and English test
scores. These are **collected by questionnaire as the experiment runs**, not produced by this
pipeline — so they will not appear in the repo, and nothing here needs to generate them.

### Predictions and analysis

Train on all of Study 1, test on Study 2, split by regime:

- **full knowledge** → high predicted probabilities, on trials where we independently know
  the participant knew the answer.
- **no knowledge** → low probabilities, *including* on the trials where they happened to
  guess correctly. This is the key dissociation: the model should not be fooled by a lucky
  guess, because it reads behaviour, not outcome. This regime also supplies the empirical
  guess rate used in §3.4.
- **partial knowledge** (paragraph-reading) → should behave like Study 1.

Then: correlate predicted probability against self-reported confidence, overall, per regime,
and within participant, with the self-report's own calibration as a reference.

**Code:** `answer_correctness/knowledge_regimes_analysis/` (`comparison_runs.py`,
`confidence_correlation.py`), `experiment_builder/preliminary_analysis.ipynb`.

### Status

Two pilots (`testrun_QA`, `second_test`) then the first real collection (`KnowQA`).
Collection is **ongoing and small** — the analysis modules' docstrings still refer to a
three-participant test run, and `MIN_PAIRS = 3`. Current Study 2 results are descriptive,
not inferential, and `preliminary_analysis.ipynb` says so.

Study 2 now carries more weight than it did in draft1: besides validating the confidence
interpretation, it supplies the guess-rate figure that the error analysis in §3.4 is built
on.

---

## 5. Open items in the draft

Diana's own `\todo`s and placeholders, recorded here because most are code work.

| Draft item | Code implication |
|---|---|
| `Y%`, `Z%` — dominant-strategy prevalence by preview group | **Already computed** — `findings.md` §1.2. Hunters 48.9%/53.3%, gatherers 58.3%/61.1% (`≥` convention). |
| `X%` — dominant strategy, all participants | The one genuinely missing number: `run_all_strategy_plots` forces `include_all=False`. One argument. |
| Clockwise / counter-clockwise prevalence | **Already established** — `findings.md` §1.1. CW 71/89, CCW 20/14 (hunters/gatherers). No classifier needed; naming the tuples in the figure would help the reader. |
| "interrupted scan completion for scan-passes shorter than four items" | **Exists** — the prefix-completion pair in `viz/visualisations_strategies.py`. Changes the result by ~2 points and flips 1.1% of participants, so the finding is robust to it (worth saying). |
| `x%` — correct responses in the no-knowledge regime | From Study 2; needs enough participants to be meaningful. |
| "why these 10 attention features and not another. Should I run a brute force try CV on each combination?" | **Not a code task — the answer is already the truth of what happened.** The ten features in `SELECT_1_COLS` were **picked by hand**, on domain grounds: few, and covering the four behaviours the design targets. They are not the output of the selection machinery, so there is no search to defend and no leakage to correct. Draft2 already says the substance in a comment ("we just tried to keep it small and cover the whole space of the idea") — it needs to move into the text. Any future reselection will also be manual. (`todo.md` T3.2) |
| "'significance' / CIs are calculated very differently here" (now commented out) | The Wald coefficient CIs ignore the L2 penalty, the class weights, and clustering by participant. A participant-clustered bootstrap is already implemented in `common/data_utils.py` but never called. |
| "some robustness checks? VIF analysis?" | VIF machinery exists in `common/`. |
| "need to explain answer building logic" | Not code — but §3.6's findings depend on it. |
| "count somehow if this discrepancy is covered by the expected 1 in 4 chance to guess" | Now the core of §3.4's guessing models. |
| "should we pre-register the data collection somehow?" | Not code. |
| Correct / wrong trial separation — WIP | `person_variance/mistake_types.py`. |

---

## 6. Paper section → code map

For deciding what has to be paper-grade. Draft2 puts the substance in Results and the
machinery in Methods, so both columns point at the same code.

| Draft2 section | Code |
|---|---|
| Results · First-scan behavior | `derived/pattern_breaking.py`, `viz/visualisations_strategies.py`, `visualisations_simplified_visits.py` |
| Results · End of trial behavior | `derived/select_confirm_last.py`, `viz/visualisations_last_label.py` |
| Results · Attention allocation | `derived/preference_matching.py`, `viz/visualisations_area_bars.py`, `visualisations_preference_correctness.py`, **`statistics/mixed_area_comparisons.py`** (paper code, settled 2026-09-05 — see §7) |
| Results · Error analysis | `notebooks/general_model_confusion.ipynb`, `person_variance/mistake_types.py` |
| Results · Participant-level coefficients | `person_variance/`, `participant_level.py` |
| Results · Text Associations | `statistics/RT_correlations/` |
| Methods · Study 1 dataset / prep / features | `data_prep/`, `derived/`, `answer_correctness/model_data.py` |
| Methods · Modeling, Models comparison | `answer_correctness/{cross_validation,evaluation_core,feature_groups}.py`, `models/{logreg,dummy}_model.py`, `answer_correctness_viz.py` |
| Methods · Feature contributions | `common/data_utils.py` (coefficient CIs), `answer_correctness_viz.py` |
| Methods · Study 2 design | `experiment_builder/{lists_builder,text_adjustments}.ipynb` |
| Methods · Study 2 data prep | `data_prep/know_qa_dataprep.py` |
| Methods · probabilities per regime, confidence | `knowledge_regimes_analysis/`, `experiment_builder/preliminary_analysis.ipynb` |

**Changed from draft1:** `pattern_breaking.py` and the strategy visualisations move from
future-directions into paper-critical. `mixed_text_answer_effects.py` is confirmed
superseded. **`mixed_area_comparisons.py` is paper code** (settled 2026-09-05) — it backs the
Attention allocation subsection, and its output folder is one of the zero-byte ones, so those
108 figures need re-running.

**Out of scope, kept as future directions** (confirmed 2026-09-04):

| | Why |
|---|---|
| `predictive_modeling/answer_RTs/` | predicting answer reading times from the paragraph text — **not working at present**. This is why draft2's "Paragraph associations" heading under Methods is empty |
| Julia and R mixed-model backends | mixed effects have their own problems and were judged not important enough to pursue for this paper |
| `statistics/mixed_text_answer_effects.py` | superseded; `RT_correlations/` is the current text–QA analysis |
| `answer_correctness/unlikely_analysis.py` | stale; the error-analysis subsection is served by `general_model_confusion.ipynb` |
| `predictive_modeling/answer_loc/` | predicting answer position |
| `answer_correctness/clusters/` | participant clustering, superseded by `person_variance/` |
| `generate_column_options.py` | the JSON feature-set factory |

"Future direction" means **kept and organized, not deleted** (`conventions.md`).

---

## 7. Open questions for Diana

1. "80% look at the answer they select" (draft2) vs ~68–71% of trials measured
   (`findings.md` §2) — participants vs trials, or a number to reconcile?
2. Is the correct-vs-distractor RT asymmetry (`findings.md` §9.1) going in the paper? Strong,
   clean, and currently living in one cell of `presentation_prep.ipynb`.

**Resolved 2026-09-04/05:**

- `L1` = native language; the project uses OneStop's L1 (native English) sample. Study 2's
  mixed linguistic backgrounds are a limitation of that study (§2).
- **Answer construction:** A and B from the critical span (B a misreading), C from the
  distractor span, D absent from the text; plausibility A > B > C > D (§2).
- **Text Associations** comes from `RT_correlations/`; `mixed_text_answer_effects.py` is
  superseded (§3.6).
- **`answer_RTs/` is out** — not working, future direction (§6).
- **Mixed effects are out** — own problems, judged not important for this paper (§6).
- **"Partial knowledge"** is the canonical name for Study 2's middle regime (§4).
- **Participant demographics** are collected by questionnaire during the experiment, not by
  this pipeline (§4).
- **"Eighty percent" vs 0.83** is not an inconsistency — it varies by testing regime and is a
  placeholder (§3).
- **Clockwise / counter-clockwise** is an observation about the data, not a classifier the
  code needs to implement (§3.1).
- **Starting-strategy duplication** is to be unified, `todo.md` T1.6 (§3.1).
- **The ordinal A > B > C > D structure will not be used in the model** — `wrong_mean`
  keeps averaging B, C and D. Considered and set aside (§2).
- **`SELECT_1_COLS` is a manual pick**, not an output of the selection machinery, so the
  CV estimate carries no selection-leakage caveat. Future reselection will also be manual
  (§5, `todo.md` T3.2).
