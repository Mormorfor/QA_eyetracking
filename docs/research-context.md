# Research context

The science behind the code, drafted from
`papers/correctness_prediction/Nature Comunications/draft2.tex` (the live paper outline) and
the code itself. Diana has reviewed the science in §§1–4. It is still a reading of her draft,
not the authoritative statement of the research: where this file and the draft disagree, the
draft wins.

> `papers/` is read-only for Claude. This file makes the code's purpose legible from inside the
> repo; it is not a substitute for the paper. `draft2.tex` is the current outline; `draft1.tex`,
> and `abstract.tex` and `paper.tex` in the parent folder, are older drafts (the last two describe
> a mixed-effects model that is not used).

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
uses the **L1 portion of OneStop** — hence the dataset key `l1` and the folder
`data/datasets/l1_onestop/`. OneStop also has an L2 (non-native) portion, which is not used
here.

This matters for the paper in two places: Study 1's sample description, and Study 2's
limitations — **Study 2 recruits participants of mixed linguistic backgrounds**, so the two
studies are not matched on language background. Draft2's `\section{Limitations}` is currently
empty; this belongs in it.

It also matters for the code: `l1` marks a **sample restriction**, not a study number. Any
future non-native data needs its own dataset record rather than reusing these paths.

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

The data follows this ordering on two independent measures:

| | A | B | C | D |
|---|---|---|---|---|
| selection rate | 84.07% | 8.97% | 4.85% | 2.11% |
| mean dwell (trials where A chosen) | 245.6 | 177.3 | 150.6 | 146.9 |
| mean fixation count | 1.231 | 0.928 | 0.803 | 0.793 |
| skip rate | 0.356 | 0.452 | 0.492 | 0.484 |

**Selection follows the full ranking; attention flattens at C ≈ D.** The mixed models say the
same: A–B, B–C and B–D are all significant, while **C–D is not**. That is exactly what the
construction predicts — B is critical-span material and gets treated as a live contender,
whereas C and D are both "not it" visually even though readers still choose between them at a
2:1 rate.

The model does not use this ordering, on purpose: `wrong_mean` averages B, C and D as equals.

### Data preparation

Excluded: repeated-reading trials and practice trials.

Because screen 4 is five disjoint text regions rather than continuous prose, every gaze
measure is computed **per region** (question + four answers). Each word is one interest
area; fixations landing outside any interest area are assigned to the nearest word.
Fixations become a temporal sequence of region symbols, then get collapsed so that
consecutive repeats become one symbol — turning the sequence into a record of *transitions
between regions*.

One targeted cleanup: a single opening look at the question — one run of fixations on one
question word, immediately followed by a move elsewhere — is removed as spillover from
screen 3 (glossary §8).

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
≈ 0.84 correct). Interpretability is a deliberate constraint, not a convenience: which
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

Headline: **~0.83 balanced accuracy** on the answer-selection screens. The Results opener says
"about eighty percent" and Models comparison 0.83; the figure varies a little between testing
regimes, and both are placeholders until the final numbers.

### 3.1 First-scan behavior

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

On the current data, with "at least half of the trials" (raw / after completion): **all
participants 60.3% / 63.1%, hunters 55.6% / 59.4%, gatherers 65.0% / 66.7%**. The group
difference runs in the direction the draft claims. Completion changes the dominant strategy of
3 of the 360 participants, so the finding does not depend on it.

Clockwise and counter-clockwise are not categories in the code: the dominant strategies are
literal tuples of screen positions, and the two most common are exactly
`(top, right, bottom, left)` and `(top, left, bottom, right)` — 79 and 23 of the hunters with
a dominant strategy, 92 and 20 of the gatherers.

"Dominant" means *most common*, not *only*: hunters use between 5 and 28 distinct opening
scans each (median 12). That belongs next to the headline.

**Code:** `features/strategies.py` makes the opening scans, the dominant strategy, the
completion and the per-trial model features. The model features use the scans without
completion; the descriptive figures use both. The figures are in `analyses/scan_strategies/`.

### 3.2 End of trial behavior

The **final fixation before answer confirmation** falls overwhelmingly on the option the
participant goes on to select (~80%), and is strongly predictive of correctness.

Interpreted as a **validation step before commitment**, not the moment of selection itself.
This is why the attention features and the last-fixation features are treated as different
kinds of evidence rather than pooled.

**Code:** `features/last_visited.py` (the last area before each press), the `LAST_*` feature
lists in `features/build.py`, and the figures in `analyses/last_visitation/`.

### 3.3 Attention allocation between answer options

Consistent with prior work on attention to subjectively preferred options: clear preference
for the answer ultimately selected — **lower skip rates, higher mean dwell durations**, etc.

**Code:** the per-area measurements in `features/area_metrics.py`; the figures and the mixed
models that test the pairwise differences in `analyses/attention_allocation/`; whether the
gaze-preferred answer is the chosen one in `features/preference.py`, plotted in
`analyses/correctness_associations/`.

The mixed models have a random intercept per participant and a text term nested within
participant. A participant reads each text once, so that term is in effect an intercept per
trial, not a text effect shared across participants — worth knowing for how Methods describes
them.

### 3.4 Error analysis

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

**Code:** `analyses/error_analysis/` — the confusion quadrants, the guessing-correction
sensitivity analysis across the three test regimes, and the guessing chance at which the two
kinds of error come out equal — and
`analyses/correctness_prediction/person_variance/mistake_types.py` for the per-participant
version. Run it with `python scripts/run_analysis.py --name error_analysis`; the walkthrough
is `src/analyses/error_analysis/overview.ipynb`.

### 3.5 Participant-level model coefficients

Which features are associated with the model performing well vs. poorly for a given person,
which features help on which trial type (correct vs. wrong), and how consistent that is
across participants — in magnitude, or at least in sign.

The best model is retrained **per participant**, leaving one trial out at a time. Participants
with very few incorrect trials score worst (few negative examples) and are excluded above a
threshold.

**Code:** `analyses/correctness_prediction/participant_level.py` and `person_variance/`, walked
through in `person_variance/overview.ipynb`.

Draft1's sentences about population-level significance are commented out in draft2, with a
`\todo` that "'significance' / CIs are calculated very differently here" — see §5.

### 3.6 Text Associations

Relates **proportion of time on each paragraph span** to **proportion of time on each answer
option and on the question** (excluding the first read-through of the question text).

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

**Code:** `analyses/text_qa_relationship/`, walked through in its `overview.ipynb`. Proportions
of time, per-participant Fisher z, FDR-corrected. Nothing runs it with saving on, so
`reports/text_qa_relationship/` is not kept up to date.

`explorations/text_answer_effects/` is an older attempt at the same question with mixed models,
and is **not** the source of these findings. Its saved output, from older data, reaches
different conclusions (`outside` the strongest predictor, `distractor` essentially never
significant, `critical`→A in both groups); do not mix its numbers into this subsection. It does
not import with the installed `pymer4`, so reviving it means updating it first. What it does
offer: it models item and participant as random effects, which the current analysis handles by
working per participant instead, and its item (`text_id_with_q`) variance is ~4.8× the
participant variance — item identity matters far more than participant identity, which supports
the item-crossed fold design.

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

**"Partial knowledge"** is the name the code uses; draft2 calls it the paragraph-reading
condition. Same regime.

Counterbalanced on both the order of regimes and which questions appear in which regime.
Plus a **1–5 confidence self-report after every trial**, which Study 1 lacked.

The counterbalancing is in `src/experiment/lists.py`: two Latin squares (regime × article
group, then which question each paragraph gets) give 27 lists; each regime list is then given
all six regime orders, for **54 ordered lists**. They were built by
`notebooks/experiment/lists_builder.ipynb`. The repo contains **no presentation code** — the
list files in `data/stimuli/new_batched_lists/` are the hand-off to the experiment software.

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

**Code:** `analyses/correctness_prediction/knowledge_regimes/`, walked through in its
`overview.ipynb`. The data preparation is `ingest/knowqa.py`.

### Status

Two pilots (`testrun_qa`, `second_test`), then the first real collection, KnowQA: 6
participants, three sessions each, 870 trials. That is small: Study 2's results are
descriptive, not inferential. Study 2 carries more weight than it did in draft1: besides
validating the confidence interpretation, it supplies the guess-rate figure that the error
analysis in §3.4 is built on.

---

## 5. Open items in the draft

Diana's own `\todo`s and placeholders, where they touch the code.

| Draft item | Code implication |
|---|---|
| `X%`, `Y%`, `Z%` — dominant-strategy prevalence, all and by group | Computed (§3.1). All 60.3% / 63.1%, hunters 55.6% / 59.4%, gatherers 65.0% / 66.7% (raw / completed). |
| Clockwise / counter-clockwise prevalence | Established (§3.1). Naming the tuples in the figure would help the reader. |
| "interrupted scan completion for scan-passes shorter than four items" | Exists (`features/strategies.py`). Changes the prevalences by 2–4 points and the dominant strategy of 3 of 360 participants, so the finding is robust to it (worth saying). |
| `x%` — correct responses in the no-knowledge regime | From Study 2; needs enough participants to be meaningful. |
| "why these 10 attention features and not another. Should I run a brute force try CV on each combination?" | Not a code task. The ten features in `SELECT_1_COLS` were **picked by hand**, on domain grounds: few, and covering the four behaviours the design targets. They are not the output of a search, so there is no search to defend and no selection leakage to correct. Draft2 says the substance in a comment ("we just tried to keep it small and cover the whole space of the idea") — it needs to move into the text. |
| "'significance' / CIs are calculated very differently here" (now commented out) | The paper's coefficient intervals come from a bootstrap that resamples whole participants and refits the actual penalised, class-weighted model (`modeling/evaluate.py::collect_logreg_coef_summaries`). |
| "some robustness checks? VIF analysis?" | VIF is in `modeling/inference.py`. |
| "need to explain answer building logic" | Not code — but §3.6's findings depend on it. |
| "count somehow if this discrepancy is covered by the expected 1 in 4 chance to guess" | The core of §3.4's guessing models. |
| "should we pre-register the data collection somehow?" | Not code. |
| Correct / wrong trial separation — WIP | `analyses/correctness_prediction/person_variance/mistake_types.py`. |

---

## 6. Paper section → code map

Draft2 puts the substance in Results and the machinery in Methods, so both point at the same
code. Paths are under `src/`.

| Draft2 section | Code |
|---|---|
| Results · First-scan behavior | `features/strategies.py`, `analyses/scan_strategies/` |
| Results · End of trial behavior | `features/last_visited.py`, `analyses/last_visitation/` |
| Results · Attention allocation | `features/area_metrics.py`, `analyses/attention_allocation/`, `features/preference.py` |
| Results · Error analysis | `analyses/error_analysis/`, `analyses/correctness_prediction/person_variance/mistake_types.py` |
| Results · Participant-level coefficients | `analyses/correctness_prediction/participant_level.py`, `.../person_variance/` |
| Results · Text Associations | `analyses/text_qa_relationship/` |
| Methods · Study 1 dataset / prep / features | `ingest/`, `features/` (`docs/data-pipeline.md`) |
| Methods · Modeling, Models comparison | `modeling/` (`folds.py`, `crossval.py`, `evaluate.py`, `feature_sets.py`, `models/`), `analyses/correctness_prediction/` — the comparison figure is `comparison.py` and `plots/model_comparison.py` |
| Methods · Feature contributions | `modeling/evaluate.py::collect_logreg_coef_summaries`, `analyses/correctness_prediction/plots/coefficients.py` |
| Methods · Study 2 design | `experiment/`, `notebooks/experiment/` |
| Methods · Study 2 data prep | `ingest/knowqa.py` |
| Methods · probabilities per regime, confidence | `analyses/correctness_prediction/knowledge_regimes/` |

**Out of scope, kept as future directions**, in `src/explorations/` (each folder's
`__init__.py` says what state it is in):

| | Why |
|---|---|
| `answer_reading_times/` | predicting answer reading times from the paragraph text — not working at present. This is why draft2's "Paragraph associations" heading under Methods is empty |
| `mixed_models/` | the Julia and R mixed-model backends: mixed effects have their own problems and were judged not important enough to pursue for this paper |
| `text_answer_effects/` | the older mixed-model Text Associations analysis (§3.6) |
| `time_course/` | before / during / after the first look at the chosen answer; unfinished, its split does not follow time yet |
| `unlikely_analysis/` | the error-analysis subsection is served by `analyses/error_analysis/` instead |
| `answer_location/` | predicting which screen position is chosen |
| `participant_clustering/` | superseded by `person_variance/` |
| `feature_search/` | generating candidate feature sets; not used for the reported model |
| `sequence_figures/` | scanpath figures |

"Future direction" means **kept and organized, not deleted** (`conventions.md`).

---

## 7. Open questions for Diana

1. "80% look at the answer they select" (draft2) vs ~68–71% of trials measured
   (`findings.md` §2) — participants vs trials, or a number to reconcile?
2. Is the correct-vs-distractor reading-time asymmetry going in the paper? Accuracy barely
   moves with time on the correct answer and falls steeply with time on the wrong ones
   (`analyses/answer_rt_comparison/`); no paper section uses it yet.
