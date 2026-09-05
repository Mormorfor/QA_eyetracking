# Pitfalls

Conceptual clarifications about this codebase — the things you need to *understand* to avoid
drawing a wrong conclusion. Durable explanations, not a bug list.

**Known bugs and planned fixes live in `docs/todo.md`.** Where an entry here has an open fix,
it says so in one line and points there rather than restating the diagnosis.

**Add to this file** when something turns out to be conceptually confusing. It is meant to grow.

---

## 1. Three grains coexist

| Grain | One row per | Where |
|---|---|---|
| **IA-level** | one **word**, in one area, in one trial, one participant | `all_participants.csv`, `hunters.csv`, `gatherers.csv` |
| **area-level** | one of the 5 screen areas, in one trial | intermediate |
| **trial-level** | one (participant, trial) | `L1_model_ready_all_features.csv` — what modeling consumes |

An IA-level frame holds ~30–60 rows per trial, and trial-level properties are **broadcast onto
every one of them**. So a trial-level value looks correct in a column and is wrong in a count
by a factor of ~39.

> **Rule: any statistical test on a trial-level property must run on a trial-level frame.**
> Recover it with `groupby([TRIAL_ID, PARTICIPANT_ID]).first()`.

Open fix: `todo.md` T3.1.

---

## 2. Every per-word mean is coverage × intensity

**The underlying fact.** All of these columns encode the *same* event for an unfixated word —
no fixation landed there. They just encode it differently:

| Column | Unfixated word | Why |
|---|---|---|
| `IA_DWELL_TIME` | `0` | dwell is the sum of fixation durations; no fixations ⇒ 0 |
| `IA_FIXATION_COUNT` | `0` | no fixations ⇒ 0 |
| `IA_FIRST_FIXATION_DURATION` | `"."` | a fixation of length zero is not a thing, so there is nothing to write |
| pupil sizes | `"."` | measured only during a fixation |

So `dwell == 0`, `count == 0` and `first_fix == "."` are three spellings of one binary fact.
`.replace(".", 0)` on dwell or count is a no-op — there is no `"."` there to replace.

**The consequence.** Averaging over *all* words in an area, zeros included, gives

```
per-word mean  =  intensity-per-read-word  ×  fixated fraction
               =  intensity  ×  (1 − skip_rate)
```

So every one of these metrics carries the coverage signal that `skip_rate` already reports,
multiplied by whatever the true per-word intensity is. Whether that matters depends on how
much the intensity term actually varies:

| metric | per-read-word value, answers A→D | spread | verdict |
|---|---|---|---|
| first fixation duration | 192.2 · 184.8 · 181.9 · 181.2 ms | **6%** | ~pure coverage — effectively a restatement of `skip_rate` |
| dwell time | 381.2 · 323.3 · 296.5 · 284.7 ms | **34%** | real intensity signal on top of coverage |
| fixation count | 1.911 · 1.693 · 1.581 · 1.536 | **24%** | real intensity signal on top of coverage |

*(Measured on `L1_model_ready_all_features.csv`, 2026-09-04. Raw metric ÷ (1 − skip_rate).)*

So for dwell and count the zeros conflate two genuine signals; for first-fixation duration
there is barely a second signal to conflate — its entire between-area difference is coverage.

**Correlation structure among the five area metrics** (answer_A): dwell ↔ count r = 0.92;
both ↔ skip_rate r ≈ −0.55; first-fix ↔ dwell r = 0.56. `area_dwell_proportion` is the most
independent (r = 0.23 with dwell), because it is normalized by the trial total rather than by
word count. Five named metrics, closer to two independent ones.

> **Decided (2026-09-04): drop `"."` for first fixation duration, on all data.**
> Implementation and blast radius: `todo.md` T3.6. Until it lands, the answer-side and
> paragraph-side columns of the same name are different measures. The pupil columns already
> drop on both paths and are fine.
>
> **Corollary: never fill this column's NaN with `0`.** A zero-length fixation does not exist,
> so the model's global `fill_value = 0.0` would put an impossible value back. It needs a
> per-column exception.
>
> **Decided (2026-09-04, `todo.md` T3.13): dwell time and fixation count stay
> coverage-inclusive.** Their coefficients mix coverage and intensity *by design* — they
> measure attention per *available* word, not per read word. Say that rather than fixing it.

### The line, and why the family is deliberately asymmetric

| | Convention | Rationale |
|---|---|---|
| dwell time, fixation count | **include** unread words as 0 | measures of *how much attention the area received* — zero is a meaningful amount of attention |
| first fixation duration, pupil size | **exclude** unread words | properties *of a fixation* — undefined when there was none |

> ⚠️ **This inconsistency is intentional. Do not unify it for tidiness.** Someone will
> eventually notice that four sibling metrics in `AREA_METRIC_COLUMNS_MODELING` handle
> unfixated words two different ways and want to make them match. The split is the considered
> position, not an oversight.

### `0` in the RT/TFD family is not ambiguous

Worth stating because it looks like it should be. Every area exists on every trial — the four
answers and the question are always on screen, and every paragraph has all three spans. So a
`0` reading time means one thing only: **that area was never fixated.** Verified — the RT,
TFD and TimeSinceOffset families contain **no NaN at all** in
`L1_model_ready_all_features.csv`, only zeros (5,284 in `RT_pure`, 9,862 in `TFD_pure`,
19,671 in `TimeSinceOffset`).

Under the coverage-inclusive convention above, that zero is real data — "spent no time reading
it" — and needs no special handling.

### Where imputation *is* a problem: the pupil family

Pupil metrics follow the exclude convention, so they are the only family carrying genuine
NaN — 10,039 cells per metric (question 5,810; answers A 257 / B 784 / C 829 / D 787), one
for every area a participant never fixated.

The model then fills every NaN with `0.0` (`logreg_model.py:24`). These are **z-scores**, so
`0` asserts *this area had exactly this participant's mean pupil size* — a specific and false
claim about an area that was never looked at.

Live in the headline model: `mean_max_fix_pupil_size_z__correct` is in `SELECT_1_COLS` and has
**257 NaN (1.32% of trials)**; the other nine features have none. That is 0.132% of the
feature matrix — too small to threaten the reported accuracy, big enough to belong in Methods.
Once T3.6 lands, first-fixation duration will acquire NaN in the same places.

**Decided 2026-09-05 (`todo.md` T3.14): the `0` fill stays at model-prep time for now, and is
documented rather than removed** — commented where it happens, counted, and reported in
Methods. The point of the entry above is unchanged: know that these cells are imputed and that
the imputed value is an assumption, not a measurement.

---

## 3. Participant-level features are computed over the trials you pass

`pattern_breaking.py` computes the dominant strategy and dominance score **over whatever
trials are in the frame handed to it**, and says so at `:247`. So the rule is:

> **Anything participant-level needs that participant's full trial set.**

### The two studies split in opposite ways — this is the thing to get right

| | Study 1 (L1) | Study 2 (KnowQA) |
|---|---|---|
| manipulation | question preview | knowledge regime |
| design | **between**-participant | **within**-participant |
| a participant appears in | exactly one group | **all three regimes, every run** |
| splitting by it | **safe** | **splits a participant's trials** |

**Study 1 — safe.** Verified 2026-09-04: 360 participants, 180 per group, and **zero**
participants with more than one `question_preview` value. `hunters.csv` and `gatherers.csv`
hold disjoint participant sets, so per-participant features come out identical whether
computed on a group file or on `all_participants.csv` — pooling the other group adds that
participant no trials.

*(An earlier version of this file claimed otherwise. It was wrong.)*

**Study 2 — not safe.** Every KnowQA participant does all three regimes in every run. So
filtering to one regime keeps the participant but drops ~two thirds of their trials, and any
per-participant feature computed on that slice is a different quantity from the one computed
on their full set. Do not reason by analogy from the hunters/gatherers case — it is the
opposite design.

*(KnowQA has a second, independent version of this: one `participant_id` also spans several
sessions. Regime and session are two different ways the same person's trials get split.)*

### Where it does bite

- **CV regime rebuilds.** `cross_validation.py` can rebuild features per regime
  (`:228`), which computes participant-level features over only that regime's slice of the
  participant's trials. This is the documented trade-off at `:206-208` — the alternative,
  passing a globally prebuilt `trial_df`, is faster but forfeits the leakage guarantee.
  Whichever is used, be aware the two notebooks driving CV use different paths, so their
  numbers need not match exactly.
- **Any trial-level filtering** that keeps a participant but drops some of their trials —
  correct-trials-only analyses, exclusion thresholds, and so on.
- **KnowQA sessions.** Here one `participant_id` really does span several sittings *within*
  the same file, so per-participant features pool across them. Pupil z-scoring is already done
  per `session_id`; the strategy features are not.
- **The prefix-completion map** (descriptive path) is learned from the group it is run on, so
  hunters and gatherers get different completion maps. Population-scoped rather than
  participant-scoped, so this one genuinely differs by frame — it shifts the dominant label
  for ~1.1% of participants. (`todo.md` T1.6 decides where the map is learned.)

**`todo.md` T3.21** lists every quantity with this shape and is where the rule gets decided:
the scope becomes an explicit flag — compute within the group being analysed, or over the
participant's true full trial set — rather than a consequence of which frame was passed.

---

## 4. "Starting strategy" exists in two variants — check which one a number used

The concept is the same everywhere: the first four tokens of the collapsed **location**
sequence, question tokens dropped — the order in which the four answer positions were first
visited. What varies is whether short scans get repaired:

| Variant | Used by |
|---|---|
| **with** prefix completion (interrupted scans shorter than four filled from the most common completion of that prefix) | the paper's descriptive prevalence figures |
| **without** completion (raw window) | the per-trial model features |

So a "dominant strategy" percentage can be either the raw or the completed one — the raw/
completed pair differ by ~2 points and flip the dominant label for ~1.1% of participants
(`findings.md` §1.2–1.3). When reading a figure or a coefficient, know which it was.

**"Dominant" means modal, not consistent.** The threshold is "their most common opening scan
covers at least half their trials" — and hunters use 6–28 distinct opening sequences each, so
a 50%-dominant participant still produced a dozen others.

> Two implementations of this currently exist (`viz/visualisations_strategies.py` and
> `derived/pattern_breaking.py`), with divergent parsing and tie-breaking. That is
> duplication to remove, not a design: **`todo.md` T1.6**. The threshold operator is also
> inconsistent between them: **`todo.md` T1.1**.

---

## 5. KnowQA's `TRIAL_INDEX` is a string

For KnowQA, `TRIAL_INDEX` is a composite (`b2l01t005`) rather than an integer — see
`glossary.md` §2 for why. Most of the pipeline is dtype-agnostic, but
`derived.reading_times.load_paragraph_fixations` coerces `TRIAL_INDEX` to int64 and drops what
won't convert, which would silently reduce the paragraph fixations to nothing.

`know_qa_dataprep.run_pipeline:884` therefore **hard-refuses** `include_paragraph=True` with
an explanatory error. That guard is correct behaviour — don't remove it to make a run go
through. **No todo item: working as intended.** Note the deeper reason is simply that
KnowQA has no paragraph data exported at all (only a third of its trials show a paragraph,
and `data_paths.py` defines no paragraph path for any Study 2 run) — the dtype issue is a
secondary blocker that would also need fixing if that ever changed.

Related: `load_all_features` pins `participant_id` to `str` because KnowQA ids are all digits
and would otherwise be inferred as int, breaking merges.

---

## 6. The vendored EyeBench code is meant to be replaced wholesale

`src/external/EyeBench/utils - paragraph feature extraction.py` came from the lab's
EyeBench/OneStop project and is kept as received. Two consequences:

- Its filename has spaces and a dash, so it is **not importable by name**. It is loaded by
  path via `importlib`, after `configs/` is aliased into `sys.modules` as `src.configs`.
- It carries **one local change**: an f-string was reflowed onto a single line, because
  multi-line f-string expressions need Python 3.12 and this environment is 3.11. Drop in a
  fresh upstream copy without re-applying that and it will not parse.

Don't patch it incrementally — the point of the arrangement is that a newer copy can replace
it entirely. Our glue lives in `src/derived/external/EyeBench/runner.py`.

Its cache can currently be written with either of two feature definitions (`todo.md` T5.9).

---

## 7. Writing to `reports/` also writes to `papers/`

`viz/plot_output.py::save_plot(paper_dirs=[...])` mirrors figures into
`papers/correctness_prediction/figures/` and tables into its `report_data/`. So "regenerate
the plots" is not a `reports/`-only operation — it changes what the paper compiles, without
anyone editing `papers/`.

This is intended and stays. Just know that it happens. Two `paper_dirs` conventions are in
use, one landing a directory deeper than the other (`todo.md` T1.5).

---

## 8. Some results exist only as pixels

**Nine of the fifteen topics under `reports/plots/` have no numbers saved anywhere** — driver
notebooks pass `print_summaries=False` and discard the returned frames, so the values survive
only inside the PNGs. On top of that, 308 PNGs across three folders are zero bytes from a
failed sync, and `RT_correlations` writes nothing at all.

Where numbers *do* exist the folder names don't match the plot folders
(`area_significance_heatmaps` ↔ `area_mixed_models`, `texts_to_answers` ↔ `slopes`), so you
cannot tell by looking whether a figure is backed by a record.

> **Two consequences for anyone working here:**
> 1. **Check `docs/findings.md` before regenerating a figure folder** — for these nine topics
>    the PNG may be the only copy of the number.
> 2. **Don't trust a populated-looking plot folder.** Some of those files are zero bytes.

Being fixed at source: `todo.md` **T4.0** is the standing requirement (every analysis persists
its numbers, with a matching `report_data/` folder per `plots/` folder); T1.3, T4.1 and T4.2
are the specific pieces.
