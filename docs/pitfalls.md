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

> **Decided (2026-09-04), and ✅ implemented in code 2026-09-23: drop `"."` for first fixation
> duration, on all data.** `create_mean_first_fix_duration` coerces with
> `pd.to_numeric(errors="coerce")`, matching what the paragraph path always did — so the
> answer-side and paragraph-side columns of the same name are **now one measure**, and
> `features.py`'s long-standing claim of a 1:1 mirror became true rather than aspirational.
> The pupil columns already dropped on both paths and are unchanged.
>
> **L1 and KnowQA are both rebuilt (2026-09-23)**; the two pilots are not, so their
> `mean_first_fixation_duration` is still on the old convention — stated rather than silent.
> On L1 the change touched **exactly 10 of 217** model-ready columns, all of them
> first-fixation-duration, leaving the headline feature set bit-identical. Measured numbers in
> `findings.md` §3.1 and its change log.
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

### Where imputation *is* a problem: the exclude-convention families

> **Updated 2026-09-23 (T3.6 landed in code).** This section used to say the pupil family was
> *the only* one carrying genuine NaN. That is no longer true: first-fixation duration now
> excludes unread words too, so it carries NaN in **exactly the same cells** — verified
> identical per area on L1. Two families, one policy.
>
> The numbers below were also re-measured, and two were wrong:
>
> * **The old "10,039 cells per metric" did not match its own breakdown.** 10,039 was the sum
>   over all **ten** columns of a metric family (5 areas + 5 derived contrasts); the
>   per-area list printed beside it summed to 8,467. Both are useful, but they are different
>   quantities and were being quoted as one.
> * **257 → 253**, and **D 787 → 785**, since the T3.18 rebuild (`findings.md` §8).

Pupil metrics and — since T3.6 — first-fixation duration follow the exclude convention, so they
are the families carrying genuine NaN, one cell for every area a participant never fixated.
Measured on L1, 2026-09-23:

| scope | cells per metric family |
|---|---|
| the five **area** columns | **8,461** — question 5,810 · A 253 · B 784 · C 829 · D 785 |
| all **ten** columns (areas + the five derived contrasts) | **10,021** |

The contrasts inherit NaN asymmetrically, which is worth knowing before quoting either number:
`__wrong_mean` is NaN only when *all three* wrong answers were unfixated (158 trials), while
`__contrast`, `__distance_furthest` and `__distance_closest` are NaN when *either* side is (383).

The model then fills every NaN with `0.0` (`logreg_model.py:24`). These are **z-scores**, so
`0` asserts *this area had exactly this participant's mean pupil size* — a specific and false
claim about an area that was never looked at.

Live in the headline model: `mean_max_fix_pupil_size_z__correct` is in `SELECT_1_COLS` and has
**253 NaN (1.30% of trials)**; the other nine features have none. That is 0.130% of the
feature matrix — too small to threaten the reported accuracy, big enough to belong in Methods.

**T3.6 has now landed in code, and it does not change that number.** `SELECT_1_COLS` contains
no first-fixation column, so the headline model still imputes exactly those 253 cells. What
changed is the *scope of the policy*: the fill now covers a second family, so the Methods
sentence T3.14 requires has two counts to state, not one. Verified on KnowQA (870 trials),
where the first-fixation family went from 0 to 295 NaN cells while the pupil family stayed at
1,180 and every `SELECT_1_COLS` count was unchanged.

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
  per `participant_id` against that dataset's own fixations (T3.20, 2026-09-23 — it was
  per `session_id` until then); the strategy features still are not scoped at all.
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

> **Resolved 2026-09-20 (`todo.md` T1.1, and most of T1.6).** Two implementations of this used
> to exist (`viz/visualisations_strategies.py` and `derived/pattern_breaking.py`) with
> divergent parsing, tie-breaking and threshold operators. **There is now one, in
> `derived/pattern_breaking.py`**, and `viz/` only plots: `build_starting_strategies` for the
> per-trial strategy, `dominant_strategy_by_participant` for the modal pick,
> `has_dominant_strategy` for the threshold (`≥`, the only operator left), and
> `build_prefix_completion_map` / `add_completed_strategy_column` for interrupted-scan
> completion. The viz copy was verified byte-identical before deletion.
>
> Two corrections fell out: the tie-breaking never actually diverged between those two
> (`glossary.md` §8 Caveat 2), but a **third** implementation in `visualisations_dominant_eye.py`
> did have an order-dependent tie-break, which moved one participant.
>
> **What still waits on T3.21** is not the duplication but the *scope* of the completion map,
> which is still learned population-wide over whatever frame it is given (T3.21 row 5).

---

## 5. KnowQA's `TRIAL_INDEX` is a string

For KnowQA, `TRIAL_INDEX` is a composite (`b2l01t005`) rather than an integer — see
`glossary.md` §2 for why. Most of the pipeline is dtype-agnostic, but
`derived.reading_times.load_paragraph_fixations` coerces `TRIAL_INDEX` to int64 and drops what
won't convert, which would silently reduce the paragraph fixations to nothing.

`know_qa_dataprep.run_pipeline` therefore **hard-refuses** `include_paragraph=True` with an
explanatory error. That guard is correct behaviour — don't remove it to make a run go through.
**No todo item: working as intended.** Note the deeper reason is simply that KnowQA has no
paragraph data exported at all (only a third of its trials show a paragraph, and
`data_paths.py` defines no paragraph path for any Study 2 run) — the dtype issue is a
secondary blocker that would also need fixing if that ever changed.

> **Changed 2026-09-23 (T6.1).** The answer pipeline has **no paragraph step left to turn on**:
> paragraph features are built by `derived/paragraph_prep.py`, and `data_csv_generation` opens
> no paragraph report at all. So the refusal is no longer protecting a flag that would do
> something — it is there to tell a caller who passes it that KnowQA has no paragraph data to
> build from, rather than silently handing back a table with no paragraph columns. The dtype
> hazard itself is unchanged and still lives in `load_paragraph_fixations`, which
> `paragraph_prep` calls.

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

## 7. Writing to `reports/` can also write to `papers/` — but it is currently switched off

`viz/plot_output.py::save_output(..., to_paper=True)` mirrors into **one** folder inside the
Overleaf repo, laid out exactly like the local tree:

```
reports/<analysis>/{figures,tables}/...                      local
papers/correctness_prediction/reports/<analysis>/{figures,tables}/...   mirrored
```

So the mirror path is the local path with a different root, and "which local file is this?" is
answerable by swapping the prefix. So "regenerate the plots" is not necessarily a
`reports/`-only operation — it can change what the paper compiles, without anyone editing
`papers/`.

> **Changed 2026-09-20.** The mirror used to write `figures/` and `report_data/` at the paper
> repo's *top level*, which put two more trees beside the drafts and split one analysis across
> both. **This code never moves or deletes anything under `papers/`** — that repo is Diana's
> and Overleaf-synced.
>
> **But those two folders are no longer there** (checked 2026-09-21). The Overleaf repo's own
> commit `fab3aa2 "ploting revamp"` deleted all **198** files in them — 177 under `figures/`,
> 21 under `report_data/`. That was a commit in `papers/`, not something `save_output` did, and
> the repo is clean, so nothing is dangling and everything is recoverable:
> `git -C papers/correctness_prediction show HEAD~1`.
>
> **Net effect right now: the paper repo holds no figures and no tables at all** — just the two
> drafts and `OLD/`. `papers/correctness_prediction/reports/` does not exist yet either, because
> the mirror is off (below). Nothing breaks, because no `\includegraphics` in either draft is
> uncommented — the drafts still reference figures only in prose. But "the mirror is paused" is
> an understatement of the current state.

> **Since 2026-09-20 (T1.3) the mirror is off.** `plot_output.PAPER_MIRROR_ENABLED = False`,
> `to_paper` defaults to `False`, and passing `to_paper=True` while the flag is off **raises**
> rather than quietly not mirroring — a caller that believes it published to the paper and did
> not is the failure this guards. Turning it back on is one constant.
>
> The reason it is off: figures are being regenerated while the numbers behind them are still
> moving (T3.1, T3.6), and Overleaf should not track that churn.

*The two conflicting `paper_dirs` conventions (T1.5) are gone with `paper_dirs` itself. For the
record, the deeper one never actually produced a `figures/figures/` tree on disk — it was a
latent bug in notebook source, not a thing that had happened. The last references to the old
API — a dead `PAPER_DIRS` constant in `person_variance/loo_runs.py`, its import in
`per_person_runs.ipynb`, two unused copies in other notebooks, and a markdown cell still
telling readers to call `save=True, paper_dirs=...` — were cleared 2026-09-21. `to_paper` is
now the only way to ask for the mirror.*

---

## 8. Some results exist only as pixels — in the *old* tree

> **Fixed at source 2026-09-20 (T1.3).** Every figure in the project is now written by
> `viz/plot_output.py::save_output`, whose `tables=` argument is **required**. A figure cannot
> be saved without its numbers being passed alongside it; a figure that genuinely has none
> passes `tables={}`, which is greppable — "this has no numbers" became a stated choice rather
> than an omission. `print_summaries` still exists, but it only controls console printing and
> no longer decides whether a result survives.
>
> T4.0's acceptance test is now **structural** rather than a convention to remember: outputs
> live at `reports/<analysis>/figures/` and `reports/<analysis>/tables/`, so a figure with no
> `tables/` sibling is visible in the same folder instead of requiring a comparison of two
> parallel trees. The old name mismatches (`area_significance_heatmaps` ↔ `area_mixed_models`,
> `texts_to_answers` ↔ `slopes`) are gone with the two-tree layout.

**The historical problem, for anyone reading an old figure folder.** Nine of the fifteen topics
under `reports/plots/` had no numbers saved anywhere — driver notebooks passed
`print_summaries=False` and discarded the returned frames, so the values survived only inside
the PNGs. `RT_correlations` wrote nothing at all (T4.1, now fixed).

**Zero-byte PNGs: 329, not the 308 usually quoted.** The three wholly-empty folders
(`area_significance_heatmaps` 108, `texts_to_answers` 165, `participant_similarity` 35, all
stamped 2026-03-17 09:37) are the well-known ones. A **second, separate failed write on
2026-03-20 12:57** left 21 more — 15 in `correctness_measures/` and 6 in
`matching_correctness/` — and those are the dangerous ones, because they sit next to
non-empty siblings in folders that look populated.

> **Still true of the old `reports/plots/` and `reports/report_data/` trees**, which are kept
> until their replacements are checked:
> 1. **Check `docs/findings.md` before deleting an old figure folder** — for those nine topics
>    the PNG may be the only copy of the number.
> 2. **Don't trust a populated-looking old plot folder.** Some of those files are zero bytes.
