# TODO

Working list for the cleanup, ordered from smallest and safest to largest.
**Nothing here is done. Nothing gets done without Diana's go-ahead per item.**

## How to use this file

Tiers run small → large. T1 is safe and mechanical; T3 changes numbers that are in the
paper; T6 is the structural rewrite we haven't designed yet.

| Column | Meaning |
|---|---|
| **Effort** | S = minutes, M = an hour or two, L = a session or more |
| **Risk** | how much can break, or how much a reported result moves |
| **Verified** | ✅ I confirmed it against stored output · ⚠️ read from code only, **check with a shell before acting** |

Claude could not execute anything on this machine while this list was written, so every
⚠️ item is a code reading, not a test result. Check those first — see §V.

### Priority rule: broken ≠ urgent

**Anything that is not used in the paper and does not run today is low priority.** Note it,
keep it, don't spend energy fixing it now. Future directions and dead ends need to end up in
the right folder after the restructure — that is the whole obligation for now. Fixing them is
work for whenever those strands are picked up again, if they are.

This applies to most of **T2**, and to the future-directions half of **T4.2**. It does *not*
apply to anything that runs today and produces a number, or to anything the public release
must be able to execute (**T5**).

### Priority rule: "just run it" is not a todo

Several things simply need to be re-run — a plot regenerated, a figure produced with a
parameter it already supports, a folder of empty PNGs refilled. Those are not tracked as
items with a fix attached, because there is nothing to design: lots of things will be run and
re-run before the paper is finished. Where an item is only a rerun, it says so.

---

## T1 · Small consistency fixes

Safe, self-contained, no reported number changes. Good warm-up batch.

### T1.1 — Make the dominance threshold `≥` everywhere ✅ S · low risk

Three functions apply the dominant-strategy threshold, and they disagree:

| Function | File | Operator |
|---|---|---|
| `proportion_with_dominant_strategy` | `viz/visualisations_strategies.py:89` | `prop > threshold` |
| `summarize_before_after` | `viz/visualisations_strategies.py:504-513` | `>= threshold` |
| `plot_dominant_strategy_counts_above_threshold` | `viz/visualisations_strategies.py:275` | `>= threshold` |

This is why the same data prints **46.1%** and reports **48.89%** for hunters. Draft2 says
"at least in half of the trials", which is `≥`.

**Do:** change `:89` to `>=`, and update the print string in `run_all_strategy_plots`
(`:686-689`, `:706-709`) from `(>{threshold}% of trials)` to `(≥{threshold}%)`.
**After:** hunters 48.9% raw / 53.3% completed; gatherers 58.3% / 61.1%.

### T1.3 — Make plot and data saving consistent and robust across the project ⚠️ M · low risk

**The requirement is one saving framework, used everywhere.** Some analyses were written as
one-off tests and were never given a saving mechanism at all; others save plots but not
numbers; others use a private `savefig` instead of `plot_output`. The result is that whether a
result survives depends on which module produced it.

Concretely, in `notebooks/vizes for purpouse/visualisations.ipynb`: `print_summaries=False` in
cell 4 (last-label) and cell 9 (matching); cells 5, 7 and 8 print nothing substantive; every
returned DataFrame is dropped. And `RT_correlations` bypasses `plot_output` entirely (T4.1).

**Do:** every analysis routes through `plot_output` for both the figure and its numbers
(`save_plot` / `save_df_csv` / `save_json`), including the ones that currently save nothing.
`print_summaries` stops being the thing that decides whether a number is kept.

This is what lets `docs/findings.md` be *regenerated* from saved CSVs rather than transcribed
from pictures. The policy and the acceptance test are **T4.0**; T1.3 is the mechanical part,
T4.1 is the one module that needs wiring from scratch.

### T1.4 — Fix phantom constant names in comments/docstrings ⚠️ S · low risk

`AREA_METRIC_COLUMNS` does not exist (it is `AREA_METRIC_COLUMNS_MODELING` /
`..._VIZES`); `AREA_LABEL_CHOICES` and `ANSWER_LABEL_CHOICES` do not exist (it is
`LABEL_CHOICES`).

Referenced at: `data_prep/data_csv_generation.py:88-90` and `:270`,
`viz/visualisations_area_matrices.py:41`, `viz/visualisations_preference_correctness.py:208`,
`viz/sequence_visualisations.py:88` and `:458`, `constants.py:146`,
`predictive_modeling/answer_loc/answer_loc_models.py:59`.

> ⚠️ **Related, and not cosmetic:** `notebooks/statistics.ipynb` cell 2 appears to *call*
> `metrics=Con.AREA_METRIC_COLUMNS`. If so that notebook raises `AttributeError` today and
> the constant was renamed after it last ran. **Verify first** — it changes this from a
> comment fix to a T2 breakage.

### T1.5 — Other small cleanups ⚠️ S · low risk

| What | Where |
|---|---|
| Dead commented-out body after a `return` | `derived/select_confirm_last.py:57-61` |
| `RT_TFD_CONTRAST_SUFFIXES` defined twice, identically, in one file | `predictive_modeling/common/feature_specs.py:38-44` and `:46-52` |
| `wilson_ci` duplicated — a nested `_wilson_ci` in a file that already imports the real one | `viz/visualisations_correctness_measures.py:608` vs `derived/correctness_measures.py:52` |
| Bare `LAST_ALL` expression statement (a no-op) | `generate_column_options.py:979` |
| `case_sensitive` parameter does nothing — both branches identical; error text says `'full'` while the filter is `pruned`/`aic` | `generate_column_options.py:831-838`, `:891` |
| Stale docstring: claims plotting lives in `src.viz.visualisations_text_answer_effects`, which does not exist | `statistics/mixed_text_answer_effects.py` |
| Docstring says `is_correct == 1`; code filters `== 0`, so the printed "469 texts / 1407 rows" describes **incorrect** trials | `experiment_builder/…/presentation_prep.ipynb` cell 9 (`collect_triples`) |
| Unused imports: `itertools` and a duplicate `os`; `dataclass`, `Callable`; `Sequence`; `Literal`; `LogisticRegression`; re-imported `numpy`/`pandas` mid-file | `data_csv_generation.py:1,14,15`, `derived/correctness_measures.py:5-6`, `derived/preference_matching.py:3`, `derived/pattern_breaking.py:5`, `common/feature_builders.py:6`, `common/data_utils.py:228-229` |
| Orphan `.pyc` for deleted modules (`corr_map_significance`, `visualisations_corr_maps`), plus `cpython-39` / `cpython-312` bytecode from retired environments | `src/**/__pycache__/` |
| Two `paper_dirs` conventions — `papers/correctness_prediction` vs `papers/correctness_prediction/figures`; one lands a level deep | `presentation_prep.ipynb` cell 31 vs `answer_prediction_paper_visualisations.ipynb` cell 2 |

### T1.6 — Unify the two "starting strategy" implementations ⚠️ M · low risk

Duplicated functionality to remove. One concept, two implementations:

| | |
|---|---|
| `viz/visualisations_strategies.py::build_strategy_dataframe` | descriptive path, **with** prefix completion |
| `derived/pattern_breaking.py::build_starting_strategies` | model-feature path, **without** completion |

**Goal: one implementation, in `derived/`, with the variation expressed as parameters** —
`window_len`, `drop_question`, and a `complete: bool` — and `viz/` calling it rather than
reimplementing it. The behavioural difference (completion for the descriptive figures, raw
window for the model features) is a legitimate parameter choice; having two code paths to
express it is not.

**What has to be reconciled while merging:**

- `_parse_seq` and the first-window logic are written twice, near-identically.
- **Tie-breaking differs.** `viz` uses `props.idxmax(axis=1)` (pandas first-in-column-order);
  `derived` uses an explicit `min(counts.items(), key=lambda kv: (-kv[1], kv[0]))`. On a tie
  they can pick different dominant strategies. Keep the explicit deterministic rule.
- **Threshold operator differs** — that is T1.1, and it lives in the same file. Do both
  together.
- The completion map is learned **population-wide over whatever frame it is given**, so it is
  group-scoped in the current descriptive path (hunters and gatherers get different maps).
  Decide whether the unified function learns it per call or takes a precomputed map — this is
  row 5 of **T3.21**, and the answer is the scope flag that item defines.

**Blocked on T3.21.** The unified function is exactly where the `scope` parameter has to live,
so settle that first or this gets built twice.

**Do this before the T3.6-era reruns**, so the paper's descriptive prevalence numbers and the
model's pattern features demonstrably come from the same code.

Defaults to preserve unless Diana says otherwise: descriptive `complete=True`, model
`complete=False`.

### T1.7 — Unify the paragraph and answer per-area metric implementations ⚠️ M · low risk

**Diana's requirement: the paragraph path and the QA path must behave identically.** Right now
they are two implementations of the same eight metrics, which is what let them silently drift.

| | |
|---|---|
| `data_prep/data_csv_generation.py` | `create_*` functions, grouped by `area_label` (question + answer_A–D) |
| `predictive_modeling/answer_RTs/features.py:72-169` | `_mean_dwell_time`, `_mean_fixations_count`, `_mean_first_fix_duration`, `_skip_rate`, `_dwell_proportion`, `_mean_pupil_z`, `_first_encounter_pupil_z`, `_num_span_visits` — grouped by `auxiliary_span_type` (critical / distractor / outside) |

`features.py:68` says outright that "each mirrors a `create_*` function in
data_csv_generation", and `:47-49` asserts the outputs "line up 1:1". Only one of the eight
actually diverged (`mean_first_fixation_duration`, → T3.6), but nothing structural prevents
the next one.

**Goal: one set of metric functions parameterized by the grouping column**, so
`area_label` and `auxiliary_span_type` are two calls rather than two codebases. That makes
identical behaviour structural instead of aspirational.

**Sequencing:** land this *with or after* T3.6, so the unified functions encode the agreed
conventions (drop for first-fixation duration and pupil, include for dwell and count —
`pitfalls.md` §2) rather than baking in today's mixture. T3.14 is settled and no longer gates
anything.

**Watch while merging:**
- `_dwell_proportion` normalizes over the trial's spans; `create_dwell_proportions` over the
  trial's areas. Same formula, different denominator scope — that difference is real and must
  survive.
- `_num_span_visits` reconstructs visits from `INTEREST_AREA_FIXATION_SEQUENCE` per trial;
  `create_simplified_visit_counts` works from the collapsed sequence columns. Check they agree
  before collapsing them into one.
- `_prepare_paragraph_ia` calls `get_participant_pupil_stats()` with no path, so it silently
  baselines against L1's answer screen — that is **T3.20** and should be fixed in the same pass.

**Pairs with T6.1.** That item moves the paragraph metric builders out of `answer_RTs/`
entirely; this one merges them with the QA builders. Same functions, same pass — decide the
destination package once.

---

### T1.8 — L1 and KnowQA `all_participants.csv` differ by a spurious column ⚠️ S · low risk

`add_base_features` ends with `out.reset_index(drop=False)`. On the L1 path
`add_zscored_pupil_columns:401` has already called `reset_index()`, so the final one adds a
stray `index` column of row numbers. On the KnowQA path that base function is excluded, so the
final reset restores `TRIAL_INDEX`/`participant_id` instead and no `index` column appears.

Net effect: the two datasets' `all_participants.csv` have different schemas by one column.
Harmless today, but it means "same pipeline, same columns" is not quite true — worth fixing
before the public release so a reader diffing the two isn't misled.

## T2 · Things that are broken today

Code that will not run. **None of it is on the paper's critical path**, which is why it went
unnoticed — and per the priority rule above, that makes this whole tier **low priority for
now**. It is recorded so nothing is lost and so the restructure files each piece in the right
place; the actual repair waits until either the strand is picked up again or the public
release forces the question (T5).

The exception is **T2.4**, which is not a breakage but a silent omission inside code that
*does* run — see the note under the table.

| # | What | Where | Verified |
|---|---|---|---|
| T2.1 | `generate_column_options.py` raises `AttributeError` unconditionally — references `fg.RT_INTERACTION_COLS`, `fg.TFD_INTERACTION_COLS`, `fg.RT_TFD_INTERACTION_COLS`, none of which exist in `feature_groups.py` any more | `:181-183`, `:401`, `:493` | ⚠️ |
| T2.2 | `answer_loc/` cannot import: `build_area_metric_pivot` imported from `data_utils` (it lives in `feature_builders`), `group_vise_train_test_split` imported from a module that doesn't define it, and called with a signature that no longer exists | `answer_loc_data.py:8-10`, `answer_loc_eval.py:10-13`, `:51-56` | ⚠️ |
| T2.3 | `fit_model_on_prepared_full_data` unconditionally calls `model.get_random_effects()` — the live logreg implements neither that nor `get_random_effect_variance_summary()`. Works only with the Julia model. | `evaluation_core.py:206`, `:240-241` | ⚠️ |
| T2.4 | `get_last_visited_feature_cols` always returns `[]` — matches prefix `last_visited_`, but the built columns are `last_before_confirm*` / `last_before_select*`. So `get_full_feature_cols` (the default whenever `feature_cols=None`) silently contains **no** last-label features. | `common/feature_specs.py:111` vs `model_data.py:456`, `:465` | ⚠️ |
| T2.5 | `notebooks/statistics.ipynb` may call the renamed `Con.AREA_METRIC_COLUMNS` — see T1.4 | `statistics.ipynb` cell 2 | ⚠️ |

T2.4 is the one with quiet consequences: any run that relied on the default feature set has
been excluding a documented feature block without saying so.

---

## T3 · Fixes that change reported numbers

**These move results that are in or headed for the paper.**

> **Most of this tier is one principle, violated repeatedly.** `docs/conventions.md`
> → *Scientific integrity of the code*: never silently discard data, never silently replace a
> problematic value, never change the estimand to dodge an edge case, prefer a loud error to a
> hidden correction. Read as a group rather than as separate bugs:
>
> | Item | The silent alteration |
> |---|---|
> | T3.5 | a can't-happen case given a silent fallback instead of an assertion |
> | T3.6 | an impossible value (zero-length fixation) averaged in as data |
> | T3.7 | a key mismatch producing all-zero RTs, indistinguishable from real zeros |
> | T3.8 | pupil stats silently falling back to a hardcoded L1 path |
> | T3.9 | validation and test regimes pooled into one reported number |
> | T3.11 | in-place coercions leaking into the saved output |
> | T3.12 / T3.14 | a blanket `fill_value = 0.0` asserting a specific false claim — **kept by decision 2026-09-05, so the fix is to stop it being *silent*: comment, count, report** |
> | T2.4 | a whole documented feature block silently absent from the default set |
>
> Each was found separately; they are the same failure mode. Fixing them individually is
> fine, but the acceptance test is the principle, not the item.

**Diana's stance (2026-09-04): correctness comes first, and numbers moving is fine at this
stage.** Don't preserve a published figure for its own sake. Make the fix, rerun, and record
the change — with particular attention to any case where a *conclusion* changes rather than
just a value, so the reason is understood rather than discovered later. Log those in the
change log at the end of `docs/findings.md`.

### Index — T3

**Item numbers are discovery order and are deliberately stable**, because they are referenced
from `pitfalls.md`, `data-pipeline.md`, `glossary.md`, `findings.md` and `research-context.md`.
They are *not* a priority order. This index is the reading order.

| # | Item | Status |
|---|---|---|
| T3.1 | Fisher tests run on the wrong grain | ✅ measured · **high impact** |
| T3.3 | Coefficient CIs wrong three ways | ⚠️ **high impact, unverified** |
| T3.21 | Participant-level accumulative features — scope flag, default `within_group` | ✅ **decided**; implementation blocks T1.6, T6.1 |
| T3.6 | Drop `"."` for first fixation duration | ✅ **decided and now unblocked — ready to run** |
| T3.14 | Missing-value policy for the exclude-convention families | ✅ **decided**: keep the 0 fill, document it |
| T3.20 | One pupil baseline per dataset; right consumer, right file (absorbs T3.8) | ✅ requirement set |
| T3.17 | Assert the invariants the pipeline assumes | ✅ · integrity rules |
| T3.18 | Does `check_text_alignment` catch a real problem? | ⚠️ investigation · **never rejects trials** |
| T3.19 | Is `last_answer_area_visited_lbl` buggy? | ⚠️ investigation |
| T3.2 | Say in Methods that features were hand-picked | ⚠️ low impact (was high — retracted); figure-2 concern checked and closed |
| T3.13 | Dwell/count stay coverage-inclusive | ✅ **resolved, no action** |
| T3.15 | Two feature-provenance paths in `cross_validation.py` | ⏸ **deferred to the restructure** → T3.21 |
| T3.4 | Smaller number-movers — holds T3.5, T3.7–T3.12 | mixed |
| ~~T3.16~~ | *(ordinal A>B>C>D structure in the model — considered and declined 2026-09-05; number retired, not reused)* | — |

### T3.1 — Fisher tests run on the wrong grain ✅ S to fix · **high impact**

`derived/correctness_measures.py` builds a trial-level frame for each plot, then hands the
**interest-area-level** frame to the significance test:

```python
# correctness_measures.py:240-256  (same shape at :286-288 and :315-317)
trial_df   = build_trial_df_for_seq_len_threshold(df, ...)   # groupby(...).first()  ✅
summary_df = summarize_binary_by_group(trial_df, ...)
test_res   = correctness_by_seq_len_threshold_test(df=df, ...)   # ❌ raw df
```

Verified against your own output: `all_participants__thresh_4__summary.csv` reports
n = 19,436 trials while the `__fisher.json` from the same call reports n = 760,628 and
`"p_value": 0.0`. **39.1× inflation**, because every word of every trial counts as an
independent observation.

**The fix is one word in three places** — pass `trial_df`, not `df`.

Recomputed at the correct grain for the threshold-4 case: OR 2.753, **p = 6e-60** (vs
reported OR 2.643, p = 0.0). That conclusion survives comfortably, but the inflation is a
fixed ~39× on n for *every* one of these tests, so any borderline result among the affected
files is currently reported as significant when it may not be.

**Blast radius:** ~96 data files and ~51 figures across
`reports/{plots,report_data}/correctness_by_seq_len_threshold`,
`…_by_trial_mean_dwell_threshold`, `…_by_back_and_forth_pattern`, plus their mirrors under
`papers/correctness_prediction/report_data/`. Bar heights, n labels and Wilson CIs on those
figures are **correct**; only the stars, p-values and odds ratios are wrong.

**Self-check:** in every affected folder, `summary.csv`'s `n` and `fisher.json`'s
`counts.*.n` disagree by ~39×.

### T3.2 — Say in Methods that the features were chosen by hand ⚠️ S · low impact

**Corrected 2026-09-05.** An earlier version of this item claimed `SELECT_1_COLS` came out of
`feature_selection.ipynb` and that the CV estimate was therefore optimistic. **That was wrong.**
Diana picked the ten features **manually**, on domain grounds — small, and covering the space
of the idea. Any future reselection will also be manual.

So there is **no automated selection inside or before the CV loop** for the headline model.
`feature_cols_by_model` takes explicit column lists, and nothing in `cross_validation.py`
consults the target when choosing features. The 0.83 does not need a leakage caveat.

**What remains is one sentence of Methods**, answering the draft2 `\todo` ("why these 10
attention features and not another"): the features were selected by hand to be few and to
cover the four behaviours the design targets — attention to correct, to wrong, to the
question, and shifts between them — not by a search procedure. Draft2 already contains the
substance of this answer in a comment; it just needs to be stated in the text.

**Checked 2026-09-05 — no issue. ✅** An earlier version of this item asked whether any
machine-selected feature set appears as a competitor in the model-comparison figure, which
would have carried optimism the hand-picked headline does not. **It does not.** Read off
`reports/report_data/answer_correctness/run_comparison/`, every run in the comparison is a
hand-specified feature set:

| run | balanced accuracy (`both`) | n features |
|---|---|---|
| `SELECTION + LAST + RT` | 0.8293 | — |
| `SELECTION + LAST` | 0.8293 | 12 |
| `LAST` | 0.8258 | 2 |
| `SELECTION - 10 features best performer` | 0.8150 | 10 |
| `BASELINE - correct+mean_wrong RT` | 0.7094 | 2 |
| `BASELINE - total_answering_RT` | 0.6348 | 1 |

No `pruned` / `aic` / `k_most_frequent` run is present. `answer_corr_prediction.ipynb` cells 12
and 17 *can* produce such runs, but none were saved into the comparison folder, so the figure
is a fair comparison of hand-picked sets against two RT baselines. Nothing to fix or caveat.

> **One mechanism worth knowing, not a defect.** `collect_and_plot_correctness_runs`
> (cell 14) does not take a curated list — it **scans a directory**
> (`reports/report_data/answer_correctness/both/logreg`, `recursive=True`, `top_n=100`) and
> plots whatever it finds. So the figure's contents are whatever has been saved there. That is
> convenient, and it means a future machine-selected run saved with `save=True` would join the
> comparison silently. Worth a glance at the folder before the final figure is generated.

The selection machinery itself (`feature_selection.ipynb`,
`generate_column_options.py`) is not used for the reported model and
`generate_column_options.py` is currently broken (T2.1) — future directions.

> **Separate and still live: encoding-induced collinearity.** Zero-filling makes
> `mean_first_fixation_duration__*` largely a restatement of `skip_rate__*` — r = −0.70 to
> −0.84, against −0.12 to −0.26 when unfixated words are excluded. It does not affect the
> manual pick (`SELECT_1_COLS` has `skip_rate__correct` and no first-fixation column), but it
> is the concrete content of the draft2 `\todo` asking for "robustness checks? VIF analysis
> to show coefficients are trustworthy?" — the honest answer being that two candidate features
> were near-duplicates by construction rather than by behaviour. See `pitfalls.md` §2, T3.6.

### T3.3 — Coefficient CIs are wrong three ways ⚠️ M · **high impact**

`wald_logreg_coef_cis` (`common/data_utils.py:233-288`) computes `cov = pinv(X'WX)` and:

1. **ignores the L2 penalty** — the coefficients are shrunk MAP estimates, the SEs are
   unpenalised-MLE SEs;
2. **ignores `class_weight="balanced"`** — the weights reweight the fit but never enter `W`;
3. **ignores clustering by participant** — `n_clusters` is hardcoded `np.nan` (`:265`, `:282`)
   and every call site defaults to `coef_ci_cluster="row"`. With ~50 trials per participant
   this understates SEs substantially.

`sig_ci` derives from these and drives the `significant_only=True` coefficient figures.
Since interpretable coefficients are a central contribution, this is the finding most likely
to draw reviewer fire — and **it is your own instinct**: draft2 comments out the
population-significance sentences next to the note that "'significance' / CIs are calculated
very differently here."

**The fix already exists and is never called:** `bootstrap_logreg_coef_cis(..., cluster=participant_ids)`
at `data_utils.py:291` (cluster branch `:319-330`), reachable via
`ci_method="bootstrap", ci_cluster="cluster"` (`logreg_model.py:132-152`).

### T3.6 — Drop `"."` for first fixation duration, on all data ✅ S to change · **decided**

**Diana's decision, 2026-09-04: dropping is canonical.** Average first-fixation duration over
the words that were actually fixated, everywhere. This aligns the answer path to the paragraph
path, which already drops.

**The reason it isn't a matter of taste: a fixation of length zero does not exist.** So a `0`
in `IA_FIRST_FIXATION_DURATION` can only ever be the missing-data sentinel, never a
measurement. Zero-filling it averages an impossible value. Contrast with `IA_DWELL_TIME` and
`IA_FIXATION_COUNT`, where 0 is real — a word that was never read genuinely received 0 ms and
0 fixations — which is why those columns are fine as they are. The pupil columns are also
impossible-at-zero and already drop on both paths. First fixation duration is the only
impossible-zero column currently being zero-filled.

**The change:** `data_csv_generation.py:490-492`

```python
# now
df[C.IA_FIRST_FIXATION_DURATION] = df[C.IA_FIRST_FIXATION_DURATION].replace(".", 0).astype(int)
# wanted
df[C.IA_FIRST_FIXATION_DURATION] = pd.to_numeric(df[C.IA_FIRST_FIXATION_DURATION], errors="coerce")
```

The `.astype(int)` has to go — NaN is not representable as int. Nothing else needs touching:
the paragraph path already coerces this way, so the `features.py:47-49` / `:68` docstrings
claiming a 1:1 mirror become true rather than needing correction.

**⚠️ One coupled decision this forces.** An area with *no* fixated words currently yields `0`;
after the change it yields **NaN** (today: 257–829 rows per answer area, 5,810 for the
question area). The model then fills NaN with `0.0` (`logreg_model.py:24`), which silently
reintroduces exactly the zero-fill we just removed — but now only for the all-skipped cases,
which is worse than doing it uniformly.

**Settled 2026-09-05 (T3.14): the `0` fill stays for now, documented.** So this item is
**unblocked and can be run.** Be clear-eyed about what that means: the raw column stops
zero-filling *every unfixated word* (which is the point — it was making per-area differences
out of skip rate), while an area with *no* fixated word at all still arrives at the model as
`0` via the global fill. The two are different in scale — the first affects nearly every row,
the second 257–829 rows per answer area — which is why fixing the first is worth doing even
with the second left standing. T3.14 lists what has to be commented and reported.

**Expected effects, to check against after the rerun:**

| | before | after (expected) |
|---|---|---|
| `mean_first_fixation_duration__answer_A` | 123.8 ms | ≈ 189 ms |
| `…__answer_B / C / D` | 101.3 / 92.4 / 93.5 | ≈ 189 each |
| `…__question` | 21.1 ms | ≈ 189, high variance, many NaN |
| corr with matching `skip_rate` | −0.70 to −0.84 | ≈ −0.12 to −0.26 |

The per-area differences in this metric should largely **disappear**, because what currently
separates them is skip rate, not fixation duration. That is a conclusion change, not just a
number change — log it (see `findings.md` change log).

**Blast radius:**
- Columns: `mean_first_fixation_duration__{answer_A..D, question}` and the derived
  `__{correct, wrong_mean, contrast, distance_furthest, distance_closest}`.
- The headline feature set is a **manual** pick, so it does not need reselecting (T3.2). But
  any *machine-selected* comparison sets built from the JSONs were pruned on the inflated
  correlations at a 0.8 threshold — if those appear in the model-comparison figure, they shift.
- `reports/plots/basic_stats_barcharts` and `basic_stats_heatmaps` include this metric, and 36
  barchart figures are mirrored into `papers/.../figures/basic_stats_barcharts`.
- `findings.md` §3.1 quotes the current answer-side values — update after the rerun.
- **A Methods sentence becomes false.** The draft says *"Words that were never fixated are
  treated as having a fixation duration of zero, number of fixations of zero, etc."* After
  this fix the fixation-duration clause no longer holds — zero counts stay, zero durations
  don't. Diana's edit to make, in `papers/`; noted here so it isn't missed.
- `SELECT_1_COLS` contains no first-fixation column, so the **headline 0.83 should not move**.
  If it does, something else is going on and that is worth knowing.
- `create_first_encounter_pupil_size:588` filters `IA_FIRST_FIXATION_DURATION > 0`; `NaN > 0`
  is False, so unfixated rows stay excluded — unchanged behaviour, and it stops depending on
  the in-place int coercion described in T3.11.

### T3.13 — ~~Decision needed~~ **RESOLVED 2026-09-04: only first fixation duration excludes unread words** ✅

**Diana's decision: dwell time and fixation count stay coverage-inclusive. No action.**
First-fixation duration (T3.6) is the only metric changing. Pupil columns already exclude and
stay that way.

The resulting asymmetry inside the metric family is **intentional** — do not "fix" it for
consistency. The line is:

| | Convention | Why |
|---|---|---|
| dwell time, fixation count | **include** unread words as 0 | measures of *how much attention the area received*; zero is a meaningful amount |
| first fixation duration, pupil size | **exclude** unread words | properties *of a fixation*; undefined when there was none |

Kept below because the measurement remains useful for the paper's robustness discussion.

---

Original analysis, retained as evidence for the decision.

`dwell == 0` and `count == 0` mean exactly what `first_fix == "."` means: no fixation landed
on that word. So including those zeros makes each per-word mean a product,
`intensity × (1 − skip_rate)`. Measured on `L1_model_ready_all_features.csv` (2026-09-04),
dividing each metric by the fixated fraction to recover per-read-word intensity:

| metric | per-read-word, answers A→D | spread |
|---|---|---|
| first fixation duration | 192.2 · 184.8 · 181.9 · 181.2 ms | 6% |
| dwell time | 381.2 · 323.3 · 296.5 · 284.7 ms | **34%** |
| fixation count | 1.911 · 1.693 · 1.581 · 1.536 | **24%** |

**So dwell and count are unlike first-fixation duration.** A word inside the chosen answer
genuinely receives more time and more fixations *when read*, not merely more often — the
composite is two real signals, not an artifact. That is why T3.6 does not automatically
extend to them.

**The choice that was made:** keep them as-is, coverage included. So their coefficients mix
coverage and intensity by design, and that is what the paper should say they measure —
attention per *available* word, not per read word. The current Methods wording already
describes this correctly for counts.

Relevant context: dwell ↔ count already correlate at r = 0.92, so they are near-duplicates
either way; `area_dwell_proportion` is the most independent of the five (r = 0.23 with dwell)
because it normalizes by trial total rather than word count.

This bears directly on the draft2 `\todo` about robustness/VIF: "the selected area wins on
every attention metric" reads as five converging results, and the correlation structure says
it is closer to two. Worth stating either way — see `findings.md` §3.1.

### T3.14 — Missing-value policy for the exclude-convention features ✅ **DECIDED 2026-09-05: keep the zero fill, document it loudly**

> **Diana's decision.** Pupil and first-fixation-duration features **keep filling with `0` at
> feature/model-prep time for now** — but the fill must be *clearly commented*, not implicit.
>
> **This unblocks T3.6**, which was waiting on this item. T3.6 can now be run.

**What this decision is and is not.** It is a decision about the *model input*, taken to keep
moving. It is **not** a claim that zero is the right value — it is not, and the code must say
so where it happens. Under `conventions.md` → *never silently replace a problematic value*,
what makes this acceptable is the word **silently**: an imputation that is written down,
counted and visible is a stated modelling choice; the same imputation undocumented is the
violation. So the whole obligation of this item is now the documenting.

**What has to be true when this is implemented:**

1. **A comment at `logreg_model.py:24`** stating: this fills columns whose zero is not a
   possible measurement (pupil z-scores, first-fixation duration); the filled value is an
   assumption, not data; it is provisional.
2. **The fill is a named, per-column-family thing**, not an anonymous `fill_value = 0.0`
   applied to everything. Even keeping the same behaviour, the code should show that someone
   chose it for these columns.
3. **The count is reported** — how many cells, in which columns, as a share of trials. Today:
   `mean_max_fix_pupil_size_z__correct` **257 NaN = 1.32% of trials**; the other nine
   `SELECT_1_COLS` features none. This number goes in Methods.
4. **Imputed cells stay identifiable** — see the `first_encounter_avg_pupil_size_z` note
   below, where 137 cells are genuinely 0 and would otherwise be indistinguishable from the
   filled ones.
5. **A note in `docs/conventions.md`** recording this as a known, accepted, provisional
   deviation, so it is not later "discovered" as a bug and silently changed.

**What to revisit, and when.** The two triggers that make this decision stop being adequate:

- **A question-area feature entering the model.** The question area is unfixated on **5,810
  trials (30%)**. Filling 1.32% of one column is a footnote; filling 30% of a feature is a
  result about the fill, not the data. Re-decide if that happens.
- **A reviewer asking.** The pupil case is arguable ("no information → assume typical" for a
  z-score). The duration case is not: after T3.6, a filled `0` is a fixation of length zero,
  which does not exist. If it is challenged, the fallback options are still in the table below.

---

Original analysis, retained as the basis for the decision and for the revisit triggers.

The two families that legitimately have no value when an area was never fixated —
**first fixation duration** (after T3.6) and **pupil size** (already) — inherit a global fill.
This is the item both T3.6 and T3.12 point at.

**Current behaviour.** `logreg_model.py:24` sets `fill_value = 0.0` and `:57-58` applies it to
every column before `StandardScaler`. There is no per-column policy and no report of how much
was filled.

**What zero asserts, per family:**

| family | `0` after filling means | verdict |
|---|---|---|
| pupil (z-scored) | "this area had this participant's mean dilation" | a specific, false claim |
| first fixation duration | a zero-length fixation | physically impossible |

**Volume.** Pupil: 10,039 NaN cells per metric — question area 5,810, answers A 257 /
B 784 / C 829 / D 787. First-fixation duration will gain NaN in exactly the same cells once
T3.6 lands. In the headline model today: `mean_max_fix_pupil_size_z__correct`, **257 NaN =
1.32% of trials**; the other nine `SELECT_1_COLS` features have none.

⚠️ **Note the question area: 5,810 trials (30%).** Any feature built on question-area pupil or
first-fixation duration is missing for nearly a third of trials. Nothing in `SELECT_1_COLS`
currently is, but the feature-selection candidate pool contains such columns — so this
interacts with T3.2 — the headline set is a manual pick and unaffected, but the
machine-selected comparison sets in the candidate pool are not.

**Also needs deciding: how NaN propagates into the derived columns.** `__correct`,
`__wrong_mean`, `__contrast`, `__distance_furthest`, `__distance_closest` are built from the
per-area values. Pandas `.mean()` skips NaN, so `wrong_mean` is only NaN when *all three*
wrong answers are unfixated, while `contrast = correct − wrong_mean` is NaN if either side is.
That means a `wrong_mean` computed over one fixated answer instead of three is currently
indistinguishable from a full one. State the rule, or carry an n-used count.

**Options, with what each costs:**

| Option | Cost / consequence |
|---|---|
| ✅ **CHOSEN** — keep the 0 fill, but **state it** as an explicit per-column choice and report the counts | cheapest; defensible for a z-score ("no information → assume typical"), indefensible for a duration |
| **Drop** trials with any NaN in the selected features | 1.32% today; ~30% if a question-area feature ever enters the set |
| **Impute** participant or item mean | for a z-scored pupil column the participant mean *is* 0, so this is circular — only meaningful for the raw pupil or duration columns |
| **Keep NaN** and use a model that handles it | `HistGradientBoostingRegressor` (already used in `answer_RTs`) does; sklearn `LogisticRegression` does not |
| Add a **`was_fixated` indicator** | `skip_rate__*` already encodes this at area level, so likely redundant — check before adding |

**One concrete argument against the silent fill:** `first_encounter_avg_pupil_size_z` has 137
cells that are *genuinely* 0 (a participant whose first-encounter pupil happened to equal
their own mean) alongside 10,039 that would be filled to 0. After filling, the two are
indistinguishable. Whatever policy is chosen, the imputed cells should stay identifiable.

**Whatever is decided, report the count in Methods** — `findings.md` §8 now carries the
current figure.

### T3.15 — Two feature-provenance paths in `cross_validation.py` ⚠️ **deferred to the restructure**

**Diana, 2026-09-05: not a decision to make now.** Everything will be run and re-run many
times before the paper's numbers are final, and the choice will most likely be moot once the
restructure has settled how features are built. Recorded because the two paths exist and
differ, not because a ruling is owed today.

**Where it goes instead:** the underlying question — *what should a participant-level feature
be computed over?* — is now **T3.21**, which treats it as one instance of a general problem
rather than a property of this one function.

**And T3.21's default answers it (2026-09-05):** `within_group`, on leakage grounds, which
points at the **per-regime rebuild** path (`:228`). So the direction is settled even though the
implementation waits for the restructure — the prebuilt-`trial_df` path (`:224`) is the one
carrying the leak, and `general_model_confusion.ipynb` is already on the right side of it.

The two paths, for the record:

| Path | How | Consequence |
|---|---|---|
| **rebuild per regime** (`:228`) | features recomputed inside each fold × regime | participant-level features (`dominance_score`, `breaks_pattern`) are computed over only that regime's slice of a participant's trials — different values than the global ones |
| **prebuilt `trial_df`** (`:224`) | a globally built table is sliced | participant-level features are the true global ones, but the leakage guarantee documented at `:206-208` no longer holds |

`answer_corr_prediction.ipynb` uses the prebuilt path; `general_model_confusion.ipynb` uses
the rebuild path. **So the same nominal model, run from two notebooks, need not produce
identical numbers** — and the paper does not currently say which it reports.

This is one instance of the participant-level frame hazard (`pitfalls.md` §3) — not the
hunters/gatherers split, which is safe. Unrelated to T3.2, since the headline features are a
manual pick.

**When it comes back:** whichever path survives the restructure, the paper should state which
one produced the reported numbers. Until then, be aware that the same nominal model run from
`answer_corr_prediction.ipynb` and from `general_model_confusion.ipynb` need not agree.

### T3.17 — Assert the invariants the pipeline currently assumes ✅ S–M · **direct application of the integrity rules**

Two facts Diana confirmed on 2026-09-05 are treated by the code as *possibilities to handle*
rather than *invariants to check*. Under `conventions.md` → *Make assumptions explicit* and
*Prefer errors over hidden corrections*, they should be assertions.

**(a) Every trial appears in every feature block.** All eight merges in
`model_data.build_trial_level_model_df` are `how="left"`, so a trial missing from any block
silently gains NaN columns, which the model then fills with `0.0`. But **a trial should never
be present in one block and absent from another** — they all derive from the same trial set.

*Do:* after each merge, assert the row count is unchanged and no key went unmatched. If one
ever does, the run should stop and name the block and the trial — not impute.

**(b) Every trial has a confirmed selection.** `data_csv_generation.py:232` computes
`is_correct` as an equality, so a NaN `selected_answer_position` compares unequal and becomes
`is_correct = 0`. **That case cannot occur** — participants cannot leave a trial without
confirming. So the fallback is not a lenient policy, it is an unreachable branch that would
silently score a data error as a wrong answer.

*Do:* assert `selected_answer_position` is non-null before computing `is_correct`. Replaces
the "decide whether to exclude" framing in T3.5 — there is nothing to exclude.

**Related, same shape:** T3.7 (run-based RT silently produces all-zeros on a key mismatch) is
the same missing-assertion problem on a different join. Worth doing in one pass — a small
`assert_full_coverage(left, merged, name)` helper used at every join in the pipeline.

> Both of these are cheap and neither changes a number *if the invariants hold*. If an
> assertion does fire, that is a finding, not a regression — log it in the `findings.md`
> change log.

### T3.18 — Verify that `check_text_alignment` catches a real problem ⚠️ M · **investigation**

`know_qa_dataprep.check_text_alignment` exists to catch a stored-text bug: a displaced double
quote adding a token the display never showed, which would shift every interest-area boundary
after it by one word, so each area's measures pick up a word belonging to its neighbour.

**Diana is not certain that bug is real** (2026-09-05). Before trusting the guard — or acting
on anything it flags — establish whether the misalignment actually occurs.

**The test she wants:** for trials the check flags as problematic, **compare the number of
interest areas against the same question in L1.** If the flagged KnowQA trial has one more IA
than the identical L1 item, the displaced token is real and the guard is doing its job. If the
counts match, the check is firing on something else and its diagnosis is wrong.

**Whatever the answer, flagged trials are not discarded** (Diana, 2026-09-05 — *we don't throw
away data*). So the question is not "keep or reject", it is **"is the boundary shifted, and if
so, fix the alignment"**:

- **If the misalignment is real** — repair the stored text so the interest-area boundaries
  match what was displayed, and recompute those trials' area measures. The trials stay in.
- **If it is not real** — the guard's diagnosis is wrong and it should stop acting as a
  gate. Downgrade it to a reported count, or remove it.

Check what `check_text_alignment` currently *does* with the trials it flags, since a guard that
excludes them is a silent-exclusion violation of the integrity rules regardless of which way
the investigation lands. If it excludes, that behaviour goes whatever else happens.

### T3.19 — Thoroughly check `last_answer_area_visited_lbl` ⚠️ M · **suspected bug**

**Diana suspects this feature is buggy** (2026-09-05).

`LAST_VISITED_LABEL` / `last_answer_area_visited_lbl` is the "last *answer* area fixated,
stepping back if the final fixation landed on the question" variant, built by
`create_last_area_and_location_visited` in Stage 1. It is one of the three last-visitation
perspectives the paper describes, so a bug here touches a Results claim.

Note it is a *different* mechanism from its two siblings: `last_lbl_before_select` and
`last_lbl_before_confirm` come from `derived/select_confirm_last.py` via the click timestamps,
while this one is derived from the fixation sequence. So the three are not guaranteed
consistent, and checking them against each other is a good starting test.

Not currently in `SELECT_1_COLS`, so the headline model is unaffected — but
`findings.md` §2's ~68–71% figure comes from the *before-confirm* variant, not this one, and
the paper's "80%" claim needs to be attributed to whichever variant actually produced it.

### T3.20 — One pupil baseline per dataset, and every consumer reaching for the right one ✅ S–M · **requirement**

**Diana's requirement, 2026-09-05:** *each dataset creates a pupil-stats file of its own, and
whatever needs z-scored pupils uses the correct one.* That is the whole item; the two bugs
below are just the places where it currently isn't true.

A participant's pupil z-score is `(pupil − participant_mean) / participant_sd`, where the mean
and SD come from a **baseline set of fixations** (`derived/pupil_norm.py::compute_participant_pupil_stats`).
Which fixations go into that baseline is a scientific choice, and right now it is made by
whichever default argument happens to be in scope.

**Two concrete failures:**

1. **The paragraph path silently baselines against L1's answer screen.**
   `get_participant_pupil_stats` defaults `fixations_path=FIX_ANSWERS_PATH`, and
   `answer_RTs/features.py:199` calls it with no path at all. So paragraph-span pupil z-scores
   are normalised by the participant's *answer-screen* pupil distribution, for any dataset.
   (This is the old **T3.8**, now folded in here.) Whether answer-screen fixations are the
   right baseline for paragraph reading is a real question — the two screens differ in
   luminance and in task — but it must be an argued choice, not a default.

2. **KnowQA never writes its own `Auxiliary/participant_pupils.csv`.**
   `add_zscored_pupil_columns` is excluded from KnowQA's registry list — correctly, since
   Stage 0 already z-scores per session — so the block in `main()` that writes that file never
   runs. But `run_pipeline:931` still passes a `pupil_stats_path` for it. On disk the KnowQA
   copy is **80 bytes with an mtime older than every sibling**: a leftover from an earlier run,
   not output of the current pipeline. Anything that reads it is reading stale L1-era numbers.

**Do:**
- Each dataset's pipeline writes its own pupil-stats file, named and pathed per dataset, or
  the pipeline states explicitly that this dataset does not need one (KnowQA's case) and the
  unused `pupil_stats_path` argument goes.
- `get_participant_pupil_stats` loses its dataset-specific default. Callers pass the baseline
  they mean; no path means an error, not L1.
- Record, per dataset, **which fixations the baseline is computed over** — answer screen,
  paragraph screen, or both — in `docs/data-pipeline.md`.

Numbers move for any paragraph pupil feature. Not in `SELECT_1_COLS`, so the headline model is
unaffected; `answer_RTs` and the `RT_correlations` proportions do use these columns.

### T3.21 — Participant-level accumulative features: one rule, and a flag ⚠️ M · **methodological, cross-cutting**

**Diana's requirement, 2026-09-05:** wherever a feature is accumulated over a participant's
trials, it must be possible to **choose** whether it is computed *within the group being
analysed* or over the participant's *true full set of trials* — and that choice should be an
explicit flag, not an accident of which frame was passed in.

> **Default decided 2026-09-05: `within_group`, to avoid train/test leakage.**
>
> The reasoning is sound and worth writing down, because it is the argument that settles the
> whole item: a globally-computed `dominance_score` for a participant is a summary of *all*
> their trials, **including the ones in the test fold**. Attach it to a training row and the
> model has seen a function of the held-out data. It is a small leak — one scalar per
> participant, diluted over ~50 trials — but it is a real one, and it is exactly the kind a
> reviewer can name in one sentence. `within_group` has no such exposure.
>
> This also resolves the old T3.15 in favour of the **per-regime rebuild** path
> (`cross_validation.py:228`), which is the one that already computes within-fold. The
> prebuilt-`trial_df` shortcut (`:224`) is the leaking one; T5.11 removes the speed argument
> that made it attractive.

**Two consequences to accept along with the default**, neither fatal, both worth stating in
Methods rather than discovering later:

1. **`within_group` makes the feature noisier**, because it is estimated from fewer trials —
   and *how much* noisier varies by fold, since folds are not all the same size. A dominance
   score over 12 trials is a worse estimate than one over 50. This is the honest trade: less
   leakage, more variance. Carrying `n_strategy_trials` alongside the score (point 4 below) is
   what makes that variance visible instead of invisible.
2. **It changes what the feature means, per analysis.** Under `within_group`, a KnowQA
   participant's dominance score in the *full-knowledge* regime is their consistency *within
   that regime* — which is the more interesting quantity anyway if the question is whether
   strategy shifts across regimes.

**Where the default should be overridden:** descriptive, non-predictive figures. When the
paper reports "X% of participants have a dominant starting strategy", there is no train/test
split and no leakage to avoid — the honest number is the global one over each participant's
whole trial set. So: **`within_group` for anything feeding a model, `global` for descriptive
statistics**, and the flag is what makes the difference legible instead of accidental.

The hazard is always the same shape: a function computes a per-participant quantity from
"whatever rows it was given", and is then called with a subset. Nothing errors; the numbers
are simply about a different thing than the name suggests. `derived/pattern_breaking.py`
already says so in a docstring — *"computed over the trials present in `df`, so pass the full
(unfiltered) trial set"* — which is a convention held by comment, i.e. not held.

**Every place this arises.** Read from the code 2026-09-05; ⚠️ = not executed.

| # | Quantity | Where it is computed | Accumulated over | Currently |
|---|---|---|---|---|
| 1 | `dominant_starting_strategy`, `dominance_score`, `n_strategy_trials` | `pattern_breaking.py::_dominant_from_strategies` | all of a participant's trials in the frame passed | **whatever frame is passed** ⚠️ |
| 2 | `breaks_pattern_*`, `strategy_distance_*`, `breaks_x_dominance_*` | `pattern_breaking.py::build_trial_level_pattern_features` | derived from 1, so inherits its scope | same ⚠️ |
| 3 | the same, inside the model | `answer_correctness/model_data.py:443` — called on `df`, so scope = whatever `build_trial_level_model_df` was handed | | see 6 |
| 4 | participant pupil mean / SD (the z-score baseline) | `pupil_norm.py::compute_participant_pupil_stats` | the fixation report it is given | **T3.20** — a separate axis (*which screen*), same shape of problem |
| 5 | prefix-completion map (`prefix2full`) | `visualisations_strategies.py::build_prefix_completion_map_from_series` | **population-wide over the frame given**, not per participant | group-scoped today: hunters and gatherers learn different maps ⚠️ |
| 6 | every participant-level feature under CV | `cross_validation.py:224` vs `:228` — prebuilt table sliced, or rebuilt per regime | global, or per-regime | **both paths in use** (old T3.15) |
| 7 | every participant-level feature under the KnowQA regime split | `knowledge_regimes_analysis/comparison_runs.py` — a prebuilt `new_features` table is sliced by regime | **global across the participant's whole session** | ⚠️ **this is the sharp case** — see below |
| 8 | per-person feature means | `person_variance/accuracy_characterization.py:55` | the `trial_df` passed | describes that subset ⚠️ |
| 9 | per-person feature↔outcome correlations | `person_variance/univariate_consistency.py:71` | the `trial_df` passed | same ⚠️ |
| 10 | per-person LOO accuracy and coefficients | `participant_level.py` | the cached global `READY_ALL_FEATURES_PATH` | **global** — the one place that is unambiguous ✅ |
| 11 | within-participant confidence↔probability correlations | `knowledge_regimes_analysis/confidence_correlation.py:318` | one `run` at a time, by construction | scoped **on purpose**, and the docstring says why ✅ |
| 12 | per-participant dominant eye, and the eye × strategy crosstab | `visualisations_dominant_eye.py:48-83` | the frame given | ⚠️ |
| 13 | `TRIAL_ANSWERS`, diffed against the participant's previous trial's cumulative answer log | `button_clicks_processing.py:120` | **sequential** — needs the participant's trials complete and in order | L1 only (KnowQA passes `all_answers_is_cumulative=False`) ⚠️ |

**Two different kinds are in that table**, and they need different treatment:

- **Aggregate** (1–3, 5, 8, 9, 12): computed from a set of trials. Subsetting changes the
  value. This is where the flag belongs.
- **Sequential** (13): computed from the trial *order*. Subsetting doesn't just change the
  value, it corrupts it — a diff against the wrong predecessor is wrong, not differently
  scoped. No flag makes sense here; it needs an assertion that the participant's trials arrive
  complete and ordered (fits the `assert_full_coverage` helper in T3.17).

**Why row 7 is the sharp one.** L1's hunters/gatherers split is **between**-participant, so
group-scoped and global are the same thing for any per-participant aggregate — which is why
that split is safe (`pitfalls.md` §3). KnowQA's three knowledge regimes are **within**-participant:
every participant does all three. So "this participant's dominance score" has two genuinely
different meanings there — *across their whole session* or *within this regime* — and they
answer different questions. Today it is silently the first, because a globally-built table is
sliced. If the question is ever *does a person's scanning strategy shift between knowledge
regimes*, the global version cannot answer it, and will look like a null result.

**What to build:**

1. A single explicit parameter — working name `scope: {"global", "within_group"}` — on every
   function in the aggregate list. **Default `within_group`** (decided above); `global` has to
   be asked for, and descriptive figures ask for it.
2. When `scope="global"`, the function takes the full trial set as a separate argument rather
   than inferring it from the frame being featurised. Inferring is what fails silently.
3. The computed value carries its scope: either a suffix on the column name, or a recorded
   attribute the feature table keeps, so a saved feature file can be asked what it contains.
4. `n_strategy_trials` (and the equivalent count for any other aggregate) is **kept and
   reported**, not just used as a denominator — it is what makes a shrunken scope visible.
5. Where a scope is chosen on scientific grounds rather than convenience, write the reason
   down in `docs/conventions.md` or the module docstring.

**Sequencing:** this must be settled before T1.6 (unifying the two starting-strategy
implementations), because the unified function is exactly where the flag lives. It also
subsumes the old T3.15 and interacts with T6.1.

### T3.4 — Smaller number-movers (contains T3.5 and T3.7–T3.12) ⚠️

| # | What | Where | Note |
|---|---|---|---|
| T3.5 ✅ | **A trial can never lack a confirmed selection** (Diana, 2026-09-05) — so the `is_correct = 0` fallback for a missing selection is dead code, not a silent exclusion. It should be an **assertion**, not a fallback: if a NaN selection ever appears, the run must stop. Folded into T3.17 | `data_csv_generation.py:232` | assert, don't handle |
| — | *(the first-fixation-duration coercion was here; promoted to its own item — see **T3.6** above)* | | |
| T3.7 | Run-based RT fails silently to all-zeros on a `(participant_id, TRIAL_INDEX)` key mismatch — row still written, indistinguishable from a real zero. Compounded: `button_clicks_data.csv` is only rebuilt when asked or missing, so a stale table is reused quietly | `derived/reading_times.py:229`, `:250`, `:271-273`; `data_csv_generation.py:1482` | Add an assertion on join coverage |
| T3.8 → | **Folded into T3.20.** `get_participant_pupil_stats` defaults to a hardcoded L1 fixation path; `answer_RTs/features.py:199` calls it with no path, so paragraph-span pupil z-scores are baselined against L1's answer screen regardless of dataset | `derived/pupil_norm.py:56-62` | fix as part of T3.20 |
| T3.9 ✅ | **DECIDED 2026-09-05: keep the `val_*` regimes, and report them.** They are not dropped. Since the logreg tunes no hyperparameters they are effectively a second test set rather than a validation set — which is fine once they are *shown* rather than folded into an average. The actual fix is therefore narrower than the original framing: `summary_overall_df` should stop averaging **all six** regimes into one `mean_balanced_accuracy`, because that single number silently mixes val and test. Report per regime; if a headline average is wanted, average the three the paper reports | `cross_validation.py:398-410`, `:445-457`, `:1023-1034` | Keep all seven regimes; stop pooling them into one figure |
| T3.10 | Fold-level CIs use `se = std/sqrt(n_folds)`, treating overlapping folds as independent → anti-conservative. Also an unweighted mean over folds regardless of each fold's `n_eval`, and a silent fallback to z=1.96 for any `ci` outside {.90,.95,.99} | `cross_validation.py:876-901` | These are the CIs on the comparison figure |
| T3.11 | Five group-feature functions mutate the caller's frame in place, creating an undocumented ordering dependency (`create_first_encounter_pupil_size` only works because `create_mean_first_fix_duration` already coerced a column to int). Also leaks `area_skipped` and `"."→0` coercions into the saved output | `data_csv_generation.py:458, 475, 490, 513-514, 533`, `:1221` | Running a subset via `group_function_names=[...]` can compare `str > int` |
| T3.12 ✅ | **The global `fill_value = 0.0` is wrong for exclude-convention columns.** Measured 2026-09-04: the pupil family is the only one carrying real NaN (10,039 cells per metric — question 5,810, answers A 257 / B 784 / C 829 / D 787). Filling a *z-score* with 0 asserts "this area had this participant's mean pupil size" for an area never looked at. Live in the headline model: `mean_max_fix_pupil_size_z__correct` has **257 NaN (1.32% of trials)**, the other nine `SELECT_1_COLS` features have none — 0.132% of the feature matrix. T3.6 will add first-fixation duration to the same problem. **Not** an issue for RT/TFD/dwell/count: those have zero NaN and their zeros are real data (see below) | `logreg_model.py:24`, `:57-58` | **Fix tracked as T3.14** — per-column fill policy, covering both families. Report the imputation count in Methods either way |
| T3.20 ✅ | **Every dataset owns its own pupil baseline, and every consumer uses the right one** — full item as its own section above | `pupil_norm.py`, `know_qa_dataprep.py:415`/`:931`, `answer_RTs/features.py:199` | Requirement set 2026-09-05. Absorbs T3.8 |
| — | *Retracted 2026-09-04:* this row previously claimed `0` in the RT/TFD family was ambiguous between "read for zero ms", "never read" and "region absent". Wrong — **every area exists on every trial** (zero NaN in `mean_dwell_time`/`skip_rate` across all five answer areas, and all three paragraph spans always present), and the RT/TFD/TimeSinceOffset families contain **no NaN at all**. So `0` there means exactly one thing: never fixated. Under the coverage-inclusive convention that is real data needing no handling | — | no action |

---

## T4 · Outputs and artifacts

### T4.0 — Standing requirement: every analysis persists its numbers, not just its figures ✅ measured · **agenda item**

**The rule to reach:** a result is not saved until its *numbers* are on disk in a readable
form. A PNG is a rendering, not a record. Concretely:

- every plotting call also writes its returned frame via `plot_output.save_df_csv` /
  `save_json`;
- no analysis relies on `print_summaries` to preserve a number — printing is for the person at
  the keyboard, persistence is for everything else;
- all plotting goes through `plot_output` so paths, naming and the `papers/` mirroring stay
  consistent;
- **acceptance test:** every folder under `reports/plots/` has a corresponding folder under
  `reports/report_data/`, under the *same* name.

**Current state, measured 2026-09-04.** Nine of fifteen plot topics have no saved numbers:

| `reports/plots/` topic | numbers on disk? |
|---|---|
| `answer_correctness` | ✅ `answer_correctness` |
| `total_answering_RT_normalized` | ✅ same name |
| `area_significance_heatmaps` | ⚠️ yes, but named `area_mixed_models` |
| `correctness_measures` | ⚠️ yes, but split across five `correctness_by_*` folders |
| `texts_to_answers` | ⚠️ yes, but named `slopes` |
| `strategies` | ❌ none |
| `dominant_eye` | ❌ none |
| `last_label_before_confirm` | ❌ none |
| `time_segments` | ❌ none |
| `matching_correctness` | ❌ none |
| `simpl_visit_matrices` | ❌ none |
| `basic_stats_barcharts` | ❌ none |
| `basic_stats_heatmaps` | ❌ none |
| `feature_selection` | ❌ none |
| `participant_similarity` | ❌ none (abandoned strand) |

That list is not abstract — it is exactly the set of results that had to be read off images to
write `docs/findings.md`: the first-scan strategy counts (§1.1), strategy variety (§1.5), the
dominant-eye crosstab (§1.6), last-area-before-confirm (§2), time segments (§4) and preference
matching (§6).

**And `RT_correlations` appears in neither tree** — it has no folder under `plots/` or
`report_data/`, because it writes nothing at all. See T4.1.

Even where numbers exist, the folder names don't correspond, so "does this figure have saved
numbers?" can't be answered by looking. Renaming to match is part of this item.

Related: T1.3 (one saving framework, used everywhere), T4.1 (`RT_correlations`, the module
that saves nothing at all), T4.2 (the zero-byte PNGs — a rerun). T4.0 is the policy those
implement.

**Downstream of this: `docs/findings.md` gets rebuilt.** That file is currently a transcription
of numbers read out of PNGs and stored stdout, unverified by Diana and explicitly marked not to
be quoted. Once every analysis writes its numbers to `reports/report_data/`, the ledger can be
**regenerated from those CSVs** rather than transcribed — at which point it becomes a citable
record instead of a stopgap. Treat that regeneration as part of finishing T4.0.

### T4.1 — `RT_correlations` writes nothing ⚠️ M · **blocks a paper subsection**

`RT_correlations/plots.py:188` does a raw `fig.savefig(save_path, ...)` with a
caller-supplied path, and `text_associations.ipynb` never supplies one. So the **current,
authoritative** text–QA analysis — a Results subsection — has no regenerable figures and no
saved tables; its results exist only as cell outputs in a 1.2 MB notebook.

**Do:** route it through `plot_output.save_plot` / `save_df_csv` like every other module.

### T4.2 — 308 zero-byte PNGs ✅ S · **just needs re-running**

Three folders contain only empty files, all stamped identically at **2026-03-17 09:37:02** —
one failed sync or copy, not a code bug. **No fix to design: re-run the plots.** Backing CSVs
survived (`report_data/area_mixed_models/`, 216 files; `slopes/`, 42), so nothing is lost.

- `reports/plots/area_significance_heatmaps/` — 108 files, from `mixed_area_comparisons`.
  **Re-run these; not low priority.** That module was classified as paper code on 2026-09-05
  (it backs the *Attention allocation* subsection — see T6), so these figures are needed.
- `reports/plots/texts_to_answers/` — 165 files, `mixed_text_answer_effects`. Future
  directions → low priority.
- `reports/plots/participant_similarity/` — 35 files. Abandoned clustering strand → low
  priority.

Also 0 bytes from a separate failed write:
`matching_correctness/…/correctness_by_matching__num_loc_visits.png`.

### T4.3 — `reports/` is tracked in git ⏸ **not now**

**Diana, 2026-09-05: don't bother with this for now.**

Recorded so it isn't rediscovered: `.gitignore` covers `/data/` and `/data_raw/` but not
`/reports/`, so a **63 MB pickle** (`report_data/per_person_corr_loo_results/`) and ~115 MB of
PNGs are in the repo history. Nothing breaks. It only becomes a question at public release,
and the remedy (history rewriting) is a git operation, so it is yours whenever you want it.

### T4.4 — Accumulated output cruft ⏸ **deferred to after the restructure**

**Diana, 2026-09-05: once the restructure has happened we will see what to do with all the old
outdated material, this included.** Nothing here gets touched before then — the restructure is
what decides which outputs are still meaningful. Inventory only:

- `report_data/answer_correctness/feature_columns/old/` — 104 JSONs beside 5 live ones
- `report_data/answer_correctness/answer_correctness.zip` — orphan snapshot in a live folder
- `archive/df_with_features_{g,h}.csv` — 2.5 GB, gitignored, local dead weight
- `papers/.../figures/answer_correctness/cross_val_comp/` — June `cross_val_comparison_stage*.png`
  interleaved with August `cross_val_comparison_*_test_stage*.png`, indistinguishable by name;
  `balanced_accuracy_comparison.png` is byte-identical to the `new_item` variant
  ⚠️ *(figures live under `papers/` — mirroring stays automatic, but any deletion there is Diana's)*

---

## T5 · Runs from scratch

Required by the public release. Currently the repo runs only on Diana's machine.

| # | What | Effort |
|---|---|---|
| T5.1 | Add `__init__.py` throughout — there are none anywhere in `src/` | S |
| T5.2 | One import convention. `src.predictive_modeling…` and bare `predictive_modeling…` / `viz.plot_output` coexist, sometimes in one file (`generate_column_options.py:11-13` vs `:18`). Works only because notebooks push two paths onto `sys.path` | M |
| T5.3 | Migrate ~40 hardcoded `"../reports/..."` literals to `PROJECT_ROOT` via `plot_output` (agreed convention: run from repo root). `visualisations_correctness_measures.py` alone repeats two of them 13 times | M |
| T5.4 | `environment.yml` pinning Python 3.11 + dependencies. Note the vendored EyeBench file needs 3.11 (reflowed f-string) | S |
| T5.5 | README: what the project is, how to run it, the `L1` = native-speaker naming, and the pipeline build order | M |
| T5.6 | Document and enforce the build order — `answer_RTs.features` must write the paragraph cache before `answer_correctness.model_data` reads it, and vice versa through `READY_ALL_FEATURES_PATH`. Currently only a `FileNotFoundError` message | S |
| T5.7 ✅ | **Decided:** `data_raw/full` and both `data_raw/tsv` subfolders are deliberate symlinks to OneStop data held outside the repo (not ours to redistribute). The release ships **instructions telling users where to place their own OneStop download**. Remaining work is writing those instructions and making sure the pipeline fails with a clear message when the data isn't there yet — no defensive validation beyond that one message | S |
| T5.8 ✅ | **Decided 2026-09-05.** `all_participants_with_practice.csv` (~3.9 GB) is a leftover from `extract_text.ipynb` and is **being kept** — do not propose deleting it again. `Auxiliary/paragraph_RT_run_based.csv`: whether anything still reads it is **to be answered during the restructure**, not now. The two pilots (`testrun_QA`, `second_test`) should stay **runnable**, not archive-only — they may need their naming brought into line with the KnowQA conventions first | S |
| T5.9 | Two EyeBench entry points write the same cache with different feature definitions (`runner.py` defaults `fix_ptb_pos_double_mapping=True`, `paragraph_trial_features.py`'s `__main__` defaults `False`, reproducing an upstream bug where every `ptb_pos_*` comes out 0). The cache cannot say which produced it | M |
| T5.10 | `experiment_builder/data_prep_new_exp.ipynb` overwrites `know_qa_dataprep.py`'s output with the older identity scheme if run with `RUN_NAME="KnowQA"`. Mark superseded or remove the collision | S |
| T5.11 | Efficiency, not correctness: `evaluate_one_fold_on_regimes` fits a fresh model **inside** the eval-regime loop on identical training data — 6 identical fits per fold, 60 redundant fits per model per run. Hoisting it removes the *speed* incentive for the prebuilt-`trial_df` shortcut (T3.15), which is what made a scope question into a performance trade-off — see T3.21 | S |

---

## T6 · Structure plan

> ### → Designed and agreed: `docs/restructure-map.md` ✅ **2026-09-05**
>
> Target tree top to bottom, a destination for every current file, the five oversized files
> split, the dataset registry that replaces `data_paths.py`, the `data/` and `reports/`
> layouts, and a seven-stage order with the T-items that land in each.
>
> **That document is the plan; this section is the raw material it was built from.** When the
> two disagree, the map wins — and the disagreement is a bug to fix here.
>
> Agreed does **not** mean started: no stage runs without being proposed first.

The aggressive-but-staged reorganization, sequenced in the map as Stages 0 → G. **T6.1 below
is absorbed as Stage C**, which is where the correctness work concentrates.

> **A stated goal of the restructure, not a side effect:** bringing the codebase into line
> with the *Scientific integrity of the code* rules in `docs/conventions.md` — no silent
> exclusions, no silent value substitution, no estimand quietly redefined to avoid an edge
> case, loud errors in place of hidden corrections, assumptions written down.
>
> This is a **sweep, not a list of known bugs.** The T3 items are the instances found so far
> by reading a fraction of the code; the restructure should look for the pattern everywhere —
> every `fillna`, `replace`, `dropna`, `errors="coerce"`, bare `except`, `.get(..., default)`,
> silent `how="left"`, and every filter without a stated scientific reason. Each one is either
> justified in a comment or removed.

### T6.1 — Separate paragraph preprocessing from QA preprocessing, and from RT prediction ⚠️ M–L · **structural, already specified**

**Diana's requirement, 2026-09-05.** Three things are currently tangled that should be three
things. Stated as three separations:

**(a) Paragraph feature extraction must not live inside the RT-prediction module.**
Paragraph-span features are built by `predictive_modeling/answer_RTs/features.py` and cached
at `PARAGRAPH_SPAN_FEATURES_PATH`. The **answer reading-time prediction is likely abandoned**;
the **extraction is wanted and is load-bearing**. Its consumers today:

| Consumer | Uses |
|---|---|
| `answer_correctness/model_data.py:279-313, :447` | paragraph dwell proportions, merged into the trial-level model frame |
| `statistics/RT_correlations/proportions.py:96-122` | the same cache — and this is the **current, live** text↔QA analysis |

So retiring `answer_RTs/` as a modelling strand must not retire the extraction with it. Move
the metric builders to a neutral home (a `paragraph/` package under `derived/`, or wherever
T1.7's unified metric functions land) and leave `answer_RTs/` holding only its models.

**(b) The QA data prep must stop producing paragraph-derived columns.**
`data_csv_generation` → `derived/reading_times.py::build_rt_and_tfd(include_paragraph=True)`
reads the paragraph IA and fixation reports and merges paragraph-region `RT_*` / `TFD_*` /
`TimeSinceOffset_*` columns into the QA table. Those columns should be **built separately and
joined in where they are needed** — at feature-construction time, not baked into the QA
preprocessing output.

Two things fall out of this that are wanted independently:

- **The merge is `how="inner"`** (`reading_times.py:607-608`). A trial with answer data but no
  paragraph data disappears from the RT/TFD table without a word. That is a silent exclusion —
  a direct violation of the integrity rules, and the same missing-assertion shape as T3.7 and
  T3.17. Separating the two tables removes the merge that does it; whatever join replaces it
  gets `assert_full_coverage`.
- **KnowQA has no paragraphs at all**, so `include_paragraph=False` is not an edge case there,
  it is the normal state. Today that is expressed as a flag threaded through three call layers
  (`data_csv_generation:1278, 1319, 1408, 1575` → `reading_times:492`). With the two preps
  separated, KnowQA simply doesn't run the paragraph one.

**(c) One paragraph pipeline, one output, one owner.** After (a) and (b): paragraph IA and
fixation reports → paragraph preprocessing → one paragraph feature table per dataset. QA
preprocessing reads no paragraph inputs. Anything wanting both joins them explicitly, on
`(participant_id, TRIAL_INDEX)`, with coverage asserted.

**Interacts with:** T1.7 (the two per-area metric implementations to unify are precisely the QA
one and the paragraph one — do these together), T3.20 (the paragraph path's pupil baseline),
T3.7 and T3.17 (the join assertions), T5.6 (the build order this simplifies), and T5.8
(`paragraph_RT_run_based.csv`, whose fate is a question this separation answers).

**Sequencing:** this is the first concrete piece of the restructure, and it is designed enough
to start from. It should not start before T3.21, because the paragraph feature functions are
also where a scope flag would have to appear.

---

Inputs already gathered:

- **Paper-critical:** `answer_correctness/` core, `person_variance/`,
  `statistics/RT_correlations/`, `derived/` (incl. `pattern_breaking`), the strategy and
  correctness viz modules, `knowledge_regimes_analysis/`, `data_prep/know_qa_dataprep.py`,
  `experiment_builder/`.
- **Future directions, keep organized:** `answer_loc/`, `answer_correctness/clusters/`,
  `statistics/mixed_text_answer_effects.py`, `answer_correctness/unlikely_analysis.py`, the
  Julia and R backends, `generate_column_options.py`, `answer_RTs/` (all confirmed out,
  2026-09-05).
- **Paper-critical (added 2026-09-05):** `statistics/mixed_area_comparisons.py`. **Classified
  as paper code**, not a future direction — its pairwise area comparisons back the *Attention
  allocation* Results subsection (`research-context.md` §3.3). Two things follow, and they are
  the whole reason the classification mattered:
  - the restructure files it with the live code, not with the parked strands;
  - **its 108 figures are among the zero-byte ones (T4.2), so they need re-running** — and
    since it is now paper code, that rerun is the part of T4.2 that is *not* low priority. The
    backing CSVs survived (`report_data/area_mixed_models/`, 216 files), so it is a rerun, not
    a recovery.
- **Open shape questions:** whether Study 1 and Study 2 code should be separated at the top
  level; where notebook-only analyses (`presentation_prep.ipynb`'s correct-vs-distractor RT
  asymmetry, `longest_alternating_run`) should live in `src/`; whether `viz/` stays one flat
  folder of 14 modules; what `archive/` keeps after four generations of `plots` →
  `new_plots` → `third_plots` → `reports/plots`.

---

## Open — waiting on Diana

Collected so a decision isn't lost in a section. Nothing below blocks the T1 batch.

**Nothing is currently blocked on a decision from Diana.** The list below is kept as the
record of what was asked and answered; the section stays so new questions have a home.

| Question | Where | Answered |
|---|---|---|
| **Missing-value policy** for first-fixation duration and pupil | T3.14 | ✅ 2026-09-05 — keep the `0` fill at model-prep time for now, but comment it clearly. **Unblocks T3.6** |
| **Scope for participant-level aggregates** — global or in-group? | T3.21 | ✅ 2026-09-05 — default **`within_group`**, to avoid train/test leakage. Resolves T3.15 toward the per-regime rebuild path |
| **`val_*` regimes** — report or drop? | T3.9 | ✅ 2026-09-05 — **don't drop them**; report them, and stop pooling all six regimes into one average |
| **`reports/` in git** | T4.3 | ✅ 2026-09-05 — **not now** |
| **`mixed_area_comparisons.py`** — paper code or future direction? | T6, `research-context.md` §3.3 | ✅ 2026-09-05 — **paper code.** Its 108 zero-byte figures move up in T4.2 |
| **Which feature sets does figure 2 plot?** | T3.2 | ✅ 2026-09-05 — **checked, not asked**: all six runs in the comparison are hand-specified; no machine-selected set is present. Question withdrawn |
| **Which provenance path backs the numbers?** | T3.15 | ⏸ deferred to the restructure |
| **`paragraph_RT_run_based.csv`** | T5.8 | ⏸ deferred — T6.1 is the item that will answer it |
| **`participant_pupils.csv` for KnowQA** | T3.20 | ✅ 2026-09-05 — each dataset writes its own |
| **Output cruft** | T4.4 | ⏸ deferred to after the restructure |

Two open items that are *investigations*, not decisions — they need running, not a ruling:
**T3.18** (is the text misalignment real?) and **T3.19** (is `last_answer_area_visited_lbl`
buggy?). And the questions still open in the other docs: the "80% look at the answer they
select" vs measured ~68–71%; whether the correct-vs-distractor RT asymmetry goes in the paper;
whether the dominant-eye × strategy association is worth testing; whether the
longest-alternation counter-evidence changes the XYXY claim (all `findings.md`); and what
`strange_trials.csv` should become (`glossary.md`).

---

## V · Verify before trusting

Everything marked ⚠️ above was read from code, not tested — there was no shell on this
machine. Each of these is a one-command check in Claude Code, and **none should be acted on
because Claude said so**:

1. `python -c "import src.predictive_modeling.answer_loc.answer_loc_data"` → expect ImportError (T2.2)
2. `python -c "from src.predictive_modeling.answer_correctness import generate_column_options as g; g.generate_all_feature_column_sets()"` → expect AttributeError (T2.1)
3. `python -c "import src.constants as C; print(hasattr(C,'AREA_METRIC_COLUMNS'))"` → expect False, which promotes T1.4 to a breakage (T2.5)
4. `from src.predictive_modeling.common.feature_specs import get_last_visited_feature_cols as f` then
   `f(load_all_features())` → expect `[]` (T2.4)
5. Load `L1_model_ready_all_features.csv` and check whether any `RT_pure_*` column is
   uniformly zero → tests whether T3.7 has already bitten
6. **T3.3 — the coefficient CIs. The one high-impact claim never checked against anything.**
   Fit the headline model, take `wald_logreg_coef_cis`, then refit with
   `ci_method="bootstrap", ci_cluster="cluster"` (already implemented,
   `data_utils.py:291`) and compare interval widths. If clustered CIs are materially wider,
   the `significant_only` coefficient figures change. This is the item most likely to affect
   the paper and the least verified.

Also worth confirming: every number in `docs/findings.md` marked **[figure]** was read off a
saved PNG and should be regenerated as text before being quoted (which T1.3 makes automatic).

---

## Keeping the docs consistent

**Rule 1 — coverage.** Every entry in `docs/pitfalls.md` and every gotcha in
`docs/data-pipeline.md` either cites a `T*` id here, or says explicitly that no action is
needed. Otherwise a described problem quietly has no owner.

*(This does not extend to the `research-context.md` §5 table — those rows map Diana's paper
`\todo`s to code implications, and most legitimately have no code item. Only rows asserting
that something in the code is wrong need an id.)*

**Rule 2 — retraction.** When a claim is withdrawn, **grep for the claim, not just its id**.
On 2026-09-05 a retracted claim (T3.2, "feature selection leaks into the CV") survived in
`research-context.md` because that row described the problem in prose without naming the
id, so an id-based sweep missed it. Search for the distinctive words — "nesting", "optimistic",
"leakage" — and check every doc, not the one you were editing.

Audited 2026-09-04; that pass found five gotchas with an existing item but no reference, two
with no item at all (now T1.8 and T3.15), and one description that had been retracted in
`pitfalls.md` but not in `data-pipeline.md` (the `skip_rate` divergence, which does not exist —
only `mean_first_fixation_duration` diverged).

Check with:

```
grep -o 'T[0-9]\+\.[0-9]\+' docs/pitfalls.md docs/data-pipeline.md | sort -u
```

and confirm each id still exists in this file.
