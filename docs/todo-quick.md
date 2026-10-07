# TODO — quick reference

One line per item in `docs/todo.md`. **Everything above the last section is index only** — any
detail, reasoning, blast radius or measured number lives in `todo.md` under the same id. The
one exception is *Your own reminders* at the end, which has no counterpart there.

Status key: **✅ DONE** = finished, nothing left for anyone · **↪️ ABSORBED** = folded into
another item, which is named · **⏭️ PUSHED** = real and understood, **deliberately not before
the restructure** (unlike *deferred*, it has a stated point at which it returns) ·
**🟡 decided** = the ruling is made, **implementation still outstanding** (green is reserved
for finished) · **ready** = nothing blocks it · **blocked** = waits on another item ·
**deferred** = deliberately not now · ⚠️ = read from code, never executed (see §V of
`todo.md`).

**Nothing here is waiting on a decision from you.** Anything not ✅ DONE or ↪️ ABSORBED is
work outstanding on Claude's side, or a deliberate deferral.

*(There is no T1.2 and no T3.16 — T1.2 never existed, T3.16 was considered and declined. Don't
go looking.)*

***

## Ready to start, nothing blocking

✅ **Nothing is ready to start, because nothing is left outside the restructure.** T1 and T2
are closed or pushed; T3's actionable items are all done; T4 is pushed; the whole of T5 is
pushed to its stages. **The next thing is T6 — stage 0, proposed first, per the map.**

What is still open is open *by decision*, not by oversight:

| | |
|---|---|
| ⏭️ pushed to a **named stage** | T5.1–T5.11 (stages A, B, D, F, G) · T2.2 and T2.6 (stage E, as `explorations/`) · T3.19 (stage E) · T3.22 (stage D) · T3.24 (stage C) |
| ⏭️ pushed to **your manual check** | T4.0 (rebuild `findings.md`) · T4.2 (re-run the empty figures) |
| ⏭️ pushed, **yours not Claude's** | T3.23 (reading) · T4.3 (a git operation) · T4.4 (after the restructure decides what still matters) |
| ✅ closed 2026-10-06 | the `collect_triples` docstring-vs-code row in T1.5 — ruled **no action** (talk already given; notebook kept for possible reuse). **Nothing is now blocked on a decision** |

*(T3.1, T3.17, T3.7, T3.9 and T3.10 have each been the head of this list and are done —
2026-09-26 and 2026-09-27. T4.2 was here and is now pushed: everything gets re-run during
the manual check.)*

***

## T1 · Small consistency fixes — safe, no reported number moves

* **✅ DONE — T1.1** — Dominance threshold is `≥` everywhere, and the five copies of the modal-strategy pick are now one function in `derived/pattern_breaking.py`. *(**done*** *2026-09-20 — hunters 48.89/53.33, gatherers 58.33/61.11; a convergence onto the numbers already quoted, no figures to regenerate)*

> ⚠️ **One regression from T1.3, found and fixed 2026-09-27.** The migration changed what the
> run summaries are *called* but not what reads them. `save_output` writes
> `model_summary__run-<run>__model-<model>__summary.csv`; both collectors still globbed for a
> bare `model_summary.csv` (`answer_correctness_viz.collect_correctness_run_reports`,
> `run_report_analysis.collect_and_plot_correctness_runs` / `collect_correctness_runs_by_mode`),
> which matches nothing under the new layout — so **the model-comparison figure, draft2's
> figure 2, could not be built.** It failed loudly (`pd.concat` on an empty list), not silently.
> Defaults now glob `model_summary*__summary.csv`, and `answer_corr_prediction.ipynb` cells 14
> and 15 read `reports/correctness_prediction/tables/` via `analysis_dir()` instead of the
> deleted `reports/report_data/` tree. **Still untested end to end** — it cannot be exercised
> until a CV run exists, which needs `COL_SAVE_PATH` rehomed (`restructure-map.md` §1.4).

* **✅ DONE — T1.3** — One `save_output()` everywhere: 81 call sites, 33 source files, 11 notebooks; `tables=` is required so a figure cannot be saved without its numbers; outputs moved to `reports/<analysis>/{figures,tables}/` with key-value filenames; Overleaf mirroring off. *(**done*** *2026-09-20 — no numbers moved; lands* *`restructure-map.md`* *§9 early and most of T5.3; closes T4.1)*

* **✅ DONE — T1.4** — Comments naming constants that no longer exist, fixed at all seven live sites. *(**done*** *2026-09-21 — documentation-only, verified by AST comparison;* *`statistics.ipynb`* *turned out to be already fixed, so this never became a breakage and T2.5 closes with it. Three of the seven were naming the wrong concept*, not just a stale name)

* **✅ DONE — T1.5** — A table of ~10 tiny cleanups: dead code, duplicate definitions, unused imports, stale docstrings, orphan `.pyc`. *(**done*** *2026-09-27 — **nine rows closed, all cosmetic, no number moved**. The `wilson_ci` row was the one that could have been more: all three nested copies were **formula-identical**, and the shared function reproduces **every stored CI in 33 saved `correctness_associations` tables to 1.1e-16**. Also removed 138 stale `.pyc` in 19 `__pycache__` dirs.* ***One row is open and needs your ruling***: *`collect_triples` in* *`presentation_prep.ipynb`* *cell 9 — its comment says* *`is_correct == 1`* *and the code filters* *`== 0`, *so the printed "469 texts" describes **incorrect** trials; fixing either side is a guess.* ***Three corrections to the item***: *`visualisations_correctness_measures.py`* *did **not** already import* *`wilson_ci`; *`pattern_breaking.py`*'s* *`Literal`* ***is*** *used, so that part was stale; and* *`case_sensitive`* ***was not a no-op*** *— only its two needle ternaries were pointless, the real fault being two messages naming* *`'full'`* *as the filter.* ***Found in passing***: *`stats/mixed_text_answer_effects.py`* *does not import at all today —* *`from pymer4.models import Lmer`* *fails on a pymer4 API move. Pre-existing and not a regression in our code; **now tracked as T2.6, ⏭️ pushed to stage E**)*

* **✅ DONE — T1.6** — Merge the two "starting strategy" implementations into one function in `derived/`. *(**done*** *— one implementation since 2026-09-20 with* *`viz/`* *only plotting; the* *`scope`* *parameter it was waiting on landed with T3.21 on 2026-09-25, and the last loose end — where the prefix-completion map is learned — was decided with it: **one map over the whole dataset**, no longer per group)*

* **✅ DONE — T1.7** — Merge the QA and paragraph implementations of the same eight per-area metrics into one set parameterized by grouping column. *(**done*** *2026-09-23 — one implementation in* *`derived/area_metrics.py`. All seven QA metrics reproduce the saved table to ≤5e-13 (CSV round-trip only) and* *`first_encounter`* *is bit-identical, so **no QA number moved**. Two things the merge turned up: the two* *`num_label_visits`* *were **not the same measure** (disagreed on **77% of trials** — the paragraph side dropped off-area fixations the answer side resolves to the nearest word; now shared, which moves paragraph counts upward), and* *`first_encounter_pupil_size`* *silently depended on row order, now an explicit* *`IA_ID`* *sort)*

* **~~T1.8~~ — removed 2026-09-27** — the stray `index` column. *(**premise expired**: all four datasets carry it now, so the schemas agree. And it is inert — row numbers, never reaches the trial-level model table, nothing reads it, ~5 MB on a 3.2 GB file. Number retired, not reused)*

> **Stage B landed 2026-10-06** (B2, the dataset registry, followed B1 the same day).
> **B1 was:** — `src/config/` and `src/lib/` exist, the five feature-set JSONs
> moved to a new top-level `configs/` folder (inputs, not results — see `configs/README.md`), and
> `COL_SAVE_PATH` resolves again, which unblocks re-running the model comparison. The dataset
> registry is in: `src/config/datasets.py` holds a `Dataset` record per dataset with
> `DATASETS` / `dataset()` / `studies()` / `pilots()`, and the eighteen four-times-repeated
> path constants are now derived from it. **Adding a dataset is one entry.** All 96 path
> constants were proved unchanged. **Stage C is next.**

## T2 · Broken today — ✅ **tier closed 2026-09-27**: five fixed, two pushed to stage E

* **✅ DONE — T2.7** — `stats/mixed_area_comparisons.py` **could not fit a single model** under the installed environment. **pandas 3.0 reads text columns as `StringDtype`**, and patsy cannot interpret that as a dtype, so `C(area_label)` raised `TypeError: Cannot interpret '<StringDtype(na_value=nan)>' as a data type` *before* any model was fitted — from a plain `read_csv`, nothing to do with how the caller loads. *(**found and fixed 2026-09-27** while pooling `attention_allocation`'s tables. This is **paper code** — it backs the Attention allocation Results subsection, and its 108 figures are the ones T4.2 says must come back — so "re-run the empty figures" was **blocked and nobody knew**: the family's outputs were last produced **2026-09-20** and every attempt since would have died. Fix: cast the three columns the formula and grouping actually use to `object` immediately before `smf.mixedlm`, which is the dtype they had under pandas 2 — restoring what the formula was written against rather than changing the model. **Verified** by re-running the family end to end)*

* **✅ DONE — T2.1** — `generate_column_options.py` references three constants `feature_groups.py` no longer has. *(**done*** *2026-09-27 — the fix was* ***removal, not restoration***. *The constants named paragraph×answer product columns (`..._critical__x__..._answer_A`); they were deleted in commit* *`c5f6470`, *and* ***nothing builds those columns — zero* *`__x__`* *columns exist in any dataset***, *so putting them back would have resurrected 31 feature sets pointing at nothing. Removed* *`_interaction_groups()`, group 5 (30 sets), group 11 (1 set) and both orchestrator calls, leaving a comment with the git ref to recover the definitions if the terms are ever built.* ***Verified***: *`generate_all_feature_column_sets(df)`* *runs end to end —* ***62 sets, 0 naming a column absent from the data***, *so §V check 2 passes. The symptom stays* ***masked*** *by T5.2 unless* *`src/`* *is on* *`sys.path`, *as the notebooks put it.* ***Found in passing***: *the default* *`COL_SAVE_PATH`* *still points at* *`reports/report_data/answer_correctness/feature_columns`, *which **no longer exists** — T1.3 moved outputs to* *`reports/<analysis>/{figures,tables}/`. *Left alone: where feature-set JSONs belong in the new layout is a restructure question, and* *`restructure-map.md`* *files this module under* *`explorations/feature_search/`)*

* **⏭️ PUSHED — T2.2** — `answer_loc/` cannot import at all. *(**pushed 2026-09-27** — Diana: "don't care about answer loc, it can stay broken". May be tidied after the restructure, which files it under* *`explorations/answer_location/`* *where* ***parked code moves as-is, broken or not***. ***Ran 2026-09-21 and it is worse than the item says***: *`answer_loc_eval`* *fails earlier on T5.2's import convention, and* *`answer_loc_models.py:109`* *passes* *`multi_class=`, removed in scikit-learn 1.8.0 — so fixing only what is listed would leave it broken)*

* **✅ DONE — T2.3** — `fit_model_on_prepared_full_data` calls random-effects methods the live logreg doesn't implement; works only with the Julia backend. *(**resolved** *2026-09-27 —* ***not a breakage***. *Diana: the function is* ***meant*** *to be Julia-only, so the fault was the name, not the behaviour. Renamed to* *`fit_julia_mixed_model_on_prepared_full_data`* *and redocumented — the docstring now says the random-effects calls are deliberate, that passing the logreg raises* *`AttributeError`* ***by design***, *and points the logistic full-data path at* *`collect_logreg_coef_summaries`* *instead.* ***It never actually broke***: *the one caller hardcodes the Julia model, the two notebooks importing it never call it, and it has written nothing to disk. Still parked — mixed effects are out of the paper. What it computes, for the record: per-participant and per-text random effects plus the varcorr summary — the person-vs-item variance question behind the item-crossed fold design)*

* **✅ DONE — T2.4** — `get_last_visited_feature_cols` always returns `[]` because of a prefix mismatch, so the **default** feature set has silently contained no last-label features. *(**done*** *2026-09-27 —* ***the fix is deliberately not a corrected prefix***. *The 16 built columns are not a usable block: each family is six mutually exclusive one-hots summing to 1, plus* *`correct`*/*`wrong`* *which are functions of the same —* *`last_before_confirm_correct`* *is **bit-identical** to* *`..._answer_A`, *since answer_A is always correct.* ***Measured***: *prefix match = 16 cols,* ***rank deficiency 7***; *`feature_groups.LAST_ALL`* *= 8 cols,* ***deficiency 0***. *So the function now filters* *`LAST_ALL`* *by presence, one source of truth rather than a second copy of the encoding.* ***No number moves***: *every driver passes* *`feature_cols`* *explicitly, so the default was never exercised by a saved run; it goes 188 → 196 columns and the 8 added fit with the expected signs (+0.93 on the correct answer).* ***Found in passing***: *the default set is rank-deficient by **46** either way — 20 in area, 18 in RT/TFD, from the* *`__contrast`* *identities. Pre-existing, latent, not fixed)*

* **✅ DONE — T2.5** — Not a breakage after all: `statistics.ipynb` already passes `Con.AREA_METRIC_COLUMNS_MODELING`. *(**done*** *2026-09-21 — fixed independently in the plots revamp; this is what kept T1.4 in the comment tier, and it unblocks T4.2's 108 figures)*

* **⏭️ PUSHED — T2.6** — `stats/mixed_text_answer_effects.py` does not import at all: `from pymer4.models import Lmer` fails. *(**pushed 2026-09-27**, the day it was found · **ran**, not read — pymer4 0.9.0 replaced the capitalised model* ***classes*** *with lowercase* ***functions***, *so* *`pymer4.models`* *now exports* *`lmer`* */* *`glmer`* */* *`lm`* */* *`glm`* *and* *`Lmer`* *exists* ***nowhere*** *in the package. **Not a regression in our code** — the environment moved under it, and* *`glmer_r_model.py:9`* *already uses the new lowercase import and works.* ***Returns at stage E***: *the module is the **superseded** text–answer strand (`RT_correlations/` is the live one), and* *`restructure-map.md`* *files it under* *`explorations/text_answer_effects/`, where **parked code moves as-is, broken or not** — so repairing it first would be work done twice. If it is ever picked up it is an **API migration, not a rename**: three* *`Lmer(...)`* *sites plus the* *`.fit()`* */* *`.coefs`* */* *`.ranef`* *result API the module reads off them)*

## T3 · Fixes that move reported numbers

> The tier is mostly one principle violated repeatedly: a silent alteration where a loud error
> belonged. Read the intro table in `todo.md` before picking items off individually.

* **✅ DONE — T3.1** — The three correctness Fisher tests now run at trial grain, reading the split the builder already computed instead of re-deriving it; `_check_trial_frame` raises on a duplicated `(participant_id, TRIAL_INDEX)`. *(**done*** *2026-09-26 — 24 analyses rerun (24 figures, 48 tables); n 760,628 → 19,436.* ***One conclusion changed**: gatherers · threshold-2 goes p 7.7e-09 → 0.189, significant → n.s., its* *`≤ 2`* *group being 116 trials. The other 23 survive at p < 1e-3. Bars/CIs verified untouched — no* *`__summary.csv`* *differs and 23 of 24 figures are byte-identical.* ***Two corrections to the item**: it was not "one word in three places" — each test re-derived its own split, and for the dwell test that re-derivation* ***was*** *the bug; and the blast radius was 24 analyses, not ~96 files / ~51 figures, a count that predated the T1.3 inversion. **XYXY is not generated at all** —* *`use_xyxy`* *defaults False — which is a scope question, not part of this item)*

* **⏭️ PUSHED — T3.23** — **Read up on clustered CIs.** *(**Diana's item, not Claude's** · reading, not code · links live at the top of* *`common/data_utils.py::wald_logreg_coef_cis`*'s *docstring — three Diana added, four academic ones Claude added below them;* ***don't tidy either set away***, *they sit next to the formula on purpose. It is what makes* ***T3.3*** *(bootstrap vs Wald, currently split by cost) and* ***T3.22*** *(clustering as precision-not-direction) decisions rather than defaults. No number depends on it)*

* **⏭️ PUSHED — T3.24** — Declare each group function's preconditions on the registry instead of relying on run order. *(**Diana's idea, raised and pushed the same day**, 2026-09-27, while scoping T3.11 · S–M.* ***A* *`requires`* *key on* *`FUNCTION_REGISTRY`, *not a separate dependencies module** — the registry already carries* *`join_columns`* *per entry, and* *`restructure-map.md`* *§6.1 moves it into* *`features/build.py`, *so the declaration travels with it; a new module would be re-homed by the same restructure.* ***Pushed*** *because stage **C** rewrites that registry, so declaring against today's version means writing it twice.* ***The set is small and closed***: *four preconditions —* *`area_label`, *`area_screen_loc`, *numeric IA columns (fixed by T3.11) and pupil* *`_z`* *columns.* ***The pupil one is a live trap***: *KnowQA deliberately drops* *`add_zscored_pupil_columns`, *so a run that also kept the two pupil group functions would fail confusingly.* ***Not urgent*** *— since T3.11 nothing silently produces a wrong number here;* *`area_metrics`* *raises and names the missing step)*

* **⏭️ PUSHED — T3.22** — Trials are treated as independent when they are nested in 360 participants and 972 items, so every count-based p here is too small. *(M–L ·* ***pushed past the restructure**, Diana 2026-09-26 — measured: ICC 0.041 by participant / **0.126 by item**, effective n ~6,000 not 19,436. A participant-clustered bootstrap* ***moves no conclusion***, so this is precision, not direction. Same root cause as* ***T3.3***; also live in* *`RT_correlations`* *(participant-only clustering, item level unhandled and larger) and ⚠️* *`mixed_area_comparisons`* *(no trial-level random effect). Waits for* *`modeling/inference.py`* *so clustered inference lands once — stage **D**)*

* **~~T3.2~~ — removed 2026-09-27** — one Methods sentence on the hand-picked features. *(**the paper's job, not the code's** — Diana. Number retired, not reused. What it established stands:* *`SELECT_1_COLS`* *is a manual pick so there is no selection-leakage caveat, and all six runs in the comparison figure are hand-specified. One code fact kept in* *`todo.md`*: *`collect_and_plot_correctness_runs`* *scans a directory rather than taking a curated list, so a machine-selected run saved later would join the figure silently)*

* **✅ DONE (paper path) — T3.3** — `collect_logreg_coef_summaries`, the full-data fit behind the paper's coefficient figures, now defaults to the participant-clustered bootstrap; cell 29 of `answer_corr_prediction.ipynb` passes it explicitly. *(**done*** *2026-09-27 — L1's 12-feature headline model: intervals* ***1.44× wider*** *than Wald and* ***all 12 stay significant***, *stable across seeds and 2k/5k resamples. The item predicted a conclusion change; there isn't one.* ***Two corrections to the diagnosis***: *clustering is the* ***smallest*** *of the three defects (a row bootstrap already recovers 1.35× of the 1.44×), and* *`ci_cluster="auto"`* *silently means* ***row***, *not cluster.* ***Three paths, three settings*** *(Diana, 2026-09-27): L1 paper figures* ***bootstrap+cluster***; *cross-validation and KnowQA* ***wald+cluster***. *To make that defensible* *`wald_logreg_coef_cis`* *was rewritten as a* ***cluster-robust sandwich carrying the L2 penalty and class weights***, *so the Wald path has none of the three defects either — it reproduces the bootstrap to* ***within 2.5% on all 12 features at ~560× the speed***. ⚠️ ***KnowQA's clustered Wald is rank-deficient***: *6 clusters for 13 parameters, so* *`rank(meat)=6`* *and two intervals come out* ***3× too narrow***. ***KnowQA reverted to bootstrap the same day***; *`get_coef_summary`* *now defaults to bootstrap+cluster and the two cost-bound callers opt out explicitly. Wald still warns when* *`n_clusters <= n_params`. **Sandwich references are in the* *`wald_logreg_coef_cis`* *docstring** (Cameron & Miller 2015, Zeileis 2006, MacKinnon et al. 2023, Freedman 2006).* ***Do not report Study 2 coefficient significance at n=6 either way***)*

* **✅ DONE — T3.4** — Umbrella for the smaller movers (T3.5, T3.7–T3.12). *(**all closed 2026-09-27**: T3.5, T3.7, T3.9, T3.10 and T3.11 fixed; T3.8 absorbed into T3.20, T3.12 into T3.14. The umbrella has nothing left under it)*

  * **✅ DONE — T3.5** — A missing confirmed selection is scored as a wrong answer; that case can't occur, so it should assert. *(absorbed into T3.17 and **done with it 2026-09-27** —* *`add_is_correct`* *raises and names the trials; measured zero occurrences on all four datasets)*

  * **✅ DONE — T3.7** — Run-based RT silently produces all-zero rows on a join-key mismatch, indistinguishable from real zeros. *(**done*** *2026-09-27 — three silent-zero paths now guarded: missing click row, **duplicated** click row, and IA_IDs resolving to no area label. First two via* *`checks.assert_full_coverage`* *(the T3.17 helper, as planned); the third needed its own check because under IA_ID drift the lookup dict is* ***populated***, *just with ids that never match. Five injected faults all raise and name the trials; clean inputs pass.* ***Correction to “it has not bitten”***: *it has — 4 second_test trials carry all-zero* *`RT_pure`. *But they sit inside a run of **10 consecutive zero-fixation trials for one participant** (a tracker dropout), so the zeros are honest. The old check missed them because it asked whether a whole **column** was uniformly zero.* ***So the guard is scoped to fixated trials*** *— zero-fixation trials legitimately have no click row, and a blanket assertion would halt a rebuild over real data; they are now **printed** instead of silent. Counts: **L1 9, KnowQA 1, second_test 10, testrun_QA 0**.* ***No value moved*** *(max abs diff 0.0 vs the saved tables).* ***Not covered***: *the paragraph sibling* *`compute_run_based_rt_from_fixations`* *has the same shape on a different input)*

  * **↪️ ABSORBED — T3.8** — *(into T3.20)*

  * **✅ DONE — T3.9** — Keep the `val_*` regimes and report them; stop averaging all six regimes into one balanced-accuracy number. *(**done*** *2026-09-27 —* ***the item's reason was half wrong***. *“Mixes val and test” barely matters: **nothing tunes**, so val and test are interchangeable held-out samples — measured **test 0.769 vs val 0.762**.* ***The real fault was averaging across the three novelty regimes***, *which ask different questions and differ hugely in size (~875 / ~875 / **97** trials per fold), unweighted.* ***Built***: *`parse_regime`* *splits a regime name into its two axes (split x novelty);* *`eval_split`* *("test" | "val" | "both", default "both" = unchanged behaviour) on all four entry points, **so the distinction survives for whenever val becomes meaningful**; and four frames instead of two — the new* *`summary_by_novelty_df`* *is **the three numbers the paper reports**, and* *`summary_overall_df`* *is now per split rather than collapsing everything. Both carry a column saying what they pooled. Also removed a **duplicate copy** of the aggregator.* ***No number moved*** *— by-regime is numerically identical, and nothing reads the overall frame. Fold means stay unweighted: that is T3.10)*

  * **✅ DONE (2 of 3; third declined) — T3.10** — Fold-level CIs treat overlapping folds as independent, average folds unweighted, and fall back silently to z=1.96 — these are the CIs on the comparison figure. *(**done*** *2026-09-27. These are the ± on* ***balanced accuracy itself***, *not the coefficient CIs (that was T3.3).* ***Fixed***: *the fold mean is now weighted by* *`n_eval`* *— fold eval sets differ by up to **43%** within a regime, so a 78-trial fold counted as much as a 120-trial one; the plain average is kept as* *`unweighted_mean_*`. *Exact pooling for* *`accuracy`, *approximate for* *`balanced_accuracy`, *which needs confusion counts the summary rows lack.* ***Fixed***: *the z fallback now raises —* *`ci=0.80`* *used to draw a **95%** interval and title it **“80% CI”**.* ***Declined***: *`se = std/sqrt(n_folds)`* *— Diana: “it is what it is”. Folds share ~80% of training data so the bars are too narrow, but there is no unbiased estimator to switch to; now commented rather than dressed up.* ***Measured***: *weighting moves balanced accuracy by at most **0.204 pp** over 10 folds)*

  * **✅ DONE — T3.11** — Five group-feature functions mutate the caller's frame, creating an undocumented ordering dependency and leaking coercions into the saved output. *(**done*** *2026-09-27 — T1.7 had already taken most of it; what was left was the side effect itself, five wrappers each coercing the shared frame on the way past.* ***Hoisted*** *to one named step before the group loop, and* ***guarded*** *so an uncoerced frame raises and names the missing step instead of comparing str with int.* ***Restructure-compatible***: *the hoisted line lives in the group-function runner that becomes* *`features/build.py`, *and* *`restructure-map.md`* *§6.1 sends the coercion to* *`ingest/readers.py`* *— one line relocates. The paragraph path already did it this way.* ***Verified***: *output identical (0 of 5 columns differ), the subset call the docstring said “still raises” now runs, the guard fires readably, both paths compute.* ***No number moved***. *The registry* *`requires`* *declaration became **T3.24**, pushed)*

  * **↪️ ABSORBED — T3.12** — The blanket `fill_value = 0.0` is wrong for the exclude-convention columns. *(into T3.14)*

* **✅ DONE — T3.6** — Stop zero-filling `"."` in first fixation duration; coerce to NaN instead. *(**done*** *2026-09-23, **L1 and KnowQA both rebuilt** — the predicted* ***conclusion change*** *happened: on L1 the A–D spread collapses **31.5 → 10.2 ms** and correlation with* *`skip_rate`* *goes **−0.70…−0.84 → −0.017…+0.035**. Diffing all 217 model-ready columns before/after, **exactly 10 changed and all 10 are first-fixation-duration** — every other column bit-identical, so the headline 0.83 cannot have moved. Only the two pilots still hold the old column; they need the* *`data_prep_new_exp.ipynb`* *path. See todo.md T3.6)*

* **✅ DONE — T3.13** — Dwell time and fixation count stay coverage-inclusive; the asymmetry inside the metric family is deliberate. *(**done*** *— no action, and do not "fix" it for tidiness)*

* **✅ DONE — T3.14** — Keep the `0` fill for pupil and first-fixation features at model-prep time, but make it named, commented, counted and reported in Methods. *(**done*** *2026-09-23 — the fill **stays blanket** (Diana amended the item's point 2: simplest data prep, keep it), but is now counted by* *`imputed_cell_counts`, located by* *`imputed_mask_`, and a NaN with no documented cause **warns** via* *`unexpected_imputed_`* *on its way to becoming 0. Coefficients **bit-identical** to the old fill, max abs diff 0.0. **Methods number: 253 cells = 0.130% of the L1 headline feature matrix.** Registered in* *`conventions.md`* *→ Register of accepted deviations)*

* **↪️ ABSORBED — T3.15** — Two feature-provenance paths in `cross_validation.py` mean the same nominal model run from two notebooks needn't agree. *(into T3.21, which settles the direction)*

* **✅ DONE — T3.17** — Turn two confirmed facts into assertions: every trial appears in every feature block, and every trial has a confirmed selection. *(**done*** *2026-09-27 — **both invariants hold on all four datasets, so no number moved**.* *`assert_full_coverage(left, right, keys, name)`* *in the new* *`src/checks.py`* *guards **eight of the nine** merges in* *`build_trial_level_model_df`, catching both silent failure modes — a key missing from a block, and a **duplicated** key, which multiplies rows rather than dropping them. **The paragraph merge is deliberately exempt** and says so: it is the one block not built from* *`df`, so partial coverage is real rather than a fault.* *`add_is_correct`* *now raises on a null position, which closes **T3.5**. Checked before the merge, not after — the item's sketched* *`(left, merged)`* *signature cannot tell an unmatched key from a matched NaN. **Verified**: zero null positions across all four datasets; 0 trials with a whole block NaN on L1; KnowQA rebuild **bit-identical** on all 200 columns; injected faults all raise; and the **CV slice path is safe** — no guard fires on regime slices, 10/50/90% random subsets, or a single participant. Helper lives at* *`src/checks.py`, the home* *`restructure-map.md`* *§3 gives* *`lib/checks.py`* *— **T3.7 is its next caller**)*

* **✅ DONE — T3.18** — Text misalignment: real, in three distinct modes, including 20 L1 trials nobody had looked at; area labels now come from screen geometry instead of the stored text. *(**done*** *2026-09-07 — two residuals noted, neither is this item)*

* **⏭️ PUSHED — T3.19** — Investigate whether `last_answer_area_visited_lbl` is buggy; cross-checking it against its two click-based siblings is the starting test. *(**pushed 2026-09-27**, Diana · M · ⚠️ · needs running, not a ruling.* ***Returns at stage E***, *which* *`restructure-map.md`* *§11 already assigns it to — that is where* *`viz/`* *dissolves into per-analysis* *`plots.py`* *and the last-visitation figures are re-run anyway, so the cross-check lands with the rerun instead of ahead of it.* ***Not on the headline path***: *the column is not in* *`SELECT_1_COLS`, so the model is unaffected either way. What does depend on it is attribution —* *`findings.md`* *§2's ~68–71% comes from the **before-confirm** sibling, not this one, so the paper's "80%" claim still needs pinning to whichever variant produced it)*

* **✅ DONE in code — T3.20** — Every dataset writes its own pupil baseline and every consumer reads the right one. *(**done*** *2026-09-23 — no default source any more (both resolvers raise unless the baseline is named);* *`pupil_norm`* *imports no dataset path at all; **every person must have a baseline** and a missing one or a non-positive SD now raises instead of yielding a silently all-NaN* *`_z`* *column; the paragraph screen computes its own baseline by **streaming** the multi-GB paragraph fixation report; KnowQA's* *`pupil_norm_unit`* *defaults to* *`"participant"`* *so it finally writes its own* *`participant_pupils.csv`* *from its own fixations.* ***⚠️ Rebuilds outstanding**: KnowQA pupil z and L1 paragraph-span pupil features both move. **No paper number affected** — no paragraph pupil column reaches any model, and* *`RT_correlations`* *never touches pupil)*

* **✅ DONE — T3.21** — Give every participant-level aggregate an explicit scope. *(**done*** *2026-09-25, all 13 rows — two parameters, not one enum:* *`scope_df`* *= which trials estimate it (default: the frame given),* *`scope_by`* *= how they are partitioned (default: per participant, pooling regimes and sessions).* *`n_strategy_trials_{with,no}_q`* *records the scope.* ***No number moved anywhere***, *L1/KnowQA/both pilots rebuilt and diffed. Rulings: completion map learned over the whole dataset (row 5); KnowQA regime comparison stays session-scoped for train/test symmetry (row 7); rows 8/9 got counts not parameters; rows 12/13 needed a sentence, and 13 turned out inert —* *`TRIAL_ANSWERS`* *is consumed by nothing. Produced the paper's missing all-participants X%: **53.6% raw / 57.2% completed**)*

## T4 · Outputs and artifacts

* **⏭️ PUSHED — T4.0** — Standing rule: a result isn't saved until its numbers are on disk. *(**mechanism done** 2026-09-20 with T1.3 —* *`tables=`* *is a required argument and the test is structural:* *`figures/`* *and* *`tables/`* *are siblings in one analysis folder. **What remains — regenerating** *`findings.md`* **from the saved CSVs — is ⏭️ pushed 2026-09-27**, Diana: the ledger gets dealt with when she* ***triple-checks everything manually after the restructure***. *Rebuilding it now would be transcribing numbers that are about to be regenerated anyway. Returns after the restructure, at which point the ⚠️ NOT VERIFIED header can come off)*

* **✅ DONE — T4.1** — `RT_correlations` used to save nothing, so a live Results subsection existed only as cell output in a 1.2 MB notebook. *(**done*** *2026-09-20 with T1.3 —* *`plot_corr_map_pair`* *now routes through* *`save_output`* *and carries r / BH-adjusted p / n with every map)*

* **⏭️ PUSHED — T4.2** — **329** zero-byte PNGs from **two** failed syncs (21 of them sit beside non-empty siblings, in live `correctness_measures` / `matching_correctness` folders). *(**pushed 2026-09-27**, Diana:* ***everything will be re-run as she checks***, *so there is nothing to re-run separately now — the figures come back from the same pass that verifies them — **note, corrected 2026-09-27: there are no empty files left to refill.** All 329 went with the old `reports/{plots,report_data}/` trees, deleted in `496f8d0`; the 108 `area_significance_heatmaps` figures are now simply *absent* and have to be generated, not overwritten. Was unblocked 2026-09-21 (the* *`statistics.ipynb`* *`AttributeError`* *is gone), so this is a deferral of effort, not a blocker: **the rerun was never attempted, so whether it completes cleanly is still untested**. The 108* *`area_significance_heatmaps`* *figures are paper code, so they are the ones that must come back)*

* **⏭️ PUSHED — T4.3** — `reports/` is tracked in git, including a 63 MB pickle. *(**pushed 2026-09-27** — Diana thought this was no longer true;* ***measured, and it is half true***. *The pickle is* ***gone from the working tree*** *(deleted in* *`496f8d0`* *"ploting revamp"), so nothing on disk carries it. But it is* ***still in git history*** *as a 63.4 MB blob, and* *`reports/`* *is* ***still tracked*** *— 1,615 files,* *`.gitignore`* *covers* *`/data/`* *and* *`/data_raw/`* *but not* *`/reports/`. *`.git`* *is* ***1.6 GB***. ***Worth knowing if repo size ever matters***: *the pickle is no longer the biggest problem —* ***notebooks are***. *History holds a 32 MB* *`visualisations.ipynb`, *a 30 MB* *`notebooks/vizes`* *blob and four copies of* *`preliminary_analysis.ipynb`* *at ~12 MB each. Still a release-time question, and still a git operation, so yours)*

* **⏭️ PUSHED — T4.4** — Old JSONs, an orphan zip, 2.5 GB of archive CSVs, indistinguishable paper figure names. *(**pushed 2026-09-27**, Diana: skip. Unchanged reasoning — the restructure is what decides which outputs are still meaningful, and* *`restructure-map.md`* *§13 already holds* *`archive/`* *out of scope for this round. Inventory only; nothing here gets touched before then)*

## T5 · Must run from scratch — ⏭️ **WHOLE TIER PUSHED to the restructure**

> **Diana, 2026-09-27**: *"it makes more sense to deal with it on a codebase that is in a state I want it."* This **confirms** `restructure-map.md` §11 rather than deferring against it — all eleven items are already assigned to stages there (**A**: T5.1 / T5.2 / T5.4 · **B**: T5.3 · **D**: T5.11 · **F**: T5.6 / T5.10 · **G**: T5.5 / T5.7 / T5.8 / T5.9). Practical meaning: **don't pick one off individually** — each lands with its stage, on the tree that stage produces. Doing T5.1–T5.3 now would write import conventions and path migrations into a layout Stages A–C are about to replace. T5.7 and T5.8 are rulings, not outstanding work.

* **✅ DONE — T5.1** — Add `__init__.py` throughout; there are none anywhere in `src/`. *(**done** 2026-10-06 — 18 added, 22 in total. Each carries one line saying what the package holds; none re-export anything, because an `__init__` that imports would create cycles in a tree this interconnected)*

* **✅ DONE — T5.2** — Settle on one import convention; two coexist and only work because notebooks push two paths onto `sys.path`. *(**done** 2026-10-06 — 21 lines in 11 files converted to the `src.`-prefixed form, and `src/` came off the path entirely. **This stopped being optional**: with `src/` on the path, making `src/stats/` a real package shadows Python's standard-library `statistics` module, and seaborn's `from statistics import NormalDist` resolved to our code — 21 modules stopped importing. See `pitfalls.md` §6)*

* **✅ DONE — T5.3** — Migrate ~40 hardcoded `"../reports/..."` literals to `PROJECT_ROOT`. *(**done** 2026-09-27 — most of it fell out of T1.3; the last four source-cell hits were cleared by hand. **Now 0 in `src/**.py` and 0 in notebook source cells**, verified by count. The ~276 remaining matches are stale printed paths inside stored notebook outputs, which are cosmetic and disappear on the next run)*

* **✅ DONE — T5.4** — Write `environment.yml`, pinned to Python 3.11. *(**done** 2026-10-06 — 15 conda + 2 pip deps, every pin matched against the live env. Pinned tightly on purpose: three of the breakages found in the cleanup were a dependency moving, not our code changing — pandas 3 → patsy (T2.7), scikit-learn 1.8 → `multi_class` (T2.2), pymer4 0.9 → `Lmer` (T2.6). ⚠️ **not yet tested on a clean machine** — that is stage G)*

* **T5.5** — Write the README: what the project is, how to run it, `L1` = native speaker, and the build order. *(M)*

* **T5.6** — Document and enforce the build order between `answer_RTs.features` and `answer_correctness.model_data`. *(S)*

* **T5.7** — Ship instructions for placing your own OneStop download, plus one clear failure message when it isn't there. *(S ·* ***🟡 decided**)*

* **T5.8** — Keep `all_participants_with_practice.csv`; keep both pilots runnable; `paragraph_RT_run_based.csv` waits for the restructure. *(S ·* ***🟡 decided**)*

* **T5.9** — Two EyeBench entry points write the same cache with different feature definitions, and the cache can't say which one made it. *(M)*

* **T5.10** — `data_prep_new_exp.ipynb` overwrites KnowQA's output with the older identity scheme; mark superseded or remove the collision. *(S)*

* **T5.11** — `evaluate_one_fold_on_regimes` refits the same model 6× per fold; hoisting it removes the speed argument for the leaking CV path. *(S · efficiency only)*

## T6 · The restructure

* **T6** — The staged reorganization; the plan of record is `docs/restructure-map.md` (Stages 0 → G). **Agreed ≠ started — no stage runs without being proposed first.**

* **✅ DONE — T6.1** — Split paragraph preprocessing out of `answer_RTs/` and out of the QA prep, so there's one paragraph table per dataset joined in explicitly. *(**done*** *2026-09-23 — new* *`derived/paragraph_prep.py`* *owns the screen;* *`build_rt_and_tfd`* *is answer-only;* *`data_csv_generation`* *opens no paragraph report; the* *`include_paragraph`* *flag and the* *`how="inner"`* *merge that silently dropped paragraph-less trials are both gone;* *`model_data.PARAGRAPH_MODEL_COLS`* *joins the paragraph table explicitly.* *`answer_RTs/features.py`* *is now a compatibility surface so its two live callers keep working. Bonus: the paragraph IA read is* *`usecols`-limited to 15 of 157 columns, which is what makes a rebuild fit in memory)*

***

## Not items, but don't forget them

* **Nothing is currently waiting on a decision from you.** The answered-questions table at the end of `todo.md` is the record of what was asked and what was ruled.

* **§V "Verify before trusting"** — six one-command checks for the ⚠️ items. **All six have now been run.** Checks 1–5 on 2026-09-21: 1, 2 and 4 reproduced (1 and 2 found *more* than their items describe), while 3 and 5 came back clean — notably **T3.7 has not bitten**, so it is a guard to add rather than damage to repair. **Check 6 (T3.3's CIs) was answered by T3.3 itself on 2026-09-27** — the clustered bootstrap is 1.44× wider than Wald and **all 12 features stay significant**, so the conclusion the check was worried about did not change. **Check 2 no longer reproduces**: T2.1 was fixed 2026-09-27, and `generate_all_feature_column_sets(df)` now completes with 62 sets and no phantom columns — but only with `src/` on `sys.path`, since T5.2's bare-import failure still masks everything in that module.

* **Doc-consistency rules** — every `pitfalls.md` / `data-pipeline.md` gotcha cites a `T*` id or says no action is needed; when retracting a claim, grep the *words*, not just the id.

***

## Your own reminders

Yours, not tracked in `todo.md` — no `T*` id, nothing checked or verified by Claude.

* [ ] Check final fixation on incoming feedback ratio. *(added 2026-09-16)*
