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

| Item      | Why it's first                                                                                                    |
| --------- | ----------------------------------------------------------------------------------------------------------------- |
| **T3.17** | cheap, and changes no number if the invariants hold                                                               |
| **T4.2**  | not a fix at all — just re-run the plots (the `statistics.ipynb` blocker is gone as of 2026-09-21, see T1.4/T2.5) |

*(T3.1 was the head of this list and is done — 2026-09-26.)*

***

## T1 · Small consistency fixes — safe, no reported number moves

* **✅ DONE — T1.1** — Dominance threshold is `≥` everywhere, and the five copies of the modal-strategy pick are now one function in `derived/pattern_breaking.py`. *(**done*** *2026-09-20 — hunters 48.89/53.33, gatherers 58.33/61.11; a convergence onto the numbers already quoted, no figures to regenerate)*

* **✅ DONE — T1.3** — One `save_output()` everywhere: 81 call sites, 33 source files, 11 notebooks; `tables=` is required so a figure cannot be saved without its numbers; outputs moved to `reports/<analysis>/{figures,tables}/` with key-value filenames; Overleaf mirroring off. *(**done*** *2026-09-20 — no numbers moved; lands* *`restructure-map.md`* *§9 early and most of T5.3; closes T4.1)*

* **✅ DONE — T1.4** — Comments naming constants that no longer exist, fixed at all seven live sites. *(**done*** *2026-09-21 — documentation-only, verified by AST comparison;* *`statistics.ipynb`* *turned out to be already fixed, so this never became a breakage and T2.5 closes with it. Three of the seven were naming the wrong concept*, not just a stale name)

* **T1.5** — A table of ~10 tiny cleanups: dead code, duplicate definitions, unused imports, stale docstrings, orphan `.pyc`. *(S ·* ***two rows closed*** *— the two* *`paper_dirs`* *conventions went with* *`paper_dirs`* *itself under T1.3, and its last stale references were cleared 2026-09-21 (dead constant in* *`loo_runs.py`, three notebook copies, one markdown cell teaching the removed API); the duplicate* *`save_plot_and_report`* */* *`maybe_save_plot`* *save paths are gone too.* ***One row is worse than written**:* *`_wilson_ci`* *is nested* ***three*** *times in* *`visualisations_correctness_measures.py`, not once)*

* **✅ DONE — T1.6** — Merge the two "starting strategy" implementations into one function in `derived/`. *(**done*** *— one implementation since 2026-09-20 with* *`viz/`* *only plotting; the* *`scope`* *parameter it was waiting on landed with T3.21 on 2026-09-25, and the last loose end — where the prefix-completion map is learned — was decided with it: **one map over the whole dataset**, no longer per group)*

* **✅ DONE — T1.7** — Merge the QA and paragraph implementations of the same eight per-area metrics into one set parameterized by grouping column. *(**done*** *2026-09-23 — one implementation in* *`derived/area_metrics.py`. All seven QA metrics reproduce the saved table to ≤5e-13 (CSV round-trip only) and* *`first_encounter`* *is bit-identical, so **no QA number moved**. Two things the merge turned up: the two* *`num_label_visits`* *were **not the same measure** (disagreed on **77% of trials** — the paragraph side dropped off-area fixations the answer side resolves to the nearest word; now shared, which moves paragraph counts upward), and* *`first_encounter_pupil_size`* *silently depended on row order, now an explicit* *`IA_ID`* *sort)*

* **~~T1.8~~ — removed 2026-09-27** — the stray `index` column. *(**premise expired**: all four datasets carry it now, so the schemas agree. And it is inert — row numbers, never reaches the trial-level model table, nothing reads it, ~5 MB on a 3.2 GB file. Number retired, not reused)*

## T2 · Broken today — all low priority, none on the paper's path

* **T2.1** — `generate_column_options.py` references three constants `feature_groups.py` no longer has. *(**ran 2026-09-21*** *— the three constants are indeed gone, but the* *`AttributeError`* *is* ***masked**: T5.2's bare-import failure fires first, so T5.2 has to land before this item's symptom is even observable)*

* **⏭️ PUSHED — T2.2** — `answer_loc/` cannot import at all. *(**pushed 2026-09-27** — Diana: "don't care about answer loc, it can stay broken". May be tidied after the restructure, which files it under* *`explorations/answer_location/`* *where* ***parked code moves as-is, broken or not***. ***Ran 2026-09-21 and it is worse than the item says***: *`answer_loc_eval`* *fails earlier on T5.2's import convention, and* *`answer_loc_models.py:109`* *passes* *`multi_class=`, removed in scikit-learn 1.8.0 — so fixing only what is listed would leave it broken)*

* **T2.3** — `fit_model_on_prepared_full_data` calls random-effects methods the live logreg doesn't implement; works only with the Julia backend. *(⚠️ · still never executed)*

* **T2.4** — `get_last_visited_feature_cols` always returns `[]` because of a prefix mismatch, so the **default** feature set has silently contained no last-label features. *(**ran 2026-09-21*** *— returns* *`[]`* *while the table carries 16* *`last_*`* *columns over 19,436 trials · still open · the one with quiet consequences)*

* **✅ DONE — T2.5** — Not a breakage after all: `statistics.ipynb` already passes `Con.AREA_METRIC_COLUMNS_MODELING`. *(**done*** *2026-09-21 — fixed independently in the plots revamp; this is what kept T1.4 in the comment tier, and it unblocks T4.2's 108 figures)*

## T3 · Fixes that move reported numbers

> The tier is mostly one principle violated repeatedly: a silent alteration where a loud error
> belonged. Read the intro table in `todo.md` before picking items off individually.

* **✅ DONE — T3.1** — The three correctness Fisher tests now run at trial grain, reading the split the builder already computed instead of re-deriving it; `_check_trial_frame` raises on a duplicated `(participant_id, TRIAL_INDEX)`. *(**done*** *2026-09-26 — 24 analyses rerun (24 figures, 48 tables); n 760,628 → 19,436.* ***One conclusion changed**: gatherers · threshold-2 goes p 7.7e-09 → 0.189, significant → n.s., its* *`≤ 2`* *group being 116 trials. The other 23 survive at p < 1e-3. Bars/CIs verified untouched — no* *`__summary.csv`* *differs and 23 of 24 figures are byte-identical.* ***Two corrections to the item**: it was not "one word in three places" — each test re-derived its own split, and for the dwell test that re-derivation* ***was*** *the bug; and the blast radius was 24 analyses, not ~96 files / ~51 figures, a count that predated the T1.3 inversion. **XYXY is not generated at all** —* *`use_xyxy`* *defaults False — which is a scope question, not part of this item)*

* **⏭️ PUSHED — T3.23** — **Read up on clustered CIs.** *(**Diana's item, not Claude's** · reading, not code · links live at the top of* *`common/data_utils.py::wald_logreg_coef_cis`*'s *docstring — three Diana added, four academic ones Claude added below them;* ***don't tidy either set away***, *they sit next to the formula on purpose. It is what makes* ***T3.3*** *(bootstrap vs Wald, currently split by cost) and* ***T3.22*** *(clustering as precision-not-direction) decisions rather than defaults. No number depends on it)*

* **⏭️ PUSHED — T3.22** — Trials are treated as independent when they are nested in 360 participants and 972 items, so every count-based p here is too small. *(M–L ·* ***pushed past the restructure**, Diana 2026-09-26 — measured: ICC 0.041 by participant / **0.126 by item**, effective n ~6,000 not 19,436. A participant-clustered bootstrap* ***moves no conclusion***, so this is precision, not direction. Same root cause as* ***T3.3***; also live in* *`RT_correlations`* *(participant-only clustering, item level unhandled and larger) and ⚠️* *`mixed_area_comparisons`* *(no trial-level random effect). Waits for* *`modeling/inference.py`* *so clustered inference lands once — stage **D**)*

* **~~T3.2~~ — removed 2026-09-27** — one Methods sentence on the hand-picked features. *(**the paper's job, not the code's** — Diana. Number retired, not reused. What it established stands:* *`SELECT_1_COLS`* *is a manual pick so there is no selection-leakage caveat, and all six runs in the comparison figure are hand-specified. One code fact kept in* *`todo.md`*: *`collect_and_plot_correctness_runs`* *scans a directory rather than taking a curated list, so a machine-selected run saved later would join the figure silently)*

* **✅ DONE (paper path) — T3.3** — `collect_logreg_coef_summaries`, the full-data fit behind the paper's coefficient figures, now defaults to the participant-clustered bootstrap; cell 29 of `answer_corr_prediction.ipynb` passes it explicitly. *(**done*** *2026-09-27 — L1's 12-feature headline model: intervals* ***1.44× wider*** *than Wald and* ***all 12 stay significant***, *stable across seeds and 2k/5k resamples. The item predicted a conclusion change; there isn't one.* ***Two corrections to the diagnosis***: *clustering is the* ***smallest*** *of the three defects (a row bootstrap already recovers 1.35× of the 1.44×), and* *`ci_cluster="auto"`* *silently means* ***row***, *not cluster.* ***Three paths, three settings*** *(Diana, 2026-09-27): L1 paper figures* ***bootstrap+cluster***; *cross-validation and KnowQA* ***wald+cluster***. *To make that defensible* *`wald_logreg_coef_cis`* *was rewritten as a* ***cluster-robust sandwich carrying the L2 penalty and class weights***, *so the Wald path has none of the three defects either — it reproduces the bootstrap to* ***within 2.5% on all 12 features at ~560× the speed***. ⚠️ ***KnowQA's clustered Wald is rank-deficient***: *6 clusters for 13 parameters, so* *`rank(meat)=6`* *and two intervals come out* ***3× too narrow***. ***KnowQA reverted to bootstrap the same day***; *`get_coef_summary`* *now defaults to bootstrap+cluster and the two cost-bound callers opt out explicitly. Wald still warns when* *`n_clusters <= n_params`. **Sandwich references are in the* *`wald_logreg_coef_cis`* *docstring** (Cameron & Miller 2015, Zeileis 2006, MacKinnon et al. 2023, Freedman 2006).* ***Do not report Study 2 coefficient significance at n=6 either way***)*

* **T3.4** — Umbrella for the smaller movers below (T3.5, T3.7–T3.12). *(⚠️)*

  * **↪️ ABSORBED — T3.5** — A missing confirmed selection is scored as a wrong answer; that case can't occur, so it should assert. *(into T3.17)*

  * **T3.7** — Run-based RT silently produces all-zero rows on a join-key mismatch, indistinguishable from real zeros. *(needs a coverage assertion ·* ***checked 2026-09-21: it has not bitten*** *— 0 of 13* *`RT_pure_*`* *columns are uniformly zero, so this is a guard to add, not damage to repair)*

  * **↪️ ABSORBED — T3.8** — *(into T3.20)*

  * **T3.9** — Keep the `val_*` regimes and report them; stop averaging all six regimes into one balanced-accuracy number. *(🟡 decided)*

  * **T3.10** — Fold-level CIs treat overlapping folds as independent, average folds unweighted, and fall back silently to z=1.96 — these are the CIs on the comparison figure.

  * **T3.11** — Five group-feature functions mutate the caller's frame, creating an undocumented ordering dependency and leaking coercions into the saved output.

  * **↪️ ABSORBED — T3.12** — The blanket `fill_value = 0.0` is wrong for the exclude-convention columns. *(into T3.14)*

* **✅ DONE — T3.6** — Stop zero-filling `"."` in first fixation duration; coerce to NaN instead. *(**done*** *2026-09-23, **L1 and KnowQA both rebuilt** — the predicted* ***conclusion change*** *happened: on L1 the A–D spread collapses **31.5 → 10.2 ms** and correlation with* *`skip_rate`* *goes **−0.70…−0.84 → −0.017…+0.035**. Diffing all 217 model-ready columns before/after, **exactly 10 changed and all 10 are first-fixation-duration** — every other column bit-identical, so the headline 0.83 cannot have moved. Only the two pilots still hold the old column; they need the* *`data_prep_new_exp.ipynb`* *path. See todo.md T3.6)*

* **✅ DONE — T3.13** — Dwell time and fixation count stay coverage-inclusive; the asymmetry inside the metric family is deliberate. *(**done*** *— no action, and do not "fix" it for tidiness)*

* **✅ DONE — T3.14** — Keep the `0` fill for pupil and first-fixation features at model-prep time, but make it named, commented, counted and reported in Methods. *(**done*** *2026-09-23 — the fill **stays blanket** (Diana amended the item's point 2: simplest data prep, keep it), but is now counted by* *`imputed_cell_counts`, located by* *`imputed_mask_`, and a NaN with no documented cause **warns** via* *`unexpected_imputed_`* *on its way to becoming 0. Coefficients **bit-identical** to the old fill, max abs diff 0.0. **Methods number: 253 cells = 0.130% of the L1 headline feature matrix.** Registered in* *`conventions.md`* *→ Register of accepted deviations)*

* **↪️ ABSORBED — T3.15** — Two feature-provenance paths in `cross_validation.py` mean the same nominal model run from two notebooks needn't agree. *(into T3.21, which settles the direction)*

* **T3.17** — Turn two confirmed facts into assertions: every trial appears in every feature block, and every trial has a confirmed selection. *(S–M · ready)*

* **✅ DONE — T3.18** — Text misalignment: real, in three distinct modes, including 20 L1 trials nobody had looked at; area labels now come from screen geometry instead of the stored text. *(**done*** *2026-09-07 — two residuals noted, neither is this item)*

* **T3.19** — Investigate whether `last_answer_area_visited_lbl` is buggy; cross-checking it against its two click-based siblings is the starting test. *(M · ⚠️ · needs running, not a ruling)*

* **✅ DONE in code — T3.20** — Every dataset writes its own pupil baseline and every consumer reads the right one. *(**done*** *2026-09-23 — no default source any more (both resolvers raise unless the baseline is named);* *`pupil_norm`* *imports no dataset path at all; **every person must have a baseline** and a missing one or a non-positive SD now raises instead of yielding a silently all-NaN* *`_z`* *column; the paragraph screen computes its own baseline by **streaming** the multi-GB paragraph fixation report; KnowQA's* *`pupil_norm_unit`* *defaults to* *`"participant"`* *so it finally writes its own* *`participant_pupils.csv`* *from its own fixations.* ***⚠️ Rebuilds outstanding**: KnowQA pupil z and L1 paragraph-span pupil features both move. **No paper number affected** — no paragraph pupil column reaches any model, and* *`RT_correlations`* *never touches pupil)*

* **✅ DONE — T3.21** — Give every participant-level aggregate an explicit scope. *(**done*** *2026-09-25, all 13 rows — two parameters, not one enum:* *`scope_df`* *= which trials estimate it (default: the frame given),* *`scope_by`* *= how they are partitioned (default: per participant, pooling regimes and sessions).* *`n_strategy_trials_{with,no}_q`* *records the scope.* ***No number moved anywhere***, *L1/KnowQA/both pilots rebuilt and diffed. Rulings: completion map learned over the whole dataset (row 5); KnowQA regime comparison stays session-scoped for train/test symmetry (row 7); rows 8/9 got counts not parameters; rows 12/13 needed a sentence, and 13 turned out inert —* *`TRIAL_ANSWERS`* *is consumed by nothing. Produced the paper's missing all-participants X%: **53.6% raw / 57.2% completed**)*

## T4 · Outputs and artifacts

* **T4.0** — Standing rule: a result isn't saved until its numbers are on disk. \*(**mechanism done** 2026-09-20 with T1.3 — `tables=` is a required argument and the test is now structural: `figures/` and `tables/` are siblings in one analysis folder. **What remains is regenerating** *`findings.md`* *from the saved CSVs, which now exist.)*

* **✅ DONE — T4.1** — `RT_correlations` used to save nothing, so a live Results subsection existed only as cell output in a 1.2 MB notebook. *(**done*** *2026-09-20 with T1.3 —* *`plot_corr_map_pair`* *now routes through* *`save_output`* *and carries r / BH-adjusted p / n with every map)*

* **T4.2** — **329** zero-byte PNGs from **two** failed syncs (21 of them sit beside non-empty siblings, in live `correctness_measures` / `matching_correctness` folders). *(S · the 108* *`area_significance_heatmaps`* *are paper code, so* ***not*** *low priority ·* ***unblocked 2026-09-21***\*: the `statistics.ipynb` `AttributeError` is gone, so it is a pure rerun again — though the rerun has not been attempted, so whether it completes is untested)\*

* **T4.3** — `reports/` is tracked in git, including a 63 MB pickle. *(**deferred*** *— a release-time question, and a git operation, so yours)*

* **T4.4** — Old JSONs, an orphan zip, 2.5 GB of archive CSVs, indistinguishable paper figure names. *(**deferred*** *until the restructure says what's still meaningful)*

## T5 · Must run from scratch — ⏭️ **WHOLE TIER PUSHED to the restructure**

> **Diana, 2026-09-27**: *"it makes more sense to deal with it on a codebase that is in a state I want it."* This **confirms** `restructure-map.md` §11 rather than deferring against it — all eleven items are already assigned to stages there (**A**: T5.1 / T5.2 / T5.4 · **B**: T5.3 · **D**: T5.11 · **F**: T5.6 / T5.10 · **G**: T5.5 / T5.7 / T5.8 / T5.9). Practical meaning: **don't pick one off individually** — each lands with its stage, on the tree that stage produces. Doing T5.1–T5.3 now would write import conventions and path migrations into a layout Stages A–C are about to replace. T5.7 and T5.8 are rulings, not outstanding work.

* **T5.1** — Add `__init__.py` throughout; there are none anywhere in `src/`. *(S)*

* **T5.2** — Settle on one import convention; two coexist and only work because notebooks push two paths onto `sys.path`. *(M)*

* **T5.3** — Migrate ~40 hardcoded `"../reports/..."` literals to `PROJECT_ROOT`. *(M)*

* **T5.4** — Write `environment.yml`, pinned to Python 3.11. *(S)*

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

* **§V "Verify before trusting"** — six one-command checks for the ⚠️ items. **Checks 1–5 were run 2026-09-21** and the outcomes were not uniform: 1, 2 and 4 reproduce (and 1 and 2 found *more* than their items describe), while 3 and 5 came back clean — notably **T3.7 has not bitten**, so it is a guard to add rather than damage to repair. **Number 6 (T3.3's CIs) has still never been run** and is the one most likely to affect the paper.

* **Doc-consistency rules** — every `pitfalls.md` / `data-pipeline.md` gotcha cites a `T*` id or says no action is needed; when retracting a claim, grep the *words*, not just the id.

***

## Your own reminders

Yours, not tracked in `todo.md` — no `T*` id, nothing checked or verified by Claude.

* [ ] Check final fixation on incoming feedback ratio. *(added 2026-09-16)*
