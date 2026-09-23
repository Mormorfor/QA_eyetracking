# TODO — quick reference

One line per item in `docs/todo.md`. **Everything above the last section is index only** — any
detail, reasoning, blast radius or measured number lives in `todo.md` under the same id. The
one exception is *Your own reminders* at the end, which has no counterpart there.

Status key: **✅ DONE** = finished, nothing left for anyone · **↪️ ABSORBED** = folded into
another item, which is named · **decided** = the ruling is made, **implementation still
outstanding** · **ready** = nothing blocks it · **blocked** = waits on another item ·
**deferred** = deliberately not now · ⚠️ = read from code, never executed (see §V of `todo.md`).

**Nothing here is waiting on a decision from you.** Anything not ✅ DONE or ↪️ ABSORBED is
work outstanding on Claude's side, or a deliberate deferral.

*(There is no T1.2 and no T3.16 — T1.2 never existed, T3.16 was considered and declined. Don't
go looking.)*

***

## Ready to start, nothing blocking

| Item      | Why it's first                                                                                                    |
| --------- | ----------------------------------------------------------------------------------------------------------------- |
| **T3.21** | the last thing blocking T1.6's `scope` parameter                                                                  |
| **T3.1**  | the fix is one word in three places, and it's high impact                                                         |
| **T3.17** | cheap, and changes no number if the invariants hold                                                               |
| **T4.2**  | not a fix at all — just re-run the plots (the `statistics.ipynb` blocker is gone as of 2026-09-21, see T1.4/T2.5) |

***

## T1 · Small consistency fixes — safe, no reported number moves

* **✅ DONE — T1.1** — Dominance threshold is `≥` everywhere, and the five copies of the modal-strategy pick are now one function in `derived/pattern_breaking.py`. *(**done*** *2026-09-20 — hunters 48.89/53.33, gatherers 58.33/61.11; a convergence onto the numbers already quoted, no figures to regenerate)*

* **✅ DONE — T1.3** — One `save_output()` everywhere: 81 call sites, 33 source files, 11 notebooks; `tables=` is required so a figure cannot be saved without its numbers; outputs moved to `reports/<analysis>/{figures,tables}/` with key-value filenames; Overleaf mirroring off. *(**done*** *2026-09-20 — no numbers moved; lands* *`restructure-map.md`* *§9 early and most of T5.3; closes T4.1)*

* **✅ DONE — T1.4** — Comments naming constants that no longer exist, fixed at all seven live sites. *(**done*** *2026-09-21 — documentation-only, verified by AST comparison;* *`statistics.ipynb`* *turned out to be already fixed, so this never became a breakage and T2.5 closes with it. Three of the seven were naming the wrong concept*, not just a stale name)

* **T1.5** — A table of ~10 tiny cleanups: dead code, duplicate definitions, unused imports, stale docstrings, orphan `.pyc`. *(S ·* ***two rows closed*** *— the two* *`paper_dirs`* *conventions went with* *`paper_dirs`* *itself under T1.3, and its last stale references were cleared 2026-09-21 (dead constant in* *`loo_runs.py`, three notebook copies, one markdown cell teaching the removed API); the duplicate* *`save_plot_and_report`* */* *`maybe_save_plot`* *save paths are gone too.* ***One row is worse than written**:* *`_wilson_ci`* *is nested* ***three*** *times in* *`visualisations_correctness_measures.py`, not once)*

* **T1.6** — Merge the two "starting strategy" implementations into one function in `derived/`. *(M ·* ***all but done*** *2026-09-20 — there is now one implementation of every dominance quantity and* *`viz/`* *only plots; what is left is purely the* *`scope`* *parameter, i.e. T3.21)*

* **✅ DONE — T1.7** — Merge the QA and paragraph implementations of the same eight per-area metrics into one set parameterized by grouping column. *(**done*** *2026-09-23 — one implementation in* *`derived/area_metrics.py`. All seven QA metrics reproduce the saved table to ≤5e-13 (CSV round-trip only) and* *`first_encounter`* *is bit-identical, so **no QA number moved**. Two things the merge turned up: the two* *`num_label_visits`* *were **not the same measure** (disagreed on **77% of trials** — the paragraph side dropped off-area fixations the answer side resolves to the nearest word; now shared, which moves paragraph counts upward), and* *`first_encounter_pupil_size`* *silently depended on row order, now an explicit* *`IA_ID`* *sort)*

* **T1.8** — L1's `all_participants.csv` carries a stray `index` column KnowQA's doesn't, so "same pipeline, same columns" isn't quite true. *(S · ready)*

## T2 · Broken today — all low priority, none on the paper's path

* **T2.1** — `generate_column_options.py` references three constants `feature_groups.py` no longer has. *(**ran 2026-09-21*** *— the three constants are indeed gone, but the* *`AttributeError`* *is* ***masked**: T5.2's bare-import failure fires first, so T5.2 has to land before this item's symptom is even observable)*

* **T2.2** — `answer_loc/` cannot import at all. *(**ran 2026-09-21*** *— reproduces,* ***and the item understates it**:* *`answer_loc_eval`* *fails earlier on T5.2's import convention, and* *`answer_loc_models.py:109`* *passes* *`multi_class=`, removed in scikit-learn 1.8.0. Fixing only what T2.2 lists would leave it broken)*

* **T2.3** — `fit_model_on_prepared_full_data` calls random-effects methods the live logreg doesn't implement; works only with the Julia backend. *(⚠️ · still never executed)*

* **T2.4** — `get_last_visited_feature_cols` always returns `[]` because of a prefix mismatch, so the **default** feature set has silently contained no last-label features. *(**ran 2026-09-21*** *— returns* *`[]`* *while the table carries 16* *`last_*`* *columns over 19,436 trials · still open · the one with quiet consequences)*

* **✅ DONE — T2.5** — Not a breakage after all: `statistics.ipynb` already passes `Con.AREA_METRIC_COLUMNS_MODELING`. *(**done*** *2026-09-21 — fixed independently in the plots revamp; this is what kept T1.4 in the comment tier, and it unblocks T4.2's 108 figures)*

## T3 · Fixes that move reported numbers

> The tier is mostly one principle violated repeatedly: a silent alteration where a loud error
> belonged. Read the intro table in `todo.md` before picking items off individually.

* **T3.1** — Fisher tests are handed the IA-level frame instead of the trial-level one, inflating n by ~39×; ~96 data files and ~51 figures carry wrong p-values (bars and CIs are fine). *(S · ready · high impact)*

* **T3.2** — Write one Methods sentence saying the ten features were hand-picked on domain grounds, not searched. *(S · low impact — the earlier leakage claim was retracted)*

* **T3.3** — Coefficient CIs ignore the L2 penalty, the class weights **and** clustering by participant; the clustered bootstrap that fixes it already exists and is never called. *(M · ⚠️ · the least-verified high-impact item)*

* **T3.4** — Umbrella for the smaller movers below (T3.5, T3.7–T3.12). *(⚠️)*

  * **↪️ ABSORBED — T3.5** — A missing confirmed selection is scored as a wrong answer; that case can't occur, so it should assert. *(into T3.17)*

  * **T3.7** — Run-based RT silently produces all-zero rows on a join-key mismatch, indistinguishable from real zeros. *(needs a coverage assertion ·* ***checked 2026-09-21: it has not bitten*** *— 0 of 13* *`RT_pure_*`* *columns are uniformly zero, so this is a guard to add, not damage to repair)*

  * **↪️ ABSORBED — T3.8** — *(into T3.20)*

  * **T3.9** — Keep the `val_*` regimes and report them; stop averaging all six regimes into one balanced-accuracy number. *(decided)*

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

* **T3.21** — Give every participant-level aggregate an explicit `scope` flag; default `within_group` for models, `global` for descriptive figures. *(M ·* ***decided*** *· blocks T1.6 and T6.1)*

## T4 · Outputs and artifacts

* **T4.0** — Standing rule: a result isn't saved until its numbers are on disk. \*(**mechanism done** 2026-09-20 with T1.3 — `tables=` is a required argument and the test is now structural: `figures/` and `tables/` are siblings in one analysis folder. **What remains is regenerating** *`findings.md`* *from the saved CSVs, which now exist.)*

* **✅ DONE — T4.1** — `RT_correlations` used to save nothing, so a live Results subsection existed only as cell output in a 1.2 MB notebook. *(**done*** *2026-09-20 with T1.3 —* *`plot_corr_map_pair`* *now routes through* *`save_output`* *and carries r / BH-adjusted p / n with every map)*

* **T4.2** — **329** zero-byte PNGs from **two** failed syncs (21 of them sit beside non-empty siblings, in live `correctness_measures` / `matching_correctness` folders). *(S · the 108* *`area_significance_heatmaps`* *are paper code, so* ***not*** *low priority ·* ***unblocked 2026-09-21***\*: the `statistics.ipynb` `AttributeError` is gone, so it is a pure rerun again — though the rerun has not been attempted, so whether it completes is untested)\*

* **T4.3** — `reports/` is tracked in git, including a 63 MB pickle. *(**deferred*** *— a release-time question, and a git operation, so yours)*

* **T4.4** — Old JSONs, an orphan zip, 2.5 GB of archive CSVs, indistinguishable paper figure names. *(**deferred*** *until the restructure says what's still meaningful)*

## T5 · Must run from scratch — required by the public release

* **T5.1** — Add `__init__.py` throughout; there are none anywhere in `src/`. *(S)*

* **T5.2** — Settle on one import convention; two coexist and only work because notebooks push two paths onto `sys.path`. *(M)*

* **T5.3** — Migrate ~40 hardcoded `"../reports/..."` literals to `PROJECT_ROOT`. *(M)*

* **T5.4** — Write `environment.yml`, pinned to Python 3.11. *(S)*

* **T5.5** — Write the README: what the project is, how to run it, `L1` = native speaker, and the build order. *(M)*

* **T5.6** — Document and enforce the build order between `answer_RTs.features` and `answer_correctness.model_data`. *(S)*

* **T5.7** — Ship instructions for placing your own OneStop download, plus one clear failure message when it isn't there. *(S ·* ***decided**)*

* **T5.8** — Keep `all_participants_with_practice.csv`; keep both pilots runnable; `paragraph_RT_run_based.csv` waits for the restructure. *(S ·* ***decided**)*

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
