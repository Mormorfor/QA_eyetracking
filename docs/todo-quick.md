# TODO — quick reference

One line per item in `docs/todo.md`. **Everything above the last section is index only** — any
detail, reasoning, blast radius or measured number lives in `todo.md` under the same id. The
one exception is *Your own reminders* at the end, which has no counterpart there.

Status key: **ready** = nothing blocks it · **decided** = the ruling is made, implementation
pending · **blocked** = waits on another item · **deferred** = deliberately not now ·
**done** = resolved, no action · ⚠️ = read from code, never executed (see §V of `todo.md`).

*(There is no T1.2 and no T3.16 — T1.2 never existed, T3.16 was considered and declined. Don't
go looking.)*

---

## Ready to start, nothing blocking

| Item | Why it's first |
|---|---|
| **T3.6** | decided and unblocked; the change is two lines |
| **T3.1** | the fix is one word in three places, and it's high impact |
| **T1.1** | fully measured, self-contained |
| **T3.17** | cheap, and changes no number if the invariants hold |
| **T4.2** | not a fix at all — just re-run the plots |

---

## T1 · Small consistency fixes — safe, no reported number moves

- **T1.1** — Three functions apply the dominant-strategy threshold with different operators; make them all `≥`. *(S · ready)*
- **T1.3** — Route every analysis's figure **and** its numbers through `plot_output`, so keeping a result stops depending on which module produced it. *(M · ⚠️)*
- **T1.4** — Fix comments naming constants that no longer exist — but check `statistics.ipynb` first, which may actually *call* one. *(S · ⚠️)*
- **T1.5** — A table of ~10 tiny cleanups: dead code, duplicate definitions, unused imports, stale docstrings, orphan `.pyc`, two `paper_dirs` conventions. *(S · ⚠️)*
- **T1.6** — Merge the two "starting strategy" implementations into one function in `derived/`, with completion as a parameter. *(M · blocked on T3.21)*
- **T1.7** — Merge the QA and paragraph implementations of the same eight per-area metrics into one set parameterized by grouping column. *(M · do with or after T3.6; pairs with T6.1)*
- **T1.8** — L1's `all_participants.csv` carries a stray `index` column KnowQA's doesn't, so "same pipeline, same columns" isn't quite true. *(S · ready)*

## T2 · Broken today — all low priority, none on the paper's path

- **T2.1** — `generate_column_options.py` raises `AttributeError` unconditionally; it references three constants `feature_groups.py` no longer has. *(⚠️)*
- **T2.2** — `answer_loc/` cannot import at all — two functions imported from the wrong module, one called with a dead signature. *(⚠️)*
- **T2.3** — `fit_model_on_prepared_full_data` calls random-effects methods the live logreg doesn't implement; works only with the Julia backend. *(⚠️)*
- **T2.4** — `get_last_visited_feature_cols` always returns `[]` because of a prefix mismatch, so the **default** feature set has silently contained no last-label features. *(⚠️ · the one with quiet consequences)*
- **T2.5** — `notebooks/statistics.ipynb` may call the renamed `Con.AREA_METRIC_COLUMNS`; if so T1.4 is a breakage, not a comment fix. *(⚠️)*

## T3 · Fixes that move reported numbers

> The tier is mostly one principle violated repeatedly: a silent alteration where a loud error
> belonged. Read the intro table in `todo.md` before picking items off individually.

- **T3.1** — Fisher tests are handed the IA-level frame instead of the trial-level one, inflating n by ~39×; ~96 data files and ~51 figures carry wrong p-values (bars and CIs are fine). *(S · ready · high impact)*
- **T3.2** — Write one Methods sentence saying the ten features were hand-picked on domain grounds, not searched. *(S · low impact — the earlier leakage claim was retracted)*
- **T3.3** — Coefficient CIs ignore the L2 penalty, the class weights **and** clustering by participant; the clustered bootstrap that fixes it already exists and is never called. *(M · ⚠️ · the least-verified high-impact item)*
- **T3.4** — Umbrella for the smaller movers below (T3.5, T3.7–T3.12). *(⚠️)*
  - **T3.5** — A missing confirmed selection is scored as a wrong answer; that case can't occur, so it should assert. *(folded into T3.17)*
  - **T3.7** — Run-based RT silently produces all-zero rows on a join-key mismatch, indistinguishable from real zeros. *(needs a coverage assertion)*
  - **T3.8** — *(folded into T3.20)*
  - **T3.9** — Keep the `val_*` regimes and report them; stop averaging all six regimes into one balanced-accuracy number. *(decided)*
  - **T3.10** — Fold-level CIs treat overlapping folds as independent, average folds unweighted, and fall back silently to z=1.96 — these are the CIs on the comparison figure.
  - **T3.11** — Five group-feature functions mutate the caller's frame, creating an undocumented ordering dependency and leaking coercions into the saved output.
  - **T3.12** — The blanket `fill_value = 0.0` is wrong for the exclude-convention columns. *(fix tracked as T3.14)*
- **T3.6** — Stop zero-filling `"."` in first fixation duration; coerce to NaN instead. *(S · **decided, unblocked, ready to run** — expect the per-area differences to largely vanish, which is a conclusion change to log)*
- **T3.13** — Dwell time and fixation count stay coverage-inclusive; the asymmetry inside the metric family is deliberate. *(**done** — no action, and do not "fix" it for tidiness)*
- **T3.14** — Keep the `0` fill for pupil and first-fixation features at model-prep time, but make it named, commented, counted and reported in Methods. *(**decided** — the obligation is now the documenting)*
- **T3.15** — Two feature-provenance paths in `cross_validation.py` mean the same nominal model run from two notebooks needn't agree. *(**deferred** into T3.21, which settles the direction)*
- **T3.17** — Turn two confirmed facts into assertions: every trial appears in every feature block, and every trial has a confirmed selection. *(S–M · ready)*
- **T3.18** — Text misalignment: real, in three distinct modes, including 20 L1 trials nobody had looked at; area labels now come from screen geometry instead of the stored text. *(**done** 2026-09-07 — two residuals noted, neither is this item)*
- **T3.19** — Investigate whether `last_answer_area_visited_lbl` is buggy; cross-checking it against its two click-based siblings is the starting test. *(M · ⚠️ · needs running, not a ruling)*
- **T3.20** — Every dataset writes its own pupil baseline and every consumer reads the right one — today the paragraph path silently baselines against L1's answer screen. *(S–M · requirement set)*
- **T3.21** — Give every participant-level aggregate an explicit `scope` flag; default `within_group` for models, `global` for descriptive figures. *(M · **decided** · blocks T1.6 and T6.1)*

## T4 · Outputs and artifacts

- **T4.0** — Standing rule: a result isn't saved until its numbers are on disk; the test is a `report_data/` folder matching every `plots/` folder (9 of 15 topics have none today). *(agenda item — `findings.md` gets regenerated from those CSVs afterwards)*
- **T4.1** — `RT_correlations` saves nothing at all, so a live Results subsection exists only as cell output in a 1.2 MB notebook. *(M · ⚠️)*
- **T4.2** — 308 zero-byte PNGs from one failed sync; no fix to design, just re-run. *(S · the 108 `area_significance_heatmaps` are paper code, so **not** low priority)*
- **T4.3** — `reports/` is tracked in git, including a 63 MB pickle. *(**deferred** — a release-time question, and a git operation, so yours)*
- **T4.4** — Old JSONs, an orphan zip, 2.5 GB of archive CSVs, indistinguishable paper figure names. *(**deferred** until the restructure says what's still meaningful)*

## T5 · Must run from scratch — required by the public release

- **T5.1** — Add `__init__.py` throughout; there are none anywhere in `src/`. *(S)*
- **T5.2** — Settle on one import convention; two coexist and only work because notebooks push two paths onto `sys.path`. *(M)*
- **T5.3** — Migrate ~40 hardcoded `"../reports/..."` literals to `PROJECT_ROOT`. *(M)*
- **T5.4** — Write `environment.yml`, pinned to Python 3.11. *(S)*
- **T5.5** — Write the README: what the project is, how to run it, `L1` = native speaker, and the build order. *(M)*
- **T5.6** — Document and enforce the build order between `answer_RTs.features` and `answer_correctness.model_data`. *(S)*
- **T5.7** — Ship instructions for placing your own OneStop download, plus one clear failure message when it isn't there. *(S · **decided**)*
- **T5.8** — Keep `all_participants_with_practice.csv`; keep both pilots runnable; `paragraph_RT_run_based.csv` waits for the restructure. *(S · **decided**)*
- **T5.9** — Two EyeBench entry points write the same cache with different feature definitions, and the cache can't say which one made it. *(M)*
- **T5.10** — `data_prep_new_exp.ipynb` overwrites KnowQA's output with the older identity scheme; mark superseded or remove the collision. *(S)*
- **T5.11** — `evaluate_one_fold_on_regimes` refits the same model 6× per fold; hoisting it removes the speed argument for the leaking CV path. *(S · efficiency only)*

## T6 · The restructure

- **T6** — The staged reorganization; the plan of record is `docs/restructure-map.md` (Stages 0 → G). **Agreed ≠ started — no stage runs without being proposed first.**
- **T6.1** — Split paragraph preprocessing out of `answer_RTs/` and out of the QA prep, so there's one paragraph table per dataset joined in explicitly. *(M–L · = Stage C · starts after T3.21)*

---

## Not items, but don't forget them

- **Nothing is currently waiting on a decision from you.** The answered-questions table at the end of `todo.md` is the record of what was asked and what was ruled.
- **§V "Verify before trusting"** — six one-command checks for the ⚠️ items. Number 6 (T3.3's CIs) is the one most likely to affect the paper and the least verified.
- **Doc-consistency rules** — every `pitfalls.md` / `data-pipeline.md` gotcha cites a `T*` id or says no action is needed; when retracting a claim, grep the *words*, not just the id.

---

## Your own reminders

Yours, not tracked in `todo.md` — no `T*` id, nothing checked or verified by Claude.

- [ ] Check final fixation on incoming feedback ratio. *(added 2026-09-16)*
