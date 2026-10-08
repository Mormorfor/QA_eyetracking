# What is deliberately left until after the restructure

Everything still open as of **2026-10-06**, and why each thing is waiting rather than done.
The companion file is `done_before_restructure.md`.

**The short version:** nothing here is an oversight. Every item was looked at, and either
assigned to a restructure stage, kept back for your manual check, or is yours to do rather
than Claude's. Each was re-verified as genuinely still open on 2026-10-06 — the ones claiming
to be broken really are, and the files claiming not to exist really don't.

---

## ✅ Nothing is waiting on a decision

The last open question closed on **2026-10-06**.

> **`collect_triples` — no action** (`presentation_prep.ipynb` cell 9). Its comment says it
> collects trials with `is_correct == 1` while the code filters `== 0`, so the printed
> "469 texts / 1,407 rows" describes **incorrect** trials. Diana's ruling: *the presentation
> has already been given, and the notebook is kept only in case something from it is reused
> later.* So the contradiction is harmless — nothing downstream reads it and no live claim
> rests on it.
>
> **The one thing to carry forward:** if anything is ever lifted out of that notebook, read the
> filter rather than the comment. Noted here rather than fixed, because editing a parked
> notebook to correct a comment is not worth a rerun of anything. (`todo.md` T1.5, last row.)

---

## 1. Work that lands inside the restructure stages

These are **not** separate jobs — each is cheaper to do as part of the stage that moves the
relevant files, and in several cases doing it earlier would mean doing it twice.

| Stage | Item | What it is |
|---|---|---|
| **A** — make it a package | ~~T5.1~~ ~~T5.2~~ ~~T5.4~~ | ✅ **Done 2026-10-06**, minus packaging — 22 `__init__.py`, one import convention (21 lines converted), `environment.yml` pinned. Verified by re-running an analysis: **45/45 tables byte-identical**. See `done_before_restructure.md` |
| **A** | **`pyproject.toml`** | ⏸ **Deferred by decision** (Diana, 2026-10-06 — *"don't want to make it pip installable quite yet"*). Until it lands, the repo reaches its code through `PYTHONPATH` rather than an install, so the `sys.path` lines in `data_paths.py` and five notebooks stay. **The rule that replaces them for now:** the repo root is the only thing on the import path, never `src/` — see `pitfalls.md` §6 |
| **B** — `config/` + `lib/` | **stale paths** | **Nine** constants in `data_paths.py` point at files or folders that aren't there. Six are old drift; three (`COL_SAVE_PATH`, `CROSS_VALIDATION_RUNS_DIR`, `PER_PERSON_LOO_RESULTS_DIR`) are live destinations orphaned when outputs moved. **`COL_SAVE_PATH` is the one that blocks things** — see §4 |
| **C** — `ingest/` + `features/` | **T3.24** | The data-prep functions depend on running in a particular order, and that order is held by convention rather than declared. Better fixed when those functions are being moved anyway |
| **D** — `modeling/` | **T5.11** | The cross-validation refits the same model six times per fold — 60 wasted fits per run. Pure speed, no correctness effect |
| **D** | **T3.22** | Trials are treated as independent when they are nested in 360 participants and 972 items, so count-based p-values are too small. A real statistical point, but it belongs with the inference code once that code is in one place |
| **E** — analyses + explorations | **T2.2**, **T2.6** | Two modules that don't import at all (verified again today). Both are parked strands. The restructure's job is to file them under `explorations/` as clearly-not-paper-code; repairing them is work for whenever those strands are picked up |
| **E** | **T3.19** | Check whether `last_answer_area_visited_lbl` is buggy. It has two siblings computed a different way, so the natural test is to cross-check the three — easiest once they sit together |
| **F** — entry points | **T5.6**, **T5.10** | Document and enforce the build order; remove a notebook that overwrites a newer pipeline's output with an older scheme |
| **G** — data + release | **T5.5**, **T5.7**, **T5.8**, **T5.9** | README (verified absent), instructions for placing your own OneStop download, pilot naming, and the two EyeBench entry points that write the same cache with different definitions |

> **Why the stages are now lower-risk than the plan says.** `restructure-map.md` calls stages C
> and D the dangerous ones *because numbers move there*. They no longer do — all of that work
> landed early (see the done-doc). What remains in C, D and E is **file movement**, which means
> "every number identical" works as the check for every remaining stage.

---

## 2. Waiting for your manual check

Two items that are explicitly **yours to drive**, by your own ruling on 2026-09-27:
*"everything will be re-run as I check."*

- **T4.0 — rebuild `findings.md` from the saved numbers.** The mechanism is done: every
  analysis now writes its numbers, so the ledger can be *regenerated* instead of transcribed
  out of pictures. Doing it now would mean transcribing numbers that are about to be
  regenerated anyway. When it happens, the ⚠️ NOT VERIFIED header can finally come off.
- **T4.2 — re-run the figures.** Note one correction: there are **no empty files left to
  refill.** The 329 zero-byte PNGs went with the old output trees. The figures that matter are
  simply *absent* and need generating, not overwriting. The blocker that made this impossible
  (**T2.7**, the modelling library failure) was found and fixed, so it will now actually run.

---

## 3. Yours rather than Claude's

- **T3.23** — read up on clustered confidence intervals, so the choice in the paper is one you
  can defend rather than one you inherited.
- **T4.3** — `reports/` is tracked in git. The 63 MB pickle is gone from the working tree but
  still in history, `.git` is about 1.6 GB, and the biggest offenders now are old notebook
  versions, not the pickle. Only matters at release, and the remedy is a git operation.
- **T4.4** — accumulated clutter: old JSONs, an orphan zip, 2.5 GB of superseded CSVs in
  `archive/`. Deferred on purpose: the restructure is what decides which outputs still mean
  anything.

---

## 4. The one thing that blocks re-running the headline model

Worth stating on its own, because it is small and easy to miss.

**~~`COL_SAVE_PATH` has no home.~~ ✅ Resolved.** It is `configs/feature_sets/`, reached as
`config/outputs.py::FEATURE_SETS_DIR`; the old `COL_SAVE_PATH` name was deleted on 2026-10-08
once the last call site moved to it (28 of them, in `column_options.py` and four notebooks —
the alias had been marked *"kept while call sites migrate"* and the migration had never
happened). The folder holds the five small files that define the paper's **model-comparison
figure** — one per model variant, each just a name and a list of feature columns:

| file | what it is |
|---|---|
| `select_1_plus_last_confirm` | 12 features — the winner, 0.8293 |
| `last_confirm_compact` | 2 features — 0.8258 |
| `select_1` | the 10 hand-picked features — 0.8150 |
| `correct_mean_wrong_RT` | 2-feature baseline — 0.7094 |
| `total_answering_RT` | 1-feature baseline — 0.6348 |

All five are recoverable from git, but they are **pre-rebuild**, so they want regenerating
rather than restoring. The open question is what they *are*: configuration you chose by hand,
or output of the feature-search machinery. They sat under `reports/` as if they were results,
but the paper depends on them as inputs — and the same notebook cell already defines three
more feature sets inline, so the code is already of two minds.

Until that's settled the cross-validation can't be re-run, which means the model-comparison
figure can't be rebuilt. **Stage B is the natural place to decide it.**

One related repair is already in place: the code that *reads* those run summaries had been
left behind when the output layout changed, so the comparison figure couldn't be built even
with the files present. Fixed, but **untested end to end** — it can't be exercised until a
cross-validation run exists.

---

## 5. Things recorded but not turned into work

Noticed while working, measured, and left for you to decide — **not** added as items, because
an unfiltered list buries the things that matter.

- **Three alternative fixes for path length**, kept as fallbacks if the current 108-character
  headroom ever stops being enough: give the output subfolder one consistent meaning (+39
  chars), hash long names, or require long-path support. Details in `restructure-map.md` §9.
- **The three-panel proportion figure** names its panels `all_participants+hunters+gatherers`
  — which, as you spotted, is redundant: all participants *is* hunters plus gatherers. Checked
  numerically: the pooled panel's correlations are the average of the other two to within
  0.00003, because the groups are equal-sized. Its confidence interval is genuinely different
  (n = 360 vs 180), so the panel isn't useless — but the *name* spells out a tautology, and it
  is the only figure in the project using that naming style. A one-line change if you want it.
- **The upstream text defect** behind T3.18 is still unfixed, because it lives in the
  presentation software, not this repo. Fully predictable — 12 of 978 stimulus rows — so
  expect roughly 1.2% of Study 2 trials to carry it as collection grows.

---

## 6. What "ready to start" means

The pre-restructure list is finished. Specifically:

- every item is **done**, **assigned to a stage**, **assigned to your check**, or **yours**, and
  since 2026-10-06 **nothing at all is waiting on a decision**;
- the code **runs**: 87 of 91 modules import, and the 4 that don't are the two parked strands
  that Stage E is meant to file away;
- the numbers are **current**: both studies rebuilt, every output regenerated, every change
  that moved a result recorded in `findings.md`'s change log;
- the integrity work that `restructure-map.md` scheduled for stages C and D has **already
  landed**, so those stages no longer move numbers and can be checked by "nothing changed".

The next step is **Stage 0 → A**, proposed before it starts, per the standing rule.
