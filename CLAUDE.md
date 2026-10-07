# QA\_eyetracking\_workspace

Research code for two eye-tracking studies on multiple-choice reading comprehension,
being prepared for publication and for public release alongside the paper.

**One-line summary:** predict whether a participant will select the *correct* answer from
their eye movements over the answer options, and interpret the model's probability as a
proxy for their latent confidence.

The paper is Diana's to write. Claude's job is the code.
**The live paper outline is the latest draft in
`papers/correctness_prediction/Nature Comunications/`.** Older drafts in that folder and in
its parent (`abstract.tex`, `paper.tex`) are retired ACM-era material.

***

## This is scientific code, not production code

**Scientific standards take priority over production-engineering standards.** Clarity,
traceability, and fidelity to the data and the methodology matter more than robustness,
convenience, or graceful failure.

**The code is allowed to fail loudly** when assumptions are violated or unexpected cases
occur. It is **not** acceptable for it to silently alter the analysis in order to keep
running.

A crash with a useful explanation is a good outcome. A run that completes while quietly
producing scientifically different results is not. The concrete rules that follow from this
are in `docs/conventions.md` under **Scientific integrity of the code** — read them before
writing or changing any analysis code.

***

## Hard rules

1. **No git operations.** No commits, no branches, no staging. Describe changes; Diana commits.
2. **No direct edits under** **`papers/`.** It is an Overleaf-synced repo with its own `.git`.
   The automatic figure mirroring done by `lib/plotting/output.py::save_output(to_paper=True)`
   is an established mechanism and stays. It is currently **switched off**
   (`PAPER_MIRROR_ENABLED = False`) -- see `pitfalls.md` section 8.
3. **Never hand-edit data files** in `data/` or `data_raw/`. Writing outputs through code
   we've agreed on is fine; opening a CSV and changing values is not.
4. **Discuss before moving any data file.** Moving them is allowed; doing it without
   agreeing the how and why first is not.

## Environment

* Conda env `QA_eyetracking_env`, **Python 3.11**
  (`C:/Users/deeth/miniconda3/envs/QA_eyetracking_env`)

* **Run from the repo root.** All paths resolve through `data_paths.PROJECT_ROOT`.
  Some `"../reports/..."` literals still survive in `src/viz/` and
  `generate_column_options.py`; they predate this convention and are being migrated out
  (`docs/todo.md` T5.3).

## Read these

@docs/conventions.md
@docs/pitfalls.md

`conventions.md` is how Diana wants Claude to work. `pitfalls.md` is what will bite you in
this specific codebase.

## The rest of the docs, on demand

| File                       | Contents                                                                                                       |
| -------------------------- | -------------------------------------------------------------------------------------------------------------- |
| `docs/todo.md`             | the working list, tiered small → large, plus a "verify before trusting" section. `docs/todo-quick.md` is its one-line-per-item index |
| `docs/done_before_restructure.md` | ✅ **start here for history.** Plain-language account of everything fixed 2026-09-20 → 10-06, grouped by what kind of problem it was. Re-verified against the code 2026-10-06 |
| `docs/todo_after_restructure.md`  | everything still open and why it is waiting — by stage, by Diana's manual check, or hers rather than Claude's. **One item needs a ruling**, flagged at the top |
| `docs/restructure-map.md`  | ✅ **AGREED 2026-09-05** — the target structure top to bottom, where every current file goes, and the staged order. **Read before touching layout, and follow it.** Still revisable, but it is the plan of record, not a suggestion. |
| `docs/findings.md`         | ⚠️ **UNVERIFIED** — a map of what results exist and where they came from, assembled by Claude from saved figures and stdout. **Not checked by Diana; do not quote from it.** Useful before recomputing anything, so you know what a figure folder holds. To be regenerated properly after the restructure. |
| `docs/research-context.md` | the science: both studies, what's claimed, paper-section → code map                                            |
| `docs/data-pipeline.md`    | four datasets, four stages, build order                                                                        |
| `docs/glossary.md`         | columns, grains, areas, metrics, contrasts, CV regimes                                                         |
| `docs/decisions/`          | dated notes on choices made during the cleanup                                                                 |

Two files this index used to promise were never written, because the restructure map absorbed
them. Recorded so nobody goes looking:

* **outputs conventions** — `reports/` and `papers/` layout and the mirroring mechanism:
  `restructure-map.md` §9, plus `todo.md` T4.0 for the standing "persist the numbers, not just
  the figure" rule.
* **live / parked / future-directions** — `restructure-map.md` §3, where the split is
  structural (`analyses/` vs `explorations/`) rather than a list, plus `research-context.md` §7.

## Layout

The restructure is **done** — stages 0 through G, finished 2026-10-07. `docs/restructure-map.md`
is the record of what the shape is and why; `README.md` is the reader-facing version.

```
src/
  config/       vocabulary and locations — column names, the dataset registry, output roots
  lib/          generic: stats, plotting, the single save path. Imports nothing above itself
  ingest/       raw vendor reports -> tidy interest-area tables
  features/     interest-area tables -> trial-level features
  modeling/     cross-validation, metrics, model wrappers, coefficient inference
  analyses/     paper code. One folder per question: compute · stats · plots
  explorations/ kept and organised, explicitly NOT backing the paper
  experiment/   how the Study 2 materials and design were generated
  vendor/       third-party code, kept as received — replace wholesale, do not patch
scripts/        build_dataset.py · run_analysis.py · make_paper_figures.py
notebooks/      drivers/ · builders/ · exploration/ · experiment/ (each has a README)
data/           datasets/{l1_onestop,knowqa,pilots/*}/{interim,features}/ · cv_folds/ · stimuli/
reports/        <analysis>/{figures,tables}/ — mirrors analyses/
papers/         Overleaf-synced; read-only from here
archive/        superseded work, kept deliberately
```

**Two rules the layout exists to enforce:**

1. **Imports only point downwards** — `lib` → `config` → `ingest` → `features` → `modeling` →
   `analyses`. Nothing imports `analyses`, and `lib` imports nothing from the project. There is
   one documented exception, noted where it lives (`analyses/correctness_prediction/run.py`).
2. **`analyses/` is paper code, `explorations/` is not**, and the path says which.

**Entry points.** `python scripts/run_analysis.py --list` says what can be run without Jupyter.
Everything under `reports/` is written by one function, `lib/plotting/output.py::save_output`,
whose `tables=` argument is required — a figure cannot be saved without its numbers.
