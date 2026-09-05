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
   The automatic figure mirroring done by `viz/plot_output.py::save_plot(paper_dirs=[...])`
   is an established mechanism and stays.
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
| `docs/todo.md`             | the working list, tiered small → large, plus a "verify before trusting" section                                |
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

## Current layout

⚠️ **This is about to change** — an aggressive-but-staged restructure is planned
(`docs/todo.md` T6). Treat this as a description of today, not a target.

```
src/
  constants.py        column-name constants; the vocabulary
  data_paths.py       the single registry of dataset locations
  data_prep/          raw reports → IA-level all_participants.csv
  derived/            feature definitions (RT, pupil, sequences, strategies)
  predictive_modeling/
    answer_correctness/   the paper's core model
    answer_RTs/           answer reading-time regression
    answer_loc/           [future directions — currently does not import]
    common/               splits, coefficient CIs, feature selection
  statistics/
    RT_correlations/    text ↔ QA relationship — CURRENT
    mixed_*.py          older strands [future directions]
  viz/                plotting; plot_output.py is the single write path
  external/EyeBench/  vendored lab code — replace wholesale, don't patch
notebooks/            analysis drivers
experiment_builder/   Study 2 design generation and data prep
data_raw/ data/       inputs and processed outputs
reports/              plots/ and report_data/ — generated outputs
papers/               Overleaf-synced; read-only for Claude
archive/              superseded work, kept deliberately
```
