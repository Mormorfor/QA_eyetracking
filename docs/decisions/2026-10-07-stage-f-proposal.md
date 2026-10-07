# Stage F — entry points

**Status: proposed 2026-10-07. Not started.** Stage E closed the same day. This is the
second-to-last stage; only G (data moves, README, run-from-scratch) follows.

---

## What this stage is for

**Today a notebook is the only way to run anything.** There is no `scripts/` directory, and of
the nine `__main__` blocks in `src/`, none belongs to a live analysis — they are in `features/`,
`ingest/` and two parked modules. So "regenerate the paper's figures" means opening Jupyter and
running cells in the right order, and the right order is not written down anywhere.

That is the gap stage F closes, and it is the one the release standard actually depends on:
`conventions.md` requires the repo to **run from scratch on a clean machine, with instructions**.
A reader who clones this cannot currently do that even if the data were present.

---

## The diagnosis, measured

### 1. 28 notebooks, and nothing distinguishes them

| kind | count | examples |
|---|---|---|
| drives a **live analysis** | 8 | `answer_corr_prediction`, `text_associations`, `statistics`, `visualisations` |
| **builds data**, run once per refresh | 5 | everything in `tests and data builders/` |
| **Study 2 material generation** | 5 | all of `experiment_builder/` |
| **parked / free-form** | 10 | `RT_pred`, `feature_selection`, `old_clustering_attempt`, `presentation_prep` |

Three of them have **zero executed cells** (`feature_selection`, `old_clustering_attempt`,
`unlikely_analysis`) — they are not "run and cleared", they have never run in their current state.

### 2. One analysis still exists only in a notebook

`plot_correctness_by_answer_a_vs_bcd_mean` — the **correct-vs-distractor reading-time
asymmetry** — is 177 lines in `presentation_prep.ipynb` cell 22. It is the result
`findings.md` §9.1 calls *"one of the most interpretable results in the project"*: time on the
correct answer is flat against accuracy (.81–.85 across the whole range) while time on the
distractors collapses it from .87 to ~.36.

`lib/plotting/output.py::ANALYSES` **already has a key for it** — `answer_rt_comparison` — and
there is no folder. It is one of only two keys without one. (The other,
`explorations/text_alignment`, backs `text_alignment_investigation.ipynb`, which is an
investigation rather than a recurring analysis; it can stay a notebook.)

> `longest_alternating_run`, the other notebook-only item the map names (§10), **is already
> lifted** — `features/build.py:371` calls `longest_alternating_answer_run` from
> `features/sequences.py`. Nothing to do.

### 3. The build order is enforced by one error message

`data-pipeline.md` §6 documents it; the only thing that *enforces* it is a `FileNotFoundError`
telling you to build the paragraph cache first. **T5.6.**

### 4. T5.10 is still live

`experiment_builder/data_prep_new_exp.ipynb` still carries `RUN_NAME` and its own warning that
running it with `RUN_NAME = "KnowQA"` overwrites `ingest/knowqa.py`'s output with the older
identity scheme. The notebook says so itself; nothing stops it.

---

## The proposal

### `scripts/` — three entry points

```
scripts/
  build_dataset.py       --dataset l1 | knowqa | testrun_qa | second_test
  run_analysis.py        --name scan_strategies [--group hunters] [--no-save]
  make_paper_figures.py  every figure in the paper, in order
```

Each is a thin `argparse` wrapper over functions that already exist — **no new logic**.
`build_dataset.py` is where the build order stops being documentation and becomes code (T5.6):
it runs ingest → paragraph features → model-ready in the one correct order, per dataset, and
refuses rather than guesses when an input is missing.

`run_analysis.py --name X` validates against `ANALYSES`, which is the same registry
`save_output` validates against — so a typo fails the same way in both places.

### `notebooks/` — three folders, by what a notebook *is*

```
notebooks/
  drivers/       thin: load, call, display. One per live analysis.
  builders/      run once per data refresh (was "tests and data builders")
  exploration/   free-form. Never the source of a reported number.
```

**Nothing is deleted and nothing is rewritten in this stage** beyond import lines — notebooks
move into the folder that describes them. Turning the drivers genuinely *thin* (lifting cell
logic into the package) is a separate pass; what this stage buys is that the distinction is
visible, and that `exploration/` carries the standing caveat.

### `analyses/answer_rt_comparison/` — lifting the one real orphan

`plot_correctness_by_answer_a_vs_bcd_mean` moves to `analyses/answer_rt_comparison/plots.py`,
with its binning/summary half in `compute.py` if they separate cleanly. This fills the
`ANALYSES` key that has no folder, and it is the difference between a headline result living in
a presentation notebook and living in the codebase.

> **It is `answer_rt_comparison`, not `correctness_associations`.** Map §5.3 draws the line:
> *a per-area quantity is a feature; a contrast between per-area quantities is an analysis* —
> and §9 distinguishes the two by what they are built from. `correctness_associations` works on
> the IA metric families; this is built from the **RT family**, which is run-based and derived
> from click timestamps. Different input, different folder, as agreed 2026-09-05.

### T5.10 — the collision

`data_prep_new_exp.ipynb` is the only way to build the two pilots, so it cannot simply go. The
fix is to remove `"KnowQA"` from its `RAW_RUNS` dict, so it can only build the pilots and the
collision is structurally impossible rather than warned about.

---

## Decided 2026-10-07

### 1. `experiment_builder/` is three things, and the generation half gets a real home

Diana: *"there are also some analysis and explorations there that would be good to organize, and
also — i want to put it somewhere where it is clear that this is the experiment source, in case i
ever design another one... True, it probably will never run again but lets put it where it would
have to go to begin with in a good repo."*

Reading the five notebooks, they are not one kind of thing:

| notebook | what it actually is | → |
|---|---|---|
| `lists_builder.ipynb` (10 defs, 40 cells) | the Latin-square design: 27 lists × 6 regime orders = 54 | **`src/experiment/lists.py`** + driver |
| `text_adjustments.ipynb` (3 defs) | stimulus rephrasing and spelling corrections | **`src/experiment/texts.py`** + driver |
| `extract_text.ipynb` (0 defs) | stimulus extraction, pure inline | driver only — nothing to lift |
| `data_prep_new_exp.ipynb` (6 defs) | **pilot ingest**, not generation. Its own header says prefer `ingest/knowqa.py` | `notebooks/builders/` + the T5.10 fix |
| `preliminary_analysis.ipynb` (3 defs) | **already a driver** — it imports `comparison_runs` and `confidence_correlation` from the live package | `notebooks/drivers/` |

**`src/experiment/` is the answer to "where is the experiment source?"** — it is the only folder
in the tree that generates *stimuli and designs* rather than consuming recordings. Its sibling
`ingest/` reads what came back; this is what went out.

**The lift is the function definitions, not the cells.** The notebooks have already run and their
outputs are committed; rewriting 40 cells of inline orchestration would be risk with no return.
The 13 `def`s move to the package where another experiment could reuse them; the orchestration
stays in the driver that ran it.

### 2. `make_paper_figures.py` is a skeleton

Diana: *"You can make it a skeleton for now. We will work on paper figures, but this comes when
the codebase is already nice and organized."* So: the structure and the registry of what belongs
in it, with the analyses that already route through `save_output` wired up and the rest listed as
explicit `TODO` entries. Not a dishonest "it works" script.

### 3. The never-run notebooks are each their own exploration — except one

Diana: *"each is its own exploration. Maybe excluding feature selection, i don't currently use it
but its purpose is to generate feature sets for my model."*

So `old_clustering_attempt` and `unlikely_analysis` → `exploration/`. **`feature_selection` →
`builders/`**: it produces an artifact other code consumes (the feature-set JSONs under
`reports/correctness_prediction/`), which is what that folder is for, and that is true whether or
not it has been run lately.

## Order of work

| # | step | note |
|---|---|---|
| 1 | `analyses/answer_rt_comparison/` — lift the RT asymmetry out of `presentation_prep` | the only code move; do it first while the stage is still small |
| 2 | `scripts/build_dataset.py` + T5.6 | the build order becomes executable |
| 3 | `scripts/run_analysis.py` | validates against `ANALYSES` |
| 4 | `notebooks/{drivers,builders,exploration}/` + the import pass | mechanical; one notebook pass, as in stage E |
| 5 | T5.10 — drop `"KnowQA"` from the pilot notebook's runs | one dict entry |
| 6 | `scripts/make_paper_figures.py` | last, because it depends on 2–4 |

---

## Acceptance test

1. **Imports:** 132/133, same single parked `pymer4` failure.
2. **`reports/` byte-identical** after `run_analysis.py --name text_qa_relationship` — the same
   check stage E passed, now through the script rather than a notebook. `manifest.json`'s
   `produced_by` will change again, to the script's module path, and that is the proof.
3. **`build_dataset.py --dataset knowqa` reproduces the model-ready table bit-identically**
   (870 × 200) — it is the cheap dataset and the one already verified post-step-6.
4. **Every `ANALYSES` key has a folder**, which is 17/17 once `answer_rt_comparison` exists
   minus the one investigation key. Today it is 15/17.
5. **No notebook left at the top of `notebooks/`** — each is in exactly one of the three folders.

---

## Explicitly not in this stage

- **Not making the drivers actually thin.** Moving a notebook into `drivers/` does not by itself
  lift its cell logic. That is a per-notebook job and several of them are stale.
- **Not fixing what the notebooks compute.** T3.19, T4.0 and T4.2 stay where they are.
- **No `pyproject.toml`** — deferred by decision; stage G if ever.
