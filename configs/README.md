# `configs/` — inputs you edit, not outputs the code produced

Hand-maintained configuration. **Nothing here is a result.** If a file in this folder changed,
it is because someone decided something — not because an analysis ran.

That is the whole distinction this folder exists to make. These files used to live under
`reports/report_data/`, which framed them as output of the feature-search machinery when the
paper actually consumes them as *input*. When the old report trees were deleted in `496f8d0`
they went with them, and the model-comparison figure could not be rebuilt until they came back.

| | |
|---|---|
| `configs/` *(here)* | **data you edit.** JSON and similar. Read by the code, never written by it |
| `src/config/` | **Python that holds the project's vocabulary** — column names, dataset locations, output roots |
| `reports/` | **everything the code produced.** Regenerable by re-running an analysis |

> ⚠️ `configs/` and `src/config/` are one letter apart. They are different things and neither
> is importable as the other — but do not create `src/configs/`: the vendored EyeBench loader
> registers that exact name in `sys.modules` (`src/external/EyeBench/paragraph_trial_features.py`,
> `_alias_config_package`), so a real package there would collide with it. See `docs/pitfalls.md` §6.

---

## `feature_sets/`

One JSON per named set of model features: `{"identifier": ..., "columns": [...]}`.

**These five are the paper's model comparison** — one row each in the comparison figure:

| file | features | what it is |
|---|---|---|
| `select_1_plus_last_confirm.json` | 12 | attention + last-fixation — the best performer |
| `last_confirm_compact.json` | 2 | last fixation only |
| `select_1.json` | 10 | the hand-picked attention set (`feature_groups.SELECT_1_COLS`) |
| `correct_mean_wrong_RT.json` | 2 | reading-time baseline |
| `total_answering_RT.json` | 1 | single-RT baseline. Note the **identifier inside says `total_answering_RT_normalized`** — the identifier is authoritative, not the filename |

They are **hand-specified**, not the output of a search. `select_1` is the same ten columns as
`SELECT_1_COLS` in code; the JSON exists so the cross-validation driver can load a set by name.
Restored from git (`496f8d0^`) on 2026-10-06 and re-checked: all 25 column names across the five
files still exist in `L1_model_ready_all_features.csv`.

`generate_column_options.py` can also *write* here — it produces machine-searched sets
(`pruned`, `aic`, …) for the parked feature-search strand. Those are exploration output and do
not belong in the paper's comparison.

> **One thing to know before regenerating the comparison figure.**
> `collect_and_plot_correctness_runs` does not take a list of runs — it **scans a directory**
> and plots whatever it finds. So a machine-searched set saved into the same results folder
> would join the figure silently. Check what is there before producing the final version.
> (`todo.md` T3.2.)
