# Pitfalls

The things that will trip you up in this codebase. Each one is a fact about how the
code works and what goes wrong if you forget it.

---

## 1. Three different "one row per" live in this project

| Grain | One row per | Files |
|---|---|---|
| **interest area** | one word, in one area, in one trial, for one participant | `interim/interest_areas.csv` |
| **area** | one of the five screen areas, in one trial | `interim/areas_answers.csv` |
| **trial** | one participant × trial | `interim/trials.csv`, `features/model_ready.csv` |

A trial has 30–60 words on screen, so an interest-area frame holds about **39 rows per
trial**. Trial-level values get copied onto every one of those rows.

This is the single easiest way to get a wrong number here. The value in the column looks
right; the count is 39× too big. A test that should see 19,436 trials sees 760,628 rows and
reports a p-value of zero.

**So: any statistical test about a trial must run on a trial-level frame.** If you only have
the interest-area frame, collapse it first with
`groupby([TRIAL_INDEX, participant_id]).first()`.

The three correctness Fisher tests take the trial frame directly and refuse to run on
anything with a duplicated `(participant_id, TRIAL_INDEX)`.

---

## 2. A per-word average mixes two different things

For a word nobody looked at, the tracker writes:

| Column | Value | Why |
|---|---|---|
| `IA_DWELL_TIME` | `0` | dwell is the sum of fixation durations, and there were none |
| `IA_FIXATION_COUNT` | `0` | same |
| `IA_FIRST_FIXATION_DURATION` | `"."` | a fixation of length zero does not exist, so there is nothing to write |
| pupil size | `"."` | only measured during a fixation |

All four say the same thing: no fixation landed here. They just spell it differently.

Now average over every word in an area. If you include the unread words as zeros, you get

```
per-word mean = (how hard the read words were read) × (what fraction got read)
```

The second half is just `skip_rate` again. So the metric carries the coverage signal twice —
once on its own and once inside every other metric.

Whether that matters depends on how much the first half actually varies. Divide each metric
by the fraction of words that were read, and you recover the per-read-word value:

| Metric | Answers A → D | Spread | What that means |
|---|---|---|---|
| first fixation duration | 192.2 · 184.8 · 181.9 · 181.2 ms | **6%** | almost pure coverage — it is `skip_rate` wearing a hat |
| dwell time | 381.2 · 323.3 · 296.5 · 284.7 ms | **34%** | real signal on top of coverage |
| fixation count | 1.911 · 1.693 · 1.581 · 1.536 | **24%** | real signal on top of coverage |

A word inside the chosen answer genuinely gets more time and more fixations *when it is
read*. First fixation duration does not — the differences between answers on that metric are
entirely about how often the words got read at all.

### Which metrics include unread words, and which don't

| | Convention | Why |
|---|---|---|
| dwell time, fixation count | **include** them as 0 | these measure *how much attention the area got*, and zero is a real amount of attention |
| first fixation duration, pupil size | **exclude** them | these are properties *of a fixation*, and there wasn't one |

**This asymmetry is on purpose. Do not "fix" it.** Four metrics sitting next to each other in
`AREA_METRIC_COLUMNS_MODELING` handle missing words two different ways, and it looks like an
oversight. It isn't.

The practical consequence: dwell time and fixation count measure attention per *available*
word, not per word read. Their coefficients mix coverage and intensity, and that is what the
paper should say they measure.

### A zero reading time is not ambiguous

All five areas exist on every trial — the question and four answers are always on screen, and
every paragraph has all three spans. So a `0` in the RT, TFD or TimeSinceOffset family means
one thing only: **nobody looked at that area.** Those families contain no NaN at all, only
zeros. That zero is real data and needs no special handling.

### Where missing values do need care

Pupil size and first fixation duration follow the exclude convention, so they are the two
families that carry genuine NaN — one cell for every area a participant never fixated.

On L1:

| | Cells per metric family |
|---|---|
| the five area columns | **8,461** — question 5,810 · A 253 · B 784 · C 829 · D 785 |
| all ten columns (areas plus the five contrasts) | **10,021** |

The contrasts inherit NaN unevenly, which matters if you quote either number. `__wrong_mean`
is NaN only when all three wrong answers went unread (158 trials). `__contrast`,
`__distance_furthest` and `__distance_closest` are NaN when *either* side is (383 trials).

**The model fills every NaN with `0`.** For a z-scored pupil that asserts "this area had
exactly this participant's average pupil size" for an area they never looked at — a specific
claim, and a false one. The fill is a deliberate choice, and `conventions.md` has the
reasoning; what matters here is that you know those cells are assumptions, not measurements.

In the headline model it is **253 cells, 1.3% of trials**, all in
`mean_max_fix_pupil_size_z__correct`. The other nine features have none.

---

## 3. Participant-level features depend on which trials you hand them

A dominance score summarises a person across their trials. Which trials? Whichever ones are
in the frame you passed.

The strategy and dominance features take two arguments that make this explicit:

| | What it decides | Default |
|---|---|---|
| `scope_df` | which trials estimate the aggregate | the frame being featurised — never wider |
| `scope_by` | how to subdivide below the participant | not at all — one value per person, pooling regimes and sessions |

The default is the narrow one, and that is deliberate: a score computed over *all* of
someone's trials includes the held-out ones, so attaching it to a training row leaks the test
set into the model. Small — one number per person — but real, and easy for a reviewer to
name.

**Read a saved dominance score by looking at its trial count first.** Every row carries
`n_strategy_trials_with_q` and `n_strategy_trials_no_q`. A proportion looks identical whether
it came from 145 trials or 46; the count is the only thing that tells them apart. On KnowQA,
145 means pooled across a person's regimes and sittings, about 48 means one regime.

### The two studies split in opposite directions

| | Study 1 (L1) | Study 2 (KnowQA) |
|---|---|---|
| what varies | question preview | knowledge regime |
| design | **between** participants | **within** participants |
| a person appears in | exactly one group | all three regimes, every session |
| splitting by it | **safe** | **splits that person's trials** |

**Study 1 is safe.** 360 participants, 180 per group, nobody in both. So a per-participant
feature comes out the same whether you compute it on `hunters` alone or on everybody —
pooling the other group adds that person no trials.

**Study 2 is not.** Filtering KnowQA to one regime keeps the participant but drops two thirds
of their trials, so anything per-participant becomes a different quantity. Do not reason by
analogy from hunters/gatherers — it is the opposite design. KnowQA also has a second version
of this: one person sits for several sessions, so regime and session are two different ways
of cutting the same person's data.

### Where this actually bites

**Cross-validation.** By default each fold is cut from the ready-made model table, where these
features are worked out from all of a person's trials, held-out ones included. That is fine for
the saved feature sets, which do not use them. For a feature set that does, pass
`rebuild_features=True` (`modeling/crossval.py`): the features are then rebuilt from each
regime's own trials. Per regime is the only scope that works, because in a new-person regime
the person has no training trials to estimate from. On KnowQA the two scopes give a different
`breaks_pattern_no_q` on 52 of 300 no-knowledge trials.

**Any filter that keeps a person but drops some of their trials** — correct-trials-only
analyses, exclusion thresholds, and so on.

**KnowQA sessions.** Pupil baselines are per person, computed over that dataset's own
fixations. Strategy features pool across sittings unless you pass
`scope_by=["session_id"]`.

---

## 4. "Starting strategy" comes in two flavours

The concept is one thing: take the collapsed sequence of screen locations, drop the question,
keep the first four, repeats included. Four visits are enough for a full circuit of the four
answer positions if the person makes one, but a position can come up twice.

What varies is whether short scans get repaired. If someone's opening pass only reached two
positions, **prefix completion** fills the rest in from the most common way that prefix gets
completed by everyone else.

| Variant | Used by |
|---|---|
| **with** completion | the paper's descriptive prevalence figures |
| **without** completion | the per-trial model features |

The two differ by about 2 percentage points and flip which strategy is "dominant" for about
1.1% of participants. So when you read a prevalence number, know which one produced it.

The completion map is learned **once over the whole dataset**, so repairing a short scan does
not depend on which group the person happens to be in. One map per dataset, never pooled
across datasets — pooling L1 with KnowQA would make Study 1's numbers depend on Study 2 data.

**"Dominant" means most common, not consistent.** The test is that a person's single most
frequent opening scan covers at least half their trials. Hunters use between 6 and 28
distinct opening sequences each, so someone at exactly 50% still produced a dozen others.

Everything lives in `features/strategies.py`: `build_starting_strategies` for the per-trial
value, `dominant_strategy_by_participant` for the modal pick, `has_dominant_strategy` for the
threshold, `build_prefix_completion_map` and `add_completed_strategy_column` for the repair.
The plotting code only plots.

---

## 5. KnowQA's trial id is a string, and it has no paragraph screen

For L1, `TRIAL_INDEX` is an integer counter. For KnowQA it is a composite string like
`b2l01t005` — batch, list, trial — because one person sits for several sessions, so a plain
counter no longer identifies a trial. `glossary.md` has the full reasoning.

Most of the pipeline does not care about the type. Where it has to — joining a fixation report
to a word table — the ids are read as text and `fixations.match_key_types` then gives them the
word table's types: an integer `TRIAL_INDEX` for L1, text for KnowQA. A value that cannot be
converted stops the run; nothing is dropped to make the join work.

**KnowQA has no paragraph data at all.** Only the answer screen was exported. Paragraph
features are built by `features/paragraph/spans.py`, which KnowQA simply does not run;
`has_paragraph` is `False` on its dataset record. If you ask the KnowQA pipeline for paragraph
features it refuses and says why.

Related: anything reading a feature table pins `participant_id` to text, because KnowQA's ids
are all digits and would otherwise be read as integers, breaking every join.

---

## 6. Never name a folder after a standard-library module

`src/statistics/` shadowed Python's own `statistics` module. The symptom was 21 unrelated
modules failing to import because seaborn's `from statistics import NormalDist` found ours
instead.

The folder is now `src/stats/`, and the import path carries only the repo root — everything
is reached as `src.…`. **Putting `src/` back on the path re-arms this**, whether through
`.env`, a notebook cell, a `conftest.py`, or a `sys.path.insert`. The failure surfaces inside
a third-party library, nowhere near the cause.

Every directory and module name under `src/` has been checked against the standard library
and against every installed package. There are no other collisions. Worth re-running that
check whenever a new top-level folder appears.

---

## 7. A Windows console cannot print half of what this project prints

Windows consoles default to cp1252. The project prints `✓`, `…`, `≤`, `≥`, `→`, `★`, `β`, `±`
and more — 23 lines across 12 modules. A full rebuild once ran to completion, wrote every
output file, and then died on its final `print("✓ Done.")` with a `UnicodeEncodeError`.
Nothing was lost; nothing said so either.

`src/__init__.py` sets stdout and stderr to UTF-8 with `errors="replace"`. That touches the
console only — pandas writes files with its own encoding and is unaffected.

**Do not delete that block.** It looks like clutter and it is the only thing between a reader
on Windows and a crash on their first run. Stripping the characters instead would be a data
change: `≤ 4` is a category label that reaches saved tables and figure legends, so renaming
it would quietly invalidate comparisons against anything saved earlier.

---

## 8. The vendored EyeBench code is meant to be swapped out whole

`src/vendor/eyebench/utils - paragraph feature extraction.py` came from the lab's
EyeBench/OneStop project and is kept exactly as received. Two consequences:

- Its filename has spaces and a dash, so it cannot be imported by name. It is loaded by path,
  after its `configs/` folder is registered under the name it expects.
- It carries **one** local edit: an f-string reflowed onto a single line, because multi-line
  f-string expressions need Python 3.12 and this project runs 3.11. Drop in a fresh upstream
  copy without re-applying that and it will not parse.

Do not patch it incrementally — the whole point is that a newer copy can replace it.

---

## 9. Writing figures can also write into the paper, and that switch is off

Everything under `reports/` is written by one function, `lib/plotting/output.py::save_output`.
Given `to_paper=True` it also mirrors into the Overleaf repo, laid out identically:

```
reports/<analysis>/{figures,tables}/...
papers/correctness_prediction/reports/<analysis>/{figures,tables}/...
```

Same path, different root — so "which local file is this?" is answered by swapping the prefix.

**The mirror is currently off.** `PAPER_MIRROR_ENABLED` is `False`, `to_paper` defaults to
`False`, and asking for `to_paper=True` while the flag is off **raises** rather than quietly
doing nothing — a caller that thinks it published to the paper and didn't is the failure that
guards against. Turning it back on is one constant.

Nothing in this repo ever moves or deletes anything under `papers/`. That repo is Diana's and
Overleaf syncs it.

---

## 10. A figure cannot be saved without its numbers

`save_output` requires a `tables=` argument. A figure that genuinely has no numbers behind it
— an illustration, a scanpath example — passes `tables={}`, which you can grep for. "This has
no numbers" is a stated choice, never an accident.

Outputs live at `reports/<analysis>/figures/` and `reports/<analysis>/tables/`, so a figure
with no `tables/` sibling is visible in the same folder rather than requiring you to compare
two parallel trees.

This exists because results used to survive only inside PNG files. Nine of fifteen topics had
no numbers saved anywhere, and a sync failure left 329 image files at zero bytes with nothing
to regenerate them from. If you ever need something from that old tree, it is in git history
rather than on disk — and anything you recover from it predates the current data, so it tells
you what a number used to be, never what it is.
