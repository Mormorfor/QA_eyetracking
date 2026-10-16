# A walk through the code

A reading order for going through the whole thing by hand. Written for someone sitting down
with coffee, not for someone who already knows where everything is.

*This is a scaffold for your review week, not project documentation. Bin it when you're done.*

---

## Three things to know before you open a single file

Everything else will make sense if these do, and nothing will if they don't.

**1. The same data exists at three different "sizes", and mixing them up is the main way a
number goes wrong here.**

- One row per **word** — every word on the screen, for every trial, for every person. About 39
  rows per trial. File: `interest_areas.csv`.
- One row per **area** — the question and the four answer options, so five rows per trial.
  File: `areas_answers.csv`.
- One row per **trial** — one person answering one question. This is what the model reads.
  File: `trials.csv`.

Each feature is stored in the file for its size, so a trial-level fact like "this person got it
right" is stored once, in `trials.csv`.

The trap is when the three are joined back together. `features/build.py::load_model_input(ds)`
does this, and in the result a trial-level fact is copied onto all ~39 word rows of its trial.
Count those rows and you think you have 39 times more data than you do. So any test about trials
must run on a table with one row per trial.

**2. `L1` means "native English speaker".** Not "level 1", not "study 1". It's which slice of the
OneStop corpus this is. Study 1 is the L1 data; Study 2 is called KnowQA.

**3. `answer_A` is always the correct answer.** The letters are meanings, not positions. Where
the correct answer physically sat on screen is randomised per trial and lives in a separate
column. So "answer_A" never tells you where someone looked on screen — only what they looked at.

---

## What the code does, in one breath

A person reads a paragraph, sees a question and four options, looks around, and presses a button.
The eye tracker records where their eyes were, moment by moment. The code turns that into one row
of numbers per trial, and then asks: **can those numbers predict whether they got it right?**

Everything in `src/` is one step of that journey. So the reading order below is just **following
one trial from the eye tracker to the paper.**

---

## The route

Nine stops. Line counts are there so you can budget — they're the real sizes.

### Stop 1 — The words the project uses · `src/config/` (1,160 lines)

**Start here. It's the dictionary.**

- `columns.py` (509) — every column name in the project, with comments explaining the ones that
  aren't obvious. Read it like a glossary, not like code. Nothing happens in this file.
- `datasets.py` (554) — where every data file lives. Four datasets (L1, KnowQA, two pilots), each
  described once.
- `screens.py` (75), `outputs.py` (19) — small, skim them.

**You'll know this stop worked** if you can say what `skip_rate__contrast` means without looking
it up. (It's the correct answer's skip rate minus the average skip rate of the three wrong
answers. Positive means more of the correct answer's words went unread.)

---

### Stop 2 — Generic helpers · `src/lib/` (860 lines)

Nothing here knows anything about eye-tracking. Confidence intervals, star annotations, and the
one function that saves figures.

The only one worth real attention is **`plotting/output.py` (619)** — every single figure and
table in the project goes through it. It's also the reason you can't save a picture without
saving the numbers behind it: the `tables=` argument is required.

---

### Stop 3 — Raw recordings become tidy words · `src/ingest/` (3,490 lines)

This is where the eye tracker's exports get cleaned up.

Read in this order:
1. `readers.py` (24) — trivial, 30 seconds.
2. `geometry.py` (129) — works out which screen area each word is in, from where it sits on the
   screen. Small and worth understanding. Where this disagrees with counting words in the stored
   texts, position wins and the correction is printed.
3. `build.py` (524) — the conductor. It reads the two raw reports, once each, and calls
   everything else in order.
4. `registry.py` (946) + `base_features.py` (266) — the list of measurements and how they're
   built. **Read `registry.py` properly.** Each feature is one entry: the function that builds
   it, the columns it reads (`needs`), and whether it works per word, per area or per trial. A
   second list, `_PRODUCES`, says which columns each feature adds, and that decides which of the
   three output files each column goes in. It is the closest thing the project has to a table of
   contents.
5. `clicks.py` (418) — reconstructs *when* the person pressed buttons, from the message log.
   Fiddly but self-contained.
6. `knowqa.py` (1,160) — **skip on the first pass.** Study 2's preparation: it converts and cleans
   the raw exports, then runs the same `build.py`. Much of its length is about identifying a
   trial, because one Study 2 participant can come back for several sessions. Come back to it
   when you care about Study 2.

**Output of this stop:** three files — `interest_areas.csv` (one row per word, 23 columns),
`areas_answers.csv` (one row per area, 19 columns) and `trials.csv` (one row per trial, 89
columns).

---

### Stop 4 — Words become one row per trial · `src/features/` (3,860 lines)

**This is the most important stop, and where I'd spend the most time.** Every number the model
sees is made here.

1. `area_metrics.py` (421) — the per-area measurements: how long each area was looked at, how
   many fixations, what share of its words were skipped, pupil size. Read this one slowly.
2. `fixations.py` (184) — reads a fixation report, and puts each fixation on a word: the word it
   landed on, or the nearest one. Short. Everything that needs single fixations — the next two
   files, the pupil baseline, the last look before a button press — goes through it, so they all
   agree on where a fixation was.
3. `reading_times.py` (339) — three different ways of timing an area: total fixation time, time
   spent on visits to it, and first look to last look. Read the top of the file first; the three
   are easy to mix up.
4. `sequences.py` (485) — turns fixations into a sequence of areas, like "A, B, A, C", and a few
   statistics of it.
5. `strategies.py` (709) — the opening scan: which order they first visited the four options.
   Backs a whole section of the paper.
6. `pupil.py` (295), `last_visited.py` (145), `preference.py` (133), `scope.py` (74) — smaller,
   each does one thing.
7. `build.py` (775) — assembles all of the above into the final table. Read it **last**, once you
   know what it's assembling. It measures nothing new from the eye data: it turns the per-area
   values into columns, compares the correct answer with the wrong ones, and joins everything on
   the trial.
8. `paragraph/spans.py` (285) — the same measurements, but over the paragraph screen instead.

`src/checks.py` (58) belongs with this stop too: the check `build.py` runs before its joins, that
every trial has exactly one match.

**Output of this stop:** `model_ready.csv` — one row per trial, 219 columns, 19,436 rows.

---

### Stop 5 — The prediction machinery · `src/modeling/` (3,730 lines)

Generic. It doesn't know this is about eye-tracking — you could point it at anything.

1. `feature_sets.py` (407) — named lists of which columns a model uses. The paper's model uses
   ten of them, picked by hand (`SELECT_1_COLS`).
2. `folds.py` (396) — which trials train and which are held out. Ten stored folds per group;
   each gives every person × paragraph pair a regime: new paragraph, new person, or both. **Read
   the docstring of `attach_fold_regimes`** — the join leaves out, without a message, the trials
   a fold file has no regime for (none, with the current files).
3. `crossval.py` (692) — the testing loop: fit on each fold's training trials, score on each
   held-out regime. Each fold is cut from the ready-made model table; `rebuild_features=True`
   rebuilds the features per regime instead, which only matters for the opening-scan features
   (explained at the top).
4. `evaluate.py` (587) — fit-and-score on one split, the single full-data fit the paper's
   coefficients come from, and the summary tables.
5. `inference.py` (397) — confidence intervals on the coefficients. **Worth scrutiny.** Two
   methods, used for different jobs: the bootstrap (slow, no approximation — the paper's
   coefficients) and the Wald formula (fast, an approximation — inside cross-validation).
6. `models/logreg_model.py` (301) — the actual model. Smaller than you'd expect. The note at the
   top on how missing values become 0 is worth reading slowly.
7. `selection.py` (343), and the other models in `models/` (dummy baseline, plus two used only
   by an exploration) — skim.

---

### Stop 6 — The paper's results · `src/analyses/` (~13,800 lines)

**This is the paper.** One folder per result. Each folder works out its numbers and draws its
figures, and every figure is saved with the numbers behind it under `reports/<folder>/`. The small
folders do both in `plots.py`; `attention_allocation/` and
`correctness_associations/` keep their tests in `stats.py`; the two big ones are split into several
modules, each listed at the top of the folder's `__init__.py`. `python scripts/run_analysis.py
--list` says which folders run without a notebook; the others are run from `notebooks/drivers/`.

Read them in the order the paper makes its argument:

| folder | what it shows | lines |
|---|---|---|
| `scan_strategies/` | most people (about 60%) look over the four answers in a habitual order, most often clockwise | 1,070 |
| `last_visitation/` | the last area looked at before confirming is usually the chosen answer | 210 |
| `attention_allocation/` | people look most at the answer they end up choosing | 790 |
| `correctness_associations/` | longer, more back-and-forth trials are less accurate | 1,830 |
| `correctness_prediction/` | **the headline model** — and the biggest folder | 7,690 |
| `text_qa_relationship/` | how time on the paragraph's parts relates to time on the answers | 1,970 |
| `answer_rt_comparison/` | accuracy barely moves with time on the correct answer, and falls with time on the wrong ones | 230 |

Inside `correctness_prediction/`, go: `run.py` → `plots/` → `participant_level.py` and
`person_variance/` → `knowledge_regimes/` (that last one is Study 2 and can wait). `run.py` fits
the model on one train/test split. The paper's accuracy comes from cross-validation
(`modeling/crossval.py`, run from `notebooks/drivers/answer_corr_prediction.ipynb`), and its
coefficients from one fit on all the data (`modeling/evaluate.py`).

---

### Stop 7 — Study 2's materials · `src/experiment/` (530 lines)

How the second experiment was built — the counterbalanced lists and the text preparation. Short,
and unlike everything else it *produces* something for people to look at rather than analysing
what they looked at.

---

### Stop 8 — Things kept but not used · `src/explorations/` (~7,300 lines)

**You can skip all of this** and lose nothing about the paper. It's abandoned or paused work,
kept on purpose and labelled. Two of these folders don't even run. One is unfinished:
`time_course/` was meant to compare behaviour before and after the first look at the chosen
answer, but it still splits each trial by screen order, not time (its docstring explains).

Worth five minutes just to confirm nothing in here is secretly load-bearing. It isn't — that's
checked automatically.

---

### Stop 9 — Someone else's code · `src/vendor/` 

Not ours. Kept exactly as received so it can be swapped for a newer copy. Don't review it as if
it were yours; do note that it fills missing values with zero very liberally.

---

## If you only have two hours

`config/columns.py` → `features/area_metrics.py` → `features/build.py` →
`modeling/feature_sets.py` → `analyses/correctness_prediction/run.py`.

That's the spine: what things are called, how the numbers are made, which ones the model gets,
and what it does with them.

---

## Where I'd look hardest

Not bugs I know about — places where a mistake would be quiet rather than loud.

1. **Anywhere a missing value becomes a zero.** The project does this deliberately in places and
   it's documented, but it's the single most common way a wrong number looks right. Search for
   `fillna` and ask "is zero actually true here?"
2. **Any join.** If two tables get merged and one is missing rows, the result silently gets
   blanks. Most joins in `features/build.py` are checked; not all. The paragraph join checks only
   that it adds no rows and matches at least one trial (today it matches all 19,436). The joins in
   `load_model_input`, which rebuild one wide table from the three files, are not checked at all —
   and they run every time a model table is built.
3. **`features/strategies.py` and anything "per participant".** These summarise a person across
   their trials — so the answer depends on *which* trials you handed it. There is a setting for
   that (`scope_df`, `scope_by`), and it's worth confirming each caller passes what it means.
   These features are deliberately not in the registry, because a registry entry is computed once
   over the whole dataset — which is exactly the scope that leaks. The docstring says so.
4. **`analyses/correctness_associations/stats.py`.** These tests treat every trial as independent,
   when trials are grouped within people and within texts. It makes the p-values too small. Known,
   not yet addressed.
5. **Anything where the paragraph screen and the answer screen are compared.** Both are measured
   by the same code. The two places they differ are listed at the top of `area_metrics.py` — worth
   confirming you agree those are the only differences there should be.

---

## Two maps you may want open alongside

- `docs/glossary.md` — what every column means.
- `docs/pitfalls.md` — the things that have actually gone wrong here, explained. If you only read
  one other file, read §1 and §2 of this.
