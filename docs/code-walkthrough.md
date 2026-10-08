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
  rows per trial. This is the big file.
- One row per **area** — the question and the four answer options, so five rows per trial.
- One row per **trial** — one person answering one question. This is what the model reads.

The trap: when a trial-level fact (like "this person got it right") sits in the word-level file,
it's copied onto all 39 word rows. Count those rows and you think you have 39 times more data
than you do. That has actually happened here.

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

### Stop 1 — The words the project uses · `src/config/` (880 lines)

**Start here. It's the dictionary.**

- `columns.py` (307) — every column name in the project, with comments explaining the ones that
  aren't obvious. Read it like a glossary, not like code. Nothing happens in this file.
- `datasets.py` (438) — where every data file lives. Four datasets (L1, KnowQA, two pilots), each
  described once.
- `screens.py` (110), `outputs.py` (27) — small, skim them.

**You'll know this stop worked** if you can say what `skip_rate__contrast` means without looking
it up. (It's: how much more of the correct answer's words were skipped than the wrong answers',
on average.)

---

### Stop 2 — Generic helpers · `src/lib/` (870 lines)

Nothing here knows anything about eye-tracking. Confidence intervals, star annotations, and the
one function that saves figures.

The only one worth real attention is **`plotting/output.py` (636)** — every single figure and
table in the project goes through it. It's also the reason you can't save a picture without
saving the numbers behind it: the `tables=` argument is required.

---

### Stop 3 — Raw recordings become tidy words · `src/ingest/` (3,030 lines)

This is where the eye tracker's exports get cleaned up.

Read in this order:
1. `readers.py` (23) — trivial, 30 seconds.
2. `geometry.py` (131) — works out which screen area each word is in, from where it physically
   sits. Small and worth understanding; it replaced a word-counting method that got it wrong on
   20 trials.
3. `build.py` (403) — the conductor. It calls everything else in order.
4. `registry.py` (381) + `base_features.py` (295) — the list of per-word measurements and how
   they're built.
5. `clicks.py` (518) — reconstructs *when* the person pressed buttons, from the message log.
   Fiddly but self-contained.
6. `knowqa.py` (1,282) — **skip on the first pass.** It's Study 2's version of all of the above,
   and it only differs because Study 2 identifies trials differently. Come back to it when you
   care about Study 2.

**Output of this stop:** `all_participants.csv` — one row per word. 2.9 GB.

---

### Stop 4 — Words become one row per trial · `src/features/` (4,510 lines)

**This is the most important stop, and where I'd spend the most time.** Every number the model
sees is invented here.

1. `area_metrics.py` (570) — the eight per-area measurements (how long they looked, how many
   times, what fraction they skipped…). Read this one slowly.
2. `reading_times.py` (590) — time spent on each region. Note there are *two different
   definitions* of "reading time" here and they're deliberately kept apart.
3. `sequences.py` (392) — turns fixations into a sequence of area symbols, like "A, B, A, C".
4. `strategies.py` (753) — the opening scan: which order they first visited the four options.
   Backs a whole section of the paper.
5. `pupil.py` (327), `last_visited.py` (286), `preference.py` (143), `scope.py` (111) — smaller,
   each does one thing.
6. `build.py` (915) — assembles all of the above into the final table. Read it **last**, once
   you know what it's assembling.
7. `paragraph/spans.py` (423) — the same measurements, but over the paragraph screen instead.

**Output of this stop:** `model_ready.csv` — one row per trial, 219 columns, 19,436 rows.

---

### Stop 5 — The prediction machinery · `src/modeling/` (3,750 lines)

Generic. It doesn't know this is about eye-tracking — you could point it at anything.

1. `feature_sets.py` (464) — named lists of which columns a model uses. The paper's model uses
   ten of them, picked by hand.
2. `folds.py` (400) — how the data is split for testing.
3. `crossval.py` (685) — the testing loop.
4. `evaluate.py` (620) — scoring, and turning results into tables.
5. `inference.py` (393) — confidence intervals on the coefficients. **Worth scrutiny** — there
   are three methods here and they disagree with each other on purpose.
6. `models/logreg_model.py` (292) — the actual model. Smaller than you'd expect.

---

### Stop 6 — The actual questions · `src/analyses/` (~13,000 lines)

**This is the paper.** One folder per thing the paper claims. Each folder has the same shape:
`compute.py` works out the numbers, `stats.py` tests them, `plots.py` draws them.

Read them in the order the paper makes its argument:

| folder | the claim | lines |
|---|---|---|
| `scan_strategies/` | people open with a consistent clockwise scan | 1,100 |
| `last_visitation/` | the last look before committing lands on their chosen answer | 228 |
| `attention_allocation/` | they look hardest at the option they pick | 800 |
| `correctness_associations/` | longer, more back-and-forth trials are less accurate | 1,800 |
| `correctness_prediction/` | **the headline model** — and the biggest folder | 6,800 |
| `text_qa_relationship/` | time on the paragraph relates to time on the options | 1,760 |
| `time_course/` | attention before vs during vs after the decision | 501 |
| `answer_rt_comparison/` | time on wrong options predicts errors; time on the right one doesn't | 222 |

Inside `correctness_prediction/`, go: `run.py` → `plots/` → `person_variance/` →
`knowledge_regimes/` (that last one is Study 2 and can wait).

---

### Stop 7 — Study 2's materials · `src/experiment/` (530 lines)

How the second experiment was built — the counterbalanced lists and the text preparation. Short,
and unlike everything else it *produces* something for people to look at rather than analysing
what they looked at.

---

### Stop 8 — Things kept but not used · `src/explorations/` (~5,400 lines)

**You can skip all of this** and lose nothing about the paper. It's abandoned or paused work,
kept on purpose and labelled. Two of these folders don't even run.

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
   blanks. Some joins check this now; not all.
3. **`features/strategies.py` and anything "per participant".** These summarise a person across
   their trials — so the answer depends on *which* trials you handed it. There's now an explicit
   setting for that, but it's worth confirming each caller passes what it means.
4. **`analyses/correctness_associations/stats.py`.** These tests treat every trial as independent,
   when trials are grouped within people and within texts. It makes the p-values too small. Known,
   not yet addressed.
5. **Anything where the paragraph screen and the answer screen are compared.** They used to be
   measured by two separate pieces of code that drifted apart. They're one now — worth confirming
   you agree they should be.

---

## Two maps you may want open alongside

- `docs/glossary.md` — what every column means.
- `docs/pitfalls.md` — the things that have actually gone wrong here, explained. If you only read
  one other file, read §1 and §2 of this.
