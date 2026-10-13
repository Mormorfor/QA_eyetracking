# Feature generation — three grains, one registry

**Status: approved 2026-10-13, implementation started the same day.** Raised by Diana during the code walkthrough, from
two independent observations: that `FUNCTION_REGISTRY` holds the same group features it held
before the restructure, and that `model_ready` *creates* features rather than preparing them.

Both are symptoms of one thing, and it is a thing stage D was meant to fix. This proposal
finishes that work and absorbs the pieces of it that fell through.

> **Diana's framing, 2026-10-13, which is the requirement this is written against:**
>
> *"The model_ready sequence should be busy preparing features for model ingest, and not
> creating new ones. The logic of the feature itself should happen in the same-ish space for all
> the features existing, and when I add another one it should just join the pipeline with all
> the rest."*

---

## What this is for

Today a feature can be built in one of **four** places, and which one is historical rather than
principled:

| where | how it is declared | example |
|---|---|---|
| `FUNCTION_REGISTRY`, `kind: "base"` | a registry entry | `add_is_correct` |
| `FUNCTION_REGISTRY`, `kind: "group"` | a registry entry | `create_skip_rate` |
| `features/build.py`, hand-written block | not declared anywhere | `has_xyx`, `seq_len` |
| `features/strategies.py`, called from `build.py` | not declared anywhere | `dominance_score` |

The first two are listable, nameable and skippable. The last two are a function call in the
middle of an assembler. Stage D's own diagnosis said why, and it is still the best statement of
the problem:

> **"Derived" is a date, not a category.** … Same kind of quantity, same source file, **two
> different mechanisms**, and the only thing separating them is that `FUNCTION_REGISTRY` already
> existed when the first pair was written and had stopped being the obvious place by the time
> the second arrived.

After this, there is one way to add a feature: a registry entry that declares its grain and what
it reads. That is the whole of Diana's requirement.

---

## The diagnosis, measured

All figures measured 2026-10-13 on L1, from `data/datasets/l1_onestop/`.

### 1. Nothing this pipeline computes is word-level

`all_participants.csv` is **239 columns at 40.1 interest-area rows per trial**, 2.9 GB. Sampling
250,000 rows (6,238 whole trials) and asking, per column, whether it ever varies inside a trial
and inside an area:

| what the column actually is | count | share |
|---|---|---|
| constant within a trial — **broadcast onto ~40 rows** | 108 | 46% |
| constant within an area, varies between areas | 17 | 7% |
| genuinely varies word to word | 111 | 47% |

That last row looks like a justification for the IA grain until you read it. Of the 111, **four**
were computed by this project: `IA_MAX_FIX_PUPIL_SIZE_z`, `IA_MIN_FIX_PUPIL_SIZE_z`,
`IA_AVERAGE_FIX_PUPIL_SIZE_z`, and `area_skipped` — which is not a feature at all but the
intermediate that `todo.md` **T3.11** records as leaking into the saved output. The other 107 are
raw EyeLink passthrough (`IA_THIRD_RUN_LAUNCH_SITE`, `IA_FIRST_SACCADE_ANGLE`, …) and OneStop's
linguistic annotations (`gpt2_surprisal`, `universal_pos`, `word_length`, …).

> **So the IA grain is the shape the raw report arrives in, not the shape anything computes at.**
> Every quantity this project builds is area-level or trial-level. `all_participants.csv` is "the
> raw report, plus everything we computed, broadcast across it", and that is where the 2.9 GB is.

### 2. The trial-level features are computed at IA grain anyway, then thrown away

`build_trial_level_derived_features` is handed `df` — the **interest-area frame** — and does:

```python
d = df[required].copy()                                        # ~40 rows per trial
d["_has_xyx"] = d["_seq"].apply(has_back_and_forth_xyx)        # computed per IA row
...
out = d.groupby(TRIAL_ID_COLS).agg({"_has_xyx": "first", ...}) # 39 of 40 discarded
```

Its input, `simpl_fix_by_label`, is a *trial-level* column that a registry group feature already
broadcast onto every IA row. So `has_xyx` is computed about forty times per trial from a value
that was copied forty times, and thirty-nine of each are dropped.

This matters for the argument, not just for the waste: **the split is not IA-grain versus
trial-grain.** Both halves compute against the IA frame and collapse. One is declared in the
registry and one is hardcoded in an assembler. That is the only difference.

### 3. Nothing filters the raw report, and the mechanism to do so was agreed a week ago

| | |
|---|---|
| raw `ia_Answers.csv` | **157 columns** |
| carried through to `all_participants.csv` | **157 — all of them** |
| dropped | **0** |
| added by the pipeline | 82 |

`ingest/readers.py::load_raw_answers_data` is `pd.read_csv(ia_a_path, engine="python")`. No
`usecols`. And **76 of the 157 raw columns are never named anywhere in `src/`.**

*(That 76 is a literal-name proxy — `needs` would make it exact. The one escape hatch was checked:
the only `IA_`-prefix pattern selection in the repo is in `vendor/eyebench/`, which reads the raw
reports directly through its own `_resolve_usecols` and is unaffected by a filter on this reader.)*

### 4. Three of stage D's eight acceptance criteria were never satisfiable

Stage D §3.4 — `needs` on every registry entry, feeding a read-time whitelist and an order
check — was **approved by Diana on 2026-10-06** and **never given a row in the order-of-work
table**. Steps 1–7 cover §3.1, §3.2, §3.6, §3.7, §3.8, §3.9 and the file moves. §3.4 has no step.
It fell through the gap between *what to do* and *when*.

Today **0 of 20** registry entries have `needs`; the keys are `callable`, `default_kwargs`,
`kind`. Which leaves these, as written:

> **5.** every column … is claimed by exactly one registry entry, **or is a raw passthrough in
> the whitelist** — there is no whitelist
> **6.** No orphan columns … *"Today nothing can state this"* — depended on step 4, dropped
> **8.** **The read-time filter is proved, not trusted:** one pipeline run with `usecols` and one
> without, bit-identical output — there is no `usecols` path to run

The other five criteria passed and are the load-bearing ones, so this is not a correctness
problem. But stage D is marked ✅ against a test that could never fully run, and these three are
inherited here rather than quietly dropped.

---

## The proposal

### 1. Three grains, three artifacts

Write down the grains the code already computes at, instead of collapsing all three into the
finest one.

| artifact | one row per | rows, L1 | holds |
|---|---|---|---|
| `interest_areas` | participant × trial × area × **word** | ~779,000 | the whitelisted raw columns, the three `IA_*_z` pupil columns, and the area assignment (`area_label`, `area_screen_loc`) |
| `areas` | participant × trial × **area** | 97,180 | the eight per-area metrics, visit counts, and the RT/TFD families **long** rather than wide |
| `trials` | participant × **trial** | 19,436 | sequences, the five `DERIVED_COLS`, the strategy/dominance family, last-visited, click timings, identity and condition columns |

The 108 broadcast columns move to `trials`, the 17 area-level ones to `areas`, and
`interest_areas` keeps only what genuinely varies per word.

**RT/TFD becomes area-grain, which is the one reshape worth calling out.** `aux/RT_and_TFD.csv`
is wide today — `RT_pure_answer_A`, `RT_pure_answer_B`, … — which makes it trial-grain by
accident of layout while being area-grain by meaning. Long-form puts it beside the eight metrics,
and the same pivot serves both. `build_trial_level_rt_tfd_features` stops being a separate thing
from `build_trial_level_area_features`.

**The two screens stay two artifacts** — `areas_answers` and `areas_paragraph` — per T6.1 and
Diana's ruling recorded in `config/screens.py`: *"the two screens are still generated separately
into separate artifacts, joined on `(participant_id, TRIAL_INDEX)` where something wants both."*
Unifying the grain is not a reason to reopen that.

### 2. `needs` on every entry — stage D §3.4, unchanged

Each registry entry declares every column it reads, as literal names. One declaration, three uses
now instead of two:

```python
"create_mean_pupil_size_metrics": {
    "callable": create_mean_pupil_size_metrics,
    "kind": "area",
    "needs": ["IA_AVERAGE_FIX_PUPIL_SIZE", "IA_MAX_FIX_PUPIL_SIZE", "IA_MIN_FIX_PUPIL_SIZE",
              "IA_AVERAGE_FIX_PUPIL_SIZE_z", "IA_MAX_FIX_PUPIL_SIZE_z", "IA_MIN_FIX_PUPIL_SIZE_z"],
},
```

1. **The raw whitelist.** `union(needs) & set(raw_header)` → `read_csv(usecols=...)`. Intersecting
   with the header sorts raw from produced automatically; a `_z` column is not in the header and
   drops out.
2. **The order check.** Walk the registry in its existing order; if an entry needs a column
   nothing before it has produced, raise. This turns **T3.11**'s undeclared ordering dependency
   into a loud failure *while leaving the order exactly as it is* — it verifies the list rather
   than computing one, so **T3.24** stays declined, as agreed.
3. **New: the grain check.** An entry may only need columns at its own grain or coarser. An
   `area` entry reading an IA column is legitimate (it aggregates); a `trial` entry reading an IA
   column is the mistake this whole proposal is about, and it becomes detectable.

### 3. One registry, all features

`kind` stops meaning "base or group" and starts meaning **grain**:

| `kind` | reads | produces | the runner |
|---|---|---|---|
| `row` *(today `base`)* | `interest_areas` | IA columns, in place | — |
| `area` *(today `group`)* | `interest_areas` | an area-keyed table | writes `areas_*` |
| `trial` *(today: nothing)* | `interest_areas` or `areas` | a trial-keyed table | writes `trials` |

The pipeline becomes one pass per grain, each a registry walk:

```
row   -> interest_areas.csv
area  -> areas_answers.csv · areas_paragraph.csv
trial -> trials.csv
```

Everything currently hardcoded in `features/build.py` and called from it becomes a `trial` entry:
the five `DERIVED_COLS`, the strategy/dominance family, last-visited, preference matching. They
become listable, nameable, skippable and order-checked exactly like the IA ones already are —
which was §3.1's stated purpose, and the half of it that survived step 3's audit.

> **On step 3.** Its conclusion — *"`kind: "trial"` not added, and should not be … there was
> nothing left to select"* — was reached by auditing the eight `include_*` booleans and finding
> seven dead. That settles **skippable**. It does not touch *declarable*, *nameable* or
> *listable*, which were §3.1's other three reasons, and it is why the placement problem rode out
> with the flags. The booleans were right to delete; the grain was not theirs to take with them.

### 4. File naming — the file is named for its grain

The rule: **a file is named for its grain, and its grain is its key.** Never for what it was
called first, and never for its population *alone* — a group subset is the same grain faceted, so
it carries the grain name plus a `__<facet>` suffix (§7), the way `skip_rate__correct` does. So
`trials__hunters.csv`, never `hunters.csv`.

```
data/datasets/<key>/
  interim/
    interest_areas.csv        participant × trial × area × word      (was all_participants.csv)
    areas_answers.csv         participant × trial × area             (new)
    areas_paragraph.csv       participant × trial × span             (was features/paragraph_spans.csv)
    trials.csv                participant × trial                    (new)
    aux/
      participant_pupils.csv  participant — a z-score baseline, an input not a feature
      button_clicks.csv       trial — event reconstruction, an input not a feature
  features/
    model_ready.csv           participant × trial, wide              (unchanged name)
```

Two renames are the point of the scheme rather than tidying:

- **`all_participants.csv` → `interest_areas.csv`.** The old name describes *who is in the file*,
  which is the one thing every artifact here has in common, and says nothing about its grain —
  which is the thing that has actually caused wrong numbers (`pitfalls.md` §1, T3.1).
- **`RT_and_TFD.csv` disappears** into `areas_answers.csv`, because it was only ever a separate
  file for being computed at a different time.

`aux/` keeps its two genuine inputs and loses the four tables that were features in hiding.

### 5. What `model_ready` keeps — and it is only reshaping

```
model_ready = pivot(areas_answers) + pivot(areas_paragraph) + join(trials) + contrasts
```

Nothing else. The pivot turns `mean_dwell_time` at area grain into
`mean_dwell_time__answer_A … __question`; the join attaches the trial-grain columns; the contrast
step adds `__correct`, `__wrong_mean`, `__contrast`, `__distance_furthest`, `__distance_closest`.

**The contrasts stay here — Diana's ruling, 2026-10-13.** They are the one family where "prepare
versus create" genuinely blurs: they are arithmetic over the pivot, and what makes them meaningful
is the *model's* framing (it knows which option is correct; the participant's choice is hidden),
not a property of the eye movements. They are a re-expression of five area columns into the
contrast the model is asked about, so they belong with the reshape.

### 5b. `kind` is the mechanism; `grain` is the output. They are not the same thing

**Added 2026-10-13, after measuring what the registry split alone would leave behind.**

Splitting only what the registry produces moves 20-odd columns and leaves `interest_areas` at
**97 columns, of which 75 are still trial-constant and only 18 genuinely vary word to word**
(the other 4 are 2 area-constant plus the 2 join keys). That is not the correct data
organisation; it is the same broadcast with a smaller lid on it.

The reason is a conflation. `kind` answers *how is this computed* — in place on the row, or by
aggregating a group. It has been doing double duty as *what shape is the output*, and for 75
columns those answers differ: `add_IA_screen_location` is a `row` feature by mechanism and
produces `question_len`, `n_interest_areas` and four `*_len` columns that are **trial**-grain
data. **The two stay separate concepts** (Diana, 2026-10-13).

> **The rule: grain is declared, never inferred.** It would be easy to classify by observing
> which columns happen to be constant within a trial — the measurement above does exactly that.
> That is a fact about *this dataset on this day*, not about the column, and a dataset where a
> column happened to be constant would silently acquire a different schema. Declarations are
> checked against observation (acceptance 5), never replaced by it.

**The smallest mechanism that does this.** One module-level map, three declared sources, nothing
added to the registry entry shape beyond what is already there:

```python
#: Columns living at a coarser grain than the interest-area row.
#: A column not named here is word-grain and stays on `interest_areas`.
COLUMN_GRAIN: dict[str, str]        # column -> "area" | "trial"
```

It is assembled from three places, each the natural owner of what it declares:

| source | declares | why there |
|---|---|---|
| **registry entries** — `_PRODUCES` values carry a grain per column instead of being a bare set | what each feature produces, and at what grain | the entry already declares `needs` and already lists what it produces for the order check; adding the grain reuses that one declaration rather than adding a second |
| **`ATTACH_PRODUCES`** | the RT/TFD and last-label blocks | produced by `_attach_*` in `build.py`, outside the registry, and **trial-keyed by construction** — they already merge on `(participant_id, TRIAL_INDEX)`, so the declaration records a fact the code enforces |
| **`RAW_COLUMN_GRAIN`** | raw report columns that are not per-word | nothing produces them, so no entry can own them; the report is the source and the map is the only place to say so |

Three sources rather than one because they go stale for different reasons and have different
owners — the same argument that keeps the four `needs` declarations apart.

**Default by omission is deliberate and runs in the safe direction.** An undeclared column stays
on `interest_areas`, which is where it already is: forgetting a declaration leaves a column
broadcast, which is the status quo, rather than silently moving it to a grain where its values
would be collapsed by a `.first()`. Acceptance 5 catches the omission; nothing is lost if it does
not.

### 6. `data_raw/` is never written, by anything here

**A hard guarantee, not a scoping note** (Diana, 2026-10-13). Nothing in this proposal writes,
moves, renames or deletes a file under `data_raw/`. The raw reports are read-only input and stay
byte-for-byte as the tracker exported them.

This is worth stating explicitly because the whitelist (§2) *sounds* like it removes columns. It
does not: `usecols` is an argument to `read_csv`, so the filtering happens **in memory, on the way
in**. All 157 raw columns remain on disk. Widening a `needs` list and re-running prep is the whole
of an undo, which is also why stage D §3.5 called the filter reversible.

Two consequences, both deliberate:

- **An un-whitelisted raw column is not lost, only unread.** If an analysis later wants
  `IA_REGRESSION_PATH_DURATION`, it goes in a `needs` list and reappears on the next prep run.
- **`data_raw/full` and `data_raw/tsv` are symlinks to the OneStop corpus held outside the repo**
  (`todo.md` T5.7). Writing to them would reach outside the project entirely. Read-only is the
  only safe posture, and the code should keep having no write path to that tree at all.

### 7. The hunters/gatherers split survives at every grain

**Diana, 2026-10-13: the group files keep being produced.** Hunters-only and gatherers-only
analysis is a live use, not a legacy of the IA table being too big to filter.

So the split is a property of each artifact, not of one of them:

```
interim/
  interest_areas.csv · interest_areas__hunters.csv · interest_areas__gatherers.csv
  areas_answers.csv  · areas_answers__hunters.csv  · areas_answers__gatherers.csv
  trials.csv         · trials__hunters.csv         · trials__gatherers.csv
features/
  model_ready.csv    · model_ready__hunters.csv    · model_ready__gatherers.csv
```

`__` is already this project's "faceted by" separator (`DERIVED_SEP`, and `save_output`'s facet
tags), so `trials__hunters` reads the same way `skip_rate__correct` does. It also renames
`interim/hunters.csv` to something that says which grain it is — the thing the current name
leaves out.

> **They must be written in the same run as their parent, and that is the point of listing them
> here.** This has already gone wrong: on 2026-10-13 `model_ready.csv` was rebuilt at 11:52 while
> `model_ready_hunters.csv` and `model_ready_gatherers.csv` were still from 2026-10-07 11:04 —
> five days and one `nearest_ia_queue` fix apart, with `text_qa_relationship` reading all three
> and only rebuilding a cache that is *absent*, never one that is merely stale. A split that is a
> step in the same run cannot drift from its parent; a split that is a separate rebuild can, and
> did.

**The split column is a parameter, not `question_preview`.** `_save_splits` already takes
`split_column` and a `{value: path}` map; what changes is that each grain's writer calls it rather
than only the IA one. L1 passes `question_preview`. KnowQA passes nothing — see below.

### 8. KnowQA compatibility

Study 2 differs from L1 in four ways that this proposal has to not break. None is new; all are
already settled elsewhere and are restated here as constraints.

| | KnowQA | what it constrains |
|---|---|---|
| `TRIAL_INDEX` | composite **string** (`b1l01t002`) | no builder may coerce it to int. The known offender is `load_paragraph_fixations`, which KnowQA never reaches (`pitfalls.md` §5) — the new writers must not add a second one |
| identity | `participant_id` **and** `session_id`; one person spans sittings | `trials.csv` carries both. Participant-level aggregates pool across sittings by default, with `n_strategy_trials_*` alongside — T3.21's settled design, unchanged here |
| paragraph screen | `has_paragraph=False` | **no `areas_paragraph.csv` is written at all.** Not an empty file, not a flag at a call site — the dataset record says the screen does not exist, so that pass does not run |
| group split | **no `question_preview` column** | no `__hunters` / `__gatherers` artifacts. The split is a parameter and KnowQA's is unset |

> **`regime` is not KnowQA's hunters/gatherers, and must not be wired up as if it were.** The
> manipulation is **within**-participant — every participant does all three regimes — so
> splitting by it keeps the participant and drops two thirds of their trials, which changes what
> every participant-level aggregate means (`pitfalls.md` §3; measured: estimating over the full
> set rather than the slice flips `breaks_pattern_no_q` on 52 of 300 no-knowledge trials). A
> regime-split artifact would therefore be a different quantity wearing the same column names.
> If one is ever wanted it needs `scope_by` set deliberately, not a default.

> **Resolved 2026-10-13 — recorded because the failure was silent.** KnowQA's dataset folder was
> briefly nested inside L1's, at `data/datasets/l1_onestop/knowqa/`, while `dataset("knowqa")`
> pointed at `data/datasets/knowqa/`. Diana moved it back the same day; verified afterwards that
> every path property on both records resolves (L1 219 columns, KnowQA 200).
>
> **What it exposed is worth keeping:** the registry had no way to say it was broken. Every
> `Dataset` path property returned a `Path` that simply did not exist, and nothing noticed until
> a read failed deep inside pandas with a `FileNotFoundError` naming a path no one had typed.
> `restructure-map.md` §7 anticipated exactly this — *"the stale-constant class of bug
> disappears, because paths are derived from one root and can be validated in one loop at import
> or in a test"* — and that validation loop was never written. **One such check belongs in step 1
> of this proposal**, where it costs nothing: walk `DATASETS`, assert each declared artifact
> either exists or is one this dataset legitimately does not have (KnowQA and `areas_paragraph`),
> and fail loudly naming the dataset and the path.

---

## Where every builder goes

| today | grain | becomes |
|---|---|---|
| 9 × `kind: "base"` | `row` | unchanged, renamed kind |
| `create_mean_area_dwell_time`, `_fix_count`, `_first_fix_duration`, `create_skip_rate`, `create_dwell_proportions` | `area` | registry entries, output to `areas_*` |
| `create_mean_pupil_size_metrics`, `create_first_encounter_pupil_size` | `area` | " |
| `create_simplified_visit_counts` | `area` | " (visit counts are per area) |
| `build_screen_rt_tfd` / `build_rt_and_tfd` | `area` | " — long form, absorbing `RT_and_TFD.csv` |
| `features/paragraph/spans.py::build_span_metrics` | `area` | `areas_paragraph` — already this shape |
| `create_fixation_sequence_tags`, `create_simplified_fixation_tags` | **`trial`** | registry entries writing `trials`; stop being broadcast |
| `create_last_area_and_location_visited` | **`trial`** | " |
| `build_trial_level_derived_features` | **`trial`** | five entries, or one — see *least sure* below |
| `build_trial_level_pattern_features` | **`trial`** | registry entry, keeping `scope_df` / `scope_by` (T3.21) |
| `features/preference.py::compute_trial_matching` | **`trial`** | registry entry |
| `build_trial_level_last_visited_features` (one-hots) | — | **stays in `model_ready`** — a reshape of a `trials` column |
| `build_trial_level_area_features` (pivot) | — | **stays in `model_ready`** |
| `add_answer_correct_wrong_contrast_columns` | — | **stays in `model_ready`** (Diana, 2026-10-13) |
| `build_trial_level_rt_tfd_features` | — | **dissolves** into the area pivot |

---

## What is inherited here

Each of these was agreed, measured or scheduled before and is not new work being invented:

| item | state before | here |
|---|---|---|
| **§3.4** `needs` + read-time filter | approved 2026-10-06, never scheduled | proposal §2 |
| **§3.1** the third `kind` | reversed by step 3 on flag grounds | proposal §3 |
| **§3.2 / step 4** the produced-column map | dropped — its *deletion pass* deleted nothing | revived **as observation only**: diff the column set around each call. The map is what acceptance 2 needs; the deletion pass that failed is not part of this |
| **Stage D acceptance 5, 6, 8** | unsatisfiable as written | acceptance 2, 3 below — runnable because `needs` exists |
| **T3.11** `area_skipped` leak + ordering dependency | open | the leak ends (it stops being written); the ordering becomes the order check |
| **T3.24** declare preconditions | declined, pushed to stage C | satisfied *without* a dependency graph, exactly as §3.4 argued |

---

## Order of work

Each step is separately verifiable, and the riskiest is last.

| # | step | moves numbers? |
|---|---|---|
| **1** ✅ | `needs` on all 20 existing entries; the order check; the whitelist behind a flag, **off**. **Done 2026-10-13.** Order check passes on the registry order unchanged. Added the `config/datasets.py` path validation §7 promised and never got — tested against the misplaced-KnowQA case, which it catches. **One correction to §3.4, found here:** the whitelist is the union of **four** declarations, not one; the registry is not the raw read's only consumer | no |
| **2** ✅ | Prove the whitelist, then turn it on. **Done 2026-10-13, proved three ways** — see below. Switched on as `build.main(filter_raw_columns=True)`, with `False` kept so the comparison stays repeatable. **38 of 157 raw columns are read; 118 of `all_participants.csv`'s 239 columns go** | **no — verified** |
| **3** ✅ | `kind` renamed `base`→`row`, `group`→`{area, trial}`. **Done 2026-10-13**, and it took two passes — see below. Final state **9 row + 8 area + 3 trial**; both resolvers work on both paths, the order check passes, `model_ready` re-verified bit-identical. Fixed the fallout in `notebooks/builders/data_prep_new_exp.ipynb`, which filtered on `kind == "base"` — and while there, repaired its pre-existing `dcg.FUNCTION_REGISTRY` reference, broken since the registry moved out of `build.py` in the restructure | no |
| **4a** ✅ | **Registry-produced columns — all twenty entries, `row` included.** `_PRODUCES` values carry a grain per column; `split_by_grain` reads `COLUMN_GRAIN` rather than the collected frames' keys. This is where `kind` and grain come apart: the 11 area/trial entries produce ~20 columns at the grain their `kind` implies, and the 9 `row` entries produce trial-grain ones — `text_id`, `is_correct`, `question_len`, the four `*_len`, `n_interest_areas`, `total_words_on_screen` — which `kind: "row"` can now say without lying | **no — rejoin is the check** |
| **4b** ✅ | **Trial-keyed attach outputs.** `ATTACH_PRODUCES` declares the RT/TFD and last-label blocks as trial-grain — true by construction, since `_attach_*` already merges on the trial pair. The largest single block of the 75 | no |
| **4c** ✅ | **Raw trial-level columns.** `RAW_COLUMN_GRAIN` declares the report's own non-per-word columns — `ANSWER_PRESS_NUMBER`, `CONFIRM_FINAL_ANSWER_RT`, `EYE_TRACKED`, `INTEREST_AREA_FIXATION_SEQUENCE`, and the identity/condition columns. Nothing produces these, so no entry can own them and the map is the only honest home | no |
| **4d** ✅ | Write the artifacts; `model_ready` rebuilt from them. **Done 2026-10-13** — see below | **no — verified** |
| **5** ✅ | Move the hardcoded builders into the registry as `trial` entries. **Done 2026-10-13 — for the five that should move, and only those.** `DERIVED_COLS` became five `trial` entries in `features/sequences.py`: values and dtypes identical on full L1, and **7.3× faster** (13.2 s → 1.8 s), because the old block parsed the sequence once per interest-area row and discarded 38 of every 39. The strategy family and preference matching were examined and **ruled out** — see below | no |
| **6** ✅ | Delete the emptied assembler blocks; `model_ready` reads the grains. **Done 2026-10-13** — `build_trial_level_derived_features` deleted (53 lines, zero callers); the five are now *carried* off the frame, not recomputed; `DERIVED_COLS` made canonical in `features/sequences.py` and imported by `modeling/feature_sets.py` instead of defined twice; `load_model_input()` reads the three grains and `build_dataset.py` uses it. **`model_ready` 19,436 × 219 with 0 missing / 0 extra / 0 value / 0 null / 0 dtype differences.** `aux/RT_and_TFD.csv` is **not** retired — see below | no |

### 4a–4c as built, 2026-10-13 ✅

All three declared and verified against **full L1** (760,628 rows, 19,436 trials). 104 columns
carry an explicit grain; everything else is word-grain by omission.

| | `interest_areas` | `areas` | `trials` |
|---|---|---|---|
| before | 121 cols | — | — |
| after 4a (registry) | 77 | 97,180 × 19 | 19,436 × 30 |
| after 4b (attach) | 45 | " | 19,436 × 62 |
| **after 4c (raw)** | **760,628 × 23** | **97,180 × 19** | **19,436 × 84** |
| rows per trial | 39.1 | **5.00** | **1.00** |

**Trial-constant columns still on `interest_areas`: zero.** All 23 that remain are genuinely
per-word, or the three join keys.

Acceptance, run on the full data at each stage:

- **[5] declarations vs observation — PASS.** Every column declared `trial` is constant within
  `(participant_id, TRIAL_INDEX)`, every `area` one within that plus `area_label`. No declaration
  was assigned by observation; observation only checked them.
- **[6] rejoin — PASS.** `interest_areas ⋈ areas ⋈ trials` reproduces the 760,628 × 121 frame with
  **0 value, 0 null and 0 dtype mismatches**.
- **[7] join keys — PASS.** **[2] lossless — PASS.** **[8] computation untouched** — the only diff
  under `src/features/` is Diana's in-flight `nearest_ia_queue` work, byte-identical to before this
  step started.
- `model_ready` re-verified at 19,436 × 219, 0 differing; imports 130/131.

**The exactness of 5.00 and 1.00 is the signal worth keeping.** A declaration that was wrong by
one column would show up as a fractional row count per trial long before the rejoin failed.

### Step 7 — the broadcast tables retired, 2026-10-13 ✅

Not in the original plan: the proposal wrote the grains *alongside* `all_participants.csv`,
`hunters.csv` and `gatherers.csv` and left retiring them for "once every consumer reads the
grains directly". Diana asked for that to be finished.

**The consumer list was bigger than the three modules I had named.** Two `src/` readers, yes —
but also **13 notebooks**, including `answer_corr_prediction.ipynb` and `visualisations.ipynb`.
What I had called "consumers" earlier were mostly *parameter names*: `attention_allocation` and
`answer_rt_comparison` take `all_participants` as an argument and never open a file. The drivers
were the readers.

**Equivalence, established before anything moved.** The rejoined grains against
`all_participants.csv`: 121 shared columns, **0 value, 0 null, 0 dtype differences**. The 118
columns the whitelist drops are raw passthrough, and nothing under `analyses/` reads one — the two
greps that hit were a plot frame's own `selected_answer` column and `screen("paragraph")`.
Confirmed end to end by running two analyses both ways: **18 tables byte-identical**.

**What changed:**

| | |
|---|---|
| `scripts/run_analysis.py` | reads the grains; its hunters/gatherers branch also **fixed** — it read `ds.root / "hunters.csv"`, a path that has not existed since stage G, so that branch raised `FileNotFoundError` |
| `analyses/text_qa_relationship/data.py` | sources are now facets of the grains; cache paths **fixed** — they named `L1_model_ready_{name}_features.csv`, so every `load_features("hunters")` was silently rebuilding from the 1.5 GB table instead of reading the 20 MB cache sitting under the restructured name |
| 13 notebooks | 20 load lines converted to `load_model_input(get_dataset("l1"), facet=...)` |
| `ingest/build.py` | `write_broadcast_table`, **default `False`** — otherwise the next prep run recreates 5.8 GB |
| `config/datasets.py` | `check_dataset_paths` asserts the three grains, not `all_participants` |
| two parked explorations | annotated, not rewired, per *keep, don't delete* |

**Result: 5,847.8 MB deleted**, plus 49.8 MB of superseded `.pre_t321.bak` files and two stale
root logs naming pre-restructure paths. The nine grain artifacts that replace them are **331 MB**.

> **Regenerable, but not from git.** `data/` is gitignored, so these were not recoverable the way
> the old `reports/` trees were — `build_dataset.py --dataset l1` rebuilds them from `data_raw/`,
> and `write_broadcast_table=True` restores the broadcast form specifically.

### Step 6 — one thing deliberately left alone, 2026-10-13

The step said "`aux/RT_and_TFD.csv` goes". **It has not gone**, and that is a decision rather than
an oversight.

Its 30 columns now live in `trials.csv`, so it is redundant *as an input* — and a grep confirms
it has **no readers at all**: every hit is a writer or a path constant. But it is still written by
the answer pipeline and by KnowQA's, and "nothing in `src/` reads it" is not the same as "nobody
opens it". It is an auxiliary table in `data/`, which puts its removal closer to a data decision
than a code one.

So the write stays, the duplication is recorded here, and retiring it is a one-line change in
`ingest/build.py` whenever Diana wants it. Nothing depends on that happening.

### Step 5 — two families that do *not* move, 2026-10-13

The step as written said "move the hardcoded builders into the registry". Two of the three
should not, and finding out why is the useful part.

**The strategy / dominance family — ruled out on leakage grounds.**
`build_trial_level_pattern_features` takes `scope_df` / `scope_by`, threaded through
`build_trial_level_model_df` so `cross_validation.py` can rebuild per regime. That is the
mechanism **T3.21's `within_group` default depends on**. A registry entry runs once during prep,
over the whole dataset — which is the `global` scope, *the leaking one*. Registering it would
silently undo a decision taken specifically to avoid a leak, and nothing would fail while it did.

> **The rule that generalises:** anything accumulated over a participant's trials stays where its
> scope can still be chosen. A registry entry has one scope, fixed at prep time.

Diana, 2026-10-13: *"lets keep it to its own analysis folder for now."* Recorded in
`features/strategies.py`'s docstring, where someone proposing to register it will hit it.
*(The file itself stays in `features/` — `features/build.py` imports it, and `features/` may not
import `analyses/`. If moving it is meant literally, that is a layering question to settle first.)*

**Preference matching — ruled out, and a dead family removed with it.**
Diana, 2026-10-13: *"I don't want pref matching to ever become features."* It is live as an
*analysis* — `analyses/correctness_associations/plots.py` calls `compute_trial_matching` directly,
once per metric and direction — and it is a parameterised family rather than one feature.

Two things implied otherwise and are gone: a **dead import** in `features/build.py` (imported,
never called), and a line in `get_derived_feature_cols` collecting `pref_matching__*` columns that
had **never returned anything**, because nothing produces that prefix and `model_ready.csv` has no
such column among its 219. That is the T2.4 shape — a feature family that is silently always
empty. Removing it changed nothing: the function returns the same five columns.

### 4d as built, 2026-10-13 ✅

**Nine artifacts, not twelve.** Three grains × three populations. The twelfth was never real:
`areas_paragraph` belongs to the paragraph pipeline, which `ingest/build.py` does not run, and
`model_ready` sits in `features/`, not `interim/`.

Produced by splitting the existing `all_participants.csv` rather than by a fresh prep run —
the split is a projection of data already on disk, and the full two-hour rebuild would have
produced byte-identical artifacts from byte-identical input. `build.main(dataset=ds)` now writes
them as part of prep, so the next real run produces them natively.

| | |
|---|---|
| `interest_areas.csv` | 760,628 × 23 · 100.9 MB |
| `areas_answers.csv` | 97,180 × 19 · 20.9 MB |
| `trials.csv` | 19,436 × 84 · 43.3 MB |
| + `__hunters` / `__gatherers` of each | split integrity verified: the two halves sum to the parent at every grain |

**Acceptance — PASS.** `model_ready` rebuilt from the three written tables is **19,436 × 219 with
0 missing, 0 extra, 0 value, 0 null and 0 dtype differences** against the one on disk.

**Two numbers worth keeping:**

- **5,847.7 MB → 330.2 MB, 17.7× smaller.** `all_participants.csv` + `hunters.csv` +
  `gatherers.csv` against the nine grain artifacts.
- **38 s → 3 s** to read. Rejoining the three to reproduce the broadcast frame takes 1 s more, so
  even the worst case — a consumer that wants everything — is faster than reading the old table.

**Nothing was removed.** `all_participants.csv`, `hunters.csv` and `gatherers.csv` are untouched
and still authoritative; the grains are written alongside. Retiring them is step 6, once every
consumer reads the grains directly.

**One bug, found by writing rather than by reasoning.** The split writer derived the facet column
name from merge suffixes, which is right when the split column is absent from a grain and wrong
when it is present — and `question_preview` is itself trial-grain, so it is present on `trials`
and absent from the other two. It crashed on the first write. Fixed by renaming the facet to a
name that cannot collide, so the rule does not have to be right in two cases.

**Why 4a–4c are separate.** Each declares a different population from a different owner, and each
can be verified on its own by the same rejoin test — split, rejoin, compare to the broadcast frame
including nulls and dtypes. Doing them together would mean one large diff whose failure could be
in any of three declarations. The column budget, measured 2026-10-13, is:

| | columns |
|---|---|
| after 4a — trial-constant residue on `interest_areas` | 75 |
| ...of which **4b** (RT/TFD + last-label) accounts for | the largest block |
| ...and **4c** (raw trial-level + `row`-feature trial outputs) the rest | |
| genuinely word-varying, which stays | **18** |
| area-constant | 2 |
| join keys, carried by every grain | 2 |
| **total before the split** | **97** |

**Step 4 is the one to be careful with**, and also the one with the cleanest test: every value
already exists, so the only question is whether it survives being stored at a different grain.

### Step 2's proof, as run 2026-10-13

The proposal asked for one prep run with `usecols` and one without. That is two hours and
overwrites `interim/`, so it was run as **three cheaper checks that between them cover more**
than the single comparison would have:

| | check | result |
|---|---|---|
| 1 | All 20 registry features on a 2,496-trial sample read both ways — same fixation subset and same pupil-stats frame handed to both, so the IA column set is the only difference | 89 shared columns, **0 differing** |
| 2 | `model_ready` built from a 121-column interest-area frame vs the full 239, **all 19,436 trials**, both in memory | 19,436 × 219 both; 217 columns compared; **max abs diff 0.0** |
| 3 | Of the 118 columns dropped from `all_participants.csv`, how many are named in `analyses/`, `features/` or `modeling/` | **none** — the two greps that hit were a plot frame's own `selected_answer` and `screen("paragraph")` |

Check 2 is the one that matters, and it is worth re-running after any change to a `needs` list:
it is the cheap form of the full comparison and catches the same mistake.

**Two things the proof turned up that are worth keeping:**

- **The order check earns itself immediately.** Running a subset without
  `create_fixation_sequence_tags` failed at run time with `KeyError: 'fix_by_label'` from deep
  inside a group feature. `check_registry_order` predicted exactly that, statically, naming both
  missing columns and the entry that wanted them. That is the T3.11 failure mode, caught where
  the mistake is.
- **`is_correct` is both a raw column and a produced one** — the only such column of the 157.
  OneStop ships its own `is_correct` (bool) alongside `correct_answer_position`; `add_is_correct`
  ignores it and derives the target from `selected == correct` as an int.

  **Verified 2026-10-13 on all 760,628 rows / 19,436 trials: the two agree everywhere, zero
  disagreements.** So the whitelist dropping the raw copy is harmless, and deriving rather than
  trusting is the better behaviour anyway — the derivation carries T3.17's null assertion, which
  a shipped boolean cannot: a null position would silently read as `False` there and score a data
  fault as a wrong answer.

  What remains true is narrow: §3.4's claim that intersecting with the header "sorts raw from
  produced by itself" has exactly one column where both are true. It misled **my own test**,
  which classified it as raw-not-whitelisted and dropped it, producing a `KeyError` I briefly
  took for a pipeline fault. Decided 2026-10-13: leave it dropped, no assertion added — the
  check has been run once and L1's corpus will not drift.

### Step 3 took two passes, and the second one matters

The first pass was a blanket `group` → `area` rename. It was wrong twice, and both are recorded
because the second is the interesting one.

**A regression I introduced.** The rename updated `"kind": "group"` and `== "group"` but missed
`!= "group"` in the *explicit-list* branch of both resolvers — so `resolve_base_functions(names)`
and `resolve_group_functions(names)` raised *"not registered as a base feature"* for every name.
That is precisely the path **KnowQA** and the pilots notebook use (they pass a filtered list to
skip `add_answer_text_columns`), so Study 2's build was broken for about twenty minutes. Caught by
reproducing it rather than by reading.

**A mislabelling the rename created.** Three of the eleven are not area-grain at all —
`create_last_area_and_location_visited`, `create_fixation_sequence_tags` and
`create_simplified_fixation_tags` join on the trial pair alone. `join_columns` had encoded the
real grain all along; `kind` was assigned by hand and a blanket rename flattened the distinction.

**Why it would have been expensive.** Step 4 splits the artifacts *by this field*. Three
trial-grain blocks would have been written into `areas_answers.csv` — at the wrong grain, silently,
with the right values. `check_kinds_match_join_columns()` now asserts the two agree, so it cannot
come back.

**A method note, because it nearly produced a false alarm.** Comparing a freshly built frame
against a **saved CSV** showed 127 pupil `_z` columns "differing" — all of them float noise from
the save/load round trip, identical to six decimals. The control (full frame vs the same saved
CSV) showed the same 127, which is what proved it was the round trip and not the whitelist.
**Compare in memory against a control, not against a CSV**, and on a tolerance rather than exact
equality.

---

## Acceptance test

Written so that every line can actually be run, which is the thing stage D's could not claim.

1. **`model_ready.csv` is bit-identical**, L1 (19,436 × 219) and KnowQA (870 trials): shape,
   column order, numeric max-abs-diff `0.0`, zero non-numeric cells differing.
2. **Every column is claimed exactly once.** Each column of each artifact is either produced by
   exactly one registry entry (observed by diffing the column set around its call), an identity
   column, or a raw passthrough in the whitelist. *(Inherits stage D 5 and 6.)*
3. **The read-time filter is proved, not trusted:** one full prep run with `usecols` and one
   without, bit-identical output. *(Inherits stage D 8.)*
4. **The order check passes** on the existing registry order, unchanged — it verifies, it does
   not reorder.
5. **The grain check passes**, in both directions:
   - no `trial` entry reads an IA-grain column;
   - **every declaration is checked against observation.** For each column `COLUMN_GRAIN` calls
     `trial`-grain, assert it really is constant within `(participant_id, TRIAL_INDEX)` on both
     studies, and likewise for `area`. Observation never *assigns* the grain — it only catches a
     declaration that is wrong. A column observed constant but undeclared is reported, not moved.
6. **Rejoining reproduces the broadcast frame exactly** — `interest_areas ⋈ areas ⋈ trials`
   against the pre-split frame, compared on **shape, column set, values, null positions and
   dtypes**. Nulls and dtypes are named because a `.first()` collapse and a merge can both
   silently change them: an all-null trial column survives a groupby differently from a mixed
   one, and an int column that acquires a null becomes float. This is the test each of 4a–4c
   passes on its own.
7. **Join keys are present at every grain.** `interest_areas`, `areas` and `trials` each carry
   `(participant_id, TRIAL_INDEX)`, and `areas` additionally carries `area_label`. Asserted
   rather than assumed — the rejoin cannot be checked if a key is missing, so this fails first
   and says so.
8. **Computation is untouched.** No feature function's body changes in step 4; the diff is
   declarations and the splitter. Verified by `git diff` over `src/features/` showing no change
   inside any `create_*` / `add_*` body.
9. **Imports:** 130/131, same single parked `pymer4` failure.
10. **Layering** unchanged, including the documented `ingest/` → `features/` exception.
11. **`reports/` unchanged** — verified with `git diff` on the tracked tree, **not** a
   filename-keyed snapshot, which reports 0/N when an output is renamed. That has happened once.

**The one deliberate difference:** `area_skipped` stops being written to `interest_areas`.
**Verified safe 2026-10-13:** it is written and consumed entirely inside `area_metrics.py` — the
per-word `IA_DWELL_TIME == 0` indicator whose mean *is* `skip_rate` — and nothing outside that
module reads it. The only other occurrences in the repo are the column name appearing inside two
notebooks' **stored output**, not their code. So it is an intermediate that leaks (T3.11), and
ending the leak costs nothing.

---

## Explicitly not in this

- **Merging the two screens' pipelines.** T6.1 and `config/screens.py` settled that they stay
  separate artifacts. This changes their grain, not their number.
- **Writing to `data_raw/` in any way.** Not merely out of scope — a standing guarantee, §6. The
  whitelist is read-time, so all 157 raw columns stay on disk and widening `needs` is the whole
  of an undo.
- **A participant-grain artifact.** Deferred by decision; the case for one (pupil baselines,
  dominance, and anything else accumulated per person) is recorded in
  `todo_after_restructure.md` §2.
- **The `Screen` interface question** — whether `drop_leading_question` and
  `sequence_vocabularies` belong on that record. Open by decision, 2026-10-13, pending a call-site
  read; see `todo_after_restructure.md` §2.
- **Re-running the paper's figures.** That is Diana's manual check (T4.0, T4.2).

---

## Decided 2026-10-13

Four questions this draft raised, answered by Diana the same day.

| | |
|---|---|
| **The contrast columns** | **Stay in `model_ready`** — they are a reshape, not a measurement (§5) |
| **`DERIVED_COLS`** | **Five separate registry entries**, not one. Each of `seq_len`, `has_xyx`, `has_xyxy`, `longest_alt_answer_run`, `trial_mean_dwell` is its own quantity and declares its own `needs`. They share one cached parse of the sequence rather than parsing five times — the saving that made "one entry" tempting, without the dishonesty of calling five quantities one feature |
| **The group splits** | **Keep producing them**, at every grain (§7) |
| **A participant grain** | **Not now.** Three grains, and participant-level aggregates stay stored per trial with their `n_*` counts (T3.21). Unifying them later is worth doing — recorded in `todo_after_restructure.md` §2 rather than folded in here, because doing it in the same pass would mean two unsettled designs moving at once |
| **How many raw columns the filter drops** | **Not a question worth answering in advance** (Diana: *"what does it matter how much it will drop — we shall see"*). The number was only ever a size argument; the structural one does not depend on it, and step 2's bit-identical check is what makes a wrong `needs` list a failed check rather than a wrong number. The real count falls out of step 1 |
| **Split files at every grain, or only the large ones** | **Every grain.** Twelve artifacts, not four |
| **One `areas` table with a `screen` column** | **No — paragraphs and QA stay separate**, as T6.1 settled. Not revisited |

---

## What I am least sure about

**Nothing structural — all three open questions were ruled on 2026-10-13** (see above).

What remains is execution risk, concentrated in one place: **`needs` has to be right for twenty
entries, and getting one wrong drops a column.** That is why the filter ships *off* in step 1 and
is not switched on until step 2 has proved the two runs byte-identical. Until that check passes,
treat every `needs` list as unverified.
