# Stage D proposal — one feature generator, one way to select features

**Status: 2026-10-06 — steps 1 and 6 are ✅ done; steps 2–5 and 7 await approval.**

| step | state |
|---|---|
| **1** — the §5.4 layering fix | ✅ done. `features/build.py` → `ingest/{base_features,registry}.py`. Registry order byte-identical, `all_participants.csv` bit-identical |
| **5** — `include_*` → `Dataset` | ✅ done 2026-10-07. The last flag is gone; `dataset=` replaces it, and the paragraph warning became a raise |
| **4** — the produced-column map | ❌ **audited 2026-10-07: do not build**, and the deletion pass that replaced it deleted **nothing** — the audit's own "dead" counts were wrong. What landed: the `LAST_*` vocabulary moved to its producer, killing the `common/` → `answer_correctness/` upward import |
| **3** — the trial-level registry kind | ✅ done 2026-10-07, **by deleting the flags instead.** 8 booleans → 1; `kind: "trial"` not added, and should not be. Numbers unchanged |
| **2c** — the collapsed-area-sequence step (§3.9) | ✅ done 2026-10-07. The answer screen now uses the primitives the paragraph screen already used. Numbers unchanged |
| **2** — `features/build.py` | ✅ done 2026-10-07. 13 functions + 5 column vocabularies moved; `common/feature_builders.py` deleted. Numbers unchanged |
| **2b** — the `Screen` record (§3.8) | ✅ done 2026-10-06. Four duplicated orchestration pieces became one each. Numbers unchanged |
| **6** — one fixation-to-area rule, RT off the clicks table | ✅ done, **out of order, on Diana's instruction** (*"I don't care that it changes numbers, I want this whole mess unified"*). Two bugs found; see §3.6. Logged in `findings.md` |
| 2, 3, 4, 5, 7 | not started |

> **The reordering has a consequence worth stating.** Step 6 was placed last precisely so that
> everything before it could be proved number-neutral first. Running it second means **steps 2–5
> must now prove neutrality against a post-step-6 baseline**, not against the tables on disk at
> the start of the stage. The baseline to compare against is whatever the next full prep rerun
> produces — and **that rerun has not happened**: `data/L1_based_data/all_participants.csv`
> (09-23) still carries the old RT columns, so L1's model-ready table is stale by design until
> Diana reruns prep. KnowQA *was* rebuilt and is current.

All numbers below were measured against the code on 2026-10-06.

---

## What this stage is for

> **Diana, 2026-10-06:** *"the main defining characteristic of 'derived' features is that they
> are later additions after the generator was already written — this is a type of redundant
> separation this restructure aims to remove. I want the feature generation (all of it)
> unified in structure and easily modifiable as far as feature selection."*

Two goals, and the second is the harder one:

1. **One mechanism** for generating every feature, at every grain.
2. **Feature selection that is easy to change** — one place to say which features exist, one
   place to say which a model uses, and no third place that has to agree with the first two.

The map already points this way: the target tree gives `features/build.py` the job *"assemble
the model-ready trial table"*. What it does not say is how, and the "how" is the whole stage.

---

## The diagnosis, measured

### 1. "Derived" is a date, not a category — Diana is right, and it is demonstrable

`DERIVED_COLS` is five columns of 219:

```
seq_len · has_xyx · has_xyxy · longest_alt_answer_run · trial_mean_dwell
```

They are **sequence statistics**. Their implementations sit in `features/sequences.py` — the
same module as `create_simplified_fixation_tags` and `create_simplified_visit_counts`, which
are registry entries. Same kind of quantity, same source file, **two different mechanisms**,
and the only thing separating them is that `FUNCTION_REGISTRY` already existed when the first
pair was written and had stopped being the obvious place by the time the second arrived.

The old folder was called `derived/` for the same reason. Stage C dissolved the folder; the
*mechanism* split it was named after is still there.

### 2. Two mechanisms do one job

|  | IA-level (`ingest/build.py` → `all_participants.csv`) | trial-level (`model_data.py` → `L1_model_ready_all_features.csv`) |
|---|---|---|
| mechanism | `FUNCTION_REGISTRY`: 20 entries, `name → {callable, default_kwargs, kind}` | **8 hardcoded `include_*` booleans** in a function signature |
| select by | a list of names | editing the signature, or passing flags at every call site |
| adding one | one dict entry | a parameter, an `if` block, a column list in `feature_groups.py`, and every call site |
| per-dataset config | `Dataset.skip_base_features` (stage C) | nothing — **the flags are copy-pasted** |

**The flag block is repeated at 7 call sites** (`knowqa.py`, `cross_validation.py`,
`participant_level.py`, `run_model_bundles.py` ×3, `model_data.py`'s own three `make_*_dataset`
helpers). That is precisely the per-dataset repetition the `Dataset` record was built to remove
on the IA side, alive and well on the trial side. `include_paragraph_features=False` in
`knowqa.build_features` **is `has_paragraph` spelled by hand** — the stage C proposal flagged
this and deferred it; it belongs here.

### 3. Feature selection has six surfaces, and two of them disagree by construction

| # | surface | size | what it decides |
|---|---|---|---|
| 1 | `FUNCTION_REGISTRY` | 20 entries | which IA features get **generated** |
| 2 | the `include_*` booleans | 8 flags | which trial features get **generated** |
| 3 | `answer_correctness/feature_groups.py` | **35 named lists** | which columns a model **uses** |
| 4 | `common/feature_specs.py` | 6 lists + 6 `get_*_feature_cols(df)` predicates | same, again |
| 5 | `answer_RTs/model_data.py` | 5 more `get_*_feature_cols(df)` predicates | same, for the RT model |
| 6 | `configs/feature_sets/*.json` | 5 files | explicit curated column lists |

Plus `generate_column_options.py` (1,003 lines) which *generates combinations* over #3.

**#3 and #4 are measurably redundant.** Verified 2026-10-06:

```
feature_groups.DERIVED_COLS             == feature_specs.DERIVED_BASE_FEATURES   True
feature_groups.PATTERN_COLS             == feature_specs.PATTERN_FEATURE_COLS    True
feature_groups.RT_TFD_CONTRAST_SUFFIXES == feature_specs.RT_TFD_CONTRAST_SUFFIXES True
feature_groups.RT_TFD_PARAGRAPH_REGIONS == feature_specs.RT_TFD_PARAGRAPH_REGIONS True
```

and `feature_specs.py` imports `feature_groups` **upward** (`common/` reaching into
`answer_correctness/`), which its own header comment already flags as temporary and points at
this stage to fix.

### 4. Two different things are both called "a feature set"

This is the conceptual knot under #3–#6, and naming it is most of the fix:

| | what it is | changes when | example |
|---|---|---|---|
| **(a) what the generator produces** | mechanical — a consequence of which metrics × which areas exist | you add a feature | `PER_QUESTION_COLS = [f"{m}__question" for m in METRIC_COLUMNS]` |
| **(b) what a model consumes** | a **scientific choice**, and sometimes a *result* | you decide something | `SELECT_1_COLS` — commented *"manually curated"*; it is the output of feature selection, not an input to generation |

`feature_groups.py` holds both, interleaved, in one 35-list file. Half of it should fall out of
the registry automatically; the other half must stay hand-written, because it encodes decisions.
Today a new feature means editing both halves by hand and hoping they agree.

---

## The proposal

### 3.1 One registry, three kinds

**Keep `FUNCTION_REGISTRY` and extend it.** `conventions.md` says it "is enough machinery for
this project" — so the answer is one more `kind`, not a second system.

| `kind` | consumes | produces | how the runner merges it |
|---|---|---|---|
| `row` *(today `"base"`)* | the IA frame | IA columns, in place | — |
| `group` | the IA frame | an area- or trial-keyed table | broadcast back onto IA rows |
| **`trial`** *(new)* | the IA frame | a trial-keyed table | merged onto the trial core |

The pipeline becomes one pass per kind:

```
row    -> all_participants.csv  (IA grain)
group  ->        "
trial  -> L1_model_ready_all_features.csv  (trial grain)
```

**This is exactly what happens today.** What changes is that the third pass is a registry walk
instead of eight `if include_x:` blocks, so the trial-level features become declarable,
nameable, skippable and listable the same way the IA ones already are.

**Order stays a plain list.** T3.24 is declined (Diana, 2026-10-06) — no dependency graph, no
topological sort. A default runner executes them in the order they appear, as now.

### 3.2 ~~What each entry produces is observed, not declared~~ — ❌ **audited and dropped, 2026-10-07**

**This step should not be built.** Audited at Diana's instruction, the same way step 3 was, and its
premise does not hold.

**The premise was:** the mechanical half of `feature_groups.py` is hand-written and should instead
be derived from a map of which builder produced which column.

**Measured. It is already derived.** Of the 35 named sets in `feature_groups.py`:

| | count |
|---|---|
| already built by a comprehension or concatenation | **23** |
| hand-written literals | 10 |
| other | 2 |

And the 10 hand-written ones are not the kind a produced-column map could supply. They are either
**vocabulary** — `PARAGRAPH_SPANS`, `RT_TFD_PARAGRAPH_REGIONS`, `RT_TFD_VARIANTS`, the `LAST_*`
one-hot names: facts about what the screen contains — or **decisions**, namely `SELECT_1_COLS`,
which its own comment calls *"manually curated"*.

**And the observing already exists, done better.** `feature_specs.get_*_feature_cols(df)` select
columns by **pattern-matching the frame it is handed** (`<metric>__<area>`, the five contrast
suffixes), filtered to what is actually present. That is "observe, don't declare" — implemented,
and at a better moment: when you have a frame, rather than requiring a build artifact to exist.
A persisted map would make importing `feature_groups` depend on a pipeline having been run, which
is exactly wrong for a repo that must run from scratch on a clean machine.

#### What the audit found instead, which is real

**1. Nine of the 35 sets have no consumer anywhere** in `src/` or `notebooks/`:

```
ALL_FEATURES_NO_LAST (94 cols)   PATTERN_COLS            PATTERN_COLS_WITH_INTERACTIONS
METRIC_COLUMNS                   PATTERN_DISTANCE_COLS   PATTERN_INTERACTION_COLS
RT_TFD_NON_ANSWER_REGIONS        RT_TFD_PER_ANSWER_REGIONS   RT_TFD_VARIANTS
```

*(Correction to something said earlier the same day: I cited `ALL_FEATURES_NO_LAST` as "94 cols,
24 RT-derived" when reporting what step 6 left stale. The set is dead, so nothing was stale
through it. The conclusion it supported is unchanged — the headline model still has 0 RT features.)*

**2. Seventeen of the 26 live sets have exactly one consumer: `generate_column_options.py`.** So
`feature_groups.py` is not a shared feature-set registry. It is overwhelmingly one 1,003-line
script's private vocabulary, with about four genuinely shared names
(`SELECT_1_COLS`, `LAST_CONFIRM_COMPACT`, `PARAGRAPH_SPANS`, `PARAGRAPH_BASED`).

**3. `feature_specs.py` is 178 lines holding one live function.** Of its 12 exports: 4 constants
have no consumer, 2 more are consumed only by `feature_groups.py` — which defines its own
identical copies, so those are name collisions rather than uses — and 3 of the 6 predicates have
no consumer. `get_area_feature_cols` and `get_derived_feature_cols` are called only by the two
`make_*_dataset` helpers, which themselves have no callers. **Only `get_full_feature_cols` is
load-bearing**, with 4 consumers.

#### The deletion pass, 2026-10-07 — and the audit above was wrong twice

Diana approved a deletion pass. Doing it found that **almost nothing should be deleted**, because
both "unused" counts above were measured against *external* references only.

**Correction 1 — `feature_specs.py`.** "11 of 12 exports dead" is wrong. All six constants feed the
predicates, and all five predicates are called by `get_full_feature_cols`. The module is fully
live; it simply has one public entry point and eleven internal details. Nothing to delete.

**Correction 2 — `feature_groups.py`.** "9 of 35 sets dead" is wrong. Six of those nine are
internal dependencies of live sets: `ALL_FEATURES_NO_LAST` builds `ALL_FEATURES`, `METRIC_COLUMNS`
builds `AREA_COLS`, and so on. Deleting them would have broken sets that *are* used. Computing the
transitive closure instead gives **three** candidates.

**And those three should stay too.** `PATTERN_INTERACTION_COLS`, `PATTERN_COLS_WITH_INTERACTIONS`
and `PATTERN_DISTANCE_COLS` are each documented **opt-in affordances**, with comments that say so:
*"Kept separate from PATTERN_COLS / the aggregate sets so they are only added on request, e.g.
`feature_cols = FG.GENERAL_FEATURES + FG.PATTERN_INTERACTION_COLS`"*, and for the distance set,
*"kept as a separate opt-in group -- typically swapped in instead of breaks_pattern"*. They are
unreferenced **because they exist to be typed at a call site**. `conventions.md` says keep, not
delete, and each already explains its own absence of callers.

> **The lesson, and it has now cost time twice in one session** (see also step 2's constant-chasing):
> "is this used?" must be answered against the **transitive closure**, including the defining module
> itself. Measuring external references alone makes live code look dead.

#### What was actually done: the upward import, which is the real section 3.3 item

The eight `LAST_*` one-hot column lists moved from `answer_correctness/feature_groups.py` to
`features/build.py`, beside `build_trial_level_last_visited_features`, **which is the function that
produces those columns**. That is section 3.3's distinction applied: output vocabulary lives with
the generator, not with the module that picks among it.

This removes the inversion. `common/feature_specs.py` needed exactly one name from
`feature_groups` -- `LAST_ALL` -- and reached up into a modelling package for it; its own header
called that *"deliberate and temporary"*. Both it and `feature_groups` now read these from
`features/build`, which is downstream for both, and **`common/` no longer imports from
`answer_correctness/` at all**.

`feature_groups` re-exports the eight names so the seven call sites that say `FG.LAST_*` are
unaffected -- verified the objects are identical, `SELECT_1_COLS + LAST_CONFIRM_COMPACT` is still
12 features, `ALL_FEATURES` still 98, `GENERAL_FEATURES` still 70. KnowQA pipeline bit-identical
(33,830 x 420, max abs diff 0.0). 97 of 98 modules import.

**Still open, and structural rather than a deletion:** 17 of the 26 live sets in `feature_groups.py`
have exactly one consumer, `generate_column_options.py`. That file is one script's private
vocabulary wearing the name of a shared registry. Moving them into that script is a stage E/F
question, not this one.

### 3.3 Split "what exists" from "what a model uses"

| | lives in | holds |
|---|---|---|
| **(a) generated** | `features/registry.py` | the registry; `produces`; the derived column families |
| **(b) chosen** | `modeling/feature_sets.py` | `SELECT_1_COLS`, `GENERAL_FEATURES`, the opt-in splits — the decisions, hand-written, each with its *why* |

`feature_groups.py` + `feature_specs.py` merge into (b), which is what map §5.1 already says —
**with the refinement that their mechanical half leaves for (a) rather than moving.** The upward
import disappears because `common/` does.

The five `configs/feature_sets/*.json` stay exactly as they are: they are curated column lists,
already inputs rather than results, already outside `src/`.

### 3.4 `needs`, and a read-time column filter — ✅ **Diana, 2026-10-06**

> *"I want to keep the ones that any code uses — this should be easy as no dynamic generation is
> involved, we just declare which raw cols are needed for the feature builder."*

**One declaration, two uses.** Each registry entry gains `needs`: every column it reads, written
out as literal names. Nothing else is declared.

```python
"create_mean_pupil_size_metrics": {
    "callable": create_mean_pupil_size_metrics,
    "kind": "group",
    "needs": ["IA_AVERAGE_FIX_PUPIL_SIZE", "IA_MAX_FIX_PUPIL_SIZE", "IA_MIN_FIX_PUPIL_SIZE",
              "IA_AVERAGE_FIX_PUPIL_SIZE_z", "IA_MAX_FIX_PUPIL_SIZE_z", "IA_MIN_FIX_PUPIL_SIZE_z"],
},
```

**Use 1 — the raw whitelist, and the split is automatic:**

```python
whitelist = union(entry["needs"]) & set(raw_report_header)   # -> read_csv(usecols=...)
```

Intersecting with the report header sorts raw from produced by itself, so nothing has to be
hand-classified. A produced column like `..._z` simply is not in the header and drops out.

**Use 2 — the order check**, walking the registry in its existing order:

```python
available = set(whitelist)
for name, entry in FUNCTION_REGISTRY.items():
    missing = set(entry["needs"]) - available
    if missing:
        raise KeyError(f"{name} needs {sorted(missing)}, which nothing before it produces")
    before = set(df.columns)
    df = run(entry, df)
    available |= set(df.columns) - before     # observed, not declared
```

This turns T3.11's undeclared ordering dependency into a loud failure **while leaving the order
exactly as it is** — it verifies the list rather than computing one, so T3.24 stays declined.

**On dynamic names — an earlier draft of this section overstated the problem twice.**
`features/pupil.py` builds its output names as `f"{col}_z"`, and the first draft treated that as a
hazard for both uses. It is neither:

* it cannot affect the **whitelist**, because no raw report column ends in `_z` — verified
  2026-10-06: every dynamically-constructed column name in live `src/` is a `_z` name, and the
  337-column raw header contains none;
* it cannot affect **`needs`** either, because writing `"IA_MAX_FIX_PUPIL_SIZE_z"` in a list is
  easy no matter how the code assembles that string at runtime. Declaring a name and constructing
  a name are different problems.

The one place it *would* bite is auto-deriving what each entry **produces** — and the fix there is
the line above: **observe it** by diffing the column set around the call, rather than declaring it.
That removes the problem instead of working around it, and it keeps §3.2's column families honest
for free, since an observed map cannot drift from the code.

**The mechanism already exists in this codebase**, which is the strongest argument for it.
`features/paragraph/spans.py:100` declares `IA_USECOLS` — 13 columns — and passes it to
`read_csv(usecols=...)` for the paragraph report. What `needs` changes is *where the declaration
lives*: today it is one module-level list carrying a hand-written note (`# needed by
compute_reading_times for the span-based RT`) to track which function wanted what. Attaching the
declaration to the function removes the need for that note, and makes the answer-screen path work
the way the paragraph path already does.

**Measured, KnowQA, 2026-10-06:** the raw IA report has **337 columns**, **329** survive into
`all_participants.csv`, and **35** are referenced by live `src/` code — so the filter drops **294
at read time**. `readers.py` loads with `engine="python"`, so this is a parse-time and memory win
as much as a disk one.

**Two things to get right:**

1. **The whitelist is per report, not global.** `features/paragraph/spans.py` opens
   `IA_PARAGRAPH_PATH` with its own `read_csv`, and the vendored EyeBench extractor reads many
   columns that are dead on the answer side (`IA_REGRESSION_*`, `IA_SKIP`, `IA_RUN_COUNT`,
   `IA_SELECTIVE_REGRESSION_PATH_DURATION`). Applying the answer-screen list to the paragraph path
   would break the vendored extractor quietly.
2. **Prove the filter, do not trust it** — one pipeline run with `usecols` and one without,
   bit-identical output. Cheap, since it is the existing acceptance harness, and it is what makes
   "we dropped 294 columns" a verified statement rather than a hopeful one.

**This changes `all_participants.csv`'s schema**, which two comments in `area_metrics.py` currently
treat as something to preserve. Deliberate break; those comments get updated with it.

### 3.5 Word grain stays, and dropping columns is reversible

✅ **Diana, 2026-10-06** on both halves:

> *"Removing columns we don't yet use is fine. I'll stop removing a column if I come up with
> something new and rerun."* … *"I guess we can keep word level for future redundancy even though
> it's very little."*

So the artifact keeps one row per word, and the filter drops what nothing reads. The two settle
together because **the filter is a one-line change and the raw data is untouched**: adding a column
back is editing one `needs` list and re-running prep. `data_raw/` keeps all 337 raw columns
regardless.

Worth stating once, for the record rather than as an objection: after filtering, word grain
preserves the **17** columns that live code reads, all of which have already been consumed by
aggregation before the file is written. The 222 unused ones — regressions, spillover, run
structure, landing positions — are what a future *word-level* reading analysis would want, and they
live in `data_raw/`, not here. So word grain in the artifact is a **convenience** (look at words
without re-running prep) rather than an archive. Naming that now so nobody later reads "we kept
word grain for future analyses" and assumes the columns came with it.

### 3.6 One fixation-to-area rule, and RT off the button-clicks table — ✅ **Diana, 2026-10-06**

> *"I want everywhere to try nearest IA if current isn't present, and I want to end up with RT
> independent of button clicks functionality."*

**The rule, one line, every screen and every consumer:**

```
area_of(fixation) = lookup( CURRENT_FIX_INTEREST_AREA_ID  or  CURRENT_FIX_NEAREST_INTEREST_AREA )
```

#### Why this is a generalisation on the answer screen and a fix on the paragraph screen

Measured over both full fixation reports, 2026-10-06:

| | answers | paragraph |
|---|---|---|
| fixations | 718,207 | 2,400,788 |
| **exact present AND nearest disagrees** | **0** | **0** |
| exact missing, nearest fills it | 76,401 (10.6%) | 31,570 (1.3%) |
| neither available | 54 (0.008%) | 0 |

Because the two columns **never disagree when both are present**, `coalesce(exact, nearest)` is
exactly equivalent to `always nearest`. So:

* **the answer screen's mapping does not change** — it already used `NEAREST_IA`;
* **the paragraph screen gains 31,570 fixations** that its RT currently drops, so paragraph
  RT/TFD will move;
* the 54 answer fixations with neither column stay unmapped under any rule. They must be
  **counted and reported**, not silently dropped — `conventions.md`, "never silently discard data".

This also settles an inconsistency *inside* `spans.py`: `span_visit_counts` already fills from the
nearest IA ("instead of dropping them", per its docstring, *"which is what the answer screen has
always done and the paragraph screen never did"*) while `build_paragraph_rt_tfd` drops. One rule
removes the split.

#### What the rule actually collapsed — and what it did not

The first draft of this section predicted three things would fold together. **One did.**
Corrected 2026-10-06 after implementing it:

| predicted | what happened |
|---|---|
| `compute_run_based_rt` + `compute_run_based_rt_from_fixations` → **one function** | ✅ **done.** 245 lines deleted; one run-based RT for both screens |
| `scan_paragraph_fixations`'s nearest-IA queue becomes **redundant** | ❌ **wrong.** It still runs — the 2026-10-06 rebuild logged *"9,019 trials queued, 9,019 matched"*. The queue feeds `span_visit_counts`, which resolves `INTEREST_AREA_FIXATION_SEQUENCE` — a **per-trial list of ids from the IA report**, not per-fixation rows. `resolve_fixation_area` works on the fixation report, a different shape, so it cannot simply replace the queue. Reworking visit counts onto the fixation report is a real option, but it is a redesign, not a consequence of this rule |
| `span_visit_counts` uses the shared mapped sequence | ❌ **not done**, for the same reason |

So `scan_paragraph_fixations` keeps both its jobs: the streaming pupil baseline **and** the
nearest-IA queue. The ~89 lines it occupies are still genuinely paragraph-specific.

#### RT comes off the button-clicks table

`compute_run_based_rt(button_clicks_df, ...)` is retired. The clicks table was never the *source*:
`ingest/clicks.py:239 extract_fixation_timestamps_with_ia` reads the **fixation report**, extracts
`(timestamp, nearest_IA)` pairs and stores them in the clicks table as `FIXATION_TIMESTAMPS_IA`.
Verified: both routes see identical fixation sets — mean **29.87 per trial**, equal on **all
24,046** trials, none missing either way.

**Retiring it also removes a defect.** The final run's end is currently reconstructed by matching
`IA_LAST_FIXATION_TIME == last_fix_start` back into the IA-level report:

```python
if last_entry is not None and last_entry[0] == timestamps[j]:
    end_proxy = timestamps[j] + last_entry[1]
else:
    end_proxy = timestamps[j]          # the trial's final fixation contributes 0 ms
```

**That fallback fires on 46.2% of trials — 11,103 of 24,044.** Of those, **6,755** are "the nearest
IA has no row in the lookup for that trial", which is a direct consequence of mapping fixations by
*nearest* while looking durations up by *exact* id: the function is inconsistent with itself. The
fixation report carries `CURRENT_FIX_DURATION`, so the unified implementation needs neither the
match nor the fallback.

~~`FIXATION_TIMESTAMPS_IA` then has no consumer left.~~ **Wrong — checked 2026-10-06 before
deleting it.** The column is still read by `clicks.extract_last_fixations_before_clicks`, which
builds `LAST_FIXATIONS_BEFORE_SELECT` / `..._CONFIRM` — i.e. the whole `last_*` feature family.
So `extract_fixation_timestamps_with_ia` **stays**, and so does the column.

What changes is only what Diana asked for: **RT no longer reads the button-clicks table.** The
table keeps its own jobs — selection and confirm events, and the last-visited features. Those use
the same `NEAREST_IA` mapping, so they already follow the §3.6 rule.

#### Implemented 2026-10-06 — and it uncovered **two** bugs, not one

The work was done this session. What it found was more than the plan expected:

**Bug 1 — the answer path lost the final fixation on 46.2% of trials.** As predicted above:
`compute_run_based_rt` reconstructed the last run's end by matching
`IA_LAST_FIXATION_TIME == last_fix_start`, and on failure fell through to `end = last_fix_start`,
so the trial's final fixation contributed **0 ms**. 11,103 of 24,044 trials.

**Bug 2 — `compute_run_based_rt_from_fixations` truncated the final run of *every* trial.**
Found only by hand-tracing a single trial against the written definition. The run aggregation used

```python
next_start=("_next_start", "last")     # pandas .agg("last") SKIPS NaN
```

`_next_start` is NaN exactly on a trial's last fixation — that NaN is the signal meaning "nothing
follows, so end this run at `last_start + last_duration`". Because `.agg("last")` skips it, the
aggregation returned the *previous* fixation's `_next_start`, `fillna` never fired, and the run
ended at its own last fixation's **start**. **This is the function the paragraph RT has used since
T6.1**, so every paragraph trial's final run was short by its last fixation's duration.

Fixed by reading the run's genuinely last row (`groupby(...).tail(1)`) instead of `.agg("last")`,
which also covers `last_duration` — a NaN duration had the same failure mode.

> **The two bugs were partly cancelling**, which is why the first measurement of this change came
> out *smaller* (−0.99%) and contradicted the mechanism. Taking that contradiction seriously rather
> than averaging over it is what surfaced bug 2. Worth remembering: the plan above said "do not
> land this with the direction unexplained", and that instruction earned its place.

**Verified against the definition by hand.** On the worst-differing trial (`l32_390` / 20),
arithmetic over the 13 runs gives `838 + 756 + 747 + 2128 = 4469 ms` for `answer_A`. The fixed
implementation returns **4469.0**. Before the fix it returned 2496.

**Measured, L1 answer screen, old vs new on identical data (19,436 trials):**

| column | trials differing | mean old | mean new |
|---|---|---|---|
| `RT_pure_question` | 167 | 416.2 | 417.1 |
| `RT_pure_answer_A` | 3,805 | 2012.9 | 2049.6 |
| `RT_pure_answer_B` | 925 | 1569.1 | 1576.1 |
| `RT_pure_answer_C` | 817 | 1410.8 | 1416.3 |
| `RT_pure_answer_D` | 709 | 1213.0 | 1217.3 |

**Trial total run-based RT: 6621.9 → 6676.5 ms (+0.82%)** — larger, as the mechanism predicts,
because the final fixation is now counted. `TimeSinceOffset_*` and `TFD_*` are **unchanged**: they
come from `compute_reading_times` over the IA table, which this did not touch.

**What landed in the code**

| | |
|---|---|
| `features/reading_times.py` | new `resolve_fixation_area` (the one rule, with counts); new `load_fixations` (either screen, dtype-preserving so KnowQA's string TRIAL_INDEX survives); `load_paragraph_fixations` is now a thin wrapper; `compute_run_based_rt` and `_parse_fixation_pairs` **deleted** (245 lines); `build_rt_and_tfd` takes `fixations_path` instead of `button_clicks_path` |
| `ingest/build.py` | `_attach_rt_and_tfd_features` takes `fixations_path`; `main`'s canonical `fixations_path` is threaded through `_process` |
| behaviour | a fixation whose id resolves to no interest area **in a trial we analyse** now raises, instead of accruing zero and being written as a real zero. Fixations from trials outside the analysis set are counted and reported separately — on L1, 143,123 of them, from the 4,610 report-only trials |

**Still to do on this step:** rebuild the downstream model-ready tables (L1 and KnowQA) and
record the move in `findings.md`. **Nothing is removed from `ingest/clicks.py`** — see the
correction above.

#### Implemented 2026-10-06

`src/config/screens.py` holds the `Screen` record — `area_col`, `regions`, and three switches
(`include_raw_pupil`, `write_skip_indicator`, `keep_dwell_totals`) that were the only real
differences once the metrics were already shared. It imports `config/columns` and nothing else,
so the `config/` layering rule holds.

| was duplicated | now |
|---|---|
| `RT_* → TimeSinceOffset_*` rename | `reading_times.rename_rt_to_time_since_offset` — **one occurrence left in the whole tree** |
| per-screen RT/TFD assembly | `reading_times.build_screen_rt_tfd(ia, screen, ...)` — both screens call it |
| the per-area metric merge loop | `area_metrics.build_area_metrics(df, screen, visit_counts=...)` |
| coerce + scale + z-score pupil | `pupil.prepare_screen_pupil` — the paragraph version adopted, as planned |

`spans.py` 476 → 449 lines, and `prepare_paragraph_ia`, `build_span_metrics` and
`build_paragraph_rt_tfd` are now one-line calls. The line count is not the point: what changed is
that there is **one implementation of each**, taking a screen.

**Visit counts stayed per screen, by necessity.** `build_area_metrics` takes them as an argument
rather than computing them, because the answer screen counts from the simplified label sequence
and the paragraph screen by resolving the raw IA sequence against its nearest-IA queue. Those are
genuinely different inputs, not a flag.

**Verified: numbers unchanged, on all three artifacts.**

| rebuilt | result |
|---|---|
| answer `RT_and_TFD.csv` | 19,436 × 32 — numeric max abs diff **0.0**, 0 non-numeric cells differing |
| `L1_paragraph_span_features.csv` | 24,046 × 51 — numeric max abs diff **0.0**, 0 non-numeric cells differing |
| KnowQA `all_participants.csv` (full pipeline) | 33,830 × 420 — still **exactly** the same 10 RT columns differing from the pre-step-6 baseline, i.e. step 2b contributes nothing |

94 of 98 modules import, same four parked failures.

**One thing I broke and caught.** The first draft declared the three raw pupil columns in *both*
`config/screens.py` and `spans.py` — the precise drift this step exists to remove. `spans.py` now
reads the single declaration. Worth recording because it is how these files got into this state
originally: a shared list copied "just for now".

### 3.7 Dataset config, once

The eight `include_*` booleans become one `skip` list on the dataset record, matching
`skip_base_features`:

```python
Dataset(..., skip_features={"build_trial_level_paragraph_features": "no paragraph screen ..."})
```

`knowqa.build_features`'s hand-written flag block collapses to nothing, and
`include_paragraph_features` stops being a second spelling of `has_paragraph`. The remaining
genuinely per-*run* choices (`pattern_scope_df`, `pattern_scope_by`) stay arguments, for the
same reason `add_zscored_pupil_columns` stayed a runner conditional in stage C: they are
properties of how you chose to run, not of the dataset.

---

#### Step 5, implemented 2026-10-07 — the last flag

Step 3 removed seven of the eight booleans by showing they were dead or compute-only. The eighth,
`include_paragraph_features`, was the one that meant something: **does this dataset have a
paragraph screen?** That is a property of the dataset, so it is now read off the record.

```python
build_trial_level_model_df(df, dataset="knowqa")     # has_paragraph=False -> block skipped
save_all_features(df, dataset="knowqa")
```

`dataset` takes a key or a `Dataset`; `build_trial_level_model_df` and `save_all_features` have
**no `include_*` parameters left at all**. `knowqa.build_features` no longer has to remember to
pass `False` -- it names its dataset, which it already knew.

**And the warning became an assertion, as planned.** The paragraph block used to *print a caution*
when the cache shared no trial with the frame, because a caller could plausibly reach it with the
flag left on by accident. Driven by `has_paragraph`, that is no longer plausible: an empty join
now means a stale cache or disagreeing keys, so it **raises**. Verified both ways 2026-10-07 --
a `has_paragraph=True` dataset with a non-overlapping cache raises `ValueError` naming the dataset
and the path; the same frame with `dataset="knowqa"` skips the block and emits 0 paragraph columns.

**Verified: numbers unchanged.** KnowQA full pipeline 33,830 x 420, max abs diff **0.0**. L1
model-ready 19,436 x 219, numeric max abs diff **0.0**, 0 non-numeric cells differing. 97 of 98
modules import.

### 3.8 The paragraph pipeline is the QA one with different flags — ✅ **Diana, 2026-10-06**

> **Diana, 2026-10-06:** *"can the paragraph csv creator be same as QA just with a few different
> flags? I said I want them to be generated separately and joined if needed, it doesn't mean I
> want to duplicate code needlessly."*

Measured answer: **largely yes.** `spans.py` (476 lines) reimplements no formula — every metric
is the shared `am.*` builder with `SPAN_COL` instead of `area_label`, which is already the "one
flag" the question asks about. What is duplicated is the **orchestration around** them:

| | paragraph | answer |
|---|---|---|
| merge the metric parts | `build_span_metrics` — hand-written list + merge loop | `generate_new_row_features` + the registry's `join_columns` |
| `RT_* → TimeSinceOffset_*` rename | `spans.py:373` | `reading_times.py:507` — **still character-for-character identical**, checked after the §3.6 work |
| coerce + scale + z-score pupil | `prepare_paragraph_ia` | `am.coerce_ia_columns` + the registry's `add_zscored_pupil_columns` |
| load → run → merge → save | `build_paragraph_features` / `save_paragraph_features` | `main` / `_process` / `_save` |

Worth noting `prepare_paragraph_ia` is arguably the **better** of the two — it coerces once up
front, which is what T3.11 wants the answer side to move toward. Unifying means adopting the
paragraph version, not discarding it.

**Genuinely paragraph-only, and not a flag:** `scan_paragraph_fixations` and the nearest-IA queue
(~89 lines, see §3.6 — the first draft wrongly predicted these would fold away), no question area,
eight metrics rather than ten, pupil `include_raw=False`.

**The shape a fix would take** — the same idea as `Dataset.skip_base_features`, one level up:

```python
Screen(key="answers",   area_col=AREA_LABEL, regions=ANSWER_REGIONS, ...)
Screen(key="paragraph", area_col=SPAN_COL,   regions=SPANS,
       extra_steps=["scan_paragraph_fixations"], skip=[...sequence tags...])
```

One runner, two configs, **two artifacts** — the separation Diana asked for in T6.1 is untouched;
they are still generated separately and joined on `(participant_id, TRIAL_INDEX)`. What goes is
the second copy of "merge the parts, rename the columns, save the table".

✅ **Approved 2026-10-06** (*"should yes happen"*) and scheduled as **step 2b**, ahead of the
registry work, because the four duplicated orchestration pieces are self-contained and each one
can be proved neutral on its own.

**Scope, stated so it is not open-ended.** A `Screen` record carries the per-screen differences,
and the four duplicated pieces become one each:

| today, twice | after |
|---|---|
| the `RT_* → TimeSinceOffset_*` rename | one helper in `reading_times.py` |
| coerce + scale + z-score pupil | one helper, the paragraph version (it coerces once up front — the better of the two) |
| merge the per-area metric parts | one `build_area_metrics(df, screen)` |
| per-screen RT/TFD assembly | one `build_screen_rt_tfd(ia, screen, ...)` |

**Out of scope, and why.** The answer screen keeps `ingest/build.py::main` as its production
entry point. Its pipeline does much more than the paragraph one — base features, button clicks,
last-visited, the hunters/gatherers split, writing `all_participants.csv` — and it reaches trial
grain later, in `model_data`, whereas the paragraph path goes straight there. Making those one
pipeline is **step 3's** job (the registry gaining a `trial` kind), not this one. What step 2b
delivers is that the *shared core* — per-area metrics and RT/TFD — is one implementation taking a
screen, instead of two.

#### Step 2c, implemented 2026-10-07

**The duplication was sharper than this section described.** All five primitives
(`parse_ia_sequence`, `resolve_fixation_sequence`, `drop_leading_question_fixation`,
`collapse_runs`, `visit_counts_from_sequence`) plus a `nearest_ia_queue` helper **already existed**
in `area_metrics.py` and were already used by the paragraph screen. The answer screen's
`create_fixation_sequence_tags` predated them and carried its own inline copies. So this was not
"write a shared implementation" but "make the older caller use the one that exists" -- verified
line by line to be behaviourally identical first.

`sequences.build_area_sequences(df, area_cols, ...)` is the shared composition. `area_cols` is the
vocabulary: the answer screen passes two (area label, screen location) and gets a list column per
vocabulary; the paragraph screen passes its span column alone. That is the only structural
difference, so `Screen` did **not** need to regain `drop_leading_question` after all -- it is one
keyword argument at the two call sites, which is clearer than a field.

| | before | after |
|---|---|---|
| `create_fixation_sequence_tags` | 5,549 chars of inline parse/resolve/map/trim | 1,207 |
| `span_visit_counts` | 3,263 chars | 1,392 |

**A real performance bug fell out.** The answer-screen version filtered the entire fixation frame
*inside* its per-trial loop -- on L1, 19,436 boolean scans of a 718,207-row frame. That is what the
registry's `## heavy iternal data loading here... slow for now` comment was about. The shared
version builds the queue in **one groupby**. KnowQA's pipeline went 51s -> 44s.

**And a silent-failure mode is now guarded.** The two screens keyed their nearest-IA queue
differently -- `(trial, participant)` from a groupby versus `(str(participant), str(trial))` from
the paragraph scan. A mismatch looks exactly like "no off-area fixations anywhere": every one
silently dropped, no error. The shared function normalises to one stringified key convention
(KnowQA's trial index is a composite string, L1's an integer) **and counts trials that have
off-area fixations but no queue entry**, printing a warning that names the likely cause.

**Verified: KnowQA bit-identical** -- 33,830 x 420, numeric max abs diff 0.0, 0 non-numeric cells
differing, which covers all four sequence columns and both visit-count columns. 97 of 98 modules
import.

### 3.9 The fifth duplicated piece: building the collapsed area sequence — ⚠️ **open, scoped**

Step 2b unified four of **five** duplicated orchestration pieces. The fifth is the one that turns
a trial's raw fixation sequence into "which areas, in order, without repeats" — and from that,
visit counts. Both screens do it; neither shares the code.

**What a visit is, so the rest reads:** one uninterrupted look at an area. Eyes go B → C → B and
that is two visits to B. You take the sequence of areas the eyes passed through, squash
consecutive repeats, and count what is left.

**The same computation, split differently.** This is the point, and an earlier draft of this
proposal got it wrong by calling the two "different inputs":

| | answer screen | paragraph screen |
|---|---|---|
| parse the raw id sequence | `create_fixation_sequence_tags` | `span_visit_counts` |
| fill off-area fixations from the nearest IA | same function, queue built **inline** from the fixation report | same function, queue built in a **separate streaming pass** (`scan_paragraph_fixations`) because the report is multi-GB |
| map ids → areas | same function | same function |
| squash consecutive repeats | `create_simplified_fixation_tags` | same function |
| count | `create_simplified_visit_counts` | same function |

So: **three registry steps on one screen, one function on the other.** The *primitives* are
already shared — `am.parse_ia_sequence`, `resolve_fixation_sequence`, `collapse_runs`,
`visit_counts_from_sequence` are called by both. What is duplicated is the composition.

**The four real differences, which are `Screen`-shaped:**

1. **Two vocabularies vs one.** The answer screen maps each fixation to *both* an area label and a
   screen location (`fix_by_label` / `fix_by_loc`). A paragraph span has no screen location, so
   that axis is answer-only.
2. **The leading-question drop.** The answer screen discards an opening fixation on the question
   when the next one is not; the paragraph screen has no question. **This is what the
   `drop_leading_question` field was for** — removed from `Screen` on 2026-10-06 because nothing
   read it yet, and it comes back with this step.
3. **Where the nearest-IA queue comes from** — inline vs a separate streaming pass.
4. **Whether the intermediates are persisted.** The answer screen writes `fix_by_label`,
   `fix_by_loc`, `simpl_fix_by_label`, `simpl_fix_by_loc` onto `all_participants.csv`; the
   paragraph screen keeps nothing.

**Why it is bigger than the other four, and why I did not bolt it onto 2b.** Point 4 is the
reason: those four columns are **load-bearing**. `simpl_fix_by_label` is read by the scanning
strategies, by the XYX / XYXY detectors and by `model_data`. Touching how they are produced risks
changing `all_participants.csv`'s schema or their values, which is a different class of risk from
the four self-contained pieces 2b dealt with.

**Shape of the fix:** one `build_area_sequence(df, screen, nearest_by_trial, ...)` producing the
collapsed sequence plus visit counts; `Screen` regains `drop_leading_question` and gains
`sequence_vocabularies` (`("label", "loc")` vs `("span",)`); the three registry entries become
thin, and `span_visit_counts` becomes a call. The queue stays per screen — that is a real
difference in how the data has to be read, not a flag.

**Acceptance:** the four sequence columns and `num_label_visits` / `num_loc_visits` bit-identical
on KnowQA's full pipeline, and `L1_paragraph_span_features.csv` bit-identical. Both are checks
this session has already run, so the cost is time rather than new machinery.

## Where every file goes

### Into `features/`

| today | → | note |
|---|---|---|
| `features/build.py` (stage C content: 8 `add_*` + `FUNCTION_REGISTRY` + 4 accessors) | ✅ **`ingest/base_features.py` + `ingest/registry.py`** | step 1 — the §5.4 fix, done 2026-10-06. It split in two rather than joining `ingest/build.py`: the row-level builders and the recipe are different jobs |
| `model_data.py`'s six `build_trial_level_*_features` | `features/build.py` | the trial-level generators |
| `model_data.py`'s `_build_trial_core`, the `groupby().agg("first")` | `features/build.py` | the grain collapse |
| `common/feature_builders.py` (pivot, contrasts, constant/categorical builders) | `features/build.py` | already routed by map §5.1 |
| the registry itself, now all three kinds | **`features/registry.py`** | a file of its own; it is the vocabulary |

### Into `modeling/`

| today | → |
|---|---|
| `answer_correctness/feature_groups.py` (the **chosen** half) + `common/feature_specs.py` | `modeling/feature_sets.py` |
| `model_data.py`'s `make_area_only_dataset` / `make_derived_dataset` / `make_full_dataset` | `modeling/datasets.py` |
| `common/prepared_dataset.py` | `modeling/evaluate.py` |
| `common/data_utils.py` | split: CIs → `modeling/inference.py`, splits → `modeling/folds.py` |
| `answer_correctness/cross_validation.py` (48 KB) | `modeling/{folds,crossval,evaluate}.py` (map §6.3) |
| `answer_correctness/evaluation_core.py` | `modeling/evaluate.py` |
| `answer_correctness/models/**` + `answer_RTs/models/**` | `modeling/models/**` |
| `common/feature_selection.py` | `modeling/selection.py` |

### Open — needs a decision, not a default

| | question |
|---|---|
| ~~`save_all_features` / `load_all_features`~~ | ✅ **Decided 2026-10-07 — `features/build.py`** (Diana: *"putting them with the builders makes sense"*). **No re-export shim**: all 13 call sites were updated instead, so there is one name for each. **Both were renamed**, because "all features" named neither the grain nor the contents and collided with a different function of the same name in `RT_correlations`: `save_model_ready` / `load_model_ready`, matching the artifact and the `Dataset.model_ready` property that locates it. `model_data.py` is now empty and **deleted**. The deciding argument for `features/` over `analyses/`: `stats/RT_correlations` calls the save function to rebuild a missing cache, and from `analyses/` that would be an analysis importing another analysis |
| ~~the three `make_*_dataset` builders~~ | ✅ **Deleted 2026-10-07** (Diana: *"can we already do that with our other better feature building functionality? if so - get rid of them"* — yes: each was `build_trial_level_model_df` + one selector + `PreparedTrialDataset`, three lines of current machinery). Searched the whole history: each appeared in **exactly one commit, the one that defined it** — never called from anywhere, ever. `modeling/datasets.py` is gone with them |
| ~~`common/viz_utils.py`~~ | ✅ **Done 2026-10-07 — `lib/plotting/confusion.py`**, by the §4 test (it names no eye-tracking vocabulary). It needed `plot_output.py` to move first, since it calls `save_output` and `lib/` imports nothing outside `lib/`; Diana ruled *"do the plot_output move now"*, so both landed together. `common/` is gone, and `answer_loc_viz.py`'s unused import of it was deleted |
| ~~`viz/plot_output.py`~~ | ✅ **Moved 2026-10-07 to `lib/plotting/output.py`**, its §3 destination — 29 src sites across 27 files plus 5 notebooks. It imports nothing from the project, so `lib/` stays clean. **The hazard was `PROJECT_ROOT = parents[2]`**, a depth count that silently redirects every output one directory up when the file moves; it is now `parents[3]` with an assertion that the result looks like the repo root. **Verified neutral:** `PROJECT_ROOT`, `REPORTS_ROOT`, `PAPER_ROOT`, all 17 `analysis_dir()`s, `slug()`, `build_stem()` and the 10-function API are identical to a pre-move snapshot, and regenerating `text_qa_relationship` left **every table and figure byte-identical** — only `manifest.json` moved, in its `saved_at` and `produced_by` provenance fields. ✅ **`ANALYSES` / `ABBREVIATIONS` stay in `lib/`** — ruled 2026-10-07, recorded as the one accepted exception in `restructure-map.md` §4. They are project vocabulary by the §4 test, but honouring it means injecting them as config and re-pointing all 35 call sites; Diana: *"separation is supposed to make things easier, not harder."* |
| `answer_RTs/model_data.py`'s five `get_*_feature_cols` | the map parks `answer_RTs/` in `explorations/`. Do its selection surfaces migrate to the new mechanism, or does it stay as-is because it is parked? |

---

## Order of work

Each step ends with the same acceptance check, and is not finished until it passes.

| # | step | why this order |
|---|---|---|
| **1** ✅ | **`add_IA_screen_location` + `FUNCTION_REGISTRY` + the 4 accessors → `ingest/`.** `features/build.py` is emptied. | the §5.4 fix; the name had to be free first. **Done 2026-10-06** — landed as `ingest/base_features.py` + `ingest/registry.py`, not one file |
| **2** ✅ | `common/feature_builders.py` + `model_data.py`'s six builders and the grain collapse → the new `features/build.py` | pure moves. **Done 2026-10-07** — and it fixed three parked modules as a side effect, see below |
| **2b** ✅ | **the `Screen` record (§3.8)** — one rename helper, one pupil prep, one `build_area_metrics`, one `build_screen_rt_tfd`; `spans.py` becomes a thin caller | self-contained and provable piece by piece. **Done 2026-10-06, ahead of step 2** — it needed none of step 2's moves, so the registry work will inherit one implementation rather than two |
| **2c** ✅ | **the collapsed-area-sequence step (§3.9)** — the fifth duplicated piece; `Screen` regains `drop_leading_question` | left out of 2b deliberately: it touches four load-bearing columns of `all_participants.csv`, which the other four pieces did not |
| **3** ✅ | ~~`features/registry.py` — add `kind: "trial"`~~ → **audit what the flags were for.** 7 of 8 were dead or compute-only; deleted. | **Done 2026-10-07.** The registry does not grow a third kind: there was nothing left to select |
| **4** ❌ | ~~record the observed produced-column map~~ → **audited; premise does not hold.** The deletion pass deleted nothing (the "dead" sets were live or deliberate). The `LAST_*` vocabulary moved to `features/build.py` instead | **Done 2026-10-07.** Section 3.3's upward import is gone |
| **5** ✅ | `include_*` → the `Dataset` record; delete the 7 copy-pasted flag blocks | **Done 2026-10-07** — see §3.7. Step 3 had already shown 7 of the 8 booleans dead, so no `skip_features` mapping was needed: the surviving one, `include_paragraph_features`, *was* `has_paragraph` and is now read off the record. `build_trial_level_model_df` and `save_all_features` have no `include_*` parameters left |
| **6** ✅ | the fixation-to-area rule (§3.6): one mapping, one run-based RT, RT off the clicks table | **the only step that moves numbers.** Planned last so the rest could be proved neutral first; **run second instead, on Diana's instruction.** Done 2026-10-06 — see the status table at the top for what that costs |
| **7** ✅ | `modeling/` — feature_sets, datasets, folds, crossval, evaluate, inference, models, selection | **Done 2026-10-07.** `inference`, `selection`, `models/` and the `data_utils` half of `folds`/`evaluate` landed earlier in the step; this completed it. `feature_groups` + `feature_specs` → **`modeling/feature_sets.py`** (the four constants they defined twice were character-identical and unreferenced outside their own modules, so the duplicates went); `cross_validation.py` (1,329 lines) → **`folds` · `crossval` · `evaluate`**, with its three figures to `viz/visualisations_cross_validation.py` because `modeling/` must not import `viz/`; `make_*_dataset` → **`modeling/datasets.py`**. `model_data.py` is left holding only `save_all_features` / `load_all_features`, whose home is still the open question below |

**Step 2 cannot move a number** (moves only). **Steps 3–5 must not, and that is the thing to
prove** — now against a post-step-6 baseline, which means the full L1 prep rerun has to happen
before they can be verified on L1. KnowQA is already current and can verify them today. Step 7 is
the map's original stage D and is mechanical by comparison.

---

## Acceptance test

Stronger than stage C's, because this stage touches the generator rather than its location:

1. **The two model-ready tables are bit-identical** — *against a post-step-6 baseline.*
   Rebuild `L1_model_ready_all_features.csv` (19,436 × 219) and KnowQA's (870 trials) from their
   IA tables and compare: shape, column order, numeric max-abs-diff `0.0`, zero non-numeric cells
   differing. **KnowQA can be checked today** (rebuilt 2026-10-06). **L1 cannot until Diana
   reruns prep** — its `all_participants.csv` is from 09-23 and still carries the pre-step-6 RT
   columns, so an L1 comparison today would show step 6's change, not a regression.
2. **`all_participants.csv` is bit-identical** — the registry move must not change what it
   produces. ✅ **passed for step 1** (KnowQA 33,830 × 420, max diff 0.0). For steps 2–5 the
   baseline is the post-step-6 KnowQA table.
3. **Imports:** 92/96, same four parked failures.
4. **Layering:** `features/` imports nothing from `ingest/` — **with no exception this time.**
5. **The observed produced-column map is complete:** every column of `all_participants.csv` and
   of the model-ready table is claimed by exactly one registry entry, or is a raw passthrough in
   the whitelist. Nothing unaccounted for.
6. **No orphan columns:** every one of the 219 model-ready columns is claimed by exactly one
   registry entry or is an identity column. Today nothing can state this.
7. `text_qa_relationship` 45/45 tables unchanged (it reads the model-ready table).
8. **The read-time filter is proved, not trusted:** one pipeline run with `usecols` and one
   without, bit-identical output. Without this, a missed `needs` entry drops data silently.

> **Use `git diff` on `reports/`, not a scratch snapshot** — those tables are tracked, and a
> filename-keyed snapshot silently reports 0/N when an output gets renamed. That has already
> happened once.

---

## Explicitly not in this stage

- **No change to which features exist, or to any feature's definition.** If a number moves,
  stop: that means the mechanism change altered behaviour.
- **`FUNCTION_REGISTRY` keeps its ordering dependency.** T3.24 declined.
- **No dependency graph, no config system, no plugin registry.** `conventions.md` is explicit;
  the registry plus one new `kind` is the whole machinery budget.
- **`SELECT_1_COLS` and the curated JSONs are not touched.** They are decisions, and this stage
  is about mechanism.
- **`analyses/` is not created** — that is stage E.
- **No data files move** (`CLAUDE.md` hard rule 4).

---

## What I am least sure about

1. **The group kind produces one name that the pivot later expands into ~10**
   (`__answer_A` … `__contrast`). Observation at the IA stage sees the one name; the expansion
   happens later, in the trial-level pivot. So the map has two layers, and acceptance check 5 —
   every column claimed by exactly one entry — is what keeps them joined up.
2. **Still open: `trial_mean_dwell` may not be a sequence feature at all.** It is grouped with the four
   sequence statistics in `DERIVED_COLS` but it is a mean dwell time — i.e. an area-metric
   relative. Worth checking where it really belongs before registering it, rather than
   preserving an accident of the old grouping.
3. **Whether `save_all_features` belongs to `features/` or `modeling/`** — listed as open above.
   It decides whether `features/` is allowed to write a cache, which is a layering question.
4. **The 1,003-line `generate_column_options.py`.** I have not read it closely enough to say
   whether it is one concern or four. It is the largest unexamined thing in this stage.
5. **Resolved, and worth recording as a lesson rather than a worry.** This list did not contain
   "the run-based RT might not compute its own documented definition" — and both implementations
   turned out not to. The thing that caught it was a *signed direction that contradicted the
   stated mechanism*, chased rather than averaged over. Where a step is expected to move numbers,
   predicting the **direction** and checking it is worth more than predicting the magnitude.
