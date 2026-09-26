# Findings ledger

> ## ⚠️ NOT VERIFIED BY DIANA — TREAT WITH CAUTION
>
> **This file has not been checked by Diana. Nothing in it should be quoted, cited, or acted
> on as established without her confirming the specific number first.**
>
> It was assembled by Claude by reading saved notebook output and saved figures. It is useful
> as a map of what results exist and where they came from. It is **not** an authoritative
> record of what this project found.
>
> **The real ledger gets written after the restructure.** Once `todo.md` T1.3 and T4.0 land —
> every analysis persisting its numbers as CSV alongside its figures — this file can be
> *regenerated from `reports/report_data/`* instead of transcribed from pictures. That version
> will be trustworthy. This one is a stopgap for the gap that made it necessary.

The empirical results this project appears to have established, with the code and output that
produced each. Compiled from `notebooks/vizes for purpouse/`, `notebooks/statistics.ipynb`,
`notebooks/mixed_text_answer_effects.ipynb`, and the figures under `reports/plots/`.

**Why it exists at all.** Most of these numbers live *only inside saved PNGs* — the notebooks
return DataFrames and discard them, and the `viz/` driver notebooks contain almost no prose
(`visualisations.ipynb`: 17 cells, **zero** markdown). Nine of fifteen plot topics have no
saved numbers whatsoever (§10). So results the paper needs to quote are currently recoverable
only by looking at a picture, and this file is the text version of that picture.

## What to distrust, specifically

| Concern | Detail |
|---|---|
| **Figure readings are approximate** | Anything marked **[figure]** was read off a rendered PNG. Group-level totals cross-check, individual bar heights may not. |
| **Reliable transcription ≠ correct number** | **[printed]** values are faithfully copied from stored stdout — but that stdout came from code with known defects. Anything touching the Fisher tests carries the grain bug (T3.1); anything touching first-fixation duration carries the coercion problem (T3.6). |
| **Different runs, different trees** | Notebook outputs record **three** different execution roots and span 2026-01 to 2026-09. §1 (Jul) and §3 (Apr) are not from the same run of the same code. |
| **The interpretation is Claude's** | Section framings, "the pattern is…" statements and cross-links between findings are inference, not something Diana wrote or endorsed. |
| **Diana has reviewed only two sections** | §1 (first-scan) and §3.1 (attention allocation) were corrected by her. Everything else is unchecked. |

**The exception, and the most reliable material here:** a handful of figures were **computed
directly by Claude on 2026-09-04/05** from the staged feature caches
(`L1_model_ready_all_features.csv`, `L1_paragraph_span_features.csv`) rather than transcribed
— the coverage × intensity decomposition and metric correlations (§3.1 note), the pupil NaN
counts and the imputation figure (§8, now 253 cells), and the participant/group counts. Those are
reproducible from the CSVs and are dated where they appear.

## Provenance markers

| Marker | Meaning |
|---|---|
| **[printed]** | copied from the notebook's stored stdout. Faithful transcription — but see the caveat about defects above. |
| **[figure]** | read off a saved PNG. Approximate — **regenerate as text before quoting.** |
| `?` | Claude's inference |

---

## 1. First-scan behaviour

Paper: Results §"First-scan behavior". Code: `src/viz/visualisations_strategies.py`,
driven by `notebooks/vizes for purpouse/visualisations.ipynb` cell 6 with
`kind="location", window_len=4, threshold=0.5`.

A "strategy" is the **first 4 tokens of the collapsed location sequence, question tokens
dropped** — i.e. the order in which the participant first visits the four answer positions.

### 1.1 The two dominant patterns are exactly clockwise and counter-clockwise

The diamond layout is `answer_0(top)`, `answer_1(left)`, `answer_2(right)`,
`answer_3(bottom)`. So:

- **clockwise** = `(top, right, bottom, left)`
- **counter-clockwise** = `(top, left, bottom, right)`

Participants with a dominant strategy at ≥50% of trials, by which strategy dominates
**[figure]** (`reports/plots/strategies/str_above_thresh_strategy_completed_*.png`):

| strategy | reading | hunters | gatherers |
|---|---|---|---|
| `(top, right, bottom, left)` | **clockwise** | **71** | **89** |
| `(top, left, bottom, right)` | **counter-clockwise** | **20** | **14** |
| `(top, left, right, bottom)` | "Z" / reading order | 4 | 4 |
| `(left, top, right, bottom)` | left-first Z | 1 | 3 |
| **total** | | **96** (53.3% of 180) | **110** (61.1% of 180) |

**Every participant with a dominant opening scan has one of only four strategies, and the
top two are precisely the clockwise and counter-clockwise circuits of the diamond.**
CW:CCW ≈ 3.6:1 in hunters, 6.4:1 in gatherers.

The group totals cross-check against the printed summary dicts (53.33 / 61.11), so the
figure reading is sound at the group level even if individual bar heights are approximate.

### 1.2 Dominant-strategy prevalence — the paper's X / Y / Z

**[printed]** from `visualisations.ipynb` cell 6, verbatim:

> 46.1% of hunters participants had a dominant strategy (>50% of trials) before completion.
> 49.4% of hunters participants had a dominant strategy (>50% of trials) after completion.
> 56.1% of gatherers participants had a dominant strategy (>50% of trials) before completion.
> 57.8% of gatherers participants had a dominant strategy (>50% of trials) after completion.

And the `before_after_summary` dicts **[printed]**:

| | hunters | gatherers |
|---|---|---|
| participants | 180 | 180 |
| mean dominant share, raw | 0.4922 | 0.5269 |
| mean dominant share, completed | 0.5119 | 0.5414 |
| mean delta | +0.0197 | +0.0145 |
| **`raw_≥50%`** | **48.89%** | **58.33%** |
| **`comp_≥50%`** | **53.33%** | **61.11%** |
| dominant label changed by completion | 2 participants (1.11%) | 2 participants (1.11%) |
| mean % of sequences changed | 4.66% | 2.96% |

> ✅ **RESOLVED 2026-09-20 (`todo.md` T1.1) — there is now one convention.**
> The stored output above predates the fix and is kept as the record of what was printed then.
> `proportion_with_dominant_strategy` used strict `prop > threshold` while
> `summarize_before_after` and `plot_dominant_strategy_counts_above_threshold` used
> `prop >= threshold`, which is why hunters read as **46.1%** in the print and **48.89%** in
> the dict — same data. Draft2 says "at least in half of the trials", which is `≥`.
>
> All of them now go through one function, `derived/pattern_breaking.has_dominant_strategy`,
> so the printed line and the dict agree. **The canonical numbers are the `≥` ones, which the
> table above already carries:** **48.89% / 53.33%** (hunters) and **58.33% / 61.11%**
> (gatherers). The 46.1 / 49.4 / 56.1 / 57.8 quartet in the quoted stdout is superseded.
>
> The difference is 5 hunters and 4 gatherers whose dominance score is exactly 0.50.

> ⚠️ **One small unexplained drift, not caused by that fix.** Recomputing this table on
> today's code and data reproduces hunters exactly (0.4922 / 0.5119) but gives gatherers
> **0.5271 / 0.5416** against the 0.5269 / 0.5414 transcribed above — +0.0002 on both. The
> T1.1 merge is not the cause; old and new logic were checked to be bit-identical on the same
> frames. Most likely the stored output (2026-07-13) predates a data rebuild. Too small to
> matter for any claim, recorded so it is not mistaken for a regression later. Everything else
> reproduces: 180 participants per group, `changed_label_n = 2` in both.

**The group difference holds in the direction draft2 claims:** gatherers (no question
preview) are more strategy-consistent than hunters (preview) — 58.3% vs 48.9% raw.

> ✅ **The all-participants figure (draft2's `X%`) now exists — computed 2026-09-25.**
> `run_all_strategy_plots` used to hardcode `include_all=False`; it is now a parameter
> defaulting to `True`.
>
> | all 360 participants | raw | after completion |
> |---|---|---|
> | **`≥50%` of trials** | **53.6%** | **57.2%** |
>
> Between the two groups, as it must be (hunters 48.9/53.3, gatherers 58.3/61.1 — both
> unchanged). **This is the `X%` draft2 asks for**, and it is computed under the same `≥`
> convention as the group figures.
>
> ⚠️ Computed by Claude on 2026-09-25 from `all_participants.csv` and not yet checked by
> Diana — the same standing caveat as the rest of this file.

### 1.3 Completion barely moves the result — which is good news

The "interrupted scan completion" is `build_prefix_completion_map_from_series` +
`add_completed_sequence_column`: from strategies observed at full length 4, learn how each
prefix is most often completed, then fill shorter sequences.

It shifts the mean dominant share by **+0.02 (hunters) / +0.015 (gatherers)** and flips a
participant's dominant label for **2 of 180 in each group (1.1%)**. So the first-scan
finding is **robust to the completion heuristic** — worth stating in the paper, since it
pre-empts "isn't this an artifact of how you handled short scans?".

Note the completion map is learned **inside each group's loop**, so hunters and gatherers
get different prefix→completion maps — the two "after completion" numbers do not come from a
shared model. Where the map is learned is one of the things `todo.md` T1.6 has to settle when
the two implementations merge.

### 1.4 Trial-level version: the first-four-visits heatmap

This is arguably the origin figure of the whole project. **[figure]**
`reports/plots/answer_correctness/first_visits_heatmap/first4_visits_location_all_participants.png`,
row-normalized, all participants, no answer split:

| visit | Top | Left | Right | Bottom |
|---|---|---|---|---|
| 1st | **90%** | 9% | 1% | 0% |
| 2nd | 10% | 32% | **53%** | 5% |
| 3rd | 9% | 7% | 16% | **68%** |
| 4th | 6% | **47%** | 28% | 19% |

The scan starts at the top in **90%** of trials, then goes right (53%) more often than left
(32%), then bottom (68%), then left (47%). Same story as §1.1 at trial rather than
participant level.

Produced inline in `presentation_prep.ipynb` cell 31 via
`matrix_plot_simplified_visits(..., normalize="row", fmt=".0%")`, and mirrored into
`papers/correctness_prediction/figures/`. It is the **only** markdown-documented cell in
that notebook:

> ## First-4 answer visit heatmap (conference)
> Overall heatmap of the first four entries of the simplified fixation sequence,
> location-based, questions excluded, across all participants and NOT split by the
> answer selected at the end.

### 1.5 The necessary counterweight: "dominant" means modal, not exclusive

**[figure]** `reports/plots/strategies/dom_str_counts_strategy_hunters.png`: hunters use
between **6 and 28 distinct** first-4 sequences each, modal ≈ 13.

So a participant with 50% dominance still produced a dozen other openings across their
trials. **This belongs next to the headline** — "has a dominant strategy" is a statement
about the mode, not about consistency.

### 1.6 Dominant strategy × tracked eye — suggestive only, untested

**[printed]** `visualisations.ipynb` cell 8, `run_dominant_strategy_eye_analysis`. Combined
crosstab **[figure]** over the 193 participants with a dominant strategy:

| strategy | left-eye tracked | right-eye tracked |
|---|---|---|
| clockwise | 56 | 93 |
| counter-clockwise | 17 | 16 |
| other two | 4 | 7 |
| **CW share** | **72.7%** | **80.2%** |

**No significance test is run anywhere** — the function only plots and returns the
crosstab. Treat as an observation, not a result; it is not currently claimed in the paper.

> ⚠️ **Two things to know before this number is used** (both established 2026-09-20).
>
> 1. **The crosstab's population depends on a threshold that defaults to 0.**
>    `run_dominant_strategy_eye_analysis(threshold=0.0)` includes *every* participant, not
>    only those with a dominant strategy. The "193 participants with a dominant strategy"
>    heading above therefore needs checking against whatever threshold the stored run used —
>    at `threshold=0.5` hunters contribute 88, not 96, because this path uses the **raw**
>    strategy while the strategy figures use the **completed** one.
> 2. **One hunter's label changed**, `l22_53`, when the unstable
>    `sort_values(...).head(1)` tie-break was replaced with the deterministic rule. They have
>    a genuine 2-way tie at a score of 0.2037, so they appear only in the `threshold=0.0`
>    version. See the change log.

---

## 2. End-of-trial behaviour

Paper: Results §"End of trial behavior". Code: `src/viz/visualisations_last_label.py`,
`visualisations.ipynb` cell 4 (with `print_summaries=False`, so nothing was printed).

**[figure]** `reports/plots/last_label_before_confirm/all_participants__prop.png` — the last
area fixated before confirming:

| chose | n | last area = the chosen answer | each other answer | question |
|---|---|---|---|---|
| A | 16,339 | **≈68%** | ≈9–10% | ≈2.5% |
| B | 1,744 | **≈71%** | ≈8–10% | ≈2% |
| C | 942 | **≈70%** | ≈8–11% | ≈2.5% |
| D | 411 | **≈70%** | ≈7–11% | ≈2.5% |

**The last fixation before confirming lands on the chosen answer ~68–71% of the time, and
this is invariant to which answer was chosen** — so it is not an artifact of A being
correct. That invariance is the strong form of the claim and is worth stating explicitly.

> The paper currently says "80% of people look at the answer they ultimately select."
> These figures say ~68–71% of *trials*. `?` Different quantity (participants vs trials), or
> a number that needs reconciling?

---

## 3. Attention allocation between answer options

Paper: Results §"Attention allocation between answer options".
Code: `src/statistics/mixed_area_comparisons.py` via `notebooks/statistics.ipynb` cell 2;
descriptives from `visualisations.ipynb` cells 2–3.

One mixed model per (group × selected answer × metric), fixed effect per area, Holm-corrected
pairwise contrasts.

### 3.1 The selected area wins on every attention metric

**[printed]** hunters, `mean_dwell_time`, selected = A:

```
answer_A 241.24 ; answer_B 145.92 ; answer_C 126.10 ; answer_D 127.69
A–B  95.32  p_holm 0.0        ★
A–C 115.14  p_holm 0.0        ★
A–D 113.55  p_holm 0.0        ★
B–C  19.81  p_holm 2.03e-26   ★
B–D  18.23  p_holm 1.04e-22   ★
C–D  -1.59  p_holm 3.90e-01   n.s.
```

The same shape holds when B, C or D is selected (selected B: `answer_B 334.8` vs A 214.8,
C 176.6, D 180.8, with C–D n.s.; selected C: `answer_C 347.3`, A–B n.s.; selected D:
`answer_D 375.1`). `skip_rate` mirrors it inversely (selected A: `A 0.350` vs
`B 0.486 / C 0.524 / D 0.511`, all six pairs ★).

**The pattern: A > B > {C ≈ D}.** The selected option is strongly separated, and *only* C and
D are indistinguishable from each other — B sits significantly between them and A
(B–C p = 2.03e-26, B–D p = 1.04e-22; C–D n.s.).

> *Corrected 2026-09-05.* An earlier version of this file said "the three non-selected options
> are largely indistinguishable", which overstated it — B is clearly separated from C and D.

**And that gradient is exactly what the item construction predicts.** A and B are both built
from the **critical span** (B being a misinterpretation of it), C from the **distractor
span**, and D is not in the text at all (`research-context.md` §2). So B is live contender
material and gets read like one, while C and D are both "not it" visually. Selection rates
follow the full ranking — 84.07 / 8.97 / 4.85 / 2.11% — even though attention flattens at
C ≈ D.

This is a stronger statement than "more attention to preferred options": attention tracks the
*designed* plausibility structure of the item, not just the eventual choice.

> ⚠️ **"Wins on every attention metric" is weaker than it sounds — the metrics are not
> independent.** Every per-word mean includes the unfixated words as zero, so each is
> `intensity × (1 − skip_rate)`. Dividing back out (measured 2026-09-04):
>
> | metric | per-read-word, answers A→D | spread |
> |---|---|---|
> | first fixation duration | 192.2 · 184.8 · 181.9 · 181.2 ms | **6%** |
> | dwell time | 381.2 · 323.3 · 296.5 · 284.7 ms | 34% |
> | fixation count | 1.911 · 1.693 · 1.581 · 1.536 | 24% |
>
> So **`mean_first_fixation_duration`'s area differences are essentially all coverage** — and
> **T3.6 confirmed it on L1, 2026-09-23.** The prediction was that the between-area spread
> would mostly vanish; it did. Measured before → after on `L1_model_ready_all_features.csv`:
>
> | | A | B | C | D | A–D spread | corr with `skip_rate` |
> |---|---|---|---|---|---|---|
> | before (zero-filled) | 123.9 | 101.3 | 92.4 | 93.5 | **31.5 ms** | −0.70 … −0.84 |
> | after (unread excluded) | 192.0 | 185.0 | 181.8 | 181.9 | **10.2 ms** | **−0.017 … +0.035** |
>
> The correlation with skip rate is now indistinguishable from zero, which is the direct
> demonstration that the old column was a restatement of coverage. Dwell time and fixation
> count keep a real intensity signal and are not artifacts, though they correlate with each
> other at r = 0.92 and with skip rate at ≈ −0.55.
>
> *(The question area lands lower than the answers — 21.1 → 146.0 ms — but it is unfixated on
> 5,810 trials (30%), so it is now NaN there and the mean is over the remaining 70%.)*
>
> Five named metrics behave like about two independent ones — worth a sentence in the
> robustness discussion rather than presenting them as five converging results.
>
> Decided (`todo.md` T3.13): dwell time and fixation count **stay** coverage-inclusive, so
> they measure attention per *available* word. Only first-fixation duration changes.

### 3.2 The negative result: first-encounter pupil size discriminates nothing

**[printed]** all participants, selected A:

```
answer_A 2.5043 ; answer_B 2.5036 ; answer_C 2.5030 ; answer_D 2.5056
all six pairwise p_holm ≥ 0.1385   — nothing significant
```

Selected B: also nothing. Only selected C (C–D, `p_holm 2.4e-04`) and selected D
(B–D `6.99e-03`, C–D `0.0466`) show anything at all.

**Dwell, fixation count and skip rate discriminate the choice; pupil size at first encounter
does not.** This is the sharpest negative finding in the project and it is written down
nowhere. Note that a z-scored pupil feature is nonetheless in the model's 10-feature set
(`mean_max_fix_pupil_size_z__correct`) — not a contradiction (the model uses contrasts, not
area means) but worth being ready to explain.

### 3.3 Model health warnings — recorded, unaddressed

**[printed]**, pervasive: `ConvergenceWarning: The MLE may be on the boundary of the
parameter space` and `The Hessian matrix at the estimated parameter values is not positive
definite` — especially for `skip_rate` and `area_dwell_proportion`, which are bounded
proportions fitted with a Gaussian LMM. These models should not be presented as clean fits.

> ⚠️ Every figure from this analysis is a **zero-byte PNG** —
> `reports/plots/area_significance_heatmaps/`, 108 files, all stamped 2026-03-17 09:37:02.
> The backing CSVs in `reports/report_data/area_mixed_models/` (216 files) survived, so this
> is a regeneration job.

---

## 4. Time segments: before / during / after the decision

Code: `src/viz/visualisations_time_segments.py`, `visualisations.ipynb` cell 7 (no printed
output). Segments defined relative to the first fixation on the *eventually selected* answer.

**[figure]** `reports/plots/time_segments/all_participants/mean_dwell_time/`:

| segment | mean dwell |
|---|---|
| before | ≈77 ms |
| during | ≈263 ms |
| after | ≈145 ms |

**Dwell more than triples the moment the eventually-chosen answer is first fixated, and
post-decision re-checking of the other options still runs ~2× the pre-decision skim.**

This is a nice mechanistic complement to §2: the decision has a visible onset, and what
follows is verification rather than fresh search.

---

## 5. Correctness vs scan effort — one consistent story

Code: `src/viz/visualisations_correctness_measures.py`, `visualisations.ipynb` cells 10–16.
All **[printed]**.

### 5.1 Scan-path length

Threshold form, `seq_len` (cell 10), e.g. threshold 5:

| group | > 5 | ≤ 5 |
|---|---|---|
| hunters | .826 (n=5547) | .927 (n=4172) |
| gatherers | .759 (n=6003) | .897 (n=3714) |

Significant in the same direction at every threshold 2→7 for both groups.

Continuous form (cell 14), all participants:

```
len 1 .890 (n=118) · 2 .933 · 3 .922 · 4 .929 (n=2688) · 5 .902 (n=4504)
6 .875 · 7 .853 · 8 .793 · 9 .809 · 10 .769 · 11 .718 · 12 .661 · 13 .672
14 .635 · 15 .634  … 38 lengths total
```

Hunters sit above gatherers at every length (len 8: .832 vs .758; len 15: .713 vs .565).

### 5.2 Back-and-forth re-checking

| pattern | absent | present | Δ |
|---|---|---|---|
| XYX | .855 (n=13,965) | .805 (n=5,471) | −5.0 pts |
| XYXY | .847 (n=18,096) | .757 (n=1,340) | −9.0 pts |

**More vacillation predicts being wrong, and doubled alternation predicts it more strongly.**

> ⚠️ **A tension to resolve.** `presentation_prep.ipynb` cell 11 computes the *longest*
> alternating run rather than mere presence, and the two longest in the dataset (length 12)
> were **both answered correctly**, as were most of the top-50. So "alternation predicts
> error" holds on presence but may reverse at the extreme. Worth checking before the claim
> goes in as monotonic.

### 5.3 Dwell intensity and response time

Trial mean dwell (cell 13), hunters: threshold 100 → `.814 (>100)` vs `.923 (≤100)`;
threshold 400 → **`.482 (>400)`** vs `.875 (≤400)`. All participants at 400: `.529` vs `.846`.

Normalized total answering RT (cell 16), all participants:

```
0.046 .924 (n=1157) · 0.099 .923 (n=6140) · 0.151 .877 · 0.203 .799 · 0.256 .736
0.308 .703 · 0.361 .673 · 0.413 .581 · 0.465 .586 · 0.518 .514  …
```

**§5 is all one finding: longer, slower, more re-visiting trials are less accurate — the
"hard item / low confidence" signature.** This is the behavioural basis of the whole
confidence interpretation, and it is the thing the RT baselines in the model comparison are
built from.

---

## 6. Preference matching — the tempering result

Code: `src/derived/preference_matching.py`, `src/viz/visualisations_preference_correctness.py`,
`visualisations.ipynb` cell 9 (`print_summaries=False`).

"Matching" = the answer with the extreme value of a metric is the one selected.

**[figure]** `reports/plots/matching_correctness/polarity/all_participants/correctness_by_matching__mean_dwell_time.png`:

> matching **n=12,093**, acc ≈ .847 · not_matching **n=7,343**, acc ≈ .826
> Fisher **p=1.57e-04**, **OR = 1.16**

Two separate things fall out:

1. **The longest-dwelt answer is the selected answer in 12,093/19,436 = 62.2% of trials.**
2. **But whether it matches barely moves accuracy — OR 1.16, ~2 percentage points.**
   Significant only because n ≈ 19k.

Point 2 is an important limit on any "gaze reveals the answer" framing: gaze reveals the
*choice* well, and the *correctness of that choice* only weakly on this measure. Draft2's
"clear preference given to the answer ultimately selected" is supported by point 1; it should
not be extended to correctness on the strength of this test.

> `reports/plots/matching_correctness/polarity/all_participants/correctness_by_matching__num_loc_visits.png`
> is **0 bytes** — a silently failed write.

---

## 7. Text spans → answer dwell

Code: `src/statistics/mixed_text_answer_effects.py`, `notebooks/mixed_text_answer_effects.ipynb`
cells 5–6. Random-slope `Lmer` models, `separated=False`, `correlated=True`. All **[printed]**.

Pooled, all participants — β (effect of dwell on that text span on dwell on that answer):

| answer | critical | distractor | outside |
|---|---|---|---|
| A | 0.089 *** | 0.040 ** | 0.158 *** |
| B | 0.079 *** | 0.012 | 0.168 *** |
| C | 0.023 | 0.024 * | **0.176 ***** |
| D | 0.039 ** | 0.015 | 0.151 *** |

By group, answer A: hunters critical 0.092 (p=.00118), gatherers critical 0.107 (p=5.4e-04).

**The headline here is that `outside` — generic reading effort on the rest of the paragraph —
is the strongest and only universally significant predictor (β ≈ 0.12–0.20, all ***).**
`critical` matters for A and B and fades for C/D. `distractor` is essentially never
significant. That reads as a **general engagement effect**, not targeted span→answer routing.

> **Superseded (confirmed 2026-09-04).** This module is an older, half-abandoned attempt.
> The current text–QA analysis is `statistics/RT_correlations/` via
> `notebooks/text_associations.ipynb`, and draft2's Text Associations claims come from
> there, not from here. The numbers above are recorded for reference only — **do not quote
> them alongside the draft2 findings**, and do not treat the disagreement (distractor→C
> marginal rather than significant; critical→A in both groups rather than preview-only) as a
> contradiction needing resolution. Different method, retired strand.
>
> Kept because two things in it may still be worth reviving: the random-effects treatment of
> item and participant (which `RT_correlations` handles by aggregating to participant level
> instead), and the variance decomposition below.

Also recorded: `text_id_with_q` random-effect variance is **~4.8×** the `participant_id`
variance (13,724 vs 2,869, residual 27,618) for hunters answer A. **Item identity matters far
more than participant identity** — a useful justification for the item-crossed fold design.

Model health **[printed]**: `boundary (singular) fit` on every random-slope model; one
convergence failure (`max|grad| = 0.00296`, tol 0.002) for separated answer-B hunters; and
random-effect correlations of `-0.998` in one model and `+0.926` in the next. The separated
models cannot be fitted with correlated slopes at all — cell 5 carries the comment
`#not enough observations if both values are true`.

---

## 8. Basic descriptives

**[printed]** `presentation_prep.ipynb` cells 7–8:

```
Correct participant-trial pairs: 16339 / 19436
Percent correct: 84.07%

selected_answer_label  n_selected  percent_selected
A                           16339          84.07
B                            1744           8.97
C                             942           4.85
D                             411           2.11
```

Confirms: **19,436 trials, 84.07% correct** — the base rate the balanced-accuracy framing
rests on. Also confirms that `selected_answer_label == "A"` ⟺ correct.

### Imputation in the headline model — a Methods number

Re-measured 2026-09-07 on `SELECT_1_COLS` over the 19,436 trials, after the T3.18 rebuild:

| feature | cells imputed as `0` |
|---|---|
| `mean_max_fix_pupil_size_z__correct` | **253 (1.30% of trials)** |
| the other nine | 0 |

**253 cells, 0.130% of the feature matrix.** All 253 are trials where the correct answer was
never fixated, so pupil size is undefined; the model's `fill_value = 0.0` then asserts the
participant's *mean* dilation. Small enough not to threaten the reported accuracy, and the
kind of figure a reviewer will ask for. See `todo.md` T3.12.

> **Was 257 before T3.18 (measured 2026-09-04).** The four recovered cells are all in the ten
> trials whose area labels were wrong: with the labels corrected, the option that actually is
> the correct answer *had* been fixated on those trials, so a real pupil measurement exists
> where an imputed `0` used to stand. Verified by attribution — those ten trials now carry no
> NaN at all in this column, and the remaining 19,426 carry exactly 253, so 253 + 4 = 257.
> **The same 257 is still quoted in `pitfalls.md` §2 and `todo.md` T3.12 / T3.14** and wants
> the same correction.

No other family contributes: RT, TFD, TimeSinceOffset, dwell time, fixation count and
skip rate all have **zero NaN** — their zeros are genuine "never read" measurements.

---

## 9. Findings that exist only inside a notebook

`presentation_prep.ipynb` defines its own analyses with no `src/` equivalent. The first is a
strong result that would be lost if that notebook were archived.

### 9.1 The asymmetry: dwelling on the correct answer is free; dwelling on distractors is fatal

`plot_correctness_by_answer_a_vs_bcd_mean`, cell 22–23 **[printed]**. Accuracy binned by
normalized RT on answer A vs. the mean over B/C/D:

| bin centre | RT on A (correct) | mean RT on B/C/D |
|---|---|---|
| ~0.2 | .841 (n=13,900) | .865 (n=17,835) |
| ~0.6 | .847 (n=4,096) | **.591** (n=1,330) |
| ~1.0 | .814 (n=888) | **.474** (n=194) |
| ~1.4 | .826 (n=305) | **.388** (n=49) |
| ~1.8 | .838 (n=105) | **.364** (n=11) |

**Time on the correct answer is flat with respect to accuracy (.81–.85 across the whole
range). Time on the distractors collapses accuracy from .87 to ~.36.**

Cell 24–25 breaks it out per distractor: B .895→.430, C .877→.538, D .870→.533, while
A stays .84 ± .04.

This is one of the most interpretable results in the project — it says the signal is
*distractor engagement*, not correct-answer engagement — and it lives in one cell of a
presentation notebook. `?` Is this in the paper? It reads like it should be.

### 9.2 Other notebook-only machinery

| What | Where | Note |
|---|---|---|
| `longest_alternating_run` — length of the longest two-option alternation | cell 11 | finer-grained than `src`'s presence-only XYX/XYXY. Max observed 12. |
| `collect_triples` — per-item matched participant triples (shortest / longest scan / most correct-answer visits) | cell 9 | ⚠️ **docstring says `is_correct == 1`; the code filters `== 0`.** The printed "469 texts / 1407 rows" therefore describes **incorrect** trials. |
| RT-contrast exemplar mining, matched within item | cells 14, 16–21 | yielded only 2 matched pairs |
| Last-before-confirm quadruple matching (one participant per last-area A/B/C/D on the same item) | cell 26 | slide construction |
| `save_participant_trial_rows` — export one trial's IA + fixation rows | cell 2 | wrote named exemplars `LOW_RT_CONTRAST - l32_263_trial25`, `LAST_D - l57_288_trial45` to a folder **outside the repo** (`E:\Technion\QA PROJECT\scanpath examples\`) |

---

## 10. Cross-cutting hazards

1. **Results exist only as pixels — for nine of the fifteen plot topics.** Measured
   2026-09-04: `strategies`, `dominant_eye`, `last_label_before_confirm`, `time_segments`,
   `matching_correctness`, `simpl_visit_matrices`, `basic_stats_barcharts`,
   `basic_stats_heatmaps` and `feature_selection` have **no** counterpart under
   `reports/report_data/`. That is exactly the set of results in §1.1, §1.5, §1.6, §2, §4 and
   §6 above that had to be read off images. `RT_correlations` writes nothing to either tree.
   Fix: `todo.md` T4.0 (standing requirement) plus T1.3, T4.1.
2. **308 zero-byte PNGs**, all stamped 2026-03-17 09:37:02, wiping every figure from
   `mixed_area_comparisons` (§3) and `mixed_text_answer_effects` (§7), plus
   `participant_similarity`. Backing CSVs survived.
3. **`RT_correlations/` writes nothing at all** — and it is the likely source of draft2's
   Text Associations claims (§7).
4. **Three different execution roots** recorded in notebook outputs, meaning these were last
   run from different checkouts:
   `E:\Technion\QA PROJECT\programming\QA_eyetracking_workspace\`,
   `E:\QA PROJECT\programming\QA_eyetracking_workspace\`,
   `C:\Users\deeth\PycharmProjects\QA_eyetracking\`.
   So the §1 strategy numbers (2026-07-13) and the §3 mixed models (2026-04-19) are not from
   the same run of the same tree.
5. **Two `paper_dirs` conventions:** `["papers/correctness_prediction"]` in
   `presentation_prep.ipynb` vs `["papers/correctness_prediction/figures"]` in
   `answer_prediction_paper_visualisations.ipynb` — one of them lands a level deep.
6. **Stale docstring:** `mixed_text_answer_effects.py` says its plotting lives in
   `src.viz.visualisations_text_answer_effects`. That module does not exist; the plotting is
   inside the statistics module itself.

---

## Change log

Every change that alters a result goes here. Correctness beats number stability
(`docs/conventions.md`), so numbers are expected to move — the point of this log is that a
**conclusion** change is understood when it happens, not rediscovered later.

One row per change. Fill `Conclusion?` with **value only** or **conclusion changed**, and if
the latter, say what the new conclusion is.

| Date | Change | Affected | Old → New | Conclusion? |
|---|---|---|---|---|
| 2026-09-20 | **T1.3** — every figure and its numbers now go through one `save_output`; outputs move to `reports/<analysis>/{figures,tables}/` with key-value filenames | the whole `reports/` tree; 33 source files, 11 notebooks | paths and filenames change; **no computed value changes** | **no change.** Verified rather than assumed: loading `all_participants.csv` plainly and via the regeneration driver's category-typed path give byte-identical group counts (16051 / 3385 at threshold 4). |
| 2026-09-20 | **Stale saved numbers caught by the regeneration** — the old `report_data/` CSVs predate the current `all_participants.csv` | `correctness_by_seq_len_threshold`, 4 of 12 tables | thresh 4 all-participants: `> 4` n 16054 → **16051**, `≤ 4` n 3382 → **3385**; acc .82241 → .82238 and .92726 → .92733. thresh 5 and the gatherers/hunters variants shift by 1–3 trials | **value only** — three trials sit on the boundary and the data changed under the old CSVs (they were written before the Sept text-alignment fixes). The direction and size of every effect are unchanged. Not caused by T1.3. |
| 2026-09-20 | **Y-axis label corrected** on the area label×location heatmaps — `matrix_plot_ABCD` labelled *both* axes `area_screen_loc`; the y axis is indexed by `area_label` | all 320 `basic_stats_heatmaps` figures (now `attention_allocation`) | wrong axis label → correct | **no numeric change** — the pivot was always right, only its label was wrong. Every one of those figures is replaced. |
| 2026-09-20 | `mixed_area_comparisons` now writes its fixed-effects and pairwise tables **by default** | `attention_allocation` | previously `save_tables=False` while `save_heatmaps=True` | no change to values; the tables simply now exist for every figure |
| 2026-09-20 | **`rt_col` settled on `CONFIRM_FINAL_ANSWER_RT`** (Diana). Both defaults disagreed — the wrapper said `CONFIRM_FINAL_ANSWER_RT`, the plot function said `total_answering_RT`, a column that does not exist in `all_participants.csv` | `correctness_associations/correctness_by_total_answering_rt` (3 figures + 3 tables) | the figures now bin **raw confirm-press RT (s)** instead of the length-normalized ratio these plots used before | **conclusion unchanged, but the measure changed** — `findings.md` §5.3's numbers (bin centres 0.046 … 0.518) are the *normalized* variant and do not correspond to the current figures. ⚠️ **The binning is now poor**: raw RT has a long tail, so with `n_bins=10` **18,126 of 19,436 trials fall in bin 0** (0.8–13.4 s), then 1,198, then 90. Pass `x_max` (or `bin_width`) to get a usable spread. The measure is in the filename (`rt-CONFIRM_FINAL_ANSWER_RT`) so the two variants can coexist. |
| 2026-09-20 | **The dwell-proportion maps could not run at all** — `add_text_dwell_proportions` re-merged `area_dwell_proportion__{critical,distractor,outside}` from the paragraph cache, but `model_data.build_trial_level_model_df` already merges them from that same cache, so pandas suffixed both copies `_x`/`_y` and the next lookup raised `KeyError`. Pre-existing; unrelated to T1.3 | `text_qa_relationship` — the whole proportions half of Results §3.6 | analysis **crashed** → runs | **no value change.** Verified before fixing: the columns already on the frame are bit-identical to the cache (max abs diff 0, nothing unmatched), so using them is the same quantity. The join that remains now asserts full coverage instead of letting an unmatched trial become NaN. |
| 2026-09-20 | **Text↔QA maps renamed and all variants saved.** `scaling-normalized` alone did not say *which* quantity, and the proportion maps carried no distinguishing facet at all — they would have overwritten the reading-time maps | `text_qa_relationship` | 3 figures → **7**: `measure-reading_times` × {`scaling-normalized`, `scaling-pure`} × 3 groups, plus `measure-dwell_proportions` | no value change; the missing variants simply now exist. 45 tables (r / BH-adjusted p / n per panel). Note `_drop_empty_answer_screens` removes **9 of 19,436 trials** on the proportions path — explicit, printed, and belongs in Methods. |
| 2026-09-20 | `save` now accepts `None` = "use `set_output_defaults`" on all 103 plotting functions | every plotting entry point | hard `save=True`/`save=False` defaults silently overrode the session switch — `set_output_defaults(save=True)` produced **no output at all** for `RT_correlations` | no change to any value. **Behaviour change:** `run_all_*` no longer saves merely by being called — pass `save=True` or set the session default. Two data-writing functions (`build_rt_and_tfd`, `update_combined_cv_run`) keep their hard defaults deliberately. |
| 2026-09-20 | **Dominant-eye tie-break made deterministic.** `build_dominant_strategy_by_eye` used `sort_values(...).head(1)`, whose tie-break depends on row order; it now uses the shared deterministic rule | `findings.md` §1.6's crosstab, and only at the default `threshold=0.0` | **one hunter** (`l22_53`) moves between two strategy rows. Their dominance score is 0.2037, so they are excluded from every thresholded figure — at `threshold=0.5` nothing changes at all | **value only**, and a correctness fix: the old pick was order-dependent. Scores, `n_trials` and `n_total` identical for all 360. Separately, the per-eye bar charts now derive their heights from the crosstab instead of re-tallying: identical counts, but bars tied at the same height are now ordered deterministically rather than by hash order (hunters/Right: the two bars at count 1 swap) |
| 2026-09-20 | **Dominance computation moved out of `viz/` into `derived/pattern_breaking.py`** — prefix completion, dominance gap, strategy variety, dominant-strategy counts, the raw-vs-completed summary, and the eye crosstab | nothing numeric; `viz/` keeps only plotting | — | **no change** — verified by reproducing §1.1, §1.2, §1.3 and §1.5 exactly (mean gap, variety 6–28 modal 13, counts 96/110, top-2 71/20 and 89/14). Done so these numbers can be *saved* rather than living only inside a PNG (T4.0) |
| 2026-09-20 | `todo.md` **T1.1** — dominance threshold is `≥` everywhere, and the five copies of the modal-strategy pick were merged into `derived/pattern_breaking.dominant_strategy_by_participant` + `has_dominant_strategy` | `proportion_with_dominant_strategy`'s return value and the two printed lines in `run_all_strategy_plots`. No figure changes — the two thresholded figures already used `≥` | hunters 46.11% → **48.89%** raw, 49.4% → **53.33%** completed; gatherers 56.11% → **58.33%**, 57.8% → **61.11%** | **value only, and it is a convergence** — these are now the same numbers `summarize_before_after` already reported and that §1.2 quotes as canonical. Driven by 5 hunters and 4 gatherers scoring exactly 0.50. Dominant strategy per participant unchanged for all 360; bar-plot counts (96/110) unchanged; old and new logic verified bit-identical on the same frames |
| 2026-09-23 **(L1 + KnowQA both rebuilt)** | `todo.md` **T3.6** — first fixation duration drops `"."` instead of zero-filling. `create_mean_first_fix_duration` now coerces with `pd.to_numeric(errors="coerce")`, so unread words are excluded from the mean instead of entering it as 0 ms | `mean_first_fixation_duration__{answer_A..D, question}` and its five derived contrasts | **L1 (19,436 trials), measured:** A 123.91 → **191.96**, B 101.30 → **185.01**, C 92.42 → **181.75**, D 93.54 → **181.94**, question 21.12 → **146.03**; A–D spread **31.5 → 10.2 ms**; correlation with the matching `skip_rate` **−0.70…−0.84 → −0.017…+0.035**; `__contrast` 28.15 → **9.11**. New NaN per area 253 / 784 / 829 / 785 / 5,810 — exactly the pre-change count of exact zeros. **KnowQA (870 trials):** A 150.16 → 202.26, B 130.45 → 194.97, C 125.78 → 197.06, D 129.37 → 197.07, question 44.92 → 158.32; spread 24.4 → 7.3 ms; `__contrast` 21.63 → 6.70 | **conclusion changed, as predicted.** The between-area spread across A–D collapses from **24.4 ms to 7.3 ms**: what separated the four answers on this metric was coverage (skip rate), not fixation duration. The correct-vs-wrong `__contrast` shrinks from 21.6 ms to 6.7 ms — this metric no longer discriminates the selected answer. Dwell time and fixation count still do, and they keep the coverage signal by design (T3.13). **Verified not to move anything else:** `mean_dwell_time`, `mean_fixations_count`, `skip_rate`, `area_dwell_proportion`, `first_encounter_avg_pupil_size_z` and `is_correct` are bit-identical before/after, trial count 870 → 870. `SELECT_1_COLS` contains no first-fixation column and its NaN counts are unchanged, so **the headline model is untouched** — proved on L1 by diffing all 217 columns before/after: **exactly 10 changed, and all 10 are first-fixation-duration**. Every other column, including all ten headline features, is bit-identical, so the reported 0.83 cannot have moved |
| 2026-09-23 | **T3.6 side effect — the exclude-convention families now both carry NaN**, so the model's `fill_value = 0.0` (T3.14) covers more cells than the Methods number says | `logreg_model.py`'s fill; the Methods imputation count | KnowQA: first-fixation family **0 → 295** NaN cells; pupil family unchanged at 1,180. NaN propagates into the contrasts exactly as T3.14 describes — `__wrong_mean` is NaN only when *all three* wrong answers are unfixated (5 of 870), `__contrast` when either side is (14) | **no value change to any reported figure** — none of these columns is in `SELECT_1_COLS`. But T3.14's obligation ("count it, report it in Methods") now has a second family to count. The L1 figure cannot be stated until L1 is rebuilt |
| 2026-09-23 | `todo.md` **T3.14** — the model's `fill_value = 0.0` stays a **blanket** fill (Diana's ruling, amending the item's original point 2: simplest data prep, keep it), but stops being anonymous. Every filled cell is now counted (`imputed_cell_counts`), located (`imputed_mask_`) and, where the missingness has no documented cause, **warned about** (`unexpected_imputed_`) on its way to becoming 0 | `models/logreg_model.py`; `knowledge_regimes_analysis/descriptives.py` now calls the shared counter instead of counting inline | **nothing.** Verified against a verbatim reproduction of the old blanket-`fillna(0)` path on the headline model: coefficients and intercept identical, **max absolute difference 0.0**, both on a clean frame and on one carrying an injected unexplained NaN | **no change — and that is the point.** T3.14 was a documentation obligation, not a numerical one. What it produces is the Methods number: **253 imputed cells = 0.130% of the L1 headline feature matrix** (19,436 × 10), all in `mean_max_fix_pupil_size_z__correct`. One correction to the item's own text: the 137 genuine zeros that motivated keeping the mask are **135 in `first_encounter_avg_pupil_size_z__distance_closest`** plus one each in `__contrast` / `__distance_furthest` — structural, since `distance_closest` is exactly 0 whenever two areas tie, not scattered as the doc implied |
| 2026-09-23 | **Bug found and fixed during T6.1, worth its own row.** Moving the 18 paragraph RT/TFD columns to the paragraph table briefly gave them **two owners** — the trial frame still took them from `RT_and_TFD.csv` (an `all_participants.csv` built before T6.1 has them baked in) while the paragraph join also brought them. Pandas suffixed all 18 `_x`/`_y`, so they silently stopped existing under the names every feature set uses | `L1_model_ready_all_features.csv` for one build | 217 → 235 columns, 18 of them named `*_x` and 18 `*_y` | **caught before it reached anything** — by the before/after column diff, not by reading the code. Fixed by giving the columns one owner (`build_trial_level_rt_tfd_features` is answer-only) and by making the paragraph merge **raise on any column collision** instead of letting pandas disambiguate. Same failure shape as the dwell-proportion `_x`/`_y` crash on 2026-09-20; the lesson is that *moving* a column between producers is the moment to check for two producers, and `how="left"` will never tell you |
| 2026-09-23 | `todo.md` **T1.7 + T6.1** — the answer and paragraph screens become two independent pipelines sharing one set of metric functions. `derived/area_metrics.py` holds the eight metrics parameterized by grouping column; `derived/paragraph_prep.py` owns the paragraph screen end to end, including the per-span RT/TFD that the ANSWER pipeline used to build | `L1_paragraph_span_features.csv` (rebuilt); `reading_times.build_rt_and_tfd` (answer-only); `data_csv_generation` (no paragraph input); `model_data.PARAGRAPH_MODEL_COLS` (explicit join); `answer_RTs/features.py` (compatibility surface) | **QA side: nothing — and the model-ready table is bit-identical.** Diffing all 217 columns before/after the whole refactor: **0 changed**, same column set, same 19,436 trials, base rate 0.8407, and all 18 paragraph RT/TFD columns present and non-null from their new source. All seven QA metric functions reproduce `all_participants.csv` to ≤5e-13 (CSV round-trip) and `first_encounter` bit-identically. **Paragraph side, 24,046 trials, 33 → 51 columns** (the 18 RT/TFD columns moved in from `RT_and_TFD.csv`). Of 31 shared columns, 16 identical and 15 moved: the **12 pupil z columns** −0.58…−0.81 → **≈0** (T3.20's own-screen baseline), and **`num_label_visits`** critical 3.045 → 2.934, distractor 2.341 → 2.317, outside 4.908 → 4.621 | **value only, and two of the three movers are corrections.** The pupil shift is the clearest evidence T3.20's bug was material rather than theoretical: paragraph pupil sizes had been z-scored against the *answer* screen, so they sat ~0.6–0.8 SD "below typical" — an artifact of comparing two screens, not a property of anyone's pupil. Against their own screen they centre at 0, as a z-score must. The visit-count fall was **not** predicted: unifying exposed a *third* divergence nobody had recorded — the paragraph code treated an unresolvable fixation as a **separator**, so a return to the same span across it counted as a new visit (`[outside, ·, outside, ·, outside]` scored 3). The answer screen has always merged across dropped entries; the unified version matches it. Neither column reaches any model (`PARAGRAPH_MODEL_COLS` carries only the dwell proportions and RT/TFD), so **no paper number moves**. The comparison is clean: the raw paragraph reports have not changed since 2025-05, and neither T3.6 nor T3.18 touched the paragraph path |
| 2026-09-23 | **T3.6 completed — both pilots rebuilt** (`testrun_QA`, `second_test`) through Stage 1 + Stage 2 on their existing cleaned inputs, so their older identity scheme is untouched | `data/new_exp_try_runs/*`, `data/second_test_runs/*` | **exactly 10 of 199 columns changed on each, all first-fixation-duration.** testrun_QA `__answer_A` 114.66 → **168.67**; second_test 126.47 → **213.65**. Trial counts (36, 340) and base rates (0.6389, 0.6029) unchanged | **value only.** All four datasets are now on the same convention. Invariant checked on every one: **zero impossible zeros** in the five per-area first-fixation columns. (Zeros do appear in the *contrast* columns — 80 on L1 — and are legitimate: a contrast of 0 means the two sides were equal, and `distance_closest` is 0 whenever two areas tie.) |
| 2026-09-23 | **Cross-dataset join found while rebuilding the pilots.** `save_all_features` defaults to joining the paragraph cache, and `PARAGRAPH_SPAN_FEATURES_PATH` is a single **global** path — so the two pilots, which have no paragraph screen, were being joined against **L1's** paragraph table. Pre-existing, not introduced by T6.1, though T6.1 widened it from 3 columns to 21 | both pilots' model-ready tables | 217 → **196** columns; all 21 paragraph columns were **entirely NaN**, so nothing was ever wrong numerically | **no value change — the ids happen not to collide, which is the only reason it was harmless.** It would stop being harmless the moment a future dataset reused an L1 participant id. `save_all_features` now takes `include_paragraph_features`, the pilots pass `False` (matching KnowQA, which always did), and a zero-overlap paragraph join now **warns** rather than passing silently |
| 2026-09-23 | **T3.20 completed — KnowQA rebuilt on a per-participant pupil baseline** (was per `session_id`). `clean` was re-run as well as `pipeline`, because the session path overwrites the raw pupil columns with their millimetre values in place, so the cleaned reports had to come back in area units first | `data/KnowQA_runs/*` — 870 trials | **exactly 40 of 196 columns changed, all of them pupil; zero non-pupil movers.** Trial count, column set and base rate (0.7276) unchanged. Example shifts: `mean_avg_fix_pupil_size_z__answer_A` −0.0371 → −0.0738, `..._z__question` +0.3093 → +0.2299. `Auxiliary/participant_pupils.csv` now holds **6 rows — one per person** (4000–4005), pooled across their sessions, replacing the 80-byte pre-Stage-0 leftover | **value only.** The shifts are small because a person's sessions have similar baselines, but they are real and KnowQA's pupil features now mean the same thing as L1's. Sanity-checked against double-scaling: the z columns come out mean ≈ 0, sd ≈ 0.91 |
| 2026-09-23 | **A silent-corruption path found while doing that, and closed.** `add_zscored_pupil_columns_by` does not merely add `_z` columns — it overwrites the raw pupil columns with millimetres **in place**. So a report cleaned under `pupil_norm_unit="session"` holds mm, and running the participant path over it would scale again: millimetres treated as an area and square-rooted a second time | `know_qa_dataprep.run_pipeline` | nothing — caught before running | **no change, and that is the point.** Nothing would have raised, no column would have gone missing, and every pupil z-score in the run would have been quietly wrong. There was already a guard for the *opposite* mistake (`_require_pupil_z_columns`, session over participant-cleaned); only one of the two directions was checked, and the unchecked one is the one that corrupts rather than fails. `_refuse_participant_over_session_cleaned` now covers it |
| 2026-09-23 *(L1 paragraph rebuilt; **KnowQA rebuild outstanding**)* | `todo.md` **T3.20** — pupil baselines. Three rulings: the baseline unit is **the person for every dataset** (KnowQA switches from per-`session_id`, pooling a person's sessions and all three regimes); the **paragraph screen computes its own baseline** instead of borrowing the answer screen's; and **every person must have a baseline** — a missing one, or a non-positive SD, now raises instead of yielding a silently all-NaN `_z` column | `derived/pupil_norm.py`, `answer_RTs/features.py`, `data_prep/know_qa_dataprep.py`, `data_prep/data_csv_generation.py` | **(b) measured 2026-09-23** — L1's 12 paragraph pupil z columns moved from −0.58…−0.81 to ≈0 once each screen baselined against itself; see the T1.7/T6.1 row above. **(a) still outstanding**: every KnowQA pupil z-score (session → participant baseline). L1's *answer-side* pupil z is **unchanged**: `main()` already resolved its baseline from its own answer fixations explicitly | **expect value changes only, and no paper number at all.** Re-measured blast radius: `model_data` merges only `area_dwell_proportion__{critical,distractor,outside}` from the paragraph cache, so **no paragraph pupil column reaches any model**, and `RT_correlations` never references pupil — the only consumer is the parked `answer_RTs`. The KnowQA half touches live descriptives. ⚠️ **Watch for the new coverage guard firing on the L1 rebuild**: if any participant turns out to lack a baseline, the old code was silently emitting an all-NaN `_z` column for them and that is a finding, not a regression |
| 2026-09-26 | `todo.md` **T3.1** — the three correctness Fisher tests now run at trial grain. The fix was *not* "pass `trial_df`": each test **re-derived its own split** from raw columns, and for the dwell test that re-derivation (`groupby().transform()`) was itself what kept it IA-level. They now read the split the builder already computed (`seq_len` / `has_pattern` / `trial_mean_dwell`), so the test and the bars cannot describe different partitions, and `_check_trial_frame` **raises** on a duplicated `(participant_id, TRIAL_INDEX)` | `statistics/correctness_measures_tests.py`, `derived/correctness_measures.py`. **24 analyses**: `correctness_by_seq_len_threshold` (3 groups × 4 thresholds), `…_by_trial_mean_dwell_threshold` (3 × 3), `…_by_back_and_forth` (3, XYX only) | n **760,628 → 19,436** (all-participants) and **~380,000 → 9,717 / 9,719** (gatherers / hunters). Threshold-4 all-participants: OR **2.6440 → 2.7560**, p **0.0 → 4.26e-60**. Every OR moves, most of them *up*, because the IA-level version weighted each trial by its word count. **24 of 24 `__fisher.json` rewritten; 1 of 24 figures changed; 0 of 24 `__summary.csv` changed** | **conclusion changed — in exactly one of the 24.** `seq_len_threshold · gatherers · threshold-2` goes **p 7.685e-09 → 0.1886, significant → n.s.** (OR 1.2737 → 1.4530). The reason is visible in its own summary table: the `≤ 2` group holds **116 trials** against 9,601, so .862 vs .811 is not resolvable at the true n — the old p came from inflating 116 trials into ~4,500 interest-area rows. This is precisely the borderline case the item predicted. **The other 23 all survive**, every one at p < 1e-3; the headline claims of §5.1–5.3 are unaffected. Bars, n labels and Wilson CIs were always correct — verified, not assumed: no `__summary.csv` differs, and 23 of 24 figures are byte-identical, the 24th differing only in its significance star |
| *(retracted)* | `todo.md` T3.2 — I had this as "nest feature selection inside CV folds". **`SELECT_1_COLS` is a manual pick, not machine-selected**, so there is no leakage to fix and no rerun. What remains is a Methods sentence | — | — | **no change — my error, corrected 2026-09-05** |
| *(pending)* | `todo.md` T3.3 — coefficient CIs via participant-clustered bootstrap | every coefficient figure, `significant_only` filtering | CIs widen | expected **conclusion change** on which coefficients count as significant. |
| 2026-09-06 *(reruns 09-07)* | `todo.md` T3.18 — `add_IA_screen_location` now places words on the answer screen by their interest-area rectangle (`IA_TOP` / `IA_LEFT`) instead of by counting tokens in the stored text. The token-count assignment is still computed and compared; geometry wins where they differ, and every correction is printed | every per-area metric, the fixation sequences and everything built on them (strategies, XYX/XYXY, `seq_len`), last-visited labels, per-region RT/TFD, preference matching, and all `__correct` / `__wrong_mean` / `__contrast` features — but **only on the affected trials** | L1 **54 interest areas on 10 of 19,436 trials** (0.007% of interest areas); KnowQA 4 IAs on 1 of 154; second_test 24 IAs on 6 of 361; testrun_QA none. Verified identical to the current labels on every other trial | **value only.** The corrected labels were derived independently two ways — screen geometry, and reconciling `IA_LABEL` against the stored text — which agreed 10/10 on L1's affected trials. Too few trials to move any reported figure. **Datasets rebuilt 2026-09-07** — KnowQA and both pilots re-run, L1 re-run by Diana. Fixed in the same pass: `total_answering_RT_normalized` now divides by the **measured interest-area count** rather than the stored token count, which had it ~2% low on the 10 L1 `truncated` and 7 Study 2 `merged` trials. The question of whether that word was displayed is **answered, not assumed**: it rendered off the bottom of the screen (it would start at y ≈ 1404 against a 1401 maximum for any interest area anywhere in L1), so it was never readable and the measured count is the correct denominator. **Verified after the rebuild (2026-09-07):** 19,436 trials / 760,628 interest areas / 360 participants all unchanged, base rate still 0.8407, no area falls to `unknown`, and the on-disk labels agree with an independent reconstruction on **all** 19,436 trials while differing from the old token-count logic on exactly the 10. One downstream number did move: `mean_max_fix_pupil_size_z__correct` imputations **257 → 253** (§8) |
| 2026-09-25 | **T3.21** — participant-level scope became two explicit parameters (`scope_df` = which trials estimate the aggregate, `scope_by` = how they are partitioned), defaulting to the frame being featurised, pooled per participant. `n_strategy_trials_{with,no}_q` now carried; `regime` and `session_id` carried into the trial-level table | `derived/pattern_breaking.py`, `answer_correctness/model_data.py`, `cross_validation.py` docstring; L1 and KnowQA feature tables rebuilt | L1 217 → **219** cols, KnowQA 196 → **200**. **No shared column changed value** | **no change — additive only.** Verified rather than assumed: all eight pattern features recomputed under the default scope are bit-identical to the saved L1 table across 19,436 trials, and a full-table diff of both datasets found no moved column. What the default *does* change is availability: the scope is now asked for instead of inherited. Backups at `*.pre_t321.bak`. **Measured while verifying, not previously known:** estimating over the full KnowQA set rather than the per-regime slice flips `breaks_pattern_no_q` on **52 of 300** no-knowledge trials — the size of the leak the default avoids. And per-regime dominance varies sharply within a person (participant 4000: 0.224 full-knowledge → 0.630 partial-knowledge) where the pooled score is a flat 0.372 for everyone — descriptive only at n=6, but it is the shift the pooled scope cannot show |
| 2026-09-25 | **T3.21 rows 5, 7–9, 12, 13 — the remainder.** Prefix-completion map now learned **once over the whole dataset** instead of per group (Diana's ruling); `run_all_strategy_plots` gained `include_all=True`; `person_variance` gained per-participant trial counts; rows 7, 12, 13 stated rather than changed | `viz/visualisations_strategies.py`, `person_variance/{accuracy_characterization,univariate_consistency}.py`, three docstrings; both pilots rebuilt | completed prevalence **53.33% / 61.11% unchanged**; pilots 196 → 199 cols, nothing moved | **no change — and the null is the point.** The pooled map disagrees with the old per-group maps on 9 (hunters) / 4 (gatherers) rare prefixes, but none flip a participant past the 50% threshold. So the completed figures are **insensitive to how the repair was learned**, which pre-empts the obvious objection to a repair heuristic and is worth a sentence in the paper. **One number that did not exist now does:** all-participants prevalence **53.6% raw / 57.2% completed** (§1.2) — draft2's `X%`. |

---

## Open questions for Diana

1. Is the §9.1 correct-vs-distractor RT asymmetry in the paper? It is a strong, clean result
   with no home in `src/`.
2. "80% look at the answer they select" (draft2) vs ~68–71% of trials (§2) — which?
3. Is the dominant-eye × strategy association (§1.6) worth testing, or deliberately left out?
4. §5.2: does the longest-alternation counter-evidence change the XYXY claim?

**Resolved 2026-09-04/05:**

- Text Associations comes from `RT_correlations/` via `text_associations.ipynb`;
  `mixed_text_answer_effects.py` is a superseded strand (§7).
- The `>` vs `≥` threshold inconsistency is **fixed** (`todo.md` **T1.1**, 2026-09-20): `≥`
  everywhere, via the one `has_dominant_strategy` rule. Canonical prevalences are
  hunters 48.89% / 53.33% and gatherers 58.33% / 61.11%.
- The all-participants `X%` **is wanted**. `run_all_strategy_plots` already takes
  `include_all`; it is just called with `False`. No todo item — it is a rerun, not a fix.
- Dwell time and fixation count **stay coverage-inclusive**; only first-fixation duration
  changes (`todo.md` T3.6 / T3.13).
