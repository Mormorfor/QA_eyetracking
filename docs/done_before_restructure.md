# What was fixed before the restructure

A plain-language record of the work done between **2026-09-20 and 2026-10-06**, while the
repo was still in its old shape. Written so it can be read later without reconstructing the
argument from `todo.md`.

**Why this file exists.** The restructure moves almost every file. Once that happens, "when
did this change and why" gets hard to answer from `git log` alone, because a move and an edit
look similar in a diff. This is the before-picture, in words.

**How to read it.** Grouped by *what kind of problem it was*, not by item number — the `T…`
ids are kept as cross-references into `todo.md`, where the full reasoning and the measurements
live. Every claim here was re-verified against the code on **2026-10-06**, not copied from the
todo list.

---

## Where the repo ended up

| | |
|---|---|
| Modules that import cleanly | **87 of 91** — the four that don't are deliberately parked (see the after-doc) |
| `reports/` | 1,019 files, **0** zero-byte, every figure with its numbers saved |
| Longest output path | 151 chars, so the repo checks out on Windows from any root up to **108** characters (was 60) |
| Figure index (`manifest.json`) | 710 entries, **0** broken references |
| Datasets rebuilt | L1 and KnowQA fully; both pilots through Stage 1 + 2 |

---

## 1. Things that were quietly producing wrong numbers

This was the biggest group, and the reason the cleanup happened at all. None of these crashed
— they all completed successfully and returned something plausible.

### The significance tests were counting words, not trials — **T3.1**

Three correctness tests were handed the word-level table instead of the trial-level one. Since
each trial covers roughly 39 words, every sample size was inflated about 39-fold: one test
reported n = 760,628 where the honest number was 19,436. With n that large almost anything
reaches significance.

**What changed:** the tests now read the trial-level split the plotting code had already
computed, and refuse to run on a frame with duplicate trials rather than silently testing
whatever they were given. **One result of 24 flipped to non-significant.** The rest survived.

### Unread words were being counted as fixations of length zero — **T3.6**

If a participant never looked at a word, the tracker writes `"."` for how long their first
fixation on it lasted — because there wasn't one. The code was turning that `"."` into `0` and
averaging it in. A fixation of zero milliseconds is not a thing that can happen, so this was
averaging an impossible value into a real measurement.

The effect was bigger than it sounds. It made the metric mostly a restatement of *how often
words were skipped* rather than *how long fixations lasted*.

**What changed:** unread words are now left out of that average instead of counted as zero.
The consequences are the largest single change in this whole effort — see §5.

### A displaced quote was shifting every area boundary — **T3.18**

A guard existed to catch stored text that didn't match what was displayed, but nobody had
confirmed the bug was real. It was — and in **three** different forms, not one. Twenty Study 1
trials were affected and had never been looked at.

**What changed:** rather than repairing the text, area labels now come from where each word
actually sat on the screen. Checked carefully: it reproduces the old labels on 19,426 of
19,436 trials and differs only on the ten genuinely misaligned ones.

### Pupil sizes were measured against the wrong baseline — **T3.20**

A pupil z-score needs a per-participant baseline. The paragraph-reading code was using
whichever default was lying around, which meant paragraph pupil sizes were normalised against
the *answer screen* of a different dataset. **What changed:** each dataset now writes its own
baseline, and every consumer asks for the right one.

### Joins that could silently drop or invent rows — **T3.17, T3.7, T3.5**

Several merges could quietly lose trials or fill in blanks. Two facts you confirmed — every
trial appears in every feature block, and every trial has a confirmed answer — were being
treated as situations to cope with rather than things that must be true. **What changed:**
they are now assertions. If one ever fails the run stops and says which trial, instead of
imputing and carrying on.

### Participant-level features silently meant different things — **T3.21**

Things like "this person's dominant scanning strategy" were computed over whatever rows
happened to be passed in. Analyse one knowledge regime and you'd silently get a different
quantity than the name suggested.

**What changed:** which trials feed the estimate is now an explicit argument, and the trial
count travels with the value so you can always tell. The default is the safe one — a score
built from all of someone's trials and attached to a training row leaks information from the
test set. On Study 2 this is measurable: the wrong scope flips a feature on 52 of 300 trials.

---

## 2. Things that were simply broken, and nobody knew

All of these were off the paper's critical path, which is why they went unnoticed.

- **The feature-set generator raised an error every time — T2.1.** It referenced three
  constants that had been deleted. The fix was *removal*, not restoration: those constants
  named columns nothing builds any more, so restoring them would have resurrected 31 feature
  sets pointing at nothing.
- **A whole block of features was silently missing — T2.4.** A function looked for columns
  starting `last_visited_`, but the columns are actually called `last_before_confirm…`. It
  returned an empty list, so any run using the default feature set had been quietly excluding
  the last-fixation features. It now returns all 8.
- **The mixed-effects area models could not fit a single model — T2.7.** Found late, and the
  most consequential of these: a pandas upgrade changed how text columns are stored, and the
  modelling library can't read the new form. It failed before fitting anything. This is paper
  code — it backs the *Attention allocation* Results section — so "just re-run those figures"
  had been **blocked for weeks without anyone knowing.** Fixed, and the family now runs end to
  end.
- **A helper promised more than it delivered — T2.3.** Renamed to say plainly that it only
  works with the mixed-effects backend, which is what it was always for.

---

## 3. One concept, two implementations

Wherever the same idea was written twice, the two copies had drifted.

- **Opening-scan strategy — T1.1, T1.6.** Existed in two places with different parsing,
  different tie-breaking, and different threshold operators (`>` in one, `≥` in another), so
  the same data printed 46.1% in one place and 48.9% in another. There is now **one**
  implementation. A *third* copy turned up in the dominant-eye analysis with an unstable
  tie-break that moved one participant.
- **The eight per-area metrics — T1.7.** The answer-screen and paragraph-screen versions were
  separate code, and one of the eight had silently diverged. Now one set of functions.
- **Two `wilson_ci`s and three nested copies of it — T1.5.** All formula-identical; now one.

---

## 4. Where results go, and whether they survive

Previously, whether a result was saved at all depended on which module produced it. Nine of
fifteen figure topics had **no numbers saved anywhere** — the values existed only inside the
pictures.

- **One save path, and the numbers are not optional — T1.3.** Everything goes through
  `save_output`, which *requires* you to hand it the numbers behind the figure. A figure that
  genuinely has none says so explicitly. 81 call sites, 33 files, 11 notebooks.
- **Outputs reorganised** into `reports/<analysis>/figures/` and `…/tables/`, so a figure
  missing its numbers is visible at a glance instead of needing two trees compared.
- **The text↔answer analysis saved nothing at all — T4.1.** A live Results subsection existed
  only as output cells inside a 1.2 MB notebook. Now saved like everything else.
- **Hardcoded paths removed — T5.3.** About 40 literal `"../reports/…"` paths, which only
  worked if you happened to run from the right folder. Now zero, in both code and notebooks.
- **Filenames shortened.** Names had grown to within **one character** of Windows' path limit,
  meaning the repo would only check out from a very short folder path — a genuine problem for
  a public release. One table of abbreviations (`all_participants` → `all_P`) fixed it, and
  the full names are still kept in the figure index so nothing is lost.
- **Sweeps now write one table, not hundreds.** The attention analysis was writing **600
  separate files averaging 506 bytes each**, every one a 4×4 grid. They are now 4 long tables
  with the settings as columns — the same numbers, in a form you can actually filter and
  group.

---

## 5. The numbers that moved

Full detail is in `findings.md`'s change log. In brief:

| What | Effect | Was it just a value, or a conclusion? |
|---|---|---|
| **First-fixation duration** (T3.6) | Answer areas go from 125.9 / 96.2 / 88.4 / 90.6 ms to 192.5 / 183.3 / 180.6 / 181.1. Between-area spread **37.5 → 11.9 ms**. Significant pairwise contrasts **56/72 → 26/72**, every flip in the same direction | **Conclusion changed.** "The selected area wins on every attention metric" is now **nine of ten** — this metric stops separating the areas once unread words are excluded. Which is exactly what was predicted |
| **Fisher tests at the right grain** (T3.1) | n 760,628 → 19,436 | One of 24 results became non-significant; the rest held |
| **Fold averaging weighted by size** (T3.10) | Largest shift 0.2 percentage points | Value only — no model ordering changed |
| **Stale group caches rebuilt** | Group correlation maps moved by at most 0.0014 | Value only — **zero** significance changes |
| **Confidence intervals clustered by participant** (T3.3) | Intervals widen | Affects which coefficients count as significant |

> A note on the first row, because it is the one that matters for the paper. The new values
> were confirmed three independent ways: they match a per-word calculation done by a completely
> different route; they land where this was predicted to land back in September; and a control
> metric (`mean_dwell_time`) is **identical before and after**, 57/72 either way. So it is the
> fix working, not the pipeline wobbling.

---

## 6. Three items were removed from the list

Recorded so nobody re-discovers them as open work.

- **T1.8** — a stray `index` column that differed between two datasets. Removed 2026-09-27;
  it stopped being true.
- **T3.2** — "say in Methods the features were hand-picked". An earlier version of this item
  claimed the headline model's features came out of an automated search and that the accuracy
  was therefore optimistic. **That was wrong** — they were chosen by hand on domain grounds,
  so there is no leakage and no caveat owed.
- **T3.15** — folded into T3.21, which treats the same question as one instance of a general
  problem rather than a quirk of one function.

---

## 7. Every item, accounted for

The sections above are grouped by theme and don't name every id. This is the complete list, so
nothing is left unaccounted for. **Status re-checked against the code on 2026-10-06**, not
copied from `todo.md`.

| Status | Items |
|---|---|
| ✅ **Done** — verified | **T1.1** threshold + one dominance fn · **T1.3** one save path · **T1.4** phantom constant names in comments · **T1.6** one starting-strategy impl · **T1.7** one per-area metric impl · **T2.1** generator runs · **T2.3** Julia-only helper renamed · **T2.4** missing last-fixation features · **T2.5** was never broken · **T2.7** pandas 3 / patsy · **T3.1** Fisher grain · **T3.3** clustered CIs on the paper path · **T3.6** first-fixation duration · **T3.17** invariants asserted · **T3.18** text misalignment · **T3.20** pupil baselines · **T3.21** explicit scope · **T4.1** text↔QA analysis saves · **T5.3** no hardcoded output paths · **T6.1** paragraph prep split out |
| ✅ **Done** — the T3.4 umbrella | **T3.4** collects the smaller number-movers, all closed: **T3.5** unreachable `is_correct` fallback → assertion · **T3.7** silent all-zero reading times → join assertion · **T3.8** pupil baseline (absorbed into T3.20) · **T3.9** stop pooling validation and test regimes into one average · **T3.10** fold average weighted by fold size · **T3.11** in-place mutation / ordering dependency · **T3.12** the blanket zero-fill (decided, tracked as T3.14) |
| ✅ **Decided, and the decision is in the code** | **T3.13** dwell time and fixation count stay coverage-inclusive — the asymmetry in that metric family is deliberate, **do not "fix" it for tidiness** · **T3.14** the zero-fill stays, but is named, counted and left identifiable, so it is a stated modelling choice rather than a silent one |
| ↪️ **Absorbed** | **T3.15** → **T3.21**, which treats it as one case of a general problem instead of a quirk of one function |
| ␀ **Removed from the list** | **T1.8** stopped being true · **T3.2** was based on a mistaken premise (see §6) |
| — **Never existed** | **T1.2** was never assigned · **T3.16** (using the A>B>C>D answer ordering in the model) was considered and declined. Noted so nobody goes looking |
| ✅ **Closed 2026-10-06** | **T1.5** — nine of its ten cleanups are done; the tenth, the `collect_triples` comment-vs-code contradiction, was ruled **no action**: the talk has been given and the notebook is kept only for possible reuse. Read the filter, not the comment, if anything is ever lifted out of it |

Everything not in this table is in `todo_after_restructure.md`.

---

## 8. Checks that were run rather than assumed

`todo.md` had a standing list of claims marked "read from the code, never executed". **All six
have now been run.** Two turned out differently than expected:

- One supposed breakage (`statistics.ipynb` calling a renamed constant) **was not real** — the
  notebook already used the correct name.
- One supposed-safe thing **was** real: the missing last-fixation features (T2.4).

The habit is worth keeping: of the items in that list, roughly a third did not behave the way
reading the code suggested.
