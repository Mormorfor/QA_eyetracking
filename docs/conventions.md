# Conventions — how to work in this project

Diana's standing preferences. Project-independent and durable: this file is where a new
preference goes, so `CLAUDE.md` doesn't have to change.

**Add to this file freely.** It is meant to grow.

---

## Scientific integrity of the code

**These are the most important rules in this file.** They follow from the framing in
`CLAUDE.md`: this is scientific code, so clarity, traceability and fidelity to the data
outrank robustness and convenience, and failing loudly is preferable to quietly altering the
analysis.

### Never silently discard data

Do not filter out observations, rows, trials, subjects, values, or files merely because they
are inconvenient, malformed, unexpected, or cause downstream errors. **Any exclusion must be
scientifically motivated, explicit in the code, and visible to the user.** If something is
preventing the code from working at all, it needs to be thoroughly discussed — not routed
around.

### Never silently replace problematic values

Do not automatically replace `NaN`, `Inf`, missing values, failed calculations, empty groups,
or invalid inputs with `0`, means, defaults, clipped values, or other substitutes unless that
behaviour is explicitly part of the analytical specification. **Every such case needs to be
examined and decided on by Diana.**

### Do not change the estimand to avoid edge cases

If a metric cannot be computed under its stated definition for some observations, **preserve
that fact.** Do not subtly redefine the metric, the denominator, the population, the
aggregation, or the inclusion criterion simply to obtain a numerical result.

### Prefer errors and warnings over hidden corrections

If an assumption is violated, raise an informative error or produce a clearly visible warning.
**A crash with a useful explanation is preferable to code that completes successfully while
producing scientifically different results.**

### Make assumptions explicit

Important assumptions about data structure, valid ranges, missingness, exclusions, grouping,
denominators, units, and statistical definitions should appear directly in the code or the
documentation, rather than being inferred implicitly.

### Do not introduce defensive behaviour without justification

Production-style safeguards — automatic fallback values, exception swallowing, permissive
parsing, clipping, coercion, skipping failed cases — **may be scientifically harmful.**
Introduce them only when their intended scientific meaning is clear.

> **How this sits with "don't dummy-proof" below:** they are the same principle, not opposites.
> Skip the validation that only guards against a caller mistake — wrong type, missing file,
> bad argument — because that failure is loud and immediate anyway. Write the check whenever
> the alternative is a *plausible-looking wrong number*. Defensive code that keeps a run alive
> is the thing to avoid; a check that stops a run and says why is the thing to add.

Correcting the codebase to meet these principles is a **significant part of the planned
restructure** — see `docs/todo.md` T3 and T6. Much of the existing T3 list is already
instances of exactly these violations.

---

## Code style

### Don't dummy-proof

This code has one user and one machine. It does **not** need defensive input validation,
type-checking of arguments, guards against wrong file formats, or "helpful" error handling
for situations that won't arise. Assume the caller passed the right thing.

Write the assertion **only** where a silent wrong answer is the alternative — a join whose
coverage must be total, a frame whose grain must be trial-level. Those are worth catching
because the failure mode is a plausible-looking number, not a crash. Everything else should
just crash, loudly, close to the mistake.

```python
# Not wanted
def load(path):
    if not isinstance(path, (str, Path)):
        raise TypeError(f"path must be str or Path, got {type(path)}")
    if not os.path.exists(path):
        raise FileNotFoundError(f"No such file: {path}. Did you mean...")
    ...

# Wanted
def load(path):
    return pd.read_csv(path)

# Also wanted — the failure here would be silent and wrong
merged = left.merge(right, on=KEY, how="inner")
assert len(merged) == len(left), f"join dropped {len(left) - len(merged)} rows"
```

### Don't add abstraction that wasn't asked for

No wrapper classes, no config systems, no plugin registries, no "for future extensibility".
If something is used once, write it once. The existing `FUNCTION_REGISTRY` in
`data_csv_generation.py` is enough machinery for this project.

### Prefer editing over rewriting

Change the lines that need changing. Do not reformat surrounding code, rename things in
passing, or restructure a function while fixing a bug in it — it makes the diff unreviewable
and hides the actual change.

### Comments explain *why*

The existing codebase is unusually well commented and the comments are genuinely
load-bearing — `constants.py` and `data_paths.py` explain the reasoning behind schema
decisions, and `know_qa_dataprep.py` documents why it diverges from the L1 path. Match that
standard. Don't add comments that restate the code.

---

## Working with Diana

### She writes the paper; Claude does the code

Anything under `papers/` is hers. Claude can read it to know what the code must support, and
never writes there.

### Ask rather than invent

When intent is unclear — what a name should be, whether an analysis is still wanted, which
of two definitions is canonical — ask. Do not pick a plausible answer and proceed as though
it were established. Guesses that look like facts are worse than an open question.

### Don't log speculative improvements

`docs/todo.md` is work that is going to happen, not a catalogue of everything that could be
done. If you notice an optional improvement — a feature that might carry more signal, a
refactor that isn't needed, an analysis worth trying — **say it in conversation and let Diana
decide**. Only write it into `todo.md` if she says yes.

An unfiltered todo list is worse than a short one: it buries the items that matter and makes
the whole file feel optional.

### Say when something is unverified

Distinguish "I ran this and it does X" from "I read the code and believe it does X". The
second is a hypothesis. Mark it as one. `docs/todo.md` uses ✅ / ⚠️ for exactly this.

### Propose before multi-file changes

Tool-level permission prompts cover whether a single edit happens; they don't cover whether
a ten-file refactor was the right shape. For anything spanning several files, or anything
that changes a number that appears in the paper, show the plan first.

### Correctness beats number stability — but log what moves

At this stage Diana would rather the numbers be right than unchanged. Don't hesitate to make
a correctness fix, rerun the analysis, and regenerate figures because a published value would
shift. Preserving a number for its own sake is not a goal.

What *is* required is a record. For every change that alters results, note it in the change
log at the end of `docs/findings.md`: what changed, old → new, why, and — most importantly —
**whether a conclusion changed or only a value**. A conclusion change is not a problem, but it
must be understood at the time rather than rediscovered months later when the paper is being
defended.

Fixes that move numbers are collected in `todo.md` T3.

---

## Release standards

The repo will be **public, alongside the paper**. That means:

- **It must run from scratch** on a clean machine, with instructions.
- It must **not** be defensively engineered — see above. Running from scratch and being
  idiot-proof are different goals; only the first one matters here.
- Entry points, build order, and the `L1` = native-speaker naming need documenting, because
  a reader will otherwise misread them.

## Keep, don't delete

Abandoned analyses stay in the repo as **future directions**, organized and labelled rather
than removed. `answer_loc/`, `clusters/`, the older `statistics/mixed_*.py` strands and the
Julia/R model backends are all in this category. If something genuinely has to go, it goes to
`archive/` and gets a line in `docs/status.md` saying why.

---

## Notes to future Claude

- `docs/findings.md` is the results ledger. **Read it before recomputing anything** — many
  numbers exist only inside saved PNGs, and regenerating blind risks overwriting a figure
  whose numbers nothing else records.
- `docs/pitfalls.md` is the list of things that have actually gone wrong here. Read it before
  touching feature generation or statistics.
