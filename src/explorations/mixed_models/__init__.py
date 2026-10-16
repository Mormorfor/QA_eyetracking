"""Fitting the correctness outcome with a mixed-effects model. **Parked.**

    models.py   the Julia (MixedModels) and R (lme4) backends
    run.py      the same fit-score-save runs as `analyses/correctness_prediction/run.py`,
                with a mixed model; they reuse that module's helpers and plots
    plots.py    figures of the random effects these backends estimate

Out of scope for this paper (`research-context.md` section 6). Not to be confused with
`analyses/attention_allocation/stats.py`, which is paper code: it uses a mixed model to
test whether the answers differ on a measurement, not to predict correctness.
"""
