"""Fitting the correctness outcome with a mixed-effects model. **Parked.**

    models.py        the Julia (MixedModels) and R (lme4) backends. The Julia one is the
                     paper's logistic regression plus random intercepts per participant and
                     per item, and plugs into `modeling/crossval.py` the same way
    run.py           the paper's single-split and full-fit runs, with the Julia model; they
                     reuse `analyses/correctness_prediction/run.py`'s helpers and plots
    plots.py         figures of the random effects the Julia backend estimates
    overview.ipynb   the Julia model against the logistic regression: cross-validated
                     accuracy, coefficients, variance components, one split's outputs

Needs Julia: run `src/vendor/julia_integration/julia_test.ipynb` once first.

Out of scope for this paper (`research-context.md` section 6). Not to be confused with
`analyses/attention_allocation/stats.py`, which is paper code: it uses a mixed model to
test whether the answers differ on a measurement, not to predict correctness.
"""
