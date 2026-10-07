"""Fitting the correctness outcome with a mixed-effects model. **Parked.**

Everything to do with mixed-effects *modeling* lives here (Diana, 2026-10-07): the Julia
and R backends (`models.py`) and the random-effects figures that read their output
(`plots.py`).

**Not to be confused with mixed-effects *difference statistics*.**
`analyses/attention_allocation/stats.py` fits a statsmodels `MixedLM` to test whether the
five screen areas differ -- that is **paper code**, settled 2026-09-05, and it is a
different thing that happens to share a method name.

⚠️ **Two runner functions did not make it here**, and the reason is worth knowing:
`run_full_features_correctness_julia_glmer_bundle` and `..._fit_all` are still in
`analyses/correctness_prediction/run.py`. They share **eight** helpers with the logistic
bundles -- the train/test frame builder, feature resolution, summary saving and three plot
orchestrators bound to `analysis="correctness_prediction"`. Moving them would mean either
duplicating that plumbing or having `explorations/` import `analyses/`, which the layering
rule forbids. The clean fix is to hoist the shared half into `modeling/` with the analysis
name as a parameter; that is a signature change, so it was flagged rather than done.
"""
