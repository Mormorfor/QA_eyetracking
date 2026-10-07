# notebooks/drivers

Thin drivers: load, call, display. One per live analysis.

These are the notebooks a number in the paper can come from. Each drives code in
`src/analyses/`; none of them should *define* an analysis. Where one still does, that is
a job for later -- stage F moved them, it did not make them thin.

Most of what they do is also reachable without Jupyter: `scripts/run_analysis.py --list`.
