"""The experiment source: how the Study 2 materials and design were generated.

This is the one package that produces *stimuli and designs* rather than consuming
recordings. Its sibling `ingest/` reads what came back; this is what went out.

Lifted out of `experiment_builder/` in stage F (Diana, 2026-10-07: *"i want to put it
somewhere where it is clear that this is the experiment source, in case i ever design
another one... lets put it where it would have to go to begin with in a good repo"*).

The notebooks that *ran* it are `notebooks/experiment/`. They have run, their outputs are
committed, and they will probably not run again -- what moved here is the reusable half, so
a second experiment would start from functions rather than from someone else's cells.

**Determinism:** `lists.py` seeds its RNG (42) at module level. The batch output is
reproducible only while the source CSV's row set is unchanged -- reordering or adding rows
changes the assignment even with the same seed.
"""
