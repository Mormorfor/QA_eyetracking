# notebooks/experiment

How the Study 2 materials were generated. The experiment source.

The reusable half is `src/experiment/` -- `lists.py` (the double Latin square: 27 lists x 6
regime orderings = 54) and `texts.py` (spelling and the Adv/Ele rephrasing carry-over).
These notebooks are the orchestration that ran it, kept because they record *what was
actually built*, not because they are expected to run again.

**Determinism:** the batch output is reproducible only while the source CSV's row set is
unchanged. Same seed plus a reordered or extended input is a different assignment.
