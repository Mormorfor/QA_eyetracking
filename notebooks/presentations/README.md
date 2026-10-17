# notebooks/presentations

Figures made for talks. **Never the source of a reported number.**

`presentation_prep` made the figures for a talk. Two things to know before running it:

* It writes example trials' rows as CSVs to a folder **outside the repository**.
* Its last cell saves a slide version of the first-visits matrix into
  `reports/scan_strategies/`, beside the paper's own figures for that analysis.

Its `collect_triples` docstring contradicts its filter (it says `is_correct == 1`, the code
filters `== 0`). The talk has been given, so this was left as it is; if anything is ever lifted
out of it, read the filter rather than the comment.
