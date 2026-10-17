# src/vendor/cv_folds

The two notebooks here make the cross-validation folds in `data/cv_folds/`. Only part of this
folder is vendored code:

* **`create_merge_folds_csv.ipynb`, cells 0–11, are the lab's code (Shubi's), kept as
  received.** They merge the per-fold lists of held-out paragraphs and participants
  (`HuntingIsCorrectFolds/items/` and `subjects/`) into one file,
  `HuntingIsCorrectFolds/all_folds_subjects_items.csv`.
* **The rest is this project's own, and is edited here.**
  * `make_hunters_cross_val_list.ipynb` turns that merged file into `HuntersFolds/`: one file
    per fold, in the same format as `GatherersFolds/`, with each trial's details joined in
    from the hunters' tables.
  * `create_merge_folds_csv.ipynb`, cells 12–17, check that the two groups hold out the same
    paragraphs, then build `HuntersRefolded/` and `GatherersRefolded/` — the folds the
    cross-validation uses — from `HuntersFolds/` and `GatherersFolds/`.

Building the folds from scratch runs in this order:

1. `create_merge_folds_csv.ipynb`, cells 0–11 -> `all_folds_subjects_items.csv`
2. `make_hunters_cross_val_list.ipynb` -> `HuntersFolds/`
3. `create_merge_folds_csv.ipynb`, cells 12–17 -> `HuntersRefolded/` and `GatherersRefolded/`

`GatherersFolds/` is an input; nothing in this repo makes it.
