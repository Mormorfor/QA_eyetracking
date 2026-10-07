# notebooks/builders

Run once per data refresh. These produce artifacts other things consume.

`clicks_selections_confirms` -> button clicks; `reading_times` -> the RT/TFD tables;
`create_merge_folds_csv` and `make_hunters_cross_val_list` -> the CV fold files;
`data_prep_new_exp` -> the two pilots; `feature_selection` -> feature-set JSONs for the
model (not currently used, but that is its purpose -- Diana, 2026-10-07).

The dataset builds themselves are now `scripts/build_dataset.py`.
