"""
text_qa_relationship
====================

How time on the parts of the paragraph relates to time on the answers and the
question: the paper's Text Associations results (Study 1).

Each paragraph has three parts: `critical` (holds the answer), `distractor`
(supports a wrong answer) and `outside` (the rest). A correlation map has these
three parts as rows and the four answers and the question as columns. Each cell
is the correlation, over trials, between time on that part of the paragraph and
time on that answer. The maps are made for two reading-time measures, RT and
TFD (total fixation duration; `docs/glossary.md` §10), and for dwell
proportions (each area's share of its screen's looking time).

The notebook that runs them is `notebooks/drivers/text_associations.ipynb`.
Figures and their tables are saved under `reports/text_qa_relationship/`.

Layout
------
`columns`      which columns form the rows and columns of a map, and their labels
`data`         loading the trial tables: all participants, hunters, gatherers
`correlations` the maps themselves: pooled r, and per-participant r tested vs 0
`comparisons`  testing map cells against each other, and one group against the other
`bootstrap`    the pooled r with a bootstrap over participants (a robustness check)
`proportions`  the same maps for dwell proportions instead of reading times
`plots`        annotated heatmaps, forest plots and summary tables
`report`       one-call helpers the notebook uses

Why the tests are not `scipy.stats.pearsonr` on the pooled trials
-----------------------------------------------------------------
Trials are nested in participants (360, about 54 trials each) and in texts. A
p-value that treated the ~19k trials as independent would call any |r| above
0.014 significant. The tests here take the participant as the unit (the nesting
in texts is not modelled):

    per participant p:  r_p = corr(x, y) over that participant's trials
                        z_p = arctanh(r_p)                      (Fisher z)
    vs. 0:              one-sample t-test on z_p
    cell vs. cell:      paired t-test on z_p(cell_a) - z_p(cell_b)
    group vs. group:    two-sample t-test on z_p (different people in each group)

See `correlations` and `comparisons` for the details and caveats.
"""

from src.analyses.text_qa_relationship.columns import (
    ANSWERS,
    METRICS,
    REGIONS,
    answer_cols,
    map_cell,
    pretty_labels,
    region_cols,
)
from src.analyses.text_qa_relationship.data import (
    FEATURE_SOURCES,
    load_all_participants_features,
    load_features,
    load_group_features,
)
from src.analyses.text_qa_relationship.correlations import (
    corr_map_stats,
    fisher_z_by_cluster,
    pooled_corr_map,
)
from src.analyses.text_qa_relationship.comparisons import (
    compare_cells,
    compare_groups,
    compare_rows_within_column,
)
from src.analyses.text_qa_relationship.bootstrap import cluster_bootstrap_corr_map
from src.analyses.text_qa_relationship.plots import (
    plot_contrasts,
    plot_corr_map,
    plot_corr_map_pair,
    plot_pooled_maps,
    shared_vmax,
    summarise_map,
)
from src.analyses.text_qa_relationship.report import (
    bootstrap_maps,
    group_bootstrap_maps,
    group_contrast,
    group_map_stats,
    metric_map_stats,
    plot_bootstrap_maps,
    plot_maps,
    reference_contrasts,
    region_contrasts,
)
from src.analyses.text_qa_relationship.proportions import (
    add_text_dwell_proportions,
    answer_proportion_cols,
    group_proportion_map_stats,
    load_all_proportion_features,
    load_group_proportion_features,
    load_proportion_features,
    plot_proportion_maps,
    proportion_col,
    proportion_group_contrast,
    proportion_map_stats,
    proportion_region_contrasts,
    proportion_summary,
    region_proportion_cols,
)
from src.analyses.text_qa_relationship._utils import significance_stars

__all__ = [
    # columns
    "REGIONS",
    "ANSWERS",
    "METRICS",
    "region_cols",
    "answer_cols",
    "map_cell",
    "pretty_labels",
    # data
    "FEATURE_SOURCES",
    "load_features",
    "load_all_participants_features",
    "load_group_features",
    # correlations
    "pooled_corr_map",
    "fisher_z_by_cluster",
    "corr_map_stats",
    # comparisons
    "compare_cells",
    "compare_rows_within_column",
    "compare_groups",
    # bootstrap
    "cluster_bootstrap_corr_map",
    # plots
    "plot_contrasts",
    "plot_corr_map",
    "plot_corr_map_pair",
    "plot_pooled_maps",
    "shared_vmax",
    "summarise_map",
    # report
    "metric_map_stats",
    "group_map_stats",
    "plot_maps",
    "region_contrasts",
    "reference_contrasts",
    "group_contrast",
    "bootstrap_maps",
    "group_bootstrap_maps",
    "plot_bootstrap_maps",
    # proportions
    "proportion_col",
    "region_proportion_cols",
    "answer_proportion_cols",
    "add_text_dwell_proportions",
    "load_proportion_features",
    "load_all_proportion_features",
    "load_group_proportion_features",
    "proportion_map_stats",
    "group_proportion_map_stats",
    "plot_proportion_maps",
    "proportion_region_contrasts",
    "proportion_group_contrast",
    "proportion_summary",
    # misc
    "significance_stars",
]
