"""
RT_correlations
===============

Associations between normalized reading times on the **text regions**
(distractor / critical / outside) and on the **answers / question**
(A / B / C / D / question), for both RT and TFD.

Everything the `notebooks/text_associations.ipynb` correlation-map section
needs lives here.

Layout
------
`columns`      which columns form the rows/cols of a map, and how to label them
`data`         loading the all-participants and per-group feature frames
`correlations` the maps themselves: pooled r, and per-participant r tested vs 0
`comparisons`  testing map cells against each other, and groups against groups
`bootstrap`    cluster bootstrap of the pooled r (robustness check)
`plots`        annotated heatmaps and tidy summary tables
`report`       one-call helpers the notebook uses

Why the tests are not `scipy.stats.pearsonr` on the pooled frame
---------------------------------------------------------------
Trials are nested in participants (360 x 54) and in texts, so a p-value that
assumes ~19k independent rows would call r = 0.014 significant. The tests here
use the participant as the unit of analysis:

    per participant p:  r_p = corr(x, y) over that participant's trials
                        z_p = arctanh(r_p)                      (Fisher z)
    vs. 0:              one-sample t-test on z_p
    cell vs. cell:      paired t-test on z_p(cell_a) - z_p(cell_b)
    group vs. group:    two-sample t-test on z_p (disjoint participants)

See `correlations` and `comparisons` for the details and caveats.
"""

from src.statistics.RT_correlations.columns import (
    ANSWERS,
    METRICS,
    REGIONS,
    answer_cols,
    map_cell,
    pretty_labels,
    region_cols,
)
from src.statistics.RT_correlations.data import (
    FEATURE_SOURCES,
    load_all_features,
    load_features,
    load_group_features,
)
from src.statistics.RT_correlations.correlations import (
    corr_map_stats,
    fisher_z_by_cluster,
    pooled_corr_map,
)
from src.statistics.RT_correlations.comparisons import (
    compare_cells,
    compare_groups,
    compare_rows_within_column,
)
from src.statistics.RT_correlations.bootstrap import cluster_bootstrap_corr_map
from src.statistics.RT_correlations.plots import (
    plot_contrasts,
    plot_corr_map,
    plot_corr_map_pair,
    plot_pooled_maps,
    shared_vmax,
    summarise_map,
)
from src.statistics.RT_correlations.report import (
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
from src.statistics.RT_correlations._utils import significance_stars

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
    "load_all_features",
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
    # misc
    "significance_stars",
]
