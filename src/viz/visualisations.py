"""
Public plotting API.

This module simply re-exports functions from the more focused modules:
- visualisations_area_matrices
- visualisations_area_bars_mixed
- visualisations_simplified_visits
- visualisations_strategies
- visualisations_time_segments

So existing imports like `from src import visualisations as V`
keep working unchanged.
"""


from src.viz.visualisations_area_matrices import (
    matrix_plot_ABCD,
    label_vs_loc_mat,
    run_all_area_metric_plots,
)

from src.viz.visualisations_area_bars import (
    plot_area_ci_bar,
    run_all_area_barplots,
)

from src.viz.visualisations_simplified_visits import (
    matrix_plot_simplified_visits,
    run_all_simplified_visit_matrices,
)

from src.viz.visualisations_strategies import (
    build_strategy_dataframe,
    proportion_with_dominant_strategy,
    plot_dominant_strategy_hist,
    plot_dominance_gap,
    plot_strategy_count_distribution,
    plot_dominant_strategy_counts_above_threshold,
    build_prefix_completion_map_from_series,
    add_completed_sequence_column,
    summarize_before_after,
    plot_strategies,
    run_all_strategy_plots,
)

from src.viz.visualisations_time_segments import (
    _assign_time_segment,
    add_time_segment_column,
    _plot_time_segment_bar,
    plot_time_segment_mean_dwell,
    plot_time_segment_sequence_length,
    plot_time_segment_fixation_count,
    plot_time_segment_skip_rate,
    run_all_time_segment_plots,
)

from src.viz.visualisations_dominant_eye import (
    build_dominant_strategy_by_eye,
    plot_dominant_strategies_by_eye_sorted,
    run_dominant_strategy_eye_analysis,
)

__all__ = [
    # area matrices
    "matrix_plot_ABCD",
    "label_vs_loc_mat",
    "run_all_area_metric_plots",
    # barplots + mixed models
    "plot_area_ci_bar",
    "run_all_area_barplots",
    # simplified visits
    "matrix_plot_simplified_visits",
    "run_all_simplified_visit_matrices",
    # strategies
    "build_strategy_dataframe",
    "proportion_with_dominant_strategy",
    "plot_dominant_strategy_hist",
    "plot_dominance_gap",
    "plot_strategy_count_distribution",
    "plot_dominant_strategy_counts_above_threshold",
    "build_prefix_completion_map_from_series",
    "add_completed_sequence_column",
    "summarize_before_after",
    "plot_strategies",
    "run_all_strategy_plots",
    # time segments
    "_assign_time_segment",
    "add_time_segment_column",
    "_plot_time_segment_bar",
    "plot_time_segment_mean_dwell",
    "plot_time_segment_sequence_length",
    "plot_time_segment_fixation_count",
    "plot_time_segment_skip_rate",
    "run_all_time_segment_plots",
    #eye dominance
    "build_dominant_strategy_by_eye",
    "plot_dominant_strategies_by_eye_sorted",
    "run_dominant_strategy_eye_analysis",

]