import os
from typing import Optional
import ast

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src import constants as Con


# ---------------------------------------------------------------------------
# First/Last Visits Heatmaps
# ---------------------------------------------------------------------------

def matrix_plot_simplified_visits(
    df: pd.DataFrame,
    kind: str = "location",        # "label" or "location"
    which: str = "first",          # "first" or "last"
    drop_question: bool = True,
    h_or_g: str = "hunters",
    selected: str = "A",
    figsize: tuple = (8, 5),
    save: bool = False,
    output_root: str = "../reports/plots/simpl_visit_matrices",
    show: bool = True,
) -> None:
    """
    Plot a heatmap of visit frequencies for a fixed-length window taken from
    the simplified fixation sequence.

    For each trial/participant row:
      - Take the simplified sequence (label or location).
      - Optionally remove 'question' tokens.
      - Take either:
          * the first X tokens      (which = 'first'), or
          * the last  X tokens      (which = 'last'),
        where X = 4 if drop_question=True, else X = 5.
      - Re-index those tokens as positions 0..len(window)-1.
      - Count how often each area occurs at each position across trials.

    Result: a small matrix:
      rows   = visit position (0..3 or 0..4)
      cols   = areas (answers, and optionally question)
      values = counts.

    Parameters
    ----------
    df : DataFrame
        Data filtered to a specific group (hunters/gatherers) and selected
        answer label, typically one row per (trial, participant).
    kind : {"label", "location"}
        Which sequence to visualise:
        - "label"    -> Con.SIMPLIFIED_FIX_SEQ_BY_LABEL
        - "location" -> Con.SIMPLIFIED_FIX_SEQ_BY_LOCATION
    which : {"first", "last"}
        Whether to use the first X or last X entries of the sequence.
    drop_question : bool
        If True, remove 'question' tokens before taking the window.
    h_or_g : str
        Group label: 'hunters' or 'gatherers', used in the title/filename.
    selected : str
        Selected answer label ('A', 'B', 'C', 'D'), used in the title/filename.
    figsize : tuple
        Figure size for the heatmap.
    save : bool
        If True, save the figure as a PNG under output_root.
    output_root : str
        Root directory where the plot will be saved.
    show : bool
        If True, display the plot; otherwise close it after saving.
    """
    if kind == "label":
        seq_col = Con.SIMPLIFIED_FIX_SEQ_BY_LABEL
        base_areas = list(Con.ANSWER_LABEL_CHOICES)
    elif kind == "location":
        seq_col = Con.SIMPLIFIED_FIX_SEQ_BY_LOCATION
        base_areas = list(Con.AREA_LABEL_CHOICES)
    else:
        raise ValueError("kind must be 'label' or 'location'")

    if which not in {"first", "last"}:
        raise ValueError("which must be 'first' or 'last'")

    window_len = 4 if drop_question else 5

    df_sel = (
        df[[Con.TRIAL_ID, Con.PARTICIPANT_ID, seq_col]]
        .drop_duplicates()
        .copy()
    )

    def _parse_seq(x):
        if isinstance(x, str):
            try:
                return ast.literal_eval(x)
            except Exception:
                return None
        return x

    df_sel[seq_col] = df_sel[seq_col].apply(_parse_seq)
    df_sel = df_sel[df_sel[seq_col].notna()].copy()

    def _clean_and_window(seq):
        if not isinstance(seq, (list, tuple)):
            return []
        seq = list(seq)
        if drop_question:
            seq = [tok for tok in seq if tok != "question"]
        if not seq:
            return []
        if which == "first":
            return seq[:window_len]
        else:
            return seq[-window_len:]

    df_sel["window"] = df_sel[seq_col].apply(_clean_and_window)
    df_sel = df_sel[df_sel["window"].map(len) > 0].copy()

    if df_sel.empty:
        print(
            f"[info] No non-empty windows for kind='{kind}', which='{which}', "
            f"drop_question={drop_question}, group={h_or_g}, selected={selected}."
        )
        return

    df_sel["position"] = df_sel["window"].apply(
        lambda lst: list(range(len(lst)))
    )

    df_expl = df_sel.explode("position")
    df_expl = df_expl[df_expl["position"].notna()].copy()

    df_expl["area"] = df_expl.apply(
        lambda row: row["window"][int(row["position"])],
        axis=1,
    )

    agg = (
        df_expl.groupby(["position", "area"])
        .size()
        .reset_index(name="count")
    )

    pivot = (
        agg.pivot(index="position", columns="area", values="count")
        .fillna(0)
        .sort_index()
    )

    if drop_question:
        area_order = [a for a in base_areas if a != "question"]
    else:
        area_order = base_areas

    col_order = [c for c in area_order if c in pivot.columns]
    pivot = pivot.reindex(columns=col_order)

    plt.figure(figsize=figsize)
    ax = sns.heatmap(pivot, annot=True, fmt="g", cmap="viridis")
    q_flag = " (no question)" if drop_question else " (with question)"
    ax.set_title(
        f"{which.capitalize()} {window_len} visits ({kind}){q_flag}\n"
        f"{h_or_g}, selected={selected}"
    )
    ax.set_xlabel("Area")
    ax.set_ylabel("Visit Order (position)")
    plt.tight_layout()

    if save:
        mode_dir = f"{which}_{kind}" + ("_noq" if drop_question else "_withq")
        out_dir = os.path.join(output_root, mode_dir)
        os.makedirs(out_dir, exist_ok=True)
        fname = f"{h_or_g} - {selected}.png"
        plt.savefig(os.path.join(out_dir, fname), dpi=300)

    if show:
        plt.show()
    else:
        plt.close()



def run_all_simplified_visit_matrices(
    hunters: pd.DataFrame,
    gatherers: pd.DataFrame,
    drop_question_variants: tuple = (True, False),
    kinds: tuple = ("label", "location"),
    which_list: tuple = ("first", "last"),
    answers: tuple = ("A", "B", "C", "D"),
    output_root: str = "../reports/plots/simpl_visit_matrices",
    save: bool = True,
    show: bool = True,
) -> None:
    """
    Generate visit-order heatmaps (first/last visits) for simplified sequences:
      - hunters, gatherers, and all participants
      - each selected answer (A–D)
      - label-based and/or location-based sequences
      - with and without 'question'
    """
    all_participants = pd.concat([hunters, gatherers], ignore_index=True)

    groups = {
        "hunters": hunters,
        "gatherers": gatherers,
        "all_participants": all_participants,
    }

    for drop_question in drop_question_variants:
        dq_folder = "questions_removed" if drop_question else "questions_included"

        for group_key, df in groups.items():
            group_label = "all participants" if group_key == "all_participants" else group_key

            group_out_root = os.path.join(output_root, dq_folder, group_key)

            for ans in answers:
                subset = df[df[Con.SELECTED_ANSWER_LABEL_COLUMN] == ans].copy()
                if subset.empty:
                    continue

                print(
                    f"\n{group_label.upper()} — selected {ans} "
                    f"(drop_question={drop_question})"
                )
                print("-" * 72)

                for which in which_list:
                    for kind in kinds:
                        matrix_plot_simplified_visits(
                            subset,
                            kind=kind,
                            which=which,
                            drop_question=drop_question,
                            h_or_g=group_label,
                            selected=ans,
                            figsize=(8, 5),
                            save=save,
                            output_root=group_out_root,
                            show=show,
                        )


