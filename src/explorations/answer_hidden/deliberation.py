"""Searching the *selected* naming for features of deliberation that predict correctness.

The selected version of the paper's twelve-feature model (`BASE`) describes the chosen
answer against the others. Here groups of features that describe how much the participant
deliberated are added to it, every combination of them, and also tried on their own
without the base. Every set is scored by the paper's comparison, as in `run.py`.

All the columns come from the answer-hidden table (`table.build_answer_hidden_table`):
either independent of which answer is correct, or measured on the selected answer against
the others. A low score under **no knowledge** is not a failure here: those right answers
were guesses, and a model that reads how sure someone was should not call them right.
"""

from __future__ import annotations

from itertools import combinations
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import columns as Con
from src.lib.plotting.output import save_output
from src.modeling.models.logreg_model import TrialLevelLogRegModel
from src.modeling.provenance import MODEL_FAMILY_COL, model_folder
from src.explorations.answer_hidden.cache import cached, table_digest
from src.explorations.answer_hidden.run import (
    ANALYSIS,
    compare_in_batches,
    model_key,
    load_tables,
    plot_answer_hidden_comparison,
)
from src.explorations.answer_hidden.selected_features import SEQUENCE_FEATURES, TIMING_FEATURES
from src.explorations.answer_hidden.table import SEP, hide_answers

# The selected version of the paper's twelve-feature model.
BASE_NAME = "base"


def base_cols() -> List[str]:
    """The paper's twelve features, selected version (`hide_answers(..., by="selected")`)."""
    import src.modeling.feature_sets as FS
    return hide_answers(list(FS.SELECT_1_COLS) + list(FS.LAST_CONFIRM_COMPACT), by="selected")


# Each group is one aspect of deliberating over the answers. A column can sit in the base
# too; a set takes each column once.
DELIBERATION_GROUPS: Dict[str, List[str]] = {
    # Going back and forth between two areas: at the end of the scan, or anywhere.
    "back_and_forth": ["longest_alt_answer_run", "has_xyx", "has_xyxy"],
    # How many times the areas were entered.
    "visits": [
        "seq_len",
        f"{Con.NUM_LABEL_VISITS}{SEP}selected",
        f"{Con.NUM_LABEL_VISITS}{SEP}unselected_mean",
        f"{Con.NUM_LABEL_VISITS}{SEP}{Con.QUESTION_AREA}",
    ],
    # Selecting more than once, and where the last looks before the two presses landed.
    "choice": [
        Con.NUM_OF_SELECTS,
        "last_before_select_selected",
        "last_before_confirm_selected",
        f"last_before_confirm_{Con.QUESTION_AREA}",
    ],
    # Time: the whole answer screen, time on the chosen and the other answers, and the
    # span from the first to the last look at them -- all per word on screen.
    "time": [
        Con.TOTAL_ANSWERING_RT_NORMALIZED,
        f"{Con.RT_NORMALIZED}_selected",
        f"{Con.RT_NORMALIZED}_unselected_mean",
        f"{Con.TIME_SINCE_OFFSET_NORMALIZED}_selected",
        f"{Con.TIME_SINCE_OFFSET_NORMALIZED}_unselected_mean",
    ],
    # Going back to the question.
    "question": [
        f"{Con.AREA_DWELL_PROPORTION}{SEP}{Con.QUESTION_AREA}",
        f"{Con.MEAN_FIXATIONS_COUNT}{SEP}{Con.QUESTION_AREA}",
        f"{Con.RT_NORMALIZED}_{Con.QUESTION_AREA}",
    ],
}


# The set taken forward from the search: back-and-forth, visits and time, twelve features
# like the paper's headline model. Within the noise of the best Study 1 set (0.650 against
# 0.654 balanced accuracy) and the best on Study 2's partial-knowledge regime, the one with
# the most wrong answers to test on. Picked after looking at these scores, so its scores
# are somewhat optimistic.
CHOSEN_NAME = "selected_deliberation"
CHOSEN_GROUPS = ("back_and_forth", "visits", "time")


def chosen_cols() -> List[str]:
    """The columns of the chosen set (`CHOSEN_GROUPS`)."""
    return list(dict.fromkeys(c for g in CHOSEN_GROUPS for c in DELIBERATION_GROUPS[g]))


# The features built around the selected answer (`selected_features.py`), as two groups.
SELECTED_GROUPS: Dict[str, List[str]] = {
    "selected_sequence": SEQUENCE_FEATURES,
    "selected_timing": TIMING_FEATURES,
}

# What the selected-answer groups are added to: nothing, the base, the time group alone and
# with the base, the best Study 1 set of twelve features, and everything above.
ANCHORS = (
    "",
    BASE_NAME,
    "time",
    f"{BASE_NAME}+time",
    "back_and_forth+visits+time",
    f"{BASE_NAME}+back_and_forth+visits+choice+time+question",
)


def selected_feature_sets(
    base: Sequence[str],
    anchors: Sequence[str] = ANCHORS,
    groups: Mapping[str, Sequence[str]] = SELECTED_GROUPS,
) -> Dict[str, List[str]]:
    """Each of `anchors` alone, with each of `groups`, and with all of them; and the groups
    on their own. In that order, anchor by anchor. By default the selected-answer groups
    on `ANCHORS`."""
    anchor_cols = combination_sets(base)
    added_options = [(), *((g,) for g in groups), tuple(groups)]
    sets: Dict[str, List[str]] = {}
    for anchor in anchors:
        for added in added_options:
            if not anchor and not added:
                continue
            cols = [*(anchor_cols[anchor] if anchor else []), *(c for g in added for c in groups[g])]
            sets["+".join(x for x in (anchor, *added) if x)] = list(dict.fromkeys(cols))
    return sets


def combination_sets(
    base: Sequence[str],
    groups: Mapping[str, Sequence[str]] = DELIBERATION_GROUPS,
) -> Dict[str, List[str]]:
    """Every combination of `groups`, with the base (`base+time+visits`) and without it
    (`time+visits`). The base alone is `base`."""
    sets: Dict[str, List[str]] = {}
    names = list(groups)
    for k in range(len(names) + 1):
        for combo in combinations(names, k):
            added = [c for g in combo for c in groups[g]]
            sets["+".join((BASE_NAME, *combo))] = list(dict.fromkeys([*base, *added]))
            if combo:
                sets["+".join(combo)] = list(dict.fromkeys(added))
    return sets


def run_deliberation_search(
    tables: Mapping[str, pd.DataFrame],
    sets: Mapping[str, Sequence[str]],
    *,
    model_builder: Callable[[], Any] = TrialLevelLogRegModel,
    verbose: bool = True,
    rerun: bool = False,
) -> Dict[str, pd.DataFrame]:
    """Score every set in `sets` on the answer-hidden tables by the paper's comparison,
    as `run.run_answer_hidden_comparison` does. Returns its `scores` and `folds`.

    Each set is scored and saved on its own (`cache.py`), keyed by its columns, the model
    and the tables, so a set scored before is read back and only new ones run; a search
    can grow a set at a time. A saved result is relabelled with the name it has here.
    `rerun=True` scores every set again.
    """
    tables_key = {k: table_digest(tables[k]) for k in ("l1_hidden", "knowqa_hidden")}
    model_id = model_key(model_builder)
    parts = []
    for name, cols in sets.items():
        result = cached(
            "search_set",
            {"features": list(cols), "model": model_id, "tables": tables_key},
            lambda name=name, cols=list(cols): compare_in_batches(
                {name: cols}, tables["l1_hidden"], tables["knowqa_hidden"],
                model_builder=model_builder, verbose=False,
            ),
            rerun=rerun, verbose=False,
        )
        parts.append({k: v.assign(model=name) for k, v in result.items()})
    if verbose:
        print(f"scored {len(sets)} sets")
    return {k: pd.concat([p[k] for p in parts], ignore_index=True) for k in ("scores", "folds")}


# The columns of the score table and the heatmap, in order.
SCORE_COLUMNS = [
    ("l1", "new_item"), ("l1", "new_subject"), ("l1", "both"), ("l1", "average"),
    ("knowqa", "no knowledge"), ("knowqa", "partial knowledge"),
    ("knowqa", "full knowledge"), ("knowqa", "all"),
]

# A few sets worth seeing as bars: the base, the time group alone and added to the base,
# the two best Study 1 sets of twelve features, and everything.
SHORT_LIST = [
    BASE_NAME,
    "time",
    f"{BASE_NAME}+time",
    "back_and_forth+visits+time",
    "back_and_forth+choice+time",
    f"{BASE_NAME}+back_and_forth+visits+choice+time+question",
]


def score_table(scores: pd.DataFrame, order: Optional[Sequence[str]] = None) -> pd.DataFrame:
    """One row per set: its number of features, its balanced accuracy on Study 1's three
    held-out kinds and their average, and on Study 2's three regimes and all of them
    together, and which model made the scores. In `order` if given, else best Study 1
    average first."""
    wide = scores.pivot_table(index="model", columns=["dataset", "test"], values="balanced_accuracy")
    first = scores.drop_duplicates("model").set_index("model")
    out = pd.DataFrame({"n_features": first["n_features"]})
    for ds, test in SCORE_COLUMNS:
        out[f"{ds}: {test}"] = wide[(ds, test)]
    out["with base"] = out.index.str.startswith(BASE_NAME)
    out[MODEL_FAMILY_COL] = first[MODEL_FAMILY_COL]
    if order is not None:
        return out.loc[list(order)]
    return out.sort_values("l1: average", ascending=False)


SEARCH_TITLES = {
    "groups": "Deliberation groups added to the selected version",
    "selected": "Features built around the selected answer",
    "matching": "Matching features (did the chosen answer stand out?)",
    "tracker": "What needs an eye tracker",
}


def plot_search_heatmap(
    scores: pd.DataFrame,
    *,
    search: str = "groups",
    order: Optional[Sequence[str]] = None,
    save: Optional[bool] = None,
    close: Optional[bool] = None,
):
    """Every set's balanced accuracy on every test, as one annotated heatmap: a row per
    set, in `order` or best Study 1 average at the top, sets that include the base in
    bold. Shaded from chance (0.5, white) up; a score below chance is white too and reads
    from its number. `search` names the search ("groups" or "selected") in the title and
    the file name. Saved with the score table. Returns the figure and the table."""
    table = score_table(scores, order=order)
    cols = [f"{ds}: {test}" for ds, test in SCORE_COLUMNS]
    values = table[cols].to_numpy(float)

    fig, ax = plt.subplots(figsize=(11, 0.24 * len(table) + 1.8), layout="constrained")
    ax.imshow(values, cmap="Blues", vmin=0.5, vmax=np.nanmax(values), aspect="auto")
    top = np.nanmax(values)
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            v = values[i, j]
            ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=7,
                    color="white" if v > 0.5 + 0.6 * (top - 0.5) else "black")
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([c.replace(": ", "\n").replace(" knowledge", "\nknowledge") for c in cols], fontsize=8)
    ax.xaxis.tick_top()
    ax.set_yticks(range(len(table)))
    ax.set_yticklabels([f"{name}  ({n})" for name, n in zip(table.index, table["n_features"])], fontsize=7)
    for label, with_base in zip(ax.get_yticklabels(), table["with base"]):
        label.set_fontweight("bold" if with_base else "normal")
    # Study 1 and Study 2 apart.
    ax.axvline(3.5, color="black", lw=1.5)
    ax.set_title(
        f"{SEARCH_TITLES[search]}: balanced accuracy\n"
        "(rows: feature groups, number of features in brackets; bold: with the base)",
        fontsize=10,
    )
    save_output(
        fig,
        analysis=ANALYSIS,
        plot="deliberation_search",
        tables={"scores": table.reset_index(names="model")},
        save=save,
        close=close,
        subdir=model_folder(table),
        search=search,
    )
    return fig, table


def main(save: bool = True, verbose: bool = True, rerun: bool = False) -> Dict[str, Any]:
    """Both searches, saved: a heatmap of each, and the paper's comparison figure for
    `SHORT_LIST`. Sets scored before are read back (`run_deliberation_search`) unless
    `rerun=True`. Returns the two comparisons."""
    tables = load_tables(verbose=verbose)
    groups = run_deliberation_search(tables, combination_sets(base_cols()), verbose=verbose, rerun=rerun)
    plot_search_heatmap(groups["scores"], search="groups", save=save, close=True)
    for ds in ("l1", "knowqa"):
        plot_answer_hidden_comparison(
            groups, dataset=ds, models=SHORT_LIST, save=save, close=True, models_shown="deliberation",
        )
    selected_sets = selected_feature_sets(base_cols())
    selected = run_deliberation_search(tables, selected_sets, verbose=verbose, rerun=rerun)
    plot_search_heatmap(selected["scores"], search="selected", order=list(selected_sets), save=save, close=True)
    return {"groups": groups, "selected": selected}


if __name__ == "__main__":
    main()
