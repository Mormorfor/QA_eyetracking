"""The histogram of items by how often they were guessed right, saved through `save_output`
with the tables behind it."""

from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.explorations.guessability.data import CHANCE, DATASETS, ITEM_KEY
from src.lib.plotting.output import save_output

ANALYSIS = "explorations/guessability"
BAR = "#4c72b0"
SOURCE_NAMES = {"Onestop": "OneStop", "RACE": "RACE"}
# The item table is saved without the question and answer texts, except where the text is
# the item's only key (the pilot), so that no RACE text lands in the tracked reports.
TEXT_COLS = ["question", "correct_answer"]


def plot_histogram(items: pd.DataFrame, hist: pd.DataFrame, dataset: str, *,
                   save: Optional[bool] = None, close: Optional[bool] = None):
    """One panel per source: the number of items at each share of right guesses (bars),
    against the number expected if every item were equally guessable (black) and if every
    guess were one in four (grey)."""
    sources = list(hist["source"].unique())
    fig, axes = plt.subplots(1, len(sources), figsize=(6.2 * len(sources), 4.4),
                             sharey=True, layout="constrained", squeeze=False)
    for ax, source in zip(axes[0], sources):
        h = hist[hist["source"] == source].sort_values("bin_low")
        it = items[items["source"] == source]
        discrete = (h["bin_low"] == h["bin_high"]).all()
        if discrete:
            x, width = h["bin_low"].to_numpy(), 0.1
        else:
            x, width = ((h["bin_low"] + h["bin_high"]) / 2).to_numpy(), (h["bin_high"] - h["bin_low"]).to_numpy() * 0.92
        ax.bar(x, h["items_observed"], width=width, color=BAR, label="items")
        ax.plot(x, h["items_expected_same_rate"], "o", color="k", ms=6,
                label="expected if every item were guessed right at the panel's overall rate")
        ax.plot(x, h["items_expected_chance"], "D", mfc="none", mec="#7f7f7f", ms=6,
                label=f"expected at chance ({CHANCE:.0%})")
        if discrete:
            for xi, yi in zip(x, h["items_observed"]):
                ax.annotate(f"{int(yi)}", (xi, yi), xytext=(0, 3), textcoords="offset points",
                            ha="center", va="bottom", fontsize=8)
            ax.set_xticks(x, h["bin"])
        else:
            ax.set_xticks(np.linspace(0, 1, 6))
        ax.axvline(CHANCE, color="#7f7f7f", lw=0.8, ls=":", label="one in four")
        ax.set_title(f"{SOURCE_NAMES.get(source, source)}: {len(it)} items, {int(it['n_guesses'].sum())} guesses "
                     f"({it['n_guesses'].median():.0f} per item), {h['rate'].iloc[0]:.0%} right", fontsize=10)
        ax.set_xlabel("share of the item's guesses that were right")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="#e5e5e5", lw=0.6)
        ax.set_axisbelow(True)
    axes[0, 0].set_ylabel("number of items")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=1 if len(sources) == 1 else 4,
               frameon=False, fontsize=8)
    fig.suptitle(f"{DATASETS[dataset]}\nhow often each item was guessed right", fontsize=11)

    saved_items = items.drop(columns=[c for c in TEXT_COLS if c not in ITEM_KEY[dataset]])
    save_output(fig, analysis=ANALYSIS, plot="items_by_share_guessed_right",
                tables={"histogram": hist, "items": saved_items},
                save=save, close=close, dataset=dataset)
    return fig
