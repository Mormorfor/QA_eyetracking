"""The figures. Each takes the tables `run.compute` builds and saves itself, with the tables
behind it, through `save_output`. Unless a panel says otherwise: dots are participants, points
their mean with its standard error, and the dotted line what chance gives."""

from __future__ import annotations

from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FixedLocator, NullLocator, ScalarFormatter

from src.explorations.position_bias.choice import MODELLED, TERM_LABELS
from src.explorations.position_bias.data import (
    ALL_PARTICIPANTS, GUESSING, KIND_NAMES, ORDER_NAMES, POSITIONS, STUDY_NAMES, SUBSETS,
)
from src.explorations.position_bias.scan import SPEED_THIRDS
from src.lib.plotting.output import save_output

ANALYSIS = "explorations/position_bias"
SUBSET = "subset"
COLOURS = {
    "no knowledge": "#c44e52", "partial knowledge": "#dd8452", "full knowledge": "#4c72b0",
    ALL_PARTICIPANTS: "#333333", "hunters": "#55a868", "gatherers": "#8172b3",
}
MARKERS = {
    "no knowledge": "o", "partial knowledge": "s", "full knowledge": "D",
    ALL_PARTICIPANTS: "o", "hunters": "^", "gatherers": "v",
}
# Study 1's two groups share every placement, so the placement figure shows them together.
PLACEMENT_GROUPS = {"knowqa": SUBSETS["knowqa"], "l1": [ALL_PARTICIPANTS]}
ORDER_LEVELS = [ORDER_NAMES[k] for k in (1, 2, 3, 4)]
LABELS = list(KIND_NAMES)


def _p(p: float) -> str:
    return f"p = {p:.2g}"


def _style(ax, grid: str = "y"):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis=grid, color="#E5E5E5", lw=0.8)
    ax.set_axisbelow(True)


def _test(tests: pd.DataFrame, subset: str, test: str) -> pd.Series:
    row = tests[(tests[SUBSET] == subset) & (tests["test"] == test)]
    assert len(row) == 1, f"no single test {test!r} for {subset!r}"
    return row.iloc[0]


def _means(ax, summary: pd.DataFrame, levels, *, colour="#333333", marker="o", label=None, offset=0.0,
           value="participant_mean", se="se", line=False):
    """Means with their standard errors at `levels`, from a table indexed by level."""
    s = summary.reindex(levels)
    ax.errorbar(np.arange(len(levels)) + offset, s[value], yerr=s[se], fmt=f"{marker}{'-' if line else ''}",
                color=colour, ms=6.5, lw=1.5, elinewidth=1.8, capsize=0, label=label, zorder=3)


def _dots(ax, per: pd.DataFrame, levels):
    rng = np.random.default_rng(0)
    for i, level in enumerate(levels):
        y = per.loc[per["level"] == level, "share"].to_numpy()
        ax.scatter(i + rng.uniform(-0.12, 0.12, len(y)), y, s=12, color="#9a9a9a", alpha=0.7, lw=0, zorder=1)


# --- Placement ----------------------------------------------------------------------------------

def plot_placement(t: Dict[str, pd.DataFrame], dataset_key: str, *, save: Optional[bool] = None,
                   close: Optional[bool] = None):
    """Was anything put at the top more often than chance? One heatmap per group: the share of
    placements that put each kind of answer at each position (one in four is even). Below: the
    mean length of the answer placed at each position; how often other people chose the answer
    placed there; and the share of choices each position got, against the share content alone
    would have sent there (`placement.expected_shares`). p from random re-placements."""
    groups = PLACEMENT_GROUPS[dataset_key]
    tests = t["placement_tests"]
    if len(groups) == 3:
        mosaic = [["h0", "h1", "h2"], ["length", "attract", "expected"]]
    else:
        mosaic = [["h0", "length"], ["attract", "expected"]]
    fig, ax = plt.subplot_mosaic(mosaic, figsize=(14, 8.6) if len(groups) == 3 else (11, 8.6), layout="constrained")

    kinds = t["placement_kinds"]
    for i, group in enumerate(groups):
        a = ax[f"h{i}"]
        share = kinds[kinds[SUBSET] == group].pivot(index="label", columns="position", values="share").reindex(
            index=LABELS, columns=POSITIONS)
        image = a.imshow(share.to_numpy(), cmap="RdBu_r", vmin=0.15, vmax=0.35, aspect="auto")
        for r in range(4):
            for c in range(4):
                a.text(c, r, f"{100 * share.iat[r, c]:.0f}%", ha="center", va="center", fontsize=9, color="#222222")
        a.set_xticks(range(4), POSITIONS)
        a.set_yticks(range(4), [KIND_NAMES[k].split(" (")[0] + (" (correct)" if k == "A" else "") for k in LABELS])
        a.tick_params(length=0)
        n_units = int(_test(tests, group, "each kind at each position one time in four")["n_units"])
        even = _test(tests, group, "each kind at each position one time in four")["p"]
        a_top = _test(tests, group, "the correct answer at the top one time in four")["p"]
        a.set_title(f"{group} ({n_units} placements)\neach kind even: {_p(even)}\ncorrect answer at the top: {_p(a_top)}",
                    fontsize=9.5)
    fig.colorbar(image, ax=[ax[f"h{i}"] for i in range(len(groups))], shrink=0.8, label="share of placements")

    la = t["placement_length_attractiveness"]
    exp = t["placement_expected"]
    for i, group in enumerate(groups):
        colour, marker = COLOURS[group], MARKERS[group]
        g = la[la[SUBSET] == group].set_index("position").reindex(POSITIONS)
        p_len = _test(tests, group, "answers equally long at every position")["p"]
        ax["length"].plot(range(4), g["mean_words"], f"{marker}-", color=colour, ms=6.5, lw=1.5,
                          label=f"{group} ({_p(p_len)})")
        p_att = _test(tests, group, "answers others chose equally often at every position")["p"]
        ax["attract"].plot(range(4), g["attractiveness"], f"{marker}-", color=colour, ms=6.5, lw=1.5,
                           label=f"{group} ({_p(p_att)})")
        e = exp[exp[SUBSET] == group].set_index("position").reindex(POSITIONS)
        ax["expected"].plot(range(4), e["observed_where_attractiveness"], f"{marker}-", color=colour, ms=6.5, lw=1.5,
                            label=f"{group}: chosen")
        ax["expected"].plot(range(4), e["expected_by_attractiveness"], f"{marker}--", color=colour, ms=6.5, lw=1.2,
                            mfc="white", label=f"{group}: content alone")
    ax["length"].set_ylabel("mean words in the answer placed there")
    ax["length"].set_ylim(0, None)  # from zero, so a difference of a fraction of a word looks like one
    ax["length"].set_title("Length of the answer at each position", fontsize=10)
    ax["attract"].axhline(0.25, color="k", lw=0.8, ls=":")
    ax["attract"].set_ylabel("share of other people who chose it")
    ax["attract"].set_title("How often others chose the answer at each position", fontsize=10)
    for key in ["attract", "expected"]:
        ax[key].set_ylim(0.15, 0.35)  # the heatmaps' range: ten points either side of one in four
    ax["expected"].axhline(0.25, color="k", lw=0.8, ls=":")
    ax["expected"].set_ylabel("share of choices")
    ax["expected"].set_title("Choices at each position, against content alone\n(trials whose answers others also saw)",
                             fontsize=10)
    for key in ["length", "attract", "expected"]:
        ax[key].set_xticks(range(4), POSITIONS)
        ax[key].legend(frameon=False, fontsize=8)
        _style(ax[key])
    fig.suptitle(f"{STUDY_NAMES[dataset_key]}: was anything placed at the top -- or anywhere -- more often than chance?",
                 fontsize=11)
    tables = {k: t[k] for k in ["placement_tests", "placement_kinds", "placement_length_attractiveness", "placement_expected"]}
    save_output(fig, analysis=ANALYSIS, plot="placement", tables=tables, save=save, close=close, dataset=dataset_key)
    return fig


# --- Choices --------------------------------------------------------------------------------------

def plot_choices(t: Dict[str, pd.DataFrame], dataset_key: str, *, save: Optional[bool] = None,
                 close: Optional[bool] = None):
    """Where the choices went, one row per group: (a) the position of the chosen answer;
    (b) the position of the wrong answer chosen, on wrong trials; (c) accuracy by where the
    correct answer sat; (d) accuracy by when the scan first reached the correct answer. In (c)
    and (d) the dotted line is the group's accuracy. p from random re-placements; (d) has none,
    since the order is the participant's, not the design's."""
    groups = SUBSETS[dataset_key]
    tests = t["choice_tests"]
    pos, acc = t["choice_positions"], t["choice_accuracy"]
    fig, axes = plt.subplots(len(groups), 4, figsize=(16, 3.3 * len(groups) + 0.8), layout="constrained", squeeze=False)
    for r, group in enumerate(groups):
        colour, marker = COLOURS[group], MARKERS[group]
        a = axes[r]
        allp = pos[(pos[SUBSET] == group) & (pos["trials"] == "all trials")].set_index("position")
        _means(a[0], allp, POSITIONS, colour=colour, marker=marker)
        a[0].axhline(0.25, color="k", lw=0.8, ls=":")
        a[0].set_title(f"{group}: chosen answer's position\nall four even: "
                       f"{_p(_test(tests, group, 'chosen answer: at each position equally often')['p'])}", fontsize=9.5)
        wrong = pos[(pos[SUBSET] == group) & (pos["trials"] == "wrong answers")].set_index("position")
        _means(a[1], wrong, POSITIONS, colour=colour, marker=marker)
        a[1].axhline(0.25, color="k", lw=0.8, ls=":")
        n_wrong = int(wrong["n_rows"].iloc[0])  # every wrong trial has one answer at each position
        even = _test(tests, group, "wrong answer chosen: at each position equally often")["p"]
        top = _test(tests, group, "wrong answer chosen: at the top one time in four")["p"]
        a[1].set_title(f"wrong answer chosen: its position ({n_wrong} wrong trials)\nall four even: {_p(even)}; "
                       f"at the top one in four: {_p(top)}", fontsize=9.5)
        overall = _test(tests, group, "accuracy: the same wherever the correct answer sat")["value"]
        by_pos = acc[(acc[SUBSET] == group) & (acc["by"] == "position of the correct answer")].set_index("level")
        _means(a[2], by_pos, POSITIONS, colour=colour, marker=marker)
        a[2].axhline(overall, color="k", lw=0.8, ls=":")
        a[2].set_title(f"accuracy by where the correct answer sat\nthe same everywhere: "
                       f"{_p(_test(tests, group, 'accuracy: the same wherever the correct answer sat')['p'])}", fontsize=9.5)
        by_order = acc[(acc[SUBSET] == group) & (acc["by"] == "when the correct answer was first reached")].set_index("level")
        _means(a[3], by_order, ORDER_LEVELS, colour=colour, marker=marker)
        a[3].axhline(overall, color="k", lw=0.8, ls=":")
        a[3].set_title("accuracy by when the correct answer was first reached", fontsize=9.5)
        for k, levels in enumerate([POSITIONS, POSITIONS, POSITIONS, ORDER_LEVELS]):
            a[k].set_xticks(range(len(levels)), levels)
            _style(a[k])
        # Fixed scales, so that a wiggle of a point or two is not drawn as large as an effect.
        a[0].set_ylim(0.10, 0.40)
        a[1].set_ylim(0.10, 0.40)
        for k in (2, 3):
            a[k].set_ylim(max(0.0, overall - 0.2), min(1.0, overall + 0.2))
        a[0].set_ylabel("share of trials")
        a[2].set_ylabel("accuracy")
    fig.suptitle(f"{STUDY_NAMES[dataset_key]}: where the chosen answer, the wrong answer and the correct answer sat "
                 "(points: participant mean ± SE; dotted: chance, or the group's accuracy)", fontsize=11)
    tables = {k: t[k] for k in ["choice_tests", "choice_positions", "choice_accuracy"]}
    save_output(fig, analysis=ANALYSIS, plot="choices", tables=tables, save=save, close=close, dataset=dataset_key)
    return fig


def plot_guesses(t: Dict[str, pd.DataFrame], *, save: Optional[bool] = None, close: Optional[bool] = None):
    """Study 2's guesses (no knowledge). (a) Where the guessed answer sat. (b) When the guessed
    answer was first reached, on trials that reached all four answers. (c) Each position's
    share of the answer gaze, for the answer guessed and the ones not guessed. (d) How often a
    guess was right, by where the correct answer sat. p from random re-placements."""
    dataset_key = "knowqa"
    tests = t["choice_tests"]
    per = t["choice_by_participant"]
    per = per[per[SUBSET] == GUESSING]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5), layout="constrained")

    ax = axes[0, 0]
    pos = t["choice_positions"]
    pos = pos[(pos[SUBSET] == GUESSING) & (pos["trials"] == "all trials")].set_index("position")
    _dots(ax, per[per["by"] == "position"], POSITIONS)
    _means(ax, pos, POSITIONS)
    ax.axhline(0.25, color="k", lw=0.8, ls=":")
    top = _test(tests, GUESSING, "chosen answer: at the top one time in four")
    ax.set_title(f"(a) Where the guessed answer sat\nat the top {100 * top['value']:.1f}%, one in four: {_p(top['p'])}",
                 fontsize=10)
    ax.set_ylabel("share of guesses")
    ax.set_xticks(range(4), POSITIONS)

    ax = axes[0, 1]
    order = t["choice_order"]
    order = order[(order[SUBSET] == GUESSING) & (order["trials"] == "complete scans")].set_index("order_name")
    _dots(ax, per[per["by"] == "order, complete scans"], ORDER_LEVELS)
    _means(ax, order, ORDER_LEVELS)
    ax.axhline(0.25, color="k", lw=0.8, ls=":")
    ax.set_title(f"(b) When the guessed answer was first reached\n{int(order['n_rows'].iloc[0])} trials that reached "
                 "all four answers", fontsize=10)
    ax.set_xticks(range(4), ORDER_LEVELS)
    ax.set_xlabel("order in which the answers were first reached")

    ax = axes[1, 0]
    gaze = t["gaze_by_position"]
    gaze = gaze[gaze[SUBSET] == GUESSING]
    for chosen, colour, marker, name in [(1, "#4c72b0", "o", "answer guessed"), (0, "#8c8c8c", "s", "answers not guessed")]:
        s = gaze[gaze["is_chosen"] == chosen].set_index("position")
        _means(ax, s, POSITIONS, colour=colour, marker=marker, label=name, value="gaze_share", line=True)
    ax.set_xticks(range(4), POSITIONS)
    ax.set_ylabel("share of the answer gaze before the decision")
    ax.set_title("(c) Gaze on each position", fontsize=10)
    ax.legend(frameon=False, fontsize=9)

    ax = axes[1, 1]
    acc = t["choice_accuracy"]
    acc = acc[(acc[SUBSET] == GUESSING) & (acc["by"] == "position of the correct answer")].set_index("level")
    _means(ax, acc, POSITIONS)
    overall = _test(tests, GUESSING, "accuracy: the same wherever the correct answer sat")
    ax.axhline(overall["value"], color="k", lw=0.8, ls=":")
    ax.set_xticks(range(4), POSITIONS)
    ax.set_ylim(overall["value"] - 0.2, overall["value"] + 0.2)  # the same window as in `plot_choices`
    ax.set_ylabel("share of guesses that were right")
    ax.set_title(f"(d) Right guesses, by where the correct answer sat\nthe same everywhere: {_p(overall['p'])} "
                 f"(dotted: all guesses, {100 * overall['value']:.0f}%)", fontsize=10)
    for a in axes.flat:
        _style(a)
    n = int(top["n_trials"])
    fig.suptitle(f"Study 2, no knowledge: {n} guesses by {per['participant_id'].nunique()} participants "
                 "(dots: participants; points: their mean ± SE; dotted: one in four)", fontsize=11)
    tables = {"choice_tests": tests[tests[SUBSET] == GUESSING], "choice_by_participant": per}
    save_output(fig, analysis=ANALYSIS, plot="guesses", tables=tables, save=save, close=close, dataset=dataset_key)
    return fig


# --- Models ---------------------------------------------------------------------------------------

BLOCKS = [
    ("Position, against the top", "position", ["position_left", "position_right", "position_bottom"]),
    ("Position, with length and others' choices in the model", "content",
     ["position_left", "position_right", "position_bottom"]),
    ("Order first reached, against the 1st (trials reaching all four)", "order",
     ["reached_2nd", "reached_3rd", "reached_4th"]),
    ("Top and first reached, in one model", "top_and_first", ["at_top", "reached_first"]),
    ("Length, with position", "length", ["log2_words"]),
]


def plot_models(t: Dict[str, pd.DataFrame], dataset_key: str, *, save: Optional[bool] = None,
                close: Optional[bool] = None):
    """Odds ratios of being chosen, with 95% intervals from resampling participants
    (`choice.choice_models`), one colour per modelled group. Every model holds the kind of
    answer; the kinds' own odds ratios, and that of others' choices, are in the table.

    **Read the order rows with knowledge with care**: see `choice.py`."""
    models = t["models"]
    rows = []
    for heading, model, terms in BLOCKS:
        rows.append((heading, None, None))
        rows += [(TERM_LABELS[term], model, term) for term in terms]
    groups = MODELLED[dataset_key]
    fig, ax = plt.subplots(figsize=(9, 0.34 * len(rows) + 1.9), layout="constrained")
    offsets = dict(zip(groups, np.linspace(-0.14, 0.14, len(groups)) if len(groups) > 1 else [0.0]))
    for group in groups:
        m = models[models[SUBSET] == group].set_index(["model", "term"])
        labelled = False
        for i, (_, model, term) in enumerate(rows):
            if term is None:
                continue
            r = m.loc[(model, term)]
            ax.errorbar(r["odds_ratio"], i + offsets[group],
                        xerr=[[r["odds_ratio"] - r["ci_low"]], [r["ci_high"] - r["odds_ratio"]]],
                        fmt=MARKERS[group], color=COLOURS[group], ms=6, elinewidth=1.6, capsize=0,
                        label=None if labelled else f"{group} ({int(m.loc[('position', 'position_left'), 'n_trials'])} trials)")
            labelled = True
    ax.axvline(1, color="#555555", lw=1, ls="--")
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(FixedLocator([0.25, 0.5, 1, 2, 4]))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.set_xlim(0.2, 5)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows])
    for label, (_, model, _) in zip(ax.get_yticklabels(), rows):
        if model is None:
            label.set_fontweight("bold")
    ax.set_ylim(len(rows) - 0.4, -0.7)
    ax.tick_params(axis="y", length=0)
    _style(ax, grid="x")
    ax.set_xlabel("odds ratio of being chosen\n(95% interval, resampling participants)")
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    fig.suptitle(f"{STUDY_NAMES[dataset_key]}: what makes an answer more likely to be chosen, beyond its kind (A-D)",
                 fontsize=11)
    which = {"knowqa": "Order, partial knowledge", "l1": "Order (people who read the paragraph)"}[dataset_key]
    fig.supxlabel(f"{which}: a trial reaches all four answers more often when the known answer comes late, "
                  "so these lean late.", x=0.01, ha="left", fontsize=8, color="#555555")
    save_output(fig, analysis=ANALYSIS, plot="models", tables={"models": models}, save=save, close=close,
                dataset=dataset_key)
    return fig


# --- Scan and gaze --------------------------------------------------------------------------------

def plot_scan(t: Dict[str, pd.DataFrame], dataset_key: str, *, save: Optional[bool] = None,
              close: Optional[bool] = None):
    """What the eyes do with each position, one line per group, before the decision: (a) where
    the first look at an answer lands; (b) each position's share of the gaze on the answers not
    chosen; (c) visits per trial; (d) where the eyes land when they come back down from the
    question, after the opening look."""
    groups = SUBSETS[dataset_key]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), layout="constrained")
    first, gaze, returns = t["first_looks"], t["gaze_by_position"], t["returns"]
    for group in groups:
        colour, marker = COLOURS[group], MARKERS[group]
        f = first[first[SUBSET] == group].set_index("position")
        _means(axes[0, 0], f, POSITIONS, colour=colour, marker=marker, label=group, line=True)
        g = gaze[(gaze[SUBSET] == group) & (gaze["is_chosen"] == 0)].set_index("position")
        _means(axes[0, 1], g, POSITIONS, colour=colour, marker=marker, label=group, value="gaze_share", line=True)
        r = returns[returns[SUBSET] == group].set_index("position").reindex(POSITIONS)
        axes[1, 0].plot(range(4), r["visits_per_trial"], f"{marker}-", color=colour, ms=6.5, lw=1.5, label=group)
        axes[1, 1].plot(range(4), r["share_of_returns_from_question"], f"{marker}-", color=colour, ms=6.5, lw=1.5,
                        label=group)
    axes[0, 0].set_title("(a) Where the first look at an answer lands", fontsize=10)
    axes[0, 0].set_ylabel("share of trials")
    axes[0, 1].set_title("(b) Gaze on the answers not chosen", fontsize=10)
    axes[0, 1].set_ylabel("share of the answer gaze before the decision")
    axes[1, 0].set_title("(c) Visits to each position", fontsize=10)
    axes[1, 0].set_ylabel("visits per trial")
    axes[1, 1].axhline(0.25, color="k", lw=0.8, ls=":")
    axes[1, 1].set_title("(d) Where the eyes land coming back from the question", fontsize=10)
    axes[1, 1].set_ylabel("share of returns from the question")
    for a in axes.flat:
        a.set_xticks(range(4), POSITIONS)
        a.legend(frameon=False, fontsize=8.5)
        _style(a)
    fig.suptitle(f"{STUDY_NAMES[dataset_key]}: what the eyes do with each position before the decision "
                 "(participant mean ± SE where shown)", fontsize=11)
    tables = {k: t[k] for k in ["first_looks", "gaze_by_position", "returns", "usual_starts", "usual_start_choices",
                                "deviation"]}
    save_output(fig, analysis=ANALYSIS, plot="scan", tables=tables, save=save, close=close, dataset=dataset_key)
    return fig


# --- Side findings ----------------------------------------------------------------------------------

def plot_side(t: Dict[str, pd.DataFrame], dataset_key: str, *, save: Optional[bool] = None,
              close: Optional[bool] = None):
    """Side findings, one line or point per group. (a) Which kind of answer was chosen.
    (b) How much longer the chosen answer was than its trial's average. (c) Study 2 only: the
    confidence rating when the choice went to the top answer and when it went elsewhere. (d) The
    share of choices at the top in each third of the participant's own decision times -- a trap
    rather than a finding: an answer reached early can be chosen early, so fast choices go to the
    top more often even with no pull to it at all, as Study 1 shows."""
    groups = SUBSETS[dataset_key]
    panels = ["kinds", "length", "confidence", "speed"] if dataset_key == "knowqa" else ["kinds", "length", "speed"]
    fig, axes = plt.subplots(1, len(panels), figsize=(4.4 * len(panels), 4.6), layout="constrained")
    ax = dict(zip(panels, axes))
    kinds, length, speed = t["choice_kinds"], t["choice_length"], t["speed"]
    offsets = dict(zip(groups, np.linspace(-0.2, 0.2, len(groups))))
    for group in groups:
        colour, marker = COLOURS[group], MARKERS[group]
        k = kinds[kinds[SUBSET] == group].set_index("label")
        _means(ax["kinds"], k, LABELS, colour=colour, marker=marker, label=group, offset=offsets[group])
        s = speed[speed[SUBSET] == group].set_index("speed")
        _means(ax["speed"], s, SPEED_THIRDS, colour=colour, marker=marker, label=group, offset=offsets[group],
               value="share_chose_top", se="se_top")
    ax["kinds"].axhline(0.25, color="k", lw=0.8, ls=":")
    ax["kinds"].set_xticks(range(4), ["A\ncorrect", "B\nmisreads", "C\nother\nsentence", "D\nnot in\nthe text"],
                           fontsize=8)
    ax["kinds"].set_ylabel("share of choices")
    ax["kinds"].set_title("(a) Which kind of answer was chosen", fontsize=10)

    l = length.set_index(SUBSET).reindex(groups)
    for i, group in enumerate(groups):
        ax["length"].errorbar(i, l.loc[group, "log2_words_above_trial_mean"], yerr=l.loc[group, "se"],
                              fmt=MARKERS[group], color=COLOURS[group], ms=7, elinewidth=1.8, capsize=0)
    ax["length"].axhline(0, color="k", lw=0.8, ls=":")
    ax["length"].set_xticks(range(len(groups)), [g.replace(" ", "\n", 1) for g in groups], fontsize=8.5)
    ax["length"].set_ylabel("log2 words above the trial's average")
    ax["length"].set_title("(b) Length of the chosen answer", fontsize=10)

    if "confidence" in ax:
        conf = t["confidence"]
        conf = conf[conf["target"] == "top"]
        for i, group in enumerate(groups):
            c = conf[conf[SUBSET] == group].set_index("went_there")
            for went, mfc in [(0.0, "white"), (1.0, COLOURS[group])]:
                ax["confidence"].errorbar(i + (0.12 if went else -0.12), c.loc[went, "confidence"], yerr=c.loc[went, "se"],
                                          fmt=MARKERS[group], color=COLOURS[group], mfc=mfc, ms=7, elinewidth=1.8, capsize=0)
        ax["confidence"].set_xticks(range(len(groups)), [g.replace(" ", "\n", 1) for g in groups], fontsize=8.5)
        ax["confidence"].set_ylabel(f"confidence rating (1-5)")
        ax["confidence"].set_title("(c) Confidence: choice at the top (filled)\nor elsewhere (open)", fontsize=10)

    ax["speed"].axhline(0.25, color="k", lw=0.8, ls=":")
    ax["speed"].set_xticks(range(3), [s.replace(" ", "\n") for s in SPEED_THIRDS], fontsize=8.5)
    ax["speed"].set_ylabel("share of choices at the top")
    ax["speed"].set_title(f"({'d' if 'confidence' in ax else 'c'}) A trap: choices at the top by speed\n"
                          "(early-reached answers can be chosen early)", fontsize=10)
    for a in axes:
        _style(a)
    ax["kinds"].legend(frameon=False, fontsize=8)
    fig.suptitle(f"{STUDY_NAMES[dataset_key]}: side findings (participant mean ± SE)", fontsize=11)
    tables = {k: t[k] for k in ["choice_kinds", "choice_length", "speed", "speed_tests"]}
    if dataset_key == "knowqa":
        tables |= {"confidence": t["confidence"], "confidence_tests": t["confidence_tests"]}
    save_output(fig, analysis=ANALYSIS, plot="side_findings", tables=tables, save=save, close=close,
                dataset=dataset_key)
    return fig


# --- Both studies ---------------------------------------------------------------------------------

OVERVIEW = [("knowqa", "no knowledge"), ("knowqa", "partial knowledge"), ("knowqa", "full knowledge"),
            ("l1", "hunters"), ("l1", "gatherers")]


def plot_overview(by_study: Dict[str, Dict[str, pd.DataFrame]], *, save: Optional[bool] = None,
                  close: Optional[bool] = None):
    """Both studies, every condition, side by side: (a) the share of choices at the top;
    (b) accuracy with the correct answer at the top minus elsewhere; (c) on the answers not
    chosen, the top answer's share of the gaze minus the other positions'. Participant means
    with standard errors (`choice.top_summary`); p in (a) and (b) from random re-placements."""
    measures = [("share of choices at the top", "chosen answer: at the top one time in four", 0.25),
                ("accuracy: correct answer at the top minus elsewhere",
                 "accuracy: the same with the correct answer at the top as elsewhere", 0.0),
                ("gaze on answers not chosen: top minus elsewhere", None, 0.0)]
    titles = ["(a) Choices that went to the top answer", "(b) Accuracy: correct answer at the top, minus elsewhere",
              "(c) Gaze on answers not chosen: top, minus elsewhere"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), layout="constrained")
    rows = []
    for j, ((measure, test, ref), title) in enumerate(zip(measures, titles)):
        ax = axes[j]
        for i, (key, group) in enumerate(OVERVIEW):
            s = by_study[key]["top_summary"]
            r = s[(s[SUBSET] == group) & (s["measure"] == measure)].iloc[0]
            ax.errorbar(i, r["mean"], yerr=r["se"], fmt=MARKERS[group], color=COLOURS[group], ms=7.5,
                        elinewidth=1.8, capsize=0)
            p = np.nan
            if test is not None:
                p = _test(by_study[key]["choice_tests"], group, test)["p"]
                ax.annotate(_p(p), (i, r["mean"] + r["se"]), xytext=(0, 4), textcoords="offset points",
                            ha="center", fontsize=8, color="#444444")
            rows.append({"study": STUDY_NAMES[key], SUBSET: group, "measure": measure, "mean": r["mean"],
                         "se": r["se"], "n_participants": r["n_participants"], "p_randomisation": p})
        ax.axhline(ref, color="k", lw=0.8, ls=":")
        ax.axvline(2.5, color="#BBBBBB", lw=1)
        ax.margins(x=0.1, y=0.12)  # room for the p labels
        ax.set_xticks(range(len(OVERVIEW)), [f"{STUDY_NAMES[k]}\n{g.replace(' knowledge', '')}" for k, g in OVERVIEW],
                      fontsize=8.5)
        ax.set_title(title, fontsize=10)
        _style(ax)
    axes[0].set_ylabel("share of choices")
    axes[1].set_ylabel("difference in accuracy")
    axes[2].set_ylabel("difference in share of gaze")
    fig.suptitle("Both studies, every condition: choices at the top, accuracy with the correct answer at the top, "
                 "and gaze on the top answer (participant mean ± SE)", fontsize=11)
    save_output(fig, analysis=ANALYSIS, plot="overview", tables={"overview": pd.DataFrame(rows)}, save=save, close=close)
    return fig
