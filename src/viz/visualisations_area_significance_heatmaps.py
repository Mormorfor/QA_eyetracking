import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os


def _stars_from_p(p):
    if pd.isna(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def plot_pairwise_significance_heatmap(
    pairwise,
    title,
    alpha=0.05,
    save_path=None,
    areas=None,
    show=False,
):
    """
    Heatmap for pairwise comparisons table.

    Color = -log10(p_adj_holm)
    Annotation (upper triangle) = stars + direction of diff_i_minus_j

    pairwise must have: area_i, area_j, p_adj_holm, diff_i_minus_j
    """
    if areas is None:
        canonical = ["answer_A", "answer_B", "answer_C", "answer_D"]
        present = set(pairwise["area_i"]).union(set(pairwise["area_j"]))
        areas = [a for a in canonical if a in present]
        if not areas:
            areas = sorted(present)

    areas = list(areas)
    if len(areas) < 2:
        return

    idx = {a: i for i, a in enumerate(areas)}
    n = len(areas)

    P = np.full((n, n), np.nan, dtype=float)
    E = np.full((n, n), np.nan, dtype=float)

    for _, r in pairwise.iterrows():
        a = r["area_i"]
        b = r["area_j"]
        if a not in idx or b not in idx:
            continue
        i, j = idx[a], idx[b]
        p = r.get("p_adj_holm", np.nan)
        eff = r.get("diff_i_minus_j", np.nan)

        P[i, j] = p
        P[j, i] = p
        E[i, j] = eff
        E[j, i] = -eff

    P_clip = np.clip(P, 1e-300, 1.0)
    intensity = -np.log10(P_clip)
    np.fill_diagonal(intensity, 0.0)

    fig, ax = plt.subplots(figsize=(6, 5), dpi=140)
    im = ax.imshow(intensity, aspect="equal")

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(areas, rotation=45, ha="right")
    ax.set_yticklabels(areas)
    ax.set_title(title)

    # grid
    ax.set_xticks(np.arange(-.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-.5, n, 1), minor=True)
    ax.grid(which="minor", linewidth=0.5)
    ax.tick_params(which="minor", bottom=False, left=False)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("-log10(p_adj_holm)")

    # annotate only upper triangle
    for i in range(n):
        for j in range(i + 1, n):
            p = P[i, j]
            eff = E[i, j]
            if pd.isna(p):
                continue
            if p < alpha:
                stars = _stars_from_p(p)
                sign = "↑" if eff > 0 else ("↓" if eff < 0 else "0")
                ax.text(j, i, "{}\n{}".format(stars, sign),
                        ha="center", va="center", fontsize=9)

    fig.tight_layout()

    if save_path is not None:
        out_dir = os.path.dirname(save_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")

    if show:
        plt.show()

    plt.close(fig)
