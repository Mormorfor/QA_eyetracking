import os
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
from typing import List, Optional


def project_root():
    # src/viz/save_plot.py → src → project root
    return Path(__file__).resolve().parents[2]


def save_plot(
    fig=None,
    rel_dir="",
    filename="plot",
    dpi=300,
    ext="png",
    paper_dirs=None,
    close=False,
):
    """
    Always saves to:
        reports/plots/<rel_dir>/<filename>.<ext>

    Optionally mirrors to each directory in paper_dirs:
        <paper_dir>/figures/<rel_dir>/<filename>.<ext>
    """
    root = project_root()
    rel_dir = _clean_path_part(rel_dir) or ""

    main_dir = root / "reports" / "plots" / rel_dir
    main_dir.mkdir(parents=True, exist_ok=True)

    main_path = main_dir / f"{filename}.{ext}"
    fig.savefig(main_path, dpi=dpi, bbox_inches="tight")

    saved_paths = [str(main_path)]

    if paper_dirs:
        for p in paper_dirs:
            paper_dir = root / p / "figures" / rel_dir
            paper_dir.mkdir(parents=True, exist_ok=True)

            paper_path = paper_dir / f"{filename}.{ext}"
            fig.savefig(paper_path, dpi=dpi, bbox_inches="tight")
            saved_paths.append(str(paper_path))

    if close:
        plt.close(fig)

    return saved_paths




def save_df_csv(
    df: pd.DataFrame,
    *,
    rel_dir: str = "",
    filename: str = "table",
    paper_dirs: Optional[List[str]] = None,
) -> List[str]:
    """
    Always saves to:
        reports/report_data/<rel_dir>/<filename>.csv

    Optionally mirrors to each directory in paper_dirs:
        <paper_dir>/report_data/<rel_dir>/<filename>.csv
    """
    root = project_root()
    rel_dir = _clean_path_part(rel_dir) or ""

    main_dir = root / "reports" / "report_data" / rel_dir
    main_dir.mkdir(parents=True, exist_ok=True)

    main_path = main_dir / f"{filename}.csv"
    df.to_csv(main_path, index=False)

    saved_paths = [str(main_path)]

    if paper_dirs:
        for p in paper_dirs:
            paper_dir = root / p / "report_data" / rel_dir
            paper_dir.mkdir(parents=True, exist_ok=True)

            paper_path = paper_dir / f"{filename}.csv"
            df.to_csv(paper_path, index=False)
            saved_paths.append(str(paper_path))

    return saved_paths


def _clean_path_part(part: Optional[str]) -> Optional[str]:
    if part is None:
        return None
    part = str(part).strip().strip("/\\")
    return part or None


def _answer_correctness_rel_dir(
    model_family: str,
    subdir: Optional[str] = None,
    split_tag: Optional[str] = None,
    fit_all: bool = False,
) -> str:
    parts = ["answer_correctness", model_family]

    subdir = _clean_path_part(subdir)
    split_tag = _clean_path_part(split_tag)

    if subdir:
        parts.append(subdir)

    if fit_all:
        parts.append("full_fit")
    elif split_tag:
        parts.append(split_tag)

    return "/".join(parts)