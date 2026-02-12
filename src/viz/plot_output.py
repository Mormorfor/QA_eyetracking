import os
from pathlib import Path
import matplotlib.pyplot as plt


def project_root():
    # src/viz/save_plot.py → src → project root
    return Path(__file__).resolve().parents[2]


def save_plot(
    fig=None,
    rel_dir="",
    filename="plot",
    dpi=300,
    ext="png",
    paper_dirs=None,   # list like ["papers/correctness prediction/figures", ...]
    close=False,
):
    """
    Always saves to:
        reports/plots/<rel_dir>/<filename>.<ext>

    Optionally mirrors to each directory in paper_dirs:
        <paper_dir>/<rel_dir>/<filename>.<ext>
    """
    root = project_root()

    main_dir = root / "reports" / "plots" / rel_dir
    os.makedirs(main_dir, exist_ok=True)

    main_path = main_dir / f"{filename}.{ext}"
    fig.savefig(main_path, dpi=dpi, bbox_inches="tight")

    saved_paths = [str(main_path)]


    if paper_dirs:
        for p in paper_dirs:
            paper_dir = root / p / rel_dir
            os.makedirs(paper_dir, exist_ok=True)

            paper_path = paper_dir / f"{filename}.{ext}"
            fig.savefig(paper_path, dpi=dpi, bbox_inches="tight")

            saved_paths.append(str(paper_path))

    if close:
        plt.close(fig)

    return saved_paths
