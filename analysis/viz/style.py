"""Publication styling (aligned with elyas/src/viz/style.py)."""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt

COLORS = {
    "nor": "#1769aa",
    "de": "#c2410c",
    "house": "#1769aa",
    "tree": "#15803d",
    "face": "#d97706",
    "person": "#7c3aed",
    "accent": "#0d9488",
    "muted": "#64748b",
}


def apply_pub_style() -> None:
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.linewidth": 1.0,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "grid.linestyle": "--",
        "grid.alpha": 0.35,
    })


def savefig(fig: plt.Figure, path, dpi: int = 300) -> None:
    path = __import__("pathlib").Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, facecolor="white")
    plt.close(fig)
