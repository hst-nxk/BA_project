"""Shared plotting style: one categorical order, fixed colour per persona, recessive axes."""
from __future__ import annotations

import sys

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from . import config as C

# Inside a Jupyter kernel keep the inline backend so figures render in the notebook;
# elsewhere (scripts, tests, servers) use the headless Agg backend.
if "ipykernel" not in sys.modules:
    matplotlib.use("Agg")

CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
TEXT, TEXT2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"

# colour follows the entity: each persona keeps its hue in every chart
PERSONA_COLORS = {
    "Non-starters": "#8a8984",
    "At-risk Disengaged": "#e34948",
    "Last-minute Crammers": "#eb6834",
    "Steady Independent Learners": "#2a78d6",
    "Consistent Achievers": "#1baf7a",
    "Night-owl Learners": "#6250d6",
    "Struggling Strivers": "#eda100",
    "Fading Starters": "#e87ba4",
}
OUTCOME_COLORS = {"Distinction": "#1baf7a", "Pass": "#2a78d6", "Fail": "#eda100", "Withdrawn": "#e34948"}
SEQUENTIAL = LinearSegmentedColormap.from_list("seq_blue", ["#eef4fc", "#9cc0ee", "#2a78d6", "#123d73"])
DIVERGING = LinearSegmentedColormap.from_list("div", ["#2a78d6", "#a9c8ef", "#e9e8e4", "#f2b199", "#e34948"])


def setup() -> None:
    plt.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 150, "savefig.bbox": "tight",
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": TEXT2, "axes.titlecolor": TEXT,
        "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "axes.labelsize": 10, "xtick.color": TEXT2, "ytick.color": TEXT2,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.prop_cycle": matplotlib.cycler(color=CATEGORICAL),
        "lines.linewidth": 2, "legend.frameon": False, "font.size": 10,
        "font.family": "DejaVu Sans",
    })


def save(fig, name: str) -> str:
    path = C.FIGURES / f"{name}.png"
    fig.savefig(path)
    return str(path)


setup()
