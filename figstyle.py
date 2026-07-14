"""Shared, publication-grade Matplotlib style for the paper's figures.

Key idea: author every figure at its *true* rendered width (one IEEE column,
~3.45 in) so fonts land on the page at their real point size instead of being
authored at 9 in and downscaled ~2.6x into the column (which is what made the
text unreadably small). Descriptive titles are dropped -- the LaTeX caption
carries that text; panels are tagged only "(a)"/"(b)".
"""
import matplotlib as mpl
import matplotlib.pyplot as plt

COL = 3.45          # IEEE conference single-column width (inches)
TEXT = 7.16         # full text width (two columns) for figure* if ever needed

# refined, consistent, colour-blind-friendly palette
COLORS = {
    "clean":  "#4d4d4d",   # neutral grey reference lines
    "fgsm":   "#1b6ca8",   # blue
    "pgd":    "#e8843c",   # orange
    "cw":     "#c0392b",   # red
    "ris":    "#c2185b",   # magenta  (the malicious/DNN-aware attack)
    "se":     "#7b1fa2",   # purple   (spectral-efficiency series)
    "rand":   "#2e7d32",   # green    (benign / random-RIS control)
    "jam":    "#2e7d32",   # green    (model-blind SNR jamming)
    "def":    "#00796b",   # teal     (defended model)
    "undef":  "#c0392b",   # red      (undefended model)
    "uni":    "#f5a623",   # amber    (universal, model-aware)
    "uni2":   "#f7cf5a",   # light amber (universal, surrogate)
}


def use_style():
    mpl.rcParams.update({
        "figure.dpi": 150, "savefig.dpi": 400, "savefig.pad_inches": 0.01,
        "font.family": "serif", "mathtext.fontset": "dejavuserif",
        "font.size": 8.5, "axes.titlesize": 9.0, "axes.labelsize": 9.0,
        "xtick.labelsize": 8.0, "ytick.labelsize": 8.0, "legend.fontsize": 8.0,
        "axes.linewidth": 0.7, "lines.linewidth": 1.5, "lines.markersize": 4.0,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.major.size": 2.6, "ytick.major.size": 2.6,
        "axes.grid": True, "axes.axisbelow": True,
        "grid.alpha": 0.25, "grid.linewidth": 0.5,
        "legend.frameon": False, "legend.handlelength": 1.6,
        "legend.borderpad": 0.2, "legend.labelspacing": 0.25,
        "legend.handletextpad": 0.4, "legend.columnspacing": 1.0,
    })


def despine(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def tag(ax, s):
    """Small bold panel tag '(a)' in the top-left, in-axes (no wasted margin)."""
    ax.set_title(s, loc="left", fontsize=9.0, fontweight="bold", pad=2)


def fig_2panel(h=1.95):
    fig, ax = plt.subplots(1, 2, figsize=(COL, h), constrained_layout=True)
    return fig, ax


def fig_1panel(h=2.2, w=COL):
    fig, ax = plt.subplots(figsize=(w, h), constrained_layout=True)
    return fig, ax


def fig_2panel_wide(h=2.35):
    """Double-column (figure*) 2-panel for dense figures that need the room."""
    fig, ax = plt.subplots(1, 2, figsize=(TEXT, h), constrained_layout=True)
    return fig, ax


def fig_wide_grid(nrows, ncols, h):
    """Double-column (figure*) grid; each panel gets ~half the text width so 8pt
    labels and legends fit without overlapping the data."""
    fig, ax = plt.subplots(nrows, ncols, figsize=(TEXT, h), constrained_layout=True)
    return fig, ax


def fig_2row(h=3.9):
    """Single-column, two panels stacked vertically -- each panel gets the full
    column width (readable) without claiming a double-column slot."""
    fig, ax = plt.subplots(2, 1, figsize=(COL, h), constrained_layout=True)
    return fig, ax


def figure_legend(fig, ax, ncol=3, y=1.02):
    """One shared legend above the panels (for multi-panel plots with same series)."""
    h, l = ax.get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", bbox_to_anchor=(0.5, y),
               ncol=ncol, fontsize=6.8, frameon=False)
