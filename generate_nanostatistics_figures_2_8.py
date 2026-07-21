#!/usr/bin/env python3
"""
Generate publication-oriented Nanostatistics Figures 2-8.

The script prioritizes editable vector output and a restrained visual grammar
appropriate for Journal-family manuscripts:

- 183 mm double-column artwork by default.
- Lower-case panel labels placed at the upper-left of each panel.
- Sans-serif typography kept between 5 and 7 pt at final size.
- Okabe-Ito-inspired, colour-vision-deficiency-aware palette.
- Redundant encodings through marker shape, fill, outline, and hatching.
- PDF and SVG as primary editable outputs.
- PNG or TIFF only as raster previews.
- Figure titles and long explanations are written to an external caption file,
  rather than embedded in the artwork.
- Automatic checks for minimum font size, clipping, and excessively thin lines.

The numerical summaries used in Figure 7 match the audited manuscript values.
The original stress-test environment and the later audit environment are
reported separately in Figure 6.

Usage
-----
Generate all figures in PDF, SVG, and PNG:

    python nanostatistics_Journal_figures.py

Choose formats and output directory:

    python nanostatistics_Journal_figures.py \
        --outdir ./nanostatistics_figures \
        --formats pdf svg png

Generate only selected figures:

    python nanostatistics_Journal_figures.py --figures 2 4 7

Generate an additional grayscale accessibility preview:

    python nanostatistics_Journal_figures.py --grayscale-preview

Use a custom width and height:

    python nanostatistics_Journal_figures.py \
        --width-mm 183 --height-mm 118

Notes
-----
1. The generated drawings are conceptual or manuscript-summary graphics.
   They do not replace plots generated directly from raw analytical outputs.
2. PDF or SVG should be treated as the master artwork. Raster export does not
   improve line art beyond the quality already present in the vector file.
3. The script does not install or distribute fonts. It selects an available
   sans-serif font, preferring Liberation Sans, Arial, or Helvetica.
"""

from __future__ import annotations

import argparse
import json
import re
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.font_manager import FontProperties, findfont
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle
from matplotlib.text import Text

MM_TO_IN = 1.0 / 25.4


@dataclass(frozen=True)
class JournalSpec:
    width_mm: float = 183.0
    height_mm: float = 118.0
    min_font_pt: float = 5.0
    body_font_pt: float = 5.7
    heading_font_pt: float = 6.5
    panel_font_pt: float = 7.0
    preview_dpi: int = 300
    min_line_pt: float = 0.25

    @property
    def figsize(self) -> tuple[float, float]:
        return self.width_mm * MM_TO_IN, self.height_mm * MM_TO_IN


@dataclass(frozen=True)
class FigureText:
    title: str
    caption: str
    accessibility_note: str = ""


@dataclass(frozen=True)
class SummaryValue:
    median: float
    q1: float
    q3: float


SPEC = JournalSpec()

C = {
    "ink": "#222222",
    "black": "#000000",
    "white": "#FFFFFF",
    "blue": "#0072B2",
    "sky": "#56B4E9",
    "teal": "#009E73",
    "orange": "#E69F00",
    "vermillion": "#D55E00",
    "purple": "#CC79A7",
    "yellow": "#F0E442",
    "gray": "#6F6F6F",
    "gray_m": "#B8B8B8",
    "gray_l": "#F2F2F2",
    "blue_l": "#E5F1F8",
    "teal_l": "#E4F3EE",
    "orange_l": "#FFF2D9",
    "vermillion_l": "#FBE9E2",
    "purple_l": "#F6EAF1",
    "line": "#C8C8C8",
}

RESULTS = {
    "ols": {
        "rmse": SummaryValue(0.1565, 0.1526, 0.1606),
        "mae": SummaryValue(0.1353, 0.1318, 0.1400),
        "coverage": None,
        "pvs_theta": None,
        "pvs_pred": None,
        "pvs_joint": None,
        "divergent_fits": None,
        "total_divergences": None,
    },
    "mass": {
        "rmse": SummaryValue(0.1568, 0.1530, 0.1607),
        "mae": SummaryValue(0.1354, 0.1320, 0.1401),
        "coverage": SummaryValue(0.9931, 0.9896, 0.9965),
        "pvs_theta": None,
        "pvs_pred": SummaryValue(0.7736, 0.7665, 0.7804),
        "pvs_joint": None,
        "divergent_fits": 89,
        "total_divergences": 532,
    },
    "area": {
        "rmse": SummaryValue(0.0481, 0.0460, 0.0509),
        "mae": SummaryValue(0.0354, 0.0341, 0.0370),
        "coverage": SummaryValue(0.9549, 0.9479, 0.9583),
        "pvs_theta": SummaryValue(0.8227, 0.7393, 0.8737),
        "pvs_pred": SummaryValue(0.9069, 0.8783, 0.9277),
        "pvs_joint": SummaryValue(0.7432, 0.6522, 0.8047),
        "divergent_fits": 35,
        "total_divergences": 160,
    },
}

FIGURE_TEXT = {
    2: FigureText(
        title="The physical-inferential gap",
        caption=(
            "Figure 2. The physical-inferential gap. "
            "a, An apparently smooth and statistically plausible fitted relationship. "
            "b, The same numerical relationship viewed across physically distinct regimes. "
            "c, Smooth extrapolation entering a region in which descriptor or parameter "
            "interpretation is fragile. d, Post-fitting classification of posterior or "
            "predictive support relative to a declared admissible domain. PVS makes the "
            "specified admissibility assessment inspectable but does not alter the fitted "
            "model or posterior distribution."
        ),
        accessibility_note=(
            "Blue curves denote statistical structure, purple shading denotes regime "
            "boundaries, diagonal hatching denotes fragile extrapolation, and teal circles "
            "versus vermillion crosses distinguish admissible from inadmissible support."
        ),
    ),
    3: FigureText(
        title="Three organizing principles of Nanostatistics",
        caption=(
            "Figure 3. Three organizing principles of Nanostatistics. "
            "a, Structured stochasticity treats observed dispersion as potentially carrying "
            "measurement, sample, batch, laboratory, or latent-regime structure. "
            "b, Hierarchical uncertainty decomposition separates crossed grouping factors and "
            "other uncertainty levels when supported by the design. "
            "c, Regime-aware mechanistic admissibility ties interpretation to the conditions "
            "under which descriptors and model assumptions retain physical meaning."
        ),
        accessibility_note=(
            "The panels use distinct marker shapes and labels in addition to colour. "
            "Laboratory and shared-batch categories are shown as crossed factors."
        ),
    ),
    4: FigureText(
        title="Visual definition of parameter-space, predictive, and joint PVS",
        caption=(
            "Figure 4. Visual definition of parameter-space, predictive, and joint Physical "
            "Validity Scores. Parameter-space PVS is the posterior fraction contained in the "
            "declared parameter domain. Predictive PVS is the fraction of the complete "
            "posterior-predictive draw-by-observation array contained in the declared "
            "predictive domain. Joint PVS requires simultaneous parameter and predictive "
            "admissibility using matched posterior draws. The displayed proportions are "
            "schematic and are not benchmark results."
        ),
        accessibility_note=(
            "Admissible states use filled circles or cells; inadmissible states use crosses "
            "or hatched cells. Joint admissibility is identified by a paired-link symbol."
        ),
    ),
    5: FigureText(
        title="Nanostatistical workflow from physical-system definition to auditable interpretation",
        caption=(
            "Figure 5. Nanostatistical workflow. The workflow begins with the physical system "
            "and scientifically meaningful descriptors, then considers uncertainty structure, "
            "regimes and admissible domains, inferential specification, diagnostics, and "
            "traceable interpretation. Feedback arrows indicate that diagnostic evidence may "
            "require revision of domains, models, or descriptors. The workflow is an inspection "
            "and reporting scaffold rather than an automatic decision pipeline."
        ),
        accessibility_note=(
            "Each workflow stage has a numbered label and a distinct geometric symbol. "
            "Colour reinforces but does not determine the reading order."
        ),
    ),
    6: FigureText(
        title="Synthetic stress-test benchmark",
        caption=(
            "Figure 6. Synthetic stress-test benchmark. "
            "a, Controlled generation of 150 datasets, each comprising three laboratories "
            "crossed with four shared batch categories, eight mass levels, and three "
            "within-condition replicates. Nominal mass is transformed into surface-reactive "
            "area, which drives a Hill-type mean and heteroscedastic Student-t observations. "
            "b, The same synthetic datasets are analysed using three complete workflows. "
            "c, PVS-aware diagnostics are applied after fitting relative to declared parameter "
            "and predictive domains. d, Sampling, diagnostics, within-dataset error summaries, "
            "and reproducibility records. The benchmark compares complete workflows and does "
            "not isolate individual component effects."
        ),
        accessibility_note=(
            "The three workflows are distinguished by heading, numbering, border style, and "
            "colour. PVS is explicitly marked as a post-fitting assessment."
        ),
    ),
    7: FigureText(
        title="Workflow-level diagnostic profiles",
        caption=(
            "Figure 7. Workflow-level diagnostic profiles across 150 simulations. "
            "a, Descriptor and mean structure. b, Median within-dataset RMSE and MAE with "
            "interquartile ranges. c, Median posterior predictive coverage and available PVS "
            "quantities with interquartile ranges; unavailable quantities are shown as NA. "
            "d, Simulations containing post-tuning divergent transitions. The lower error of "
            "the area-based workflow arises from the complete aligned specification, whereas "
            "PVS makes declared admissibility inspectable after fitting."
        ),
        accessibility_note=(
            "Metrics use distinct marker shapes, direct labels, and numerical values. "
            "Divergent fits use filled vermillion circles; non-divergent fits use open gray circles."
        ),
    ),
    8: FigureText(
        title="Evidence boundaries, sensitivity results, and validation pathway",
        caption=(
            "Figure 8. Evidence boundaries and validation pathway. "
            "a, Findings supported by the controlled benchmark. b, Conditions that limit "
            "interpretation. c, Claims not established by the current study. d, Required next "
            "steps from ablation and real-data reanalysis to experimental, interlaboratory, and "
            "external validation. The present work is a controlled methodological demonstration "
            "rather than empirical validation across nanoscale systems."
        ),
        accessibility_note=(
            "Supported, conditional, unsupported, and future items use different symbols, "
            "headings, and border patterns in addition to colour."
        ),
    ),
}


def select_font() -> str:
    candidates = ["Liberation Sans", "Arial", "Helvetica", "Nimbus Sans", "DejaVu Sans"]
    for candidate in candidates:
        try:
            path = findfont(FontProperties(family=candidate), fallback_to_default=False)
            if path:
                return candidate
        except Exception:
            continue
    return "DejaVu Sans"


def configure_matplotlib(font_name: str, dpi: int) -> None:
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": [font_name, "Arial", "Helvetica", "DejaVu Sans"],
        "font.size": SPEC.body_font_pt,
        "axes.labelsize": SPEC.body_font_pt,
        "axes.titlesize": SPEC.heading_font_pt,
        "xtick.labelsize": 5.2,
        "ytick.labelsize": 5.2,
        "legend.fontsize": 5.2,
        "axes.unicode_minus": False,
        "mathtext.fontset": "dejavusans",
        "figure.facecolor": C["white"],
        "savefig.facecolor": C["white"],
        "savefig.dpi": dpi,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "axes.linewidth": 0.55,
        "lines.linewidth": 0.75,
        "patch.linewidth": 0.55,
    })


def make_canvas(spec: JournalSpec) -> tuple[Figure, Axes]:
    fig = plt.figure(figsize=spec.figsize)
    ax = fig.add_axes([0.015, 0.02, 0.97, 0.965])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_aspect("auto")
    ax.axis("off")
    return fig, ax


def rounded_box(ax: Axes, x: float, y: float, w: float, h: float, *, face: str = C["white"], edge: str = C["line"], lw: float = 0.65, radius: float = 1.0, hatch: str | None = None, alpha: float = 1.0, zorder: float = 1) -> FancyBboxPatch:
    patch = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0.18,rounding_size={radius}", facecolor=face, edgecolor=edge, linewidth=lw, hatch=hatch, alpha=alpha, zorder=zorder)
    ax.add_patch(patch)
    return patch


def panel_label(ax: Axes, x: float, y: float, label: str) -> None:
    ax.text(x, y, label, fontsize=SPEC.panel_font_pt, fontweight="bold", color=C["black"], ha="left", va="top", zorder=20)


def panel_heading(ax: Axes, x: float, y: float, text: str, *, color: str = C["ink"], ha: str = "left", size: float | None = None) -> None:
    ax.text(x, y, text, fontsize=size or SPEC.heading_font_pt, fontweight="bold", color=color, ha=ha, va="top")


def text_block(ax: Axes, x: float, y: float, text: str, *, width: int = 32, size: float | None = None, color: str = C["ink"], ha: str = "left", va: str = "top", weight: str = "normal", linespacing: float = 1.12) -> None:
    wrapped = "\n".join(textwrap.fill(part.strip(), width=width, break_long_words=False) for part in text.split("\n"))
    ax.text(x, y, wrapped, fontsize=size or SPEC.body_font_pt, color=color, ha=ha, va=va, fontweight=weight, linespacing=linespacing)


def arrow(ax: Axes, x1: float, y1: float, x2: float, y2: float, *, color: str = C["gray"], lw: float = 0.9, mutation: float = 8.0, rad: float = 0.0, zorder: float = 5, style: str = "-|>") -> FancyArrowPatch:
    patch = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style, mutation_scale=mutation, linewidth=lw, color=color, connectionstyle=f"arc3,rad={rad}", zorder=zorder, shrinkA=0, shrinkB=0)
    ax.add_patch(patch)
    return patch


def pill(ax: Axes, x: float, y: float, w: float, text: str, *, face: str, edge: str | None = None, text_color: str = C["white"], hatch: str | None = None, size: float = 5.2) -> None:
    rounded_box(ax, x, y, w, 3.4, face=face, edge=edge or face, lw=0.55, radius=1.6, hatch=hatch, zorder=8)
    ax.text(x + w / 2, y + 1.7, text, fontsize=size, color=text_color, ha="center", va="center", fontweight="bold", zorder=9)


def draw_hill(ax: Axes, x0: float, y0: float, w: float, h: float, *, curve_color: str = C["blue"], transition: bool = False, uncertainty: bool = True, points: bool = True, extrapolation: bool = False, seed: int = 42) -> None:
    rng = np.random.default_rng(seed)
    ax.plot([x0, x0], [y0, y0 + h], color=C["ink"], lw=0.5)
    ax.plot([x0, x0 + w], [y0, y0], color=C["ink"], lw=0.5)
    x = np.linspace(0.05, 5.0, 180)
    y = 0.04 + 0.90 * x**3 / (0.95**3 + x**3)
    xp = x0 + (x - x.min()) / (x.max() - x.min()) * w
    yp = y0 + y / 1.06 * h
    if transition:
        lower, upper = 0.95 * 0.75, 0.95 * 1.25
        xa = x0 + (lower - x.min()) / (x.max() - x.min()) * w
        xb = x0 + (upper - x.min()) / (x.max() - x.min()) * w
        ax.add_patch(Rectangle((xa, y0), xb - xa, h, facecolor=C["purple_l"], edgecolor=C["purple"], linewidth=0.35, hatch="////", alpha=0.65, zorder=0))
    if extrapolation:
        start = x0 + 0.77 * w
        ax.add_patch(Rectangle((start, y0), 0.23 * w, h, facecolor=C["vermillion_l"], edgecolor=C["vermillion"], linewidth=0.55, hatch="\\\\", alpha=0.85, zorder=0))
    if uncertainty:
        sigma = 0.035 * (1 + 0.10 * x)
        low = y0 + (y - 2.3 * sigma) / 1.06 * h
        high = y0 + (y + 2.3 * sigma) / 1.06 * h
        ax.fill_between(xp, low, high, color=C["orange"], alpha=0.16, zorder=1)
    ax.plot(xp, yp, color=curve_color, lw=1.45, zorder=4)
    if points:
        xx = np.linspace(0.08, 4.85, 25)
        yy0 = 0.04 + 0.90 * xx**3 / (0.95**3 + xx**3)
        yy = np.clip(yy0 + rng.normal(0, 0.027 * (1 + 0.15 * xx), len(xx)), -0.02, 1.03)
        ax.scatter(x0 + (xx - x.min()) / (x.max() - x.min()) * w, y0 + yy / 1.06 * h, s=8.5, marker="o", facecolor=C["sky"], edgecolor=C["white"], linewidth=0.3, zorder=5)


def posterior_cloud(ax: Axes, x0: float, y0: float, w: float, h: float, *, seed: int = 17, n: int = 72, domain: tuple[float, float, float, float] = (0.18, 0.78, 0.18, 0.78)) -> float:
    rng = np.random.default_rng(seed)
    pts = rng.multivariate_normal([0.52, 0.50], [[0.045, 0.024], [0.024, 0.060]], n)
    pts = np.clip(pts, 0.03, 0.97)
    x1, x2, y1, y2 = domain
    ax.add_patch(Rectangle((x0 + x1 * w, y0 + y1 * h), (x2 - x1) * w, (y2 - y1) * h, facecolor=C["teal_l"], edgecolor=C["teal"], linewidth=0.65, zorder=1))
    inside = (pts[:, 0] >= x1) & (pts[:, 0] <= x2) & (pts[:, 1] >= y1) & (pts[:, 1] <= y2)
    xp = x0 + pts[:, 0] * w
    yp = y0 + pts[:, 1] * h
    ax.scatter(xp[inside], yp[inside], s=10, marker="o", facecolor=C["teal"], edgecolor=C["white"], linewidth=0.25, zorder=3)
    ax.scatter(xp[~inside], yp[~inside], s=13, marker="x", color=C["vermillion"], linewidth=0.7, zorder=4)
    return float(inside.mean())


def metric_bar(ax: Axes, x: float, y: float, w: float, value: float, *, label: str, color: str, display: str | None = None, hatch: str | None = None) -> None:
    ax.add_patch(Rectangle((x, y - 0.8), w, 1.6, facecolor=C["gray_l"], edgecolor=C["line"], linewidth=0.35))
    ax.add_patch(Rectangle((x, y - 0.8), np.clip(value, 0.0, 1.0) * w, 1.6, facecolor=color, edgecolor=color, linewidth=0.35, hatch=hatch))
    ax.text(x - 0.8, y, label, fontsize=5.1, ha="right", va="center")
    if display is not None:
        ax.text(x + w + 0.8, y, display, fontsize=5.1, ha="left", va="center", fontweight="bold")


def draw_dot_grid(ax: Axes, x: float, y: float, divergent: int, *, total: int = 150, cols: int = 15, dx: float = 1.15, dy: float = 1.12, radius: float = 0.28) -> None:
    for idx in range(total):
        row, col = divmod(idx, cols)
        is_divergent = idx < divergent
        ax.add_patch(Circle((x + col * dx, y - row * dy), radius, facecolor=C["vermillion"] if is_divergent else C["white"], edgecolor=C["vermillion"] if is_divergent else C["gray_m"], linewidth=0.45))


def draw_iqr_point(ax: Axes, x0: float, y: float, value: SummaryValue, *, scale_min: float, scale_max: float, width: float, color: str, marker: str, label: str, value_format: str = ".4f") -> None:
    def transform(v: float) -> float:
        return x0 + (v - scale_min) / (scale_max - scale_min) * width
    q1x, q3x, mx = transform(value.q1), transform(value.q3), transform(value.median)
    ax.plot([q1x, q3x], [y, y], color=color, lw=1.0)
    ax.plot([q1x, q1x], [y - 0.65, y + 0.65], color=color, lw=0.55)
    ax.plot([q3x, q3x], [y - 0.65, y + 0.65], color=color, lw=0.55)
    ax.scatter([mx], [y], s=18, marker=marker, facecolor=color, edgecolor=C["white"], linewidth=0.4, zorder=6)
    ax.text(x0 - 0.8, y, label, fontsize=5.0, ha="right", va="center")
    ax.text(x0 + width + 0.8, y, format(value.median, value_format), fontsize=5.0, ha="left", va="center", fontweight="bold")


def draw_na(ax: Axes, x: float, y: float, w: float, *, label: str) -> None:
    ax.text(x - 0.8, y, label, fontsize=5.0, ha="right", va="center", color=C["gray"])
    rounded_box(ax, x, y - 0.85, w, 1.7, face=C["gray_l"], edge=C["gray_m"], lw=0.35, radius=0.5, hatch="...")
    ax.text(x + w / 2, y, "NA", fontsize=5.0, ha="center", va="center", color=C["gray"], fontweight="bold")


def figure_2(spec: JournalSpec) -> Figure:
    fig, ax = make_canvas(spec)
    panel_y, panel_h, panel_w = 18, 74, 22.5
    xs = [2.0, 26.5, 51.0, 75.5]
    colors = [C["blue"], C["purple"], C["vermillion"], C["teal"]]
    headings = ["Observed numerical pattern", "Hidden physical organization", "Smooth extrapolation", "Admissibility-aware inspection"]
    for idx, (x, color, heading) in enumerate(zip(xs, colors, headings)):
        rounded_box(ax, x, panel_y, panel_w, panel_h, edge=color, lw=0.75)
        panel_label(ax, x + 1.2, panel_y + panel_h - 1.2, chr(ord("a") + idx))
        panel_heading(ax, x + 4.8, panel_y + panel_h - 1.5, heading, color=color, size=6.0)
    draw_hill(ax, xs[0] + 2.4, 47, 17.5, 22, transition=False, seed=1)
    pill(ax, xs[0] + 4.2, 37.2, 14.0, "smooth fitted relation", face=C["blue"])
    text_block(ax, xs[0] + 11.25, 31.3, "Numerically regular", ha="center", size=5.4, weight="bold", color=C["blue"])
    draw_hill(ax, xs[1] + 2.4, 47, 17.5, 22, transition=True, seed=1)
    ax.text(xs[1] + 6.8, 72, "Regime I", fontsize=5.2, ha="center", color=C["teal"])
    ax.text(xs[1] + 16.8, 72, "Regime II", fontsize=5.2, ha="center", color=C["purple"])
    ax.text(xs[1] + 11.25, 32, "same curve, different physical interpretation", fontsize=5.0, ha="center", color=C["gray"])
    draw_hill(ax, xs[2] + 2.4, 47, 17.5, 22, transition=True, extrapolation=True, seed=1)
    ax.text(xs[2] + 17.2, 70.5, "fragile\ninterpretation", fontsize=5.0, ha="center", va="bottom", color=C["vermillion"], fontweight="bold")
    pill(ax, xs[2] + 4.1, 35.5, 14.3, "smooth does not imply valid", face=C["vermillion"], hatch="\\")
    score = posterior_cloud(ax, xs[3] + 3.0, 50, 16.0, 18.0, seed=23)
    ax.text(xs[3] + 11.0, 70.5, r"Declared domain $\Omega_{\mathrm{phys}}$", fontsize=5.1, ha="center", color=C["teal"], fontweight="bold")
    metric_bar(ax, xs[3] + 5.2, 41, 11.5, score, label="PVS", color=C["teal"], display=f"{score:.2f}")
    ax.text(xs[3] + 11.0, 33.4, "post-fitting classification", fontsize=5.0, ha="center", color=C["gray"])
    rounded_box(ax, 20, 6.3, 60, 6.5, face=C["blue_l"], edge=C["blue"], lw=0.65, radius=1.3)
    ax.text(50, 9.55, "Parametric continuity does not guarantee mechanistic continuity.", fontsize=6.4, ha="center", va="center", color=C["blue"], fontweight="bold")
    return fig


def figure_3(spec: JournalSpec) -> Figure:
    fig, ax = make_canvas(spec)
    xs, w, y, h = [2.0, 34.0, 66.0], 31.0, 17, 76
    headings = ["Structured stochasticity", "Hierarchical uncertainty decomposition", "Regime-aware mechanistic admissibility"]
    colors = [C["orange"], C["blue"], C["purple"]]
    for idx, (x, heading, color) in enumerate(zip(xs, headings, colors)):
        rounded_box(ax, x, y, w, h, edge=color, lw=0.75)
        panel_label(ax, x + 1.2, y + h - 1.2, chr(ord("a") + idx))
        panel_heading(ax, x + 4.8, y + h - 1.5, heading, color=color, size=6.0)
    rng = np.random.default_rng(2)
    clusters = [(11, 69, C["sky"], "o", 18, 2.5, 2.0), (23, 61, C["orange"], "s", 16, 2.8, 2.4), (12, 49, C["purple"], "^", 14, 2.5, 2.4), (24, 42, C["teal"], "D", 14, 2.4, 2.1)]
    for cx, cy, color, marker, n, sx, sy in clusters:
        ax.scatter(rng.normal(cx, sx, n), rng.normal(cy, sy, n), s=11, marker=marker, facecolor=color, edgecolor=C["white"], linewidth=0.25, alpha=0.88)
    ax.text(17.5, 76, "Observed variability", fontsize=5.5, ha="center", fontweight="bold")
    labels = [("measurement", C["sky"], "o"), ("batch", C["orange"], "s"), ("laboratory", C["purple"], "^"), ("latent structure", C["teal"], "D")]
    for idx, (label, color, marker) in enumerate(labels):
        xx, yy = 6.0 + (idx % 2) * 14, 29.0 - (idx // 2) * 5.0
        ax.scatter([xx], [yy], s=16, marker=marker, color=color, edgecolor=C["white"], linewidth=0.3)
        ax.text(xx + 1.8, yy, label, fontsize=5.0, va="center")
    ax.text(17.5, 20.6, "Investigate before absorbing into one residual term", fontsize=5.1, ha="center", color=C["gray"])
    lab_xs, batch_xs = [38.5, 46.5, 54.5], [37.2, 43.8, 50.4, 57.0]
    for i, lx in enumerate(lab_xs, 1):
        rounded_box(ax, lx, 69, 6.0, 5.2, face=C["blue"], edge=C["blue"], radius=0.7)
        ax.text(lx + 3, 71.6, f"Lab {i}", fontsize=5.0, color=C["white"], ha="center", va="center", fontweight="bold")
    for j, bx in enumerate(batch_xs, 1):
        rounded_box(ax, bx, 54, 5.5, 5.2, face=C["orange"], edge=C["orange"], radius=0.7)
        ax.text(bx + 2.75, 56.6, f"B{j}", fontsize=5.0, color=C["white"], ha="center", va="center", fontweight="bold")
    for lx in [41.5, 49.5, 57.5]:
        for bx in [39.95, 46.55, 53.15, 59.75]:
            ax.plot([lx, bx], [69, 59.2], color=C["line"], lw=0.4)
    ax.text(49.5, 77, "3 laboratories × 4 shared batch categories", fontsize=5.2, ha="center", fontweight="bold")
    layers = [("measurement", C["gray"]), ("sample", C["sky"]), ("batch", C["orange"]), ("laboratory", C["purple"]), ("model", C["blue"]), ("regime", C["teal"]), ("extrapolation", C["vermillion"])]
    widths = [25, 23, 21, 19, 17, 15, 13]
    for idx, ((label, color), width) in enumerate(zip(layers, widths)):
        xx, yy = 49.5 - width / 2, 21.5 + idx * 3.4
        rounded_box(ax, xx, yy, width, 2.2, face=color, edge=color, radius=0.45)
        ax.text(49.5, yy + 1.1, label, fontsize=5.0, color=C["white"], ha="center", va="center", fontweight="bold")
    draw_hill(ax, 70.0, 45, 23.0, 24, transition=True, extrapolation=True, seed=7)
    ax.add_patch(Rectangle((70.0, 45), 23.0, 24, facecolor="none", edgecolor=C["teal"], linewidth=0.6))
    ax.text(81.5, 72, "Declared response domain", fontsize=5.2, ha="center", color=C["teal"], fontweight="bold")
    ax.text(81.5, 31.2, "Interpretation remains tied to descriptor meaning,\nregime, and scientifically justified boundaries.", fontsize=5.1, ha="center", va="top", color=C["gray"], linespacing=1.1)
    rounded_box(ax, 15.0, 6.2, 70.0, 6.5, face=C["gray_l"], edge=C["blue"], lw=0.6)
    ax.text(50, 9.45, "Structured variability  →  decomposed uncertainty  →  regime-aware interpretation", fontsize=6.2, color=C["blue"], ha="center", va="center", fontweight="bold")
    return fig


def figure_4(spec: JournalSpec) -> Figure:
    fig, ax = make_canvas(spec)
    col_x, col_w = [4.5, 29.5, 54.5, 79.5], [21, 21, 21, 16]
    for x, width, heading in zip(col_x, col_w, ["Distribution", "Declared domain", "Classification", "Score"]):
        rounded_box(ax, x, 90.5, width, 5.2, face=C["blue"], edge=C["blue"], radius=0.7)
        ax.text(x + width / 2, 93.1, heading, fontsize=5.8, color=C["white"], ha="center", va="center", fontweight="bold")
    row_y, row_h = [64.5, 39.0, 13.5], 21
    for idx, (y, name, color) in enumerate(zip(row_y, ["Parameter-space PVS", "Predictive PVS", "Joint PVS"], [C["purple"], C["blue"], C["teal"]])):
        rounded_box(ax, 0.7, y, 3.0, row_h, face=color, edge=color, radius=0.6)
        ax.text(2.2, y + row_h / 2, name, fontsize=5.1, color=C["white"], rotation=90, ha="center", va="center", fontweight="bold")
        panel_label(ax, 4.3, y + row_h - 0.2, chr(ord("a") + idx))
    f_theta = posterior_cloud(ax, 7.0, 68.0, 16.0, 12.0, seed=4)
    ax.text(15, 81.3, r"Posterior draws $\theta^{(s)}$", fontsize=5.2, ha="center", fontweight="bold")
    rounded_box(ax, 32.0, 68.0, 16.0, 12.0, face=C["teal_l"], edge=C["teal"], lw=0.65)
    ax.text(40, 75.7, r"$\Omega_{\mathrm{phys}}$", fontsize=8.0, color=C["teal"], ha="center", va="center", fontweight="bold")
    ax.text(40, 71.3, "declared parameter domain", fontsize=5.0, ha="center")
    param_states = [(57.3, 76.8, C["teal"], "o"), (61.0, 72.0, C["teal"], "o"), (64.7, 77.8, C["teal"], "o"), (68.4, 71.0, C["vermillion"], "x"), (72.0, 75.5, C["teal"], "o")]
    for x, y, color, marker in param_states:
        ax.scatter([x], [y], s=19, marker=marker, color=color, linewidth=0.8, edgecolor=C["white"] if marker == "o" else None)
    ax.text(64.7, 81.3, "inside / outside", fontsize=5.2, ha="center", fontweight="bold")
    metric_bar(ax, 83.0, 74.0, 9.0, f_theta, label=r"$\widehat{\mathrm{PVS}}_\theta$", color=C["teal"], display="schematic")
    rng = np.random.default_rng(10)
    status = rng.random((6, 12)) < 0.88
    x0, y0 = 6.5, 44.5
    for r in range(status.shape[0]):
        for c in range(status.shape[1]):
            ok = bool(status[r, c])
            ax.add_patch(Rectangle((x0 + c * 1.48, y0 + r * 1.65), 1.05, 1.15, facecolor=C["teal"] if ok else C["vermillion_l"], edgecolor=C["teal"] if ok else C["vermillion"], linewidth=0.3, hatch=None if ok else "xx"))
    f_pred = float(status.mean())
    ax.text(15, 57.0, r"Draw-by-observation array $\widetilde y_j^{(s)}$", fontsize=5.1, ha="center", fontweight="bold")
    rounded_box(ax, 32.0, 43.5, 16.0, 12.0, face=C["teal_l"], edge=C["teal"], lw=0.65)
    ax.text(40, 51.0, r"$\mathcal{Y}_{\mathrm{phys}}$", fontsize=8.0, color=C["teal"], ha="center", va="center", fontweight="bold")
    ax.text(40, 46.8, "declared predictive domain", fontsize=5.0, ha="center")
    for idx, ok in enumerate([True, True, False, True, True, False]):
        xx = 57.0 + idx * 3.0
        ax.add_patch(Rectangle((xx, 48.2), 2.0, 2.7, facecolor=C["teal"] if ok else C["vermillion_l"], edgecolor=C["teal"] if ok else C["vermillion"], linewidth=0.35, hatch=None if ok else "xx"))
    ax.text(65.0, 56.9, "status for each draw × observation", fontsize=5.0, ha="center", fontweight="bold")
    metric_bar(ax, 83.0, 49.5, 9.0, f_pred, label=r"$\widehat{\mathrm{PVS}}_{\mathrm{pred}}$", color=C["teal"], display="schematic")
    pairs = [(True, True), (True, False), (False, True), (False, False)]
    for idx, (p_ok, y_ok) in enumerate(pairs):
        xx = 7.0 + idx * 5.0
        ax.scatter([xx], [24.2], s=22, marker="o" if p_ok else "x", color=C["teal"] if p_ok else C["vermillion"], linewidth=0.8, edgecolor=C["white"] if p_ok else None)
        arrow(ax, xx + 1.1, 24.2, xx + 2.4, 24.2, color=C["gray"], lw=0.55, mutation=6)
        ax.add_patch(Rectangle((xx + 2.7, 23.2), 1.9, 1.9, facecolor=C["teal"] if y_ok else C["vermillion_l"], edgecolor=C["teal"] if y_ok else C["vermillion"], linewidth=0.35, hatch=None if y_ok else "xx"))
        counted = p_ok and y_ok
        ax.text(xx + 2.2, 19.7, "counted" if counted else "not counted", fontsize=5.0, ha="center", color=C["teal"] if counted else C["vermillion"], fontweight="bold")
    ax.text(15, 32.0, r"Matched states $(\theta^{(s)},\widetilde y_j^{(s)})$", fontsize=5.1, ha="center", fontweight="bold")
    rounded_box(ax, 31.2, 18.4, 17.6, 11.5, face=C["gray_l"], edge=C["blue"], lw=0.65)
    ax.text(40, 25.8, r"$\mathbb{I}(\theta^{(s)}\in\Omega_{\mathrm{phys}})$", fontsize=6.0, ha="center", color=C["blue"], fontweight="bold")
    ax.text(40, 22.8, "AND", fontsize=5.0, ha="center", fontweight="bold")
    ax.text(40, 20.2, r"$\mathbb{I}(\widetilde y_j^{(s)}\in\mathcal{Y}_{\mathrm{phys}})$", fontsize=6.0, ha="center", color=C["teal"], fontweight="bold")
    pill(ax, 56.0, 25.0, 13.0, "parameter event", face=C["purple"], hatch="//")
    pill(ax, 56.0, 19.5, 13.0, "predictive event", face=C["teal"])
    ax.text(70.7, 23.0, "matched by posterior draw", fontsize=5.0, ha="center", va="center", color=C["gray"])
    metric_bar(ax, 83.0, 24.0, 9.0, 0.72, label=r"$\widehat{\mathrm{PVS}}_{\mathrm{joint}}$", color=C["teal"], display="schematic")
    rounded_box(ax, 22, 4.2, 56, 5.6, face=C["vermillion_l"], edge=C["vermillion"], lw=0.55, hatch="..")
    ax.text(50, 7.0, "PVS classifies fitted support. It does not remove draws, modify the posterior, or improve prediction by itself.", fontsize=5.4, ha="center", va="center", color=C["vermillion"], fontweight="bold")
    return fig


def _workflow_stage(ax: Axes, x: float, y: float, w: float, h: float, *, number: int, title: str, color: str, symbol: str, prompt: str) -> None:
    rounded_box(ax, x, y, w, h, face=C["white"], edge=color, lw=0.7)
    ax.text(x + 1.1, y + h - 1.1, str(number), fontsize=6.4, ha="left", va="top", fontweight="bold", color=color)
    ax.text(x + w / 2, y + h - 2.0, title, fontsize=5.7, ha="center", va="top", fontweight="bold", color=color)
    ax.text(x + w / 2, y + h / 2 + 1.3, symbol, fontsize=11.0, ha="center", va="center", color=color, fontweight="bold")
    ax.text(x + w / 2, y + 3.0, textwrap.fill(prompt, width=24), fontsize=5.0, ha="center", va="center", color=C["gray"], linespacing=1.05)


def figure_5(spec: JournalSpec) -> Figure:
    fig, ax = make_canvas(spec)
    stages = [("Physical system", C["sky"], "●", "What physical mechanism and response are being studied?"), ("Mechanistic descriptors", C["teal"], "A", "Does each descriptor retain physical meaning across the domain?"), ("Uncertainty structure", C["orange"], "σ", "Which uncertainty sources can the design identify?"), ("Regimes and domains", C["purple"], "Ω", "Which regimes, boundaries, and admissible states are justified?"), ("Inferential model", C["blue"], "f", "Which statistical, Bayesian, machine-learning, or hybrid model is proportionate?"), ("Diagnostics and PVS", C["teal"], "P", "Which statistical, computational, and admissibility checks are required?"), ("Interpretation and provenance", C["gray"], "≡", "Which claims, limits, data, code, seeds, and records are traceable?")]
    top_xs, bottom_xs, top_y, bottom_y, w, h = [3.0, 27.0, 51.0, 75.0], [15.0, 39.5, 64.0], 57, 28, 21.0, 24.0
    for idx, (x, stage) in enumerate(zip(top_xs, stages[:4]), 1):
        title, color, symbol, prompt = stage
        _workflow_stage(ax, x, top_y, w, h, number=idx, title=title, color=color, symbol=symbol, prompt=prompt)
        if idx < 4:
            arrow(ax, x + w + 0.8, top_y + h / 2, top_xs[idx] - 0.8, top_y + h / 2, color=C["gray_m"], lw=0.75, mutation=7)
    arrow(ax, top_xs[-1] + w / 2, top_y - 0.8, bottom_xs[0] + w / 2, bottom_y + h + 0.8, color=C["gray_m"], lw=0.75, mutation=7, rad=0.2)
    for local_idx, (x, stage) in enumerate(zip(bottom_xs, stages[4:]), 5):
        title, color, symbol, prompt = stage
        _workflow_stage(ax, x, bottom_y, w, h, number=local_idx, title=title, color=color, symbol=symbol, prompt=prompt)
        if local_idx < 7:
            next_x = bottom_xs[local_idx - 4]
            arrow(ax, x + w + 0.8, bottom_y + h / 2, next_x - 0.8, bottom_y + h / 2, color=C["gray_m"], lw=0.75, mutation=7)
    rounded_box(ax, 4.5, 14.8, 91.0, 8.0, face=C["gray_l"], edge=C["line"], lw=0.55)
    ax.text(7.0, 20.3, "STATISTICAL AND COMPUTATIONAL", fontsize=5.4, color=C["blue"], fontweight="bold", va="center")
    ax.text(35.0, 20.3, "residuals   PPC   coverage   convergence   ESS   divergences   domain of applicability", fontsize=5.1, color=C["ink"], va="center")
    ax.text(7.0, 17.0, "PHYSICAL ADMISSIBILITY", fontsize=5.4, color=C["teal"], fontweight="bold", va="center")
    ax.text(35.0, 17.0, "parameter PVS   predictive PVS   joint PVS   domain sensitivity", fontsize=5.1, color=C["ink"], va="center")
    arrow(ax, 74.0, 26.0, 69.0, 54.0, color=C["purple"], lw=0.7, mutation=7, rad=0.15)
    ax.text(76.5, 42.0, "revise domains", fontsize=5.0, color=C["purple"], ha="center", rotation=72)
    arrow(ax, 62.0, 26.0, 50.0, 54.0, color=C["orange"], lw=0.7, mutation=7, rad=0.12)
    ax.text(58.5, 40.5, "revise model", fontsize=5.0, color=C["orange"], ha="center", rotation=63)
    arrow(ax, 51.0, 26.0, 31.0, 54.0, color=C["teal"], lw=0.7, mutation=7, rad=0.08)
    ax.text(42.5, 39.0, "reconsider descriptors", fontsize=5.0, color=C["teal"], ha="center", rotation=50)
    rounded_box(ax, 24.0, 5.0, 52.0, 5.7, face=C["blue_l"], edge=C["blue"], lw=0.55)
    ax.text(50, 7.85, "Inspection and reporting scaffold, not an automatic acceptance pipeline", fontsize=5.7, ha="center", va="center", color=C["blue"], fontweight="bold")
    return fig


def figure_6(spec: JournalSpec) -> Figure:
    fig, ax = make_canvas(spec)
    panels = [(2.0, 52.0, 47.0, 44.0, "a", "Controlled data-generating process", C["blue"]), (51.0, 52.0, 47.0, 44.0, "b", "Three compared workflows", C["orange"]), (2.0, 5.0, 47.0, 43.0, "c", "Post-fitting admissibility assessment", C["purple"]), (51.0, 5.0, 47.0, 43.0, "d", "Computation, diagnostics, and reproducibility", C["teal"])]
    for x, y, w, h, label, heading, color in panels:
        rounded_box(ax, x, y, w, h, edge=color, lw=0.75)
        panel_label(ax, x + 1.2, y + h - 1.2, label)
        panel_heading(ax, x + 4.8, y + h - 1.5, heading, color=color, size=6.0)
    tags = [("3 labs", C["purple"], 7.5), ("4 shared batches", C["orange"], 13.0), ("8 mass levels", C["blue"], 10.5), ("3 replicates", C["gray"], 9.5)]
    xx = 5.2
    for text, color, width in tags:
        pill(ax, xx, 83.5, width, text, face=color)
        xx += width + 1.0
    ax.text(25.5, 80.0, "288 observations per dataset   •   150 independent datasets", fontsize=5.1, ha="center", fontweight="bold")
    nodes = [(8.0, "Nominal\nmass", C["sky"], "m"), (19.5, "Surface-reactive\narea", C["teal"], "A"), (31.0, "Hill-type\nmean", C["purple"], "μ"), (42.5, "Student-t\nobservations", C["orange"], "y")]
    for x, label, color, symbol in nodes:
        rounded_box(ax, x - 4.2, 64.0, 8.4, 10.0, face=C["white"], edge=color, lw=0.7)
        ax.text(x, 70.5, symbol, fontsize=9.0, color=color, ha="center", va="center", fontweight="bold")
        ax.text(x, 66.2, label, fontsize=5.0, color=C["ink"], ha="center", va="center", fontweight="bold")
    for left, right in zip(nodes[:-1], nodes[1:]):
        arrow(ax, left[0] + 4.6, 69.0, right[0] - 4.6, 69.0, color=C["gray_m"], lw=0.75, mutation=7)
    ax.text(14.0, 59.7, r"$\log A=f(\log m,\mathrm{lab},\mathrm{batch},\mathrm{observation})$", fontsize=5.0, ha="center", color=C["teal"])
    ax.text(32.2, 59.7, r"$\mu(A)$: saturating Hill response", fontsize=5.0, ha="center", color=C["purple"])
    ax.text(42.0, 55.8, r"$\sigma(A)=\sigma_0(1+\rho_A A)$", fontsize=5.0, ha="center", color=C["orange"])
    workflow_xs, workflow_w = [53.7, 68.4, 83.1], 13.0
    workflow_data = [("W1", "OLS/ANOVA\nmass-based", C["gray"], ["mass descriptor", "linear mean", "fixed lab + batch", "no posterior PVS"]), ("W2", "Hierarchical Bayes\nmass-based", C["blue"], ["mass descriptor", "linear mean", "partial pooling", "Student-t", "predictive PVS"]), ("W3", "Area-based Bayes\n+ PVS-aware", C["teal"], ["area descriptor", "Hill mean", "partial pooling", "heteroscedastic Student-t", "full PVS set"])]
    for x, (code, title, color, items) in zip(workflow_xs, workflow_data):
        rounded_box(ax, x, 58.5, workflow_w, 27.0, face=C["white"], edge=color, lw=1.0 if code == "W3" else 0.65, hatch="//" if code == "W1" else None)
        rounded_box(ax, x + 0.8, 78.0, workflow_w - 1.6, 6.0, face=color, edge=color, lw=0.55)
        ax.text(x + workflow_w / 2, 82.2, code, fontsize=5.8, color=C["white"], ha="center", va="center", fontweight="bold")
        ax.text(x + workflow_w / 2, 79.6, title, fontsize=5.0, color=C["white"], ha="center", va="center", fontweight="bold")
        for idx, item in enumerate(items):
            yy = 75.2 - idx * 3.15
            marker = ["o", "s", "^", "D", "P"][idx]
            ax.scatter([x + 1.5], [yy], s=10, marker=marker, color=color, edgecolor=C["white"], linewidth=0.25)
            ax.text(x + 2.7, yy, item, fontsize=5.05, va="center")
    ax.text(74.5, 54.8, "The same synthetic datasets enter all three workflows", fontsize=5.0, ha="center", color=C["gray"])
    posterior_cloud(ax, 6.0, 18.0, 13.5, 16.0, seed=8)
    ax.text(12.8, 36.2, "posterior draws", fontsize=5.2, ha="center", fontweight="bold")
    arrow(ax, 20.5, 26.0, 25.0, 26.0, color=C["purple"], lw=0.75, mutation=7)
    rounded_box(ax, 26.0, 18.0, 10.5, 16.0, face=C["teal_l"], edge=C["teal"], lw=0.65)
    ax.text(31.25, 28.5, r"$\Omega_{\mathrm{phys}}$", fontsize=7.0, color=C["teal"], ha="center", fontweight="bold")
    ax.text(31.25, 22.2, r"$\mathcal{Y}_{\mathrm{phys}}$", fontsize=7.0, color=C["teal"], ha="center", fontweight="bold")
    arrow(ax, 37.5, 26.0, 40.7, 26.0, color=C["purple"], lw=0.75, mutation=7)
    pill(ax, 39.0, 30.5, 7.0, r"PVS$_\theta$", face=C["purple"], hatch="//")
    pill(ax, 39.0, 25.0, 7.0, r"PVS$_{\mathrm{pred}}$", face=C["blue"])
    pill(ax, 39.0, 19.5, 7.0, r"PVS$_{\mathrm{joint}}$", face=C["teal"])
    ax.text(25.5, 12.0, "POST-FITTING", fontsize=5.4, color=C["purple"], ha="center", fontweight="bold")
    ax.text(25.5, 8.9, "Support is classified; draws are not removed or altered", fontsize=5.0, color=C["gray"], ha="center")
    blocks = [
        ("Sampling", C["blue"], "4 chains × (3,000 tuning + 3,000 retained)\ntarget_accept = 0.99"),
        ("Diagnostics", C["orange"], r"$\widehat R$ • ESS$_{\mathrm{bulk}}$ • ESS$_{\mathrm{tail}}$" + "\nBFMI • divergences"),
        ("Performance", C["sky"], "within-dataset RMSE and MAE\n95% predictive coverage"),
        ("Provenance", C["teal"], "Python 3.10.20: original run\nPython 3.12.3: audit\nseed 20260630 • NetCDF • DOI"),
    ]
    card_positions = [(54.5, 27.0), (76.5, 27.0), (54.5, 12.5), (76.5, 12.5)]
    for (x, y), (title, color, body) in zip(card_positions, blocks):
        rounded_box(ax, x, y, 19.0, 12.0, face=C["white"], edge=color, lw=0.65)
        rounded_box(ax, x + 0.7, y + 7.8, 17.6, 3.4, face=color, edge=color, lw=0.5)
        ax.text(x + 9.5, y + 9.5, title, fontsize=5.2, color=C["white"], ha="center", va="center", fontweight="bold")
        ax.text(x + 9.5, y + 4.5, body, fontsize=5.0, ha="center", va="center", linespacing=1.18)
    ax.text(74.5, 8.2, "Complete workflow comparison; individual component effects are not isolated", fontsize=5.0, ha="center", color=C["gray"])
    return fig


def _mini_response_panel(ax: Axes, x0: float, y0: float, w: float, h: float, *, mode: str, color: str, seed: int) -> None:
    rng = np.random.default_rng(seed)
    xx = np.linspace(0.0, 1.0, 24)
    truth = 0.08 + 0.86 * xx**3 / (0.19 + xx**3)
    obs = np.clip(truth + rng.normal(0, 0.075, len(xx)), 0, 1)
    ax.plot([x0, x0], [y0, y0 + h], color=C["ink"], lw=0.45)
    ax.plot([x0, x0 + w], [y0, y0], color=C["ink"], lw=0.45)
    ax.scatter(x0 + xx * w, y0 + obs * h, s=7, marker="o", facecolor=C["sky"], edgecolor=C["white"], linewidth=0.25, zorder=4)
    if mode == "linear":
        fit = 0.10 + 0.77 * xx
        ax.plot(x0 + xx * w, y0 + fit * h, color=color, lw=1.15)
    else:
        ax.plot(x0 + xx * w, y0 + truth * h, color=color, lw=1.3)
        ax.fill_between(x0 + xx * w, y0 + np.clip(truth - 0.05 * (1 + xx), 0, 1) * h, y0 + np.clip(truth + 0.05 * (1 + xx), 0, 1) * h, color=C["orange"], alpha=0.16)


def figure_7(spec: JournalSpec) -> Figure:
    fig, ax = make_canvas(spec)
    xs, col_w = [4.5, 36.0, 67.5], 28.0
    titles, colors, keys = ["OLS/ANOVA mass-based", "Hierarchical Bayesian mass-based", "Area-based Bayesian + PVS-aware"], [C["gray"], C["blue"], C["teal"]], ["ols", "mass", "area"]
    for idx, (x, title, color) in enumerate(zip(xs, titles, colors)):
        rounded_box(ax, x, 91.0, col_w, 5.6, face=color, edge=color, radius=0.8)
        ax.text(x + col_w / 2, 93.8, title, fontsize=5.5, color=C["white"], ha="center", va="center", fontweight="bold")
        panel_label(ax, x - 2.0, 96.5, chr(ord("a") + idx))
    for y, label in [(79.0, "Descriptor and\nmean structure"), (60.0, "Within-dataset\nerror"), (37.0, "Coverage and\nadmissibility"), (13.5, "Sampling burden")]:
        rounded_box(ax, 0.8, y - 5.0, 2.8, 10.0, face=C["gray_l"], edge=C["line"], lw=0.4, radius=0.6)
        ax.text(2.2, y, label, fontsize=5.05, color=C["gray"], rotation=90, ha="center", va="center", fontweight="bold")
    for x, color, mode in zip(xs, colors, ["linear", "linear", "hill"]):
        _mini_response_panel(ax, x + 3.5, 72.5, 21.0, 12.5, mode=mode, color=color, seed=int(x * 10))
        pill(ax, x + 8.0, 67.0, 12.0, "mass descriptor" if mode == "linear" else "surface-reactive area", face=color, hatch="//" if mode == "linear" and color == C["gray"] else None, size=5.0)
    for x, color, key in zip(xs, colors, keys):
        rounded_box(ax, x + 1.8, 49.0, 24.4, 16.5, face=C["white"], edge=C["line"], lw=0.45)
        draw_iqr_point(ax, x + 7.2, 60.5, RESULTS[key]["rmse"], scale_min=0.0, scale_max=0.18, width=14.0, color=color, marker="o", label="RMSE")
        draw_iqr_point(ax, x + 7.2, 55.6, RESULTS[key]["mae"], scale_min=0.0, scale_max=0.18, width=14.0, color=C["orange"], marker="s", label="MAE")
        ax.plot([x + 7.2, x + 21.2], [51.8, 51.8], color=C["line"], lw=0.35)
        for val in [0.0, 0.05, 0.10, 0.15]:
            tick_x = x + 7.2 + val / 0.18 * 14.0
            ax.plot([tick_x, tick_x], [51.4, 52.2], color=C["line"], lw=0.35)
            ax.text(tick_x, 50.5, f"{val:.2f}", fontsize=5.0, ha="center", color=C["gray"])
    metric_specs = [("coverage", "95% coverage", C["sky"], "o"), ("pvs_theta", r"PVS$_\theta$", C["purple"], "s"), ("pvs_pred", r"PVS$_{\mathrm{pred}}$", C["teal"], "^"), ("pvs_joint", r"PVS$_{\mathrm{joint}}$", C["orange"], "D")]
    for x, key in zip(xs, keys):
        rounded_box(ax, x + 1.8, 25.0, 24.4, 19.5, face=C["white"], edge=C["line"], lw=0.45)
        for idx, (metric_key, label, color, marker) in enumerate(metric_specs):
            yy = 40.5 - idx * 4.2
            value = RESULTS[key][metric_key]
            if value is None:
                draw_na(ax, x + 8.2, yy, 12.0, label=label)
            else:
                draw_iqr_point(ax, x + 8.2, yy, value, scale_min=0.0, scale_max=1.0, width=12.0, color=color, marker=marker, label=label, value_format=".3f")
        ax.plot([x + 8.2, x + 20.2], [25.9, 25.9], color=C["line"], lw=0.35)
        for val in [0.0, 0.5, 1.0]:
            tick_x = x + 8.2 + val * 12.0
            ax.plot([tick_x, tick_x], [25.5, 26.3], color=C["line"], lw=0.35)
            ax.text(tick_x, 24.7, f"{val:.1f}", fontsize=5.0, ha="center", color=C["gray"])
    rounded_box(ax, xs[0] + 1.8, 5.5, 24.4, 14.5, face=C["gray_l"], edge=C["line"], lw=0.45, hatch="...")
    ax.text(xs[0] + 14.0, 12.8, "MCMC diagnostics\nnot applicable", fontsize=5.5, color=C["gray"], ha="center", va="center", fontweight="bold")
    for x, key in zip(xs[1:], keys[1:]):
        divergent, total_div = int(RESULTS[key]["divergent_fits"]), int(RESULTS[key]["total_divergences"])
        rounded_box(ax, x + 1.8, 5.5, 24.4, 14.5, face=C["white"], edge=C["line"], lw=0.45)
        draw_dot_grid(ax, x + 5.0, 17.7, divergent, dx=1.17, dy=1.10, radius=0.25)
        ax.text(x + 14.0, 7.2, f"{divergent}/150 fits   •   {total_div} total divergences", fontsize=5.05, color=C["vermillion"], ha="center", fontweight="bold")
    rounded_box(ax, 20.0, 0.6, 60.0, 3.6, face=C["orange_l"], edge=C["orange"], lw=0.5)
    ax.text(50, 2.4, "Workflow-level comparison. Descriptor, mean structure, variance, hierarchy, priors, and diagnostics vary simultaneously.", fontsize=5.0, ha="center", va="center", color=C["orange"], fontweight="bold")
    return fig


def _bullet_item(ax: Axes, x: float, y: float, text: str, *, symbol: str, symbol_color: str, width: int, size: float = 5.0) -> None:
    ax.text(x, y, symbol, fontsize=6.5, color=symbol_color, ha="center", va="center", fontweight="bold")
    ax.text(x + 2.0, y, textwrap.fill(text, width=width, break_long_words=False), fontsize=size, color=C["ink"], ha="left", va="center", linespacing=1.08)


def figure_8(spec: JournalSpec) -> Figure:
    fig, ax = make_canvas(spec)
    panels = [(2.0, 54.0, 47.0, 42.0, "a", "Supported by the controlled benchmark", C["teal"], None), (51.0, 54.0, 47.0, 42.0, "b", "Conditional interpretation", C["orange"], None), (2.0, 5.0, 47.0, 42.0, "c", "Not established", C["vermillion"], None), (51.0, 5.0, 47.0, 42.0, "d", "Required next tests", C["purple"], None)]
    for x, y, w, h, label, heading, color, hatch in panels:
        rounded_box(ax, x, y, w, h, edge=color, lw=0.75, hatch=hatch)
        panel_label(ax, x + 1.2, y + h - 1.2, label)
        panel_heading(ax, x + 4.8, y + h - 1.5, heading, color=color, size=6.0)
    supported = ["Lower within-dataset error for the aligned area-based workflow", "Posterior predictive coverage close to the nominal interval level", "Higher predictive PVS under the tested domains", "Workflow-level contrasts retained in conservative subsets", "Domain sensitivity made quantitatively inspectable", "Very small practical changes under tested moderate prior perturbations"]
    for idx, item in enumerate(supported):
        _bullet_item(ax, 6.3, 86.0 - idx * 5.6, item, symbol="+", symbol_color=C["teal"], width=50)
    conditional = ["Absolute PVS depends on the declared admissible domain", "Parameter-space PVS covered a limited Hill-model condition set", "Several positivity conditions were already satisfied by prior support", "Baseline and asymptotic-response conditions added the main post-fitting restrictions", "Convergence summaries must be interpreted alongside divergences", "Results depend on the synthetic DGP and complete workflow specification"]
    for idx, item in enumerate(conditional):
        _bullet_item(ax, 55.3, 86.0 - idx * 5.6, item, symbol="!", symbol_color=C["orange"], width=52)
    unsupported = ["Isolated causal contribution of PVS", "Isolated contribution of descriptor choice", "Universal superiority of the area-based workflow", "Empirical validation across nanoscale systems", "Universal PVS thresholds, certification, or regulatory validity", "Exhaustive assessment of every mechanistic property", "Equivalence between reproducibility and scientific validation"]
    for idx, item in enumerate(unsupported):
        _bullet_item(ax, 6.3, 38.7 - idx * 4.7, item, symbol="X", symbol_color=C["vermillion"], width=50)
    steps = [("Ablation study", C["purple"], 17), ("Real-data reanalysis", C["blue"], 20), ("Original experiments", C["teal"], 23), ("Interlaboratory comparison", C["orange"], 27), ("External validation", C["sky"], 31), ("Domain-specific calibration", C["gray"], 35)]
    base_x, base_y = 56.0, 10.0
    for idx, (label, color, width) in enumerate(steps):
        yy = base_y + idx * 5.4
        rounded_box(ax, base_x, yy, width, 3.6, face=color, edge=color, lw=0.5, radius=0.7)
        ax.text(base_x + width / 2, yy + 1.8, label, fontsize=5.0, color=C["white"], ha="center", va="center", fontweight="bold")
        if idx < len(steps) - 1:
            arrow(ax, base_x + width + 0.8, yy + 1.8, base_x + steps[idx + 1][2] - 5.0, yy + 5.4 + 1.8, color=C["gray_m"], lw=0.55, mutation=6, rad=0.1)
    rounded_box(ax, 33.0, 48.3, 34.0, 4.8, face=C["blue"], edge=C["blue"], lw=0.65)
    ax.text(50, 50.7, "CONTROLLED METHODOLOGICAL DEMONSTRATION", fontsize=5.9, color=C["white"], ha="center", va="center", fontweight="bold")
    ax.text(74.5, 50.4, "not yet external empirical validation", fontsize=5.0, color=C["purple"], ha="center", va="center", fontweight="bold")
    arrow(ax, 67.7, 50.7, 70.3, 50.7, color=C["purple"], lw=0.7, mutation=7)
    rounded_box(ax, 54.0, 0.6, 42.0, 3.6, face=C["gray_l"], edge=C["line"], lw=0.45)
    ax.text(75.0, 2.4, "Ablation factors: descriptor • nonlinear mean • heteroscedasticity • hierarchy • prior support • post-fitting admissibility", fontsize=5.0, color=C["gray"], ha="center", va="center")
    return fig


FIGURE_BUILDERS: dict[int, Callable[[JournalSpec], Figure]] = {2: figure_2, 3: figure_3, 4: figure_4, 5: figure_5, 6: figure_6, 7: figure_7, 8: figure_8}


def audit_figure(fig: Figure, spec: JournalSpec) -> list[str]:
    warnings: list[str] = []
    fig.canvas.draw()
    renderer, figure_bbox = fig.canvas.get_renderer(), fig.bbox
    for text in fig.findobj(match=Text):
        if not text.get_visible() or not text.get_text().strip():
            continue
        size = float(text.get_fontsize())
        if size < spec.min_font_pt - 1e-6:
            warnings.append(f"Text below {spec.min_font_pt:.1f} pt: {size:.2f} pt, content={text.get_text()[:60]!r}")
        try:
            bbox = text.get_window_extent(renderer=renderer)
        except Exception:
            continue
        tolerance = 2.0
        if bbox.x0 < figure_bbox.x0 - tolerance or bbox.y0 < figure_bbox.y0 - tolerance or bbox.x1 > figure_bbox.x1 + tolerance or bbox.y1 > figure_bbox.y1 + tolerance:
            warnings.append(f"Potentially clipped text: {text.get_text()[:60]!r}")
    for line in fig.findobj(match=Line2D):
        if line.get_visible():
            lw = float(line.get_linewidth())
            if 0 < lw < spec.min_line_pt - 1e-6:
                warnings.append(f"Line below {spec.min_line_pt:.2f} pt: {lw:.3f} pt")
    return warnings


def save_grayscale_preview(fig: Figure, path: Path, dpi: int) -> None:
    import io
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=dpi, bbox_inches="tight", pad_inches=0.02)
    buffer.seek(0)
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("Pillow is required for --grayscale-preview.") from exc
    Image.open(buffer).convert("L").save(path)


def save_figure(fig: Figure, outdir: Path, stem: str, *, formats: Sequence[str], dpi: int, grayscale_preview: bool) -> list[Path]:
    outdir.mkdir(parents=True, exist_ok=True)
    created: list[Path] = []
    metadata = {"Creator": "Nanostatistics publication figure generator", "Title": stem}
    for fmt in formats:
        fmt = fmt.lower()
        target = outdir / f"{stem}.{fmt}"
        if fmt in {"pdf", "svg"}:
            fig.savefig(target, format=fmt, bbox_inches="tight", pad_inches=0.02, metadata=metadata)
        elif fmt == "png":
            fig.savefig(target, format="png", dpi=dpi, bbox_inches="tight", pad_inches=0.02)
        elif fmt in {"tif", "tiff"}:
            try:
                fig.savefig(target, format="tiff", dpi=dpi, bbox_inches="tight", pad_inches=0.02, pil_kwargs={"compression": "tiff_lzw"})
            except TypeError:
                fig.savefig(target, format="tiff", dpi=dpi, bbox_inches="tight", pad_inches=0.02)
        else:
            raise ValueError(f"Unsupported format: {fmt}")
        created.append(target)
    if grayscale_preview:
        gray_path = outdir / f"{stem}_grayscale_preview.png"
        save_grayscale_preview(fig, gray_path, dpi=dpi)
        created.append(gray_path)
    return created


def write_caption_file(outdir: Path, figures: Iterable[int]) -> Path:
    path = outdir / "figure_captions_and_accessibility_notes.md"
    lines = ["# Nanostatistics Figures 2-8", "", "Figure titles and long explanations are kept outside the artwork.", ""]
    for number in figures:
        item = FIGURE_TEXT[number]
        lines.extend([f"## Figure {number}. {item.title}", "", item.caption, "", f"**Accessibility note:** {item.accessibility_note}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def write_quality_report(outdir: Path, report: dict[str, object]) -> Path:
    path = outdir / "figure_quality_report.json"
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate publication-oriented Nanostatistics Figures 2-8.")
    parser.add_argument("--outdir", type=Path, default=Path("nanostatistics_Journal_figures"))
    parser.add_argument("--figures", nargs="+", type=int, default=list(range(2, 9)), choices=list(range(2, 9)))
    parser.add_argument("--formats", nargs="+", default=["pdf", "svg", "png"], choices=["pdf", "svg", "png", "tif", "tiff"])
    parser.add_argument("--dpi", type=int, default=300, help="Raster preview resolution. Vector outputs are resolution-independent.")
    parser.add_argument("--width-mm", type=float, default=183.0)
    parser.add_argument("--height-mm", type=float, default=118.0)
    parser.add_argument("--grayscale-preview", action="store_true")
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outdir = args.outdir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    font_name = select_font()
    configure_matplotlib(font_name, args.dpi)
    spec = JournalSpec(width_mm=args.width_mm, height_mm=args.height_mm, min_font_pt=5.0, body_font_pt=5.7, heading_font_pt=6.5, panel_font_pt=7.0, preview_dpi=args.dpi, min_line_pt=0.25)
    selected = sorted(set(args.figures))
    quality_report: dict[str, object] = {"font": font_name, "width_mm": spec.width_mm, "height_mm": spec.height_mm, "raster_dpi": args.dpi, "figures": {}}
    created_files: list[Path] = []
    for number in selected:
        fig = FIGURE_BUILDERS[number](spec)
        warnings = audit_figure(fig, spec)
        safe_title = re.sub(r"[^a-z0-9]+", "_", FIGURE_TEXT[number].title.lower()).strip("_")
        stem = f"Figure_{number}_{safe_title}"
        created = save_figure(fig, outdir, stem, formats=args.formats, dpi=args.dpi, grayscale_preview=args.grayscale_preview)
        created_files.extend(created)
        quality_report["figures"][str(number)] = {"warnings": warnings, "files": [str(path.name) for path in created]}
        plt.close(fig)
    caption_path = write_caption_file(outdir, selected)
    report_path = write_quality_report(outdir, quality_report)
    created_files.extend([caption_path, report_path])
    warning_count = sum(len(quality_report["figures"][str(number)]["warnings"]) for number in selected)
    print(f"Font selected: {font_name}")
    print(f"Final artwork size: {spec.width_mm:.1f} × {spec.height_mm:.1f} mm")
    print(f"Output directory: {outdir}")
    print(f"Generated {len(created_files)} files.")
    print(f"Automated audit warnings: {warning_count}")
    for path in created_files:
        print(path.name)
    if warning_count and args.strict:
        raise SystemExit("Strict mode failed. Inspect figure_quality_report.json.")


if __name__ == "__main__":
    main()
