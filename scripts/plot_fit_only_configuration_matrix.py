#!/usr/bin/env python3

"""Plot Full and Finite Fit-only configuration-selection matrices."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1 import make_axes_locatable


PROFILES = ("36D", "48D", "60D", "72D")
K_VALUES = (1, 2, 4, 8, 12, 16, 20, 24, 28, 32)
COHORTS = (
    ("full", "Full"),
    ("strict_finite", "Finite"),
)
BEST_CONFIGURATION = ("36D", 28)


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    default_root = root / "experiments/fit_only_configuration_selection"
    parser = argparse.ArgumentParser(
        description="Plot Fit-only 4x10 profile/window selection heatmaps."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=default_root / "macro_metrics.csv",
    )
    parser.add_argument(
        "--strict-finite-input",
        type=Path,
        default=default_root / "strict_finite_macro_metrics.csv",
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=default_root / "configuration_f1_matrix",
    )
    return parser.parse_args()


def matrix(frame: pd.DataFrame, cohort: str) -> pd.DataFrame:
    selected = frame.loc[frame["cohort"].eq(cohort)].copy()
    expected = len(PROFILES) * len(K_VALUES)
    if len(selected) != expected:
        raise ValueError(
            f"{cohort} expects {expected} profile/window cells, found {len(selected)}"
        )
    result = selected.pivot(index="profile", columns="k", values="f1")
    return result.loc[list(PROFILES), list(K_VALUES)]


def text_color(value: float, minimum: float, maximum: float) -> str:
    fraction = np.clip((value - minimum) / (maximum - minimum), 0.0, 1.0)
    red, green, blue, _ = plt.get_cmap("YlGnBu")(fraction)
    luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
    return "#171717" if luminance >= 0.48 else "white"


def draw_panel(
    axis: plt.Axes,
    values: pd.DataFrame,
    *,
    title: str,
    vmin: float,
    vmax: float,
) -> matplotlib.image.AxesImage:
    image = axis.imshow(
        values.to_numpy() * 100.0,
        cmap="YlGnBu",
        vmin=vmin,
        vmax=vmax,
        aspect="auto",
    )
    axis.set_title(title, fontsize=8, pad=5)
    axis.set_xticks(range(len(K_VALUES)), [str(k) for k in K_VALUES])
    axis.set_yticks(range(len(PROFILES)), list(PROFILES))
    axis.tick_params(axis="both", length=0, pad=2)
    axis.set_xlabel(r"Prefix window $K$", labelpad=2)

    for row, profile in enumerate(PROFILES):
        for column, k in enumerate(K_VALUES):
            value = float(values.loc[profile, k]) * 100.0
            axis.text(
                column,
                row,
                f"{value:.1f}",
                ha="center",
                va="center",
                fontsize=5.7,
                color=text_color(value, vmin, vmax),
            )

    for spine in axis.spines.values():
        spine.set_linewidth(0.55)
        spine.set_color("#6B7280")
    axis.set_xticks(np.arange(-0.5, len(K_VALUES), 1), minor=True)
    axis.set_yticks(np.arange(-0.5, len(PROFILES), 1), minor=True)
    axis.grid(which="minor", color="white", linewidth=0.65)
    axis.tick_params(which="minor", bottom=False, left=False)

    profile, k = BEST_CONFIGURATION
    row = PROFILES.index(profile)
    column = K_VALUES.index(k)
    axis.add_patch(
        Rectangle(
            (column - 0.5, row - 0.5),
            1.0,
            1.0,
            fill=False,
            edgecolor="#D1495B",
            linewidth=1.1,
            zorder=4,
        )
    )
    return image


def main() -> int:
    args = parse_args()
    full_source = args.input.resolve()
    strict_source = args.strict_finite_input.resolve()
    output_prefix = args.output_prefix.resolve()
    full_frame = pd.read_csv(full_source)
    strict_frame = pd.read_csv(strict_source)
    matrices = {
        "full": matrix(full_frame, "full"),
        "strict_finite": matrix(strict_frame, "strict_finite"),
    }
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    for cohort, values in matrices.items():
        values.to_csv(
            output_prefix.with_name(
                f"{output_prefix.name}_{cohort}_f1.csv"
            )
        )
    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "font.size": 7.5,
            "axes.labelsize": 7.5,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 7.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    figure = plt.figure(figsize=(7.0, 2.35))
    grid = figure.add_gridspec(
        1,
        2,
        left=0.07,
        right=0.96,
        bottom=0.22,
        top=0.88,
        wspace=0.10,
    )
    axes = (
        figure.add_subplot(grid[0, 0]),
        figure.add_subplot(grid[0, 1]),
    )
    for index, ((cohort, title), axis) in enumerate(
        zip(COHORTS, axes, strict=True)
    ):
        values = matrices[cohort]
        plot_values = values.to_numpy().ravel() * 100.0
        vmin = float(np.floor(plot_values.min()))
        vmax = float(np.ceil(plot_values.max()))
        image = draw_panel(
            axis,
            values,
            title=f"{title} macro F1 (%)",
            vmin=vmin,
            vmax=vmax,
        )
        if index == 0:
            axis.set_ylabel("Feature profile", labelpad=2)
        else:
            axis.tick_params(axis="y", labelleft=False)
        colorbar_axis = make_axes_locatable(axis).append_axes(
            "right",
            size="5%",
            pad=0.09,
        )
        colorbar = figure.colorbar(image, cax=colorbar_axis)
        colorbar.ax.tick_params(labelsize=6.2, length=2)
    figure.text(
        0.14,
        0.045,
        "Red outline: frozen 36D/K=28 configuration (selected by Full only).",
        fontsize=6.0,
        color="#4B5563",
    )
    figure.savefig(
        output_prefix.with_suffix(".png"),
        dpi=400,
        bbox_inches="tight",
        pad_inches=0.02,
    )
    figure.savefig(
        output_prefix.with_suffix(".pdf"),
        bbox_inches="tight",
        pad_inches=0.02,
    )
    plt.close(figure)
    print(f"wrote {output_prefix.with_suffix('.png')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
