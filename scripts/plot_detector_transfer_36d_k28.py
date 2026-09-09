#!/usr/bin/env python3
"""Plot the Full and Finite 36D/K=28 detector-transfer F1 matrices."""

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


DISPLAY_NAMES = {
    "qwen25_vl_earthvqa": "Q-E",
    "qwen25_vl_lingoqa": "Q-L",
    "qwen25_vl_vqav2": "Q-V",
    "llava15_earthvqa": "L-E",
    "llava15_lingoqa": "L-L",
    "llava15_vqav2": "L-V",
    "internvl3_earthvqa": "I-E",
    "internvl3_lingoqa": "I-L",
    "internvl3_vqav2": "I-V",
}
JOB_ORDER = (
    "qwen25_vl_earthvqa",
    "qwen25_vl_lingoqa",
    "qwen25_vl_vqav2",
    "llava15_earthvqa",
    "llava15_lingoqa",
    "llava15_vqav2",
    "internvl3_earthvqa",
    "internvl3_lingoqa",
    "internvl3_vqav2",
)
COHORTS = (
    ("full", "Full"),
    ("strict_finite", "Finite"),
)


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    default_root = root / "experiments/transfer_36d_k28"
    parser = argparse.ArgumentParser(
        description="Plot Full and Finite detector-transfer F1 matrices."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=default_root / "transfer_metrics.csv",
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=default_root / "transfer_f1_matrix",
    )
    return parser.parse_args()


def matrix(
    frame: pd.DataFrame,
    *,
    cohort: str,
    sources: list[str],
    targets: list[str],
) -> pd.DataFrame:
    selected = frame.loc[frame["cohort"].eq(cohort)]
    expected = len(sources) * len(targets)
    if len(selected) != expected:
        raise ValueError(f"{cohort} expects {expected} cells, found {len(selected)}")
    output = selected.pivot(
        index="source_job",
        columns="target_job",
        values="f1",
    )
    return output.loc[sources, targets]


def color_bounds(values: pd.DataFrame) -> tuple[float, float]:
    minimum = float(values.to_numpy().min() * 100.0)
    maximum = float(values.to_numpy().max() * 100.0)
    return (
        max(0.0, 5.0 * np.floor((minimum - 1.0) / 5.0)),
        min(100.0, 5.0 * np.ceil((maximum + 1.0) / 5.0)),
    )


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
    source_labels: list[str],
    target_labels: list[str],
) -> matplotlib.image.AxesImage:
    image = axis.imshow(
        values.to_numpy() * 100.0,
        cmap="YlGnBu",
        vmin=vmin,
        vmax=vmax,
        aspect="equal",
    )
    axis.set_title(title, fontsize=8.0, pad=5)
    axis.set_xticks(range(len(target_labels)), target_labels, rotation=42, ha="right")
    axis.set_yticks(range(len(source_labels)), source_labels)
    axis.tick_params(axis="both", length=0, pad=2)
    axis.set_xlabel("Target task", labelpad=2, fontsize=7.5)

    for row in range(len(source_labels)):
        for column in range(len(target_labels)):
            value = float(values.iloc[row, column]) * 100.0
            axis.text(
                column,
                row,
                f"{value:.1f}",
                ha="center",
                va="center",
                fontsize=5.7,
                color=text_color(value, vmin, vmax),
            )
            if values.index[row] == values.columns[column]:
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

    for spine in axis.spines.values():
        spine.set_linewidth(0.55)
        spine.set_color("#6B7280")
    axis.set_xticks(np.arange(-0.5, len(target_labels), 1), minor=True)
    axis.set_yticks(np.arange(-0.5, len(source_labels), 1), minor=True)
    axis.grid(which="minor", color="white", linewidth=0.65)
    axis.tick_params(which="minor", bottom=False, left=False)
    return image


def main() -> int:
    args = parse_args()
    input_path = args.input.resolve()
    output_prefix = args.output_prefix.resolve()
    frame = pd.read_csv(input_path)
    present_sources = set(frame["source_job"])
    present_targets = set(frame["target_job"])
    sources = [job for job in JOB_ORDER if job in present_sources]
    targets = [job for job in JOB_ORDER if job in present_targets]
    if len(sources) != len(present_sources) or len(targets) != len(present_targets):
        raise ValueError("Input contains an unknown source or target task")
    source_labels = [DISPLAY_NAMES[item] for item in sources]
    target_labels = [DISPLAY_NAMES[item] for item in targets]
    matrices = {
        cohort: matrix(
            frame,
            cohort=cohort,
            sources=sources,
            targets=targets,
        )
        for cohort, _ in COHORTS
    }
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    for cohort, values in matrices.items():
        values.to_csv(
            output_prefix.with_name(f"{output_prefix.name}_{cohort}_f1.csv")
        )

    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "font.size": 7.5,
            "axes.labelsize": 7.5,
            "xtick.labelsize": 6.2,
            "ytick.labelsize": 6.2,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    figure = plt.figure(figsize=(6.0, 3.65))
    grid = figure.add_gridspec(
        1,
        2,
        left=0.07,
        right=0.96,
        bottom=0.23,
        top=0.88,
        wspace=0.14,
    )
    axes = (
        figure.add_subplot(grid[0, 0]),
        figure.add_subplot(grid[0, 1]),
    )
    for (cohort, title), axis in zip(
        COHORTS,
        axes,
        strict=True,
    ):
        values = matrices[cohort]
        vmin, vmax = color_bounds(values)
        image = draw_panel(
            axis,
            values,
            title=f"{title} F1 (%)",
            vmin=vmin,
            vmax=vmax,
            source_labels=source_labels,
            target_labels=target_labels,
        )
        if title == "Full":
            axis.set_ylabel("Source task", labelpad=2, fontsize=7.5)
        else:
            axis.tick_params(axis="y", labelleft=False)
        colorbar_axis = make_axes_locatable(axis).append_axes(
            "right",
            size="5%",
            pad=0.09,
        )
        colorbar = figure.colorbar(image, cax=colorbar_axis)
        colorbar.ax.tick_params(labelsize=6.0, length=2, pad=1)
    figure.text(
        0.07,
        0.055,
        "Q/L/I: Qwen2.5-VL/LLaVA-1.5/InternVL3; "
        "E/L/V: EarthVQA/LingoQA/VQAv2. Red outlines: in-domain diagonal.",
        fontsize=5.8,
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
