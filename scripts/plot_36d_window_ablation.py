#!/usr/bin/env python3

"""Plot Full and Finite 36D prefix-window diagnostics."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


K_VALUES = (1, 2, 4, 8, 12, 16, 20, 24, 28, 32)
FULL_COLOR = "#1F77B4"
STRICT_COLOR = "#D1495B"


def window_values(
    frame: pd.DataFrame,
    *,
    cohort: str,
) -> np.ndarray:
    selected = frame.loc[
        frame["profile"].eq("36D") & frame["cohort"].eq(cohort)
    ].copy()
    if len(selected) != len(K_VALUES):
        raise ValueError(
            f"{cohort} expected {len(K_VALUES)} 36D rows, "
            f"found {len(selected)}"
        )
    return (
        selected.set_index("k")
        .reindex(K_VALUES)["f1"]
        .to_numpy()
        * 100.0
    )


def draw_panel(
    axis: plt.Axes,
    *,
    values: np.ndarray,
    title: str,
    color: str,
    y_limits: tuple[float, float],
    y_ticks: tuple[float, ...],
    selected_position: int,
    positions: list[int],
) -> None:
    axis.plot(
        positions,
        values,
        color=color,
        linewidth=1.35,
        marker="o",
        markersize=3.6,
        markeredgewidth=0.55,
        markeredgecolor="white",
        zorder=3,
    )
    axis.axvline(
        selected_position,
        color="#4B5563",
        linestyle="--",
        linewidth=0.8,
        zorder=1,
    )
    for position, value in zip(positions, values, strict=True):
        upward = value < y_limits[1] - 0.7
        axis.annotate(
            f"{value:.1f}",
            (position, value),
            xytext=(0, 3 if upward else -7),
            textcoords="offset points",
            ha="center",
            va="bottom" if upward else "top",
            fontsize=5.3,
            color="#1F2937",
            zorder=4,
        )
    axis.text(
        selected_position + 0.10,
        y_limits[1] - 0.25,
        r"$K=28$",
        fontsize=5.8,
        color="#374151",
    )
    axis.set_ylim(*y_limits)
    axis.set_yticks(y_ticks)
    axis.set_title(title, fontsize=8.0, pad=3.0)
    axis.set_xticks(positions, [str(value) for value in K_VALUES])
    axis.set_xlim(-0.35, len(K_VALUES) - 0.65)
    axis.set_xlabel(r"Prefix window $K$", labelpad=2)
    axis.set_ylabel("Macro F1 (%)", labelpad=3)
    axis.grid(axis="y", color="#D1D5DB", linewidth=0.55, zorder=0)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_linewidth(0.65)
    axis.spines["bottom"].set_linewidth(0.65)
    axis.tick_params(length=2.5, width=0.6, pad=2)


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    experiment_root = root / "experiments" / "fit_only_configuration_selection"
    full_frame = pd.read_csv(experiment_root / "macro_metrics.csv")
    strict_frame = pd.read_csv(
        experiment_root / "strict_finite_macro_metrics.csv"
    )

    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "font.size": 7.5,
            "axes.labelsize": 7.5,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "legend.fontsize": 6.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    figure, axes = plt.subplots(1, 2, figsize=(7.0, 2.35))
    positions = list(range(len(K_VALUES)))
    selected_position = K_VALUES.index(28)
    full_values = window_values(full_frame, cohort="full")
    strict_values = window_values(strict_frame, cohort="strict_finite")

    draw_panel(
        axes[0],
        values=full_values,
        title="Full",
        color=FULL_COLOR,
        y_limits=(90.0, 94.1),
        y_ticks=(90.0, 91.0, 92.0, 93.0, 94.0),
        selected_position=selected_position,
        positions=positions,
    )
    draw_panel(
        axes[1],
        values=strict_values,
        title="Finite",
        color=STRICT_COLOR,
        y_limits=(64.5, 72.2),
        y_ticks=(65.0, 67.0, 69.0, 71.0),
        selected_position=selected_position,
        positions=positions,
    )
    figure.subplots_adjust(
        left=0.08,
        right=0.99,
        bottom=0.19,
        top=0.87,
        wspace=0.22,
    )

    output_prefix = experiment_root / "window_ablation_36d_f1"
    figure.savefig(
        output_prefix.with_suffix(".png"),
        dpi=400,
        bbox_inches="tight",
        pad_inches=0.015,
    )
    figure.savefig(
        output_prefix.with_suffix(".pdf"),
        bbox_inches="tight",
        pad_inches=0.015,
    )
    plt.close(figure)
    print(f"Wrote {output_prefix.with_suffix('.png')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
