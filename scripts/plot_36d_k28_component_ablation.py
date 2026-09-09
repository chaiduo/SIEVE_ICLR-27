#!/usr/bin/env python3

"""Plot the Fit-only 36D/K=28 layer-pair and metric ablations."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
ABLATION_ROOT = (
    REPOSITORY_ROOT
    / "experiments"
    / "ablation_36d_k28_fit_only_strict_finite"
)
COHORTS = (
    ("full", "Full", "#1F77B4"),
    ("strict_finite", "Finite", "#D1495B"),
)
PANELS = (
    (
        "Layer-pair ablation",
        (
            ("full_36d", "Full 36D"),
            ("without_pair_6_7", "w/o (6,7)"),
            ("without_pair_24_25", "w/o (24,25)"),
            ("without_pair_26_27", "w/o (26,27)"),
        ),
    ),
    (
        "Metric ablation",
        (
            ("full_36d", "Full 36D"),
            ("without_cos_sim", "w/o CosSim"),
            ("without_mean_diff", "w/o MeanDiff"),
            ("without_std_diff", "w/o StdDiff"),
            ("without_l2_distance", "w/o L2"),
        ),
    ),
)


def panel_values(
    frame: pd.DataFrame,
    configurations: tuple[tuple[str, str], ...],
    cohort: str,
) -> np.ndarray:
    indexed = (
        frame.loc[frame["cohort"].eq(cohort)]
        .set_index("configuration")["f1"]
        .reindex([name for name, _ in configurations])
    )
    if indexed.isna().any():
        missing = indexed.index[indexed.isna()].tolist()
        raise ValueError(f"Missing {cohort} values for {missing}")
    return indexed.to_numpy() * 100.0


def write_paper_table(frame: pd.DataFrame, path: Path) -> None:
    rows = []
    for panel_name, configurations in PANELS:
        for configuration, label in configurations:
            values = (
                frame.loc[frame["configuration"].eq(configuration)]
                .set_index("cohort")
            )
            full = values.loc["full"]
            finite = values.loc["strict_finite"]
            rows.append(
                {
                    "group": panel_name,
                    "ablation": label,
                    "full_f1_percent": 100.0 * float(full["f1"]),
                    "strict_finite_f1_percent": 100.0 * float(finite["f1"]),
                    "full_delta_f1_pp": float(full["delta_f1_pp"]),
                    "strict_finite_delta_f1_pp": float(finite["delta_f1_pp"]),
                }
            )
    pd.DataFrame(rows).to_csv(path, index=False)


def main() -> int:
    source = ABLATION_ROOT / "macro_metrics.csv"
    frame = pd.read_csv(source)
    expected = len(COHORTS) * 8
    if len(frame) != expected:
        raise ValueError(f"Expected {expected} macro rows, found {len(frame)}")

    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "font.size": 7.5,
            "axes.labelsize": 7.5,
            "axes.titlesize": 8.0,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "legend.fontsize": 6.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    figure, axes = plt.subplots(1, 2, figsize=(7.0, 2.35), sharey=True)
    bar_width = 0.34
    f1_values = frame["f1"].to_numpy() * 100.0
    lower_bound = max(0.0, 5.0 * np.floor((f1_values.min() - 2.0) / 5.0))
    upper_bound = min(100.0, 5.0 * np.ceil((f1_values.max() + 3.0) / 5.0))

    for axis, (title, configurations) in zip(axes, PANELS, strict=True):
        positions = np.arange(len(configurations))
        for index, (cohort, label, color) in enumerate(COHORTS):
            values = panel_values(frame, configurations, cohort)
            offset = (index - 0.5) * bar_width
            bars = axis.bar(
                positions + offset,
                values,
                width=bar_width,
                color=color,
                edgecolor="white",
                linewidth=0.45,
                label=label,
                zorder=3,
            )
            for bar, value in zip(bars, values, strict=True):
                axis.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    value + 0.18,
                    f"{value:.1f}",
                    ha="center",
                    va="bottom",
                    fontsize=5.4,
                    color="#1F2937",
                )
        axis.set_title(title, pad=4)
        axis.set_xticks(positions, [label for _, label in configurations])
        axis.tick_params(axis="x", length=0, pad=2)
        axis.grid(axis="y", color="#D1D5DB", linewidth=0.55, zorder=0)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.spines["left"].set_linewidth(0.65)
        axis.spines["bottom"].set_linewidth(0.65)
        axis.set_axisbelow(True)

    axes[0].set_ylabel("Significant-SDC F1 (%)", labelpad=3)
    axes[0].set_ylim(lower_bound, upper_bound)
    axes[0].set_yticks(np.arange(lower_bound, upper_bound + 0.1, 5.0))
    axes[1].legend(
        loc="upper right",
        frameon=False,
        handlelength=1.2,
        borderpad=0.1,
        labelspacing=0.25,
    )
    figure.tight_layout(pad=0.4, w_pad=0.55)

    output_prefix = ABLATION_ROOT / "component_ablation_36d_k28"
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
    write_paper_table(frame, output_prefix.with_name("component_ablation_paper_table.csv"))
    print(f"Wrote {output_prefix.with_suffix('.png')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
