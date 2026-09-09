#!/usr/bin/env python3
"""Plot Significant-SDC share among finite-valued labeled SDC executions."""

from __future__ import annotations

import argparse
import csv
import json
import math
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    import ujson
except ImportError:
    ujson = None


JOBS = (
    ("qwen25_vl_earthvqa", "Qwen2.5-VL-7B", "EarthVQA"),
    ("qwen25_vl_lingoqa", "Qwen2.5-VL-7B", "LingoQA"),
    ("qwen25_vl_vqav2", "Qwen2.5-VL-7B", "VQAv2"),
    ("internvl3_earthvqa", "InternVL3-8B", "EarthVQA"),
    ("internvl3_lingoqa", "InternVL3-8B", "LingoQA"),
    ("internvl3_vqav2", "InternVL3-8B", "VQAv2"),
    ("llava15_earthvqa", "LLaVA-1.5-7B", "EarthVQA"),
    ("llava15_lingoqa", "LLaVA-1.5-7B", "LingoQA"),
    ("llava15_vqav2", "LLaVA-1.5-7B", "VQAv2"),
)
DATASETS = ("EarthVQA", "LingoQA", "VQAv2")
COLORS = ("#4E79A7", "#59A96A", "#C44E52")


@dataclass(frozen=True)
class TaskResult:
    job: str
    model: str
    dataset: str
    injected_executions: int
    finite_value_sdc_labeled: int
    non_significant_sdc: int
    significant_sdc: int
    non_finite_value_sdc: int
    invalid_fault_sdc: int
    unlabeled_finite_value_sdc: int

    @property
    def significant_share_percent(self) -> float:
        if not self.finite_value_sdc_labeled:
            raise ValueError(f"{self.job} has no finite-valued labeled SDC")
        return 100.0 * self.significant_sdc / self.finite_value_sdc_labeled


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Plot Significant-SDC share among all labeled SDC."
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=root / "experiments/telemetry_50",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "experiments/figure2_significant_sdc_among_sdc",
    )
    parser.add_argument(
        "--reuse-csv",
        action="store_true",
        help="Regenerate the figure from the existing output CSV.",
    )
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    if args.workers <= 0:
        parser.error("--workers must be positive")
    return args


def _json_loader() -> Callable[[str | bytes], Any]:
    return ujson.loads if ujson is not None else json.loads


def process_task(arguments: tuple[Path, str, str, str]) -> TaskResult:
    path, job, model, dataset = arguments
    loads = _json_loader()
    injected_executions = 0
    non_significant_sdc = 0
    significant_sdc = 0
    non_finite_value_sdc = 0
    invalid_fault_sdc = 0
    unlabeled_finite_value_sdc = 0

    with path.open("rb") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                record = loads(line)
            except Exception as error:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from error
            if not bool(record.get("injected")):
                continue
            injected_executions += 1
            answers_differ = (
                str(record.get("pred_answer", ""))
                != str(record.get("clean_answer", ""))
            )
            if not answers_differ:
                continue
            fault = record.get("fault")
            try:
                before = float(fault["before"])
                after = float(fault["after"])
            except (KeyError, TypeError, ValueError, OverflowError):
                invalid_fault_sdc += 1
                continue
            if not (math.isfinite(before) and math.isfinite(after)):
                non_finite_value_sdc += 1
                continue
            try:
                significance = int(record["significance"])
            except (KeyError, TypeError, ValueError, OverflowError):
                unlabeled_finite_value_sdc += 1
                continue
            if significance not in (0, 1, 2):
                unlabeled_finite_value_sdc += 1
            elif significance == 2:
                significant_sdc += 1
            else:
                non_significant_sdc += 1

    return TaskResult(
        job=job,
        model=model,
        dataset=dataset,
        injected_executions=injected_executions,
        finite_value_sdc_labeled=non_significant_sdc + significant_sdc,
        non_significant_sdc=non_significant_sdc,
        significant_sdc=significant_sdc,
        non_finite_value_sdc=non_finite_value_sdc,
        invalid_fault_sdc=invalid_fault_sdc,
        unlabeled_finite_value_sdc=unlabeled_finite_value_sdc,
    )


def write_csv(path: Path, results: list[TaskResult]) -> None:
    fields = (
        "job",
        "model",
        "dataset",
        "injected_executions",
        "finite_value_sdc_labeled",
        "non_significant_sdc",
        "significant_sdc",
        "non_finite_value_sdc",
        "invalid_fault_sdc",
        "unlabeled_finite_value_sdc",
        "significant_share_among_finite_value_sdc_percent",
    )
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for result in results:
            row = asdict(result)
            row["significant_share_among_finite_value_sdc_percent"] = (
                result.significant_share_percent
            )
            writer.writerow(row)


def read_csv(path: Path) -> list[TaskResult]:
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != len(JOBS):
        raise ValueError(f"Expected {len(JOBS)} task rows, found {len(rows)}")
    return [
        TaskResult(
            job=row["job"],
            model=row["model"],
            dataset=row["dataset"],
            injected_executions=int(row["injected_executions"]),
            finite_value_sdc_labeled=int(row["finite_value_sdc_labeled"]),
            non_significant_sdc=int(row["non_significant_sdc"]),
            significant_sdc=int(row["significant_sdc"]),
            non_finite_value_sdc=int(row["non_finite_value_sdc"]),
            invalid_fault_sdc=int(row["invalid_fault_sdc"]),
            unlabeled_finite_value_sdc=int(
                row["unlabeled_finite_value_sdc"]
            ),
        )
        for row in rows
    ]


def plot(path_prefix: Path, results: list[TaskResult]) -> None:
    by_model = {
        model: {result.dataset: result for result in results if result.model == model}
        for model in ("Qwen2.5-VL-7B", "InternVL3-8B", "LLaVA-1.5-7B")
    }
    models = tuple(by_model)
    positions = np.arange(len(models), dtype=np.float64)
    width = 0.22
    offsets = (-width, 0.0, width)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "font.size": 8,
            "axes.labelsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    figure, axis = plt.subplots(figsize=(3.25, 2.25))
    label_padding = (1.8, 4.8, 1.8)
    for index, (dataset, color, offset) in enumerate(
        zip(DATASETS, COLORS, offsets, strict=True)
    ):
        values = [
            by_model[model][dataset].significant_share_percent
            for model in models
        ]
        bars = axis.bar(
            positions + offset,
            values,
            width=width,
            color=color,
            edgecolor="#1A1A1A",
            linewidth=0.55,
            label=dataset,
        )
        for bar, value in zip(bars, values, strict=True):
            axis.text(
                bar.get_x() + bar.get_width() / 2.0,
                value + label_padding[index],
                f"{value:.0f}%",
                ha="center",
                va="bottom",
                fontsize=5.5,
            )

    axis.set_ylim(0, 108)
    axis.set_yticks(np.arange(0, 101, 20))
    axis.set_yticklabels([f"{value}%" for value in range(0, 101, 20)])
    axis.set_ylabel("Significant SDC (%)")
    axis.set_xticks(positions, models)
    axis.grid(axis="y", color="#D1D5DB", linewidth=0.5)
    axis.set_axisbelow(True)
    axis.legend(
        loc="upper left",
        frameon=True,
        framealpha=1.0,
        edgecolor="#1A1A1A",
        fontsize=6.7,
        handlelength=1.2,
        borderpad=0.35,
        labelspacing=0.25,
    )
    for spine in axis.spines.values():
        spine.set_linewidth(0.75)
    figure.tight_layout(pad=0.25)
    figure.savefig(
        path_prefix.with_suffix(".png"),
        dpi=400,
        bbox_inches="tight",
        pad_inches=0.02,
    )
    figure.savefig(
        path_prefix.with_suffix(".pdf"),
        bbox_inches="tight",
        pad_inches=0.02,
    )
    plt.close(figure)


def main() -> int:
    args = parse_args()
    input_root = args.input_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "significant_sdc_among_finite_value_sdc.csv"
    prefix = output_dir / "significant_sdc_among_finite_value_sdc"
    if args.reuse_csv:
        results = read_csv(csv_path)
        plot(prefix, results)
        print(f"[figure2] regenerated {prefix.with_suffix('.png')}", flush=True)
        return 0

    task_arguments = [
        (input_root / job / "labels.jsonl", job, model, dataset)
        for job, model, dataset in JOBS
    ]
    missing = [str(path) for path, *_ in task_arguments if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing label files: {missing}")

    results: list[TaskResult] = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(process_task, argument): argument[1]
            for argument in task_arguments
        }
        for future in as_completed(futures):
            result = future.result()
            print(
                f"[figure2] {result.job}: "
                f"{result.significant_sdc}/{result.finite_value_sdc_labeled} "
                f"({result.significant_share_percent:.2f}%)",
                flush=True,
            )
            results.append(result)

    order = {job: index for index, (job, _, _) in enumerate(JOBS)}
    results.sort(key=lambda item: order[item.job])
    write_csv(csv_path, results)
    plot(prefix, results)
    total_significant = sum(item.significant_sdc for item in results)
    total_sdc = sum(item.finite_value_sdc_labeled for item in results)
    total_non_finite = sum(item.non_finite_value_sdc for item in results)
    total_invalid_fault = sum(item.invalid_fault_sdc for item in results)
    total_unlabeled = sum(
        item.unlabeled_finite_value_sdc for item in results
    )
    print(
        "[figure2] pooled Significant SDC among finite-valued labeled SDC: "
        f"{total_significant}/{total_sdc} ({100.0 * total_significant / total_sdc:.2f}%); "
        f"non-finite SDC={total_non_finite}; invalid-fault SDC={total_invalid_fault}; "
        f"unlabeled finite-valued SDC={total_unlabeled}",
        flush=True,
    )
    print(f"[figure2] wrote {csv_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
