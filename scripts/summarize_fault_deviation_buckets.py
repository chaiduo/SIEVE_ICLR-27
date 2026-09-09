#!/usr/bin/env python3
"""Summarize Significant-SDC prevalence by injected-value deviation."""

from __future__ import annotations

import argparse
import csv
import json
import math
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import matplotlib
import matplotlib.ticker as mticker

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
FINITE_BUCKETS = (
    ("[0,1]", 1.0),
    ("(1,1e6]", 1e6),
)
BUCKETS = (
    *(name for name, _ in FINITE_BUCKETS),
    ">1e6",
    "non_finite",
)


@dataclass
class JobResult:
    job: str
    model: str
    dataset: str
    total_rows: int
    clean_rows: int
    injected_rows: int
    valid_injected_rows: int
    invalid_label_rows: int
    invalid_fault_rows: int
    is_sdc_mismatches: int
    counts: dict[str, list[int]]
    sdc_counts: dict[str, list[int]]


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-root",
        type=Path,
        default=root / "experiments/telemetry_50",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "experiments/fault_deviation_buckets_telemetry50",
    )
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument(
        "--reuse-summary",
        action="store_true",
        help="Regenerate tables from the existing summary without rescanning JSONL.",
    )
    args = parser.parse_args()
    if args.workers <= 0:
        parser.error("--workers must be positive")
    return args


def deviation_bucket(before: Any, after: Any) -> str:
    before_value = float(before)
    after_value = float(after)
    if not math.isfinite(before_value) or not math.isfinite(after_value):
        return "non_finite"
    delta = abs(after_value - before_value)
    if not math.isfinite(delta):
        return "non_finite"
    for name, upper in FINITE_BUCKETS:
        if delta <= upper:
            return name
    return ">1e6"


def _json_loader() -> Callable[[str | bytes], Any]:
    return ujson.loads if ujson is not None else json.loads


def process_job(arguments: tuple[Path, str, str, str]) -> JobResult:
    path, job, model, dataset = arguments
    loads = _json_loader()
    counts = {bucket: [0, 0] for bucket in BUCKETS}
    sdc_counts = {bucket: [0, 0] for bucket in BUCKETS}
    total_rows = 0
    clean_rows = 0
    injected_rows = 0
    valid_injected_rows = 0
    invalid_label_rows = 0
    invalid_fault_rows = 0
    is_sdc_mismatches = 0

    with path.open("rb") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                record = loads(line)
            except Exception as error:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from error
            total_rows += 1
            if not bool(record.get("injected")):
                clean_rows += 1
                continue
            injected_rows += 1

            try:
                significance = int(record["significance"])
            except (KeyError, TypeError, ValueError, OverflowError):
                invalid_label_rows += 1
                continue
            if significance not in (0, 1, 2):
                invalid_label_rows += 1
                continue

            answers_differ = (
                str(record.get("pred_answer", ""))
                != str(record.get("clean_answer", ""))
            )
            significant = bool(answers_differ and significance == 2)
            observed_is_sdc = record.get("is_sdc")
            if observed_is_sdc is not None:
                is_sdc_mismatches += int(bool(observed_is_sdc) != answers_differ)

            fault = record.get("fault")
            if not isinstance(fault, dict):
                invalid_fault_rows += 1
                continue
            try:
                bucket = deviation_bucket(fault["before"], fault["after"])
            except (KeyError, TypeError, ValueError, OverflowError):
                invalid_fault_rows += 1
                continue

            counts[bucket][int(significant)] += 1
            if answers_differ:
                sdc_counts[bucket][int(significant)] += 1
            valid_injected_rows += 1

    return JobResult(
        job=job,
        model=model,
        dataset=dataset,
        total_rows=total_rows,
        clean_rows=clean_rows,
        injected_rows=injected_rows,
        valid_injected_rows=valid_injected_rows,
        invalid_label_rows=invalid_label_rows,
        invalid_fault_rows=invalid_fault_rows,
        is_sdc_mismatches=is_sdc_mismatches,
        counts=counts,
        sdc_counts=sdc_counts,
    )


def combine(results: list[JobResult], *, key: str) -> JobResult:
    counts = {bucket: [0, 0] for bucket in BUCKETS}
    sdc_counts = {bucket: [0, 0] for bucket in BUCKETS}
    for result in results:
        for bucket in BUCKETS:
            counts[bucket][0] += result.counts[bucket][0]
            counts[bucket][1] += result.counts[bucket][1]
            sdc_counts[bucket][0] += result.sdc_counts[bucket][0]
            sdc_counts[bucket][1] += result.sdc_counts[bucket][1]
    return JobResult(
        job=key,
        model="all",
        dataset="all",
        total_rows=sum(item.total_rows for item in results),
        clean_rows=sum(item.clean_rows for item in results),
        injected_rows=sum(item.injected_rows for item in results),
        valid_injected_rows=sum(item.valid_injected_rows for item in results),
        invalid_label_rows=sum(item.invalid_label_rows for item in results),
        invalid_fault_rows=sum(item.invalid_fault_rows for item in results),
        is_sdc_mismatches=sum(item.is_sdc_mismatches for item in results),
        counts=counts,
        sdc_counts=sdc_counts,
    )


def bucket_rows(
    result: JobResult,
    *,
    scope_type: str,
    buckets: tuple[str, ...] = BUCKETS,
) -> list[dict[str, Any]]:
    total_significant = sum(result.counts[bucket][1] for bucket in buckets)
    total_non_significant = sum(result.counts[bucket][0] for bucket in buckets)
    total_samples = total_significant + total_non_significant
    rows = []
    for order, bucket in enumerate(buckets):
        non_significant, significant = result.counts[bucket]
        total = non_significant + significant
        rows.append(
            {
                "scope_type": scope_type,
                "scope": result.job,
                "model": result.model,
                "dataset": result.dataset,
                "bucket_order": order,
                "deviation_bucket": bucket,
                "samples": total,
                "sample_share_pct": _percent(total, total_samples),
                "significant_sdc": significant,
                "non_significant_sdc": non_significant,
                "significant_global_share_pct": _percent(
                    significant, total_samples
                ),
                "non_significant_global_share_pct": _percent(
                    non_significant, total_samples
                ),
                "significant_rate_pct": _percent(significant, total),
                "non_significant_rate_pct": _percent(non_significant, total),
                "significant_capture_pct": _percent(
                    significant, total_significant
                ),
                "non_significant_distribution_pct": _percent(
                    non_significant, total_non_significant
                ),
            }
        )
    return rows


def sdc_bucket_rows(
    result: JobResult,
    *,
    scope_type: str,
    buckets: tuple[str, ...],
) -> list[dict[str, Any]]:
    total_significant = sum(result.sdc_counts[bucket][1] for bucket in buckets)
    total_non_significant = sum(
        result.sdc_counts[bucket][0] for bucket in buckets
    )
    total_sdc = total_significant + total_non_significant
    rows = []
    for order, bucket in enumerate(buckets):
        non_significant, significant = result.sdc_counts[bucket]
        samples = non_significant + significant
        rows.append(
            {
                "scope_type": scope_type,
                "scope": result.job,
                "model": result.model,
                "dataset": result.dataset,
                "bucket_order": order,
                "deviation_bucket": bucket,
                "samples": samples,
                "sample_share_pct": _percent(samples, total_sdc),
                "significant_sdc": significant,
                "significant_global_share_pct": _percent(
                    significant, total_sdc
                ),
                "non_significant_sdc": non_significant,
                "non_significant_global_share_pct": _percent(
                    non_significant, total_sdc
                ),
            }
        )
    return rows


def _percent(numerator: int, denominator: int) -> float:
    return 0.0 if denominator == 0 else 100.0 * numerator / denominator


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def global_share_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    fields = (
        "scope_type",
        "scope",
        "model",
        "dataset",
        "bucket_order",
        "deviation_bucket",
        "samples",
        "sample_share_pct",
        "significant_sdc",
        "significant_global_share_pct",
        "non_significant_sdc",
        "non_significant_global_share_pct",
    )
    return [{field: row[field] for field in fields} for row in rows]


def plot_global_shares(
    rows: list[dict[str, Any]],
    output_dir: Path,
    *,
    filename: str,
    denominator_label: str,
    log_scale: bool,
) -> None:
    selected = [row for row in rows if int(row["samples"]) > 0]
    labels = [str(row["deviation_bucket"]) for row in selected]
    significant = np.asarray(
        [float(row["significant_global_share_pct"]) for row in selected]
    )
    non_significant = np.asarray(
        [float(row["non_significant_global_share_pct"]) for row in selected]
    )
    positions = np.arange(len(selected))
    width = 0.38

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Nimbus Roman", "Times New Roman", "DejaVu Serif"],
            "font.size": 9,
            "axes.edgecolor": "black",
            "axes.linewidth": 0.7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )
    figure, axis = plt.subplots(figsize=(10.8, 4.4))
    non_significant_bars = axis.bar(
        positions - width / 2,
        non_significant,
        width,
        label="Non-significant SDC",
        color="#4C78A8",
        edgecolor="black",
        linewidth=0.4,
    )
    significant_bars = axis.bar(
        positions + width / 2,
        significant,
        width,
        label="Significant SDC",
        color="#E45756",
        edgecolor="black",
        linewidth=0.4,
    )
    if log_scale:
        axis.set_yscale("log")
        axis.set_ylim(1e-4, 100)
    else:
        maximum = float(max(significant.max(), non_significant.max()))
        axis.set_ylim(0.0, maximum * 1.18)
        axis.yaxis.set_major_formatter(
            mticker.PercentFormatter(xmax=100, decimals=0)
        )
    axis.set_ylabel(denominator_label)
    axis.set_xlabel(r"Absolute injected-value deviation $|after-before|$")
    axis.set_xticks(positions)
    axis.set_xticklabels(labels, rotation=32, ha="right")
    axis.grid(
        axis="y",
        which="both" if log_scale else "major",
        color="#D9D9D9",
        linewidth=0.5,
    )
    axis.set_axisbelow(True)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.legend(frameon=False, ncol=2, loc="upper right")
    for bars, values in (
        (non_significant_bars, non_significant),
        (significant_bars, significant),
    ):
        axis.bar_label(
            bars,
            labels=[_format_plot_percent(value) for value in values],
            padding=2,
            rotation=90,
            fontsize=6.5,
        )
    figure.tight_layout()
    figure.savefig(
        output_dir / f"{filename}.png",
        dpi=300,
        bbox_inches="tight",
    )
    figure.savefig(
        output_dir / f"{filename}.pdf",
        bbox_inches="tight",
    )
    plt.close(figure)


def _format_plot_percent(value: float) -> str:
    if value >= 1.0:
        return f"{value:.2f}%"
    if value >= 0.01:
        return f"{value:.3f}%"
    return f"{value:.4f}%"


def plot_sdc_broken_axis(
    rows: list[dict[str, Any]],
    output_dir: Path,
) -> None:
    selected = [row for row in rows if int(row["samples"]) > 0]
    display_labels = {
        "[0,1]": "[0, 1]",
        "(1,1e6]": r"$(1, 10^6]$",
        ">1e6": r"$>10^6$",
    }
    labels = [
        display_labels.get(
            str(row["deviation_bucket"]),
            str(row["deviation_bucket"]),
        )
        for row in selected
    ]
    significant = np.asarray(
        [float(row["significant_global_share_pct"]) for row in selected]
    )
    non_significant = np.asarray(
        [float(row["non_significant_global_share_pct"]) for row in selected]
    )
    positions = np.arange(len(selected))
    width = 0.38
    figure, (upper, lower) = plt.subplots(
        2,
        1,
        sharex=True,
        figsize=(3.25, 2.55),
        gridspec_kw={"height_ratios": (1, 4), "hspace": 0.06},
    )
    bar_sets = []
    for axis in (upper, lower):
        non_significant_bars = axis.bar(
            positions - width / 2,
            non_significant,
            width,
            label="Non-significant SDC",
            color="#4C78A8",
            edgecolor="black",
            linewidth=0.4,
        )
        significant_bars = axis.bar(
            positions + width / 2,
            significant,
            width,
            label="Significant SDC",
            color="#E45756",
            edgecolor="black",
            linewidth=0.4,
        )
        axis.grid(axis="y", color="#D9D9D9", linewidth=0.5)
        axis.set_axisbelow(True)
        axis.spines["right"].set_visible(False)
        axis.tick_params(axis="both", labelsize=7)
        axis.yaxis.set_major_formatter(
            mticker.PercentFormatter(xmax=100, decimals=0)
        )
        bar_sets.append((non_significant_bars, significant_bars))

    upper.set_ylim(50, 62)
    upper.set_yticks((50, 55, 60))
    upper.spines["bottom"].set_visible(False)
    upper.spines["top"].set_visible(False)
    upper.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
    upper.legend(
        frameon=False,
        ncol=2,
        loc="upper right",
        fontsize=6.5,
        handlelength=1.4,
        columnspacing=0.8,
    )

    lower.set_ylim(0, 20)
    lower.set_yticks((0, 5, 10, 15, 20))
    lower.spines["top"].set_visible(False)
    lower.set_xticks(positions)
    lower.set_xticklabels(labels, rotation=0)
    lower.set_xlabel("Fault-induced value deviation", fontsize=8)

    diagonal = 0.008
    marker = [(-1, -diagonal), (1, diagonal)]
    break_style = {
        "marker": marker,
        "markersize": 10,
        "linestyle": "none",
        "color": "black",
        "mec": "black",
        "mew": 0.8,
        "clip_on": False,
    }
    upper.plot([0, 1], [0, 0], transform=upper.transAxes, **break_style)
    lower.plot([0, 1], [1, 1], transform=lower.transAxes, **break_style)

    for bars, values in zip(bar_sets[0], (non_significant, significant)):
        upper.bar_label(
            bars,
            labels=[
                _format_plot_percent(value) if value >= 50 else ""
                for value in values
            ],
            padding=2,
            rotation=0,
            fontsize=5.8,
        )
    for bars, values in zip(bar_sets[1], (non_significant, significant)):
        lower.bar_label(
            bars,
            labels=[
                _format_plot_percent(value) if value < 50 else ""
                for value in values
            ],
            padding=2,
            rotation=0,
            fontsize=5.8,
        )

    figure.supylabel(
        "Percentage of SDC (%)",
        x=0.065,
        fontsize=8,
    )
    figure.subplots_adjust(
        left=0.20,
        right=0.98,
        bottom=0.20,
        top=0.98,
        hspace=0.06,
    )
    figure.savefig(
        output_dir
        / "finite_sdc_global_share_by_deviation_bucket_broken_axis.png",
        dpi=300,
    )
    figure.savefig(
        output_dir
        / "finite_sdc_global_share_by_deviation_bucket_broken_axis.pdf",
    )
    plt.close(figure)


def main() -> int:
    args = parse_args()
    input_root = args.input_root.resolve()
    output_dir = args.output_dir.resolve()
    work = [
        (input_root / job / "labels.jsonl", job, model, dataset)
        for job, model, dataset in JOBS
    ]
    missing = [str(path) for path, *_ in work if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing labels files: {missing}")

    summary_path = output_dir / "summary.json"
    if args.reuse_summary:
        saved = json.loads(summary_path.read_text(encoding="utf-8"))
        if any("sdc_counts" not in item for item in saved["jobs"]):
            raise ValueError(
                "Existing summary predates SDC-only counts; rescan without "
                "--reuse-summary"
            )
        results = [JobResult(**item) for item in saved["jobs"]]
    else:
        with ProcessPoolExecutor(
            max_workers=min(args.workers, len(work))
        ) as pool:
            results = list(pool.map(process_job, work))

    all_result = combine(results, key="all_jobs")
    model_results = [
        combine(
            [item for item in results if item.model == model],
            key=model,
        )
        for model in dict.fromkeys(item.model for item in results)
    ]
    for item in model_results:
        item.model = item.job
    dataset_results = [
        combine(
            [item for item in results if item.dataset == dataset],
            key=dataset,
        )
        for dataset in dict.fromkeys(item.dataset for item in results)
    ]
    for item in dataset_results:
        item.dataset = item.job

    all_rows = bucket_rows(all_result, scope_type="all")
    for item in model_results:
        all_rows.extend(bucket_rows(item, scope_type="model"))
    for item in dataset_results:
        all_rows.extend(bucket_rows(item, scope_type="dataset"))
    for item in results:
        all_rows.extend(bucket_rows(item, scope_type="job"))

    write_csv(output_dir / "deviation_buckets.csv", all_rows)
    write_csv(
        output_dir / "aggregate_buckets.csv",
        [row for row in all_rows if row["scope_type"] == "all"],
    )
    finite_buckets = tuple(
        bucket for bucket in BUCKETS if bucket != "non_finite"
    )
    finite_rows = bucket_rows(
        all_result,
        scope_type="all",
        buckets=finite_buckets,
    )
    for item in model_results:
        finite_rows.extend(
            bucket_rows(item, scope_type="model", buckets=finite_buckets)
        )
    for item in dataset_results:
        finite_rows.extend(
            bucket_rows(item, scope_type="dataset", buckets=finite_buckets)
        )
    for item in results:
        finite_rows.extend(
            bucket_rows(item, scope_type="job", buckets=finite_buckets)
        )
    write_csv(output_dir / "finite_deviation_buckets.csv", finite_rows)
    write_csv(
        output_dir / "aggregate_finite_buckets.csv",
        [row for row in finite_rows if row["scope_type"] == "all"],
    )
    finite_global_rows = global_share_rows(finite_rows)
    write_csv(
        output_dir / "finite_global_shares.csv",
        finite_global_rows,
    )
    aggregate_finite_global_rows = [
        row for row in finite_global_rows if row["scope_type"] == "all"
    ]
    write_csv(
        output_dir / "aggregate_finite_global_shares.csv",
        aggregate_finite_global_rows,
    )
    plot_global_shares(
        aggregate_finite_global_rows,
        output_dir,
        filename="finite_global_share_by_deviation_bucket",
        denominator_label="Share of all finite injected runs (%)",
        log_scale=True,
    )
    finite_sdc_rows = sdc_bucket_rows(
        all_result,
        scope_type="all",
        buckets=finite_buckets,
    )
    for item in model_results:
        finite_sdc_rows.extend(
            sdc_bucket_rows(
                item,
                scope_type="model",
                buckets=finite_buckets,
            )
        )
    for item in dataset_results:
        finite_sdc_rows.extend(
            sdc_bucket_rows(
                item,
                scope_type="dataset",
                buckets=finite_buckets,
            )
        )
    for item in results:
        finite_sdc_rows.extend(
            sdc_bucket_rows(
                item,
                scope_type="job",
                buckets=finite_buckets,
            )
        )
    write_csv(
        output_dir / "finite_sdc_global_shares.csv",
        finite_sdc_rows,
    )
    aggregate_finite_sdc_rows = [
        row for row in finite_sdc_rows if row["scope_type"] == "all"
    ]
    write_csv(
        output_dir / "aggregate_finite_sdc_global_shares.csv",
        aggregate_finite_sdc_rows,
    )
    finite_sdc_total = sum(
        int(row["samples"]) for row in aggregate_finite_sdc_rows
    )
    plot_global_shares(
        aggregate_finite_sdc_rows,
        output_dir,
        filename="finite_sdc_global_share_by_deviation_bucket",
        denominator_label=(
            f"Share of all finite SDC runs (N={finite_sdc_total:,})"
        ),
        log_scale=False,
    )
    plot_sdc_broken_axis(aggregate_finite_sdc_rows, output_dir)
    summary = {
        "protocol": {
            "source": str(input_root),
            "jobs": [job for job, _, _ in JOBS],
            "population": "all injected rows across fit/calibration/test",
            "deviation": "abs(after - before)",
            "significant_sdc": (
                "pred_answer != clean_answer and significance == 2"
            ),
            "non_significant_sdc": (
                "pred_answer != clean_answer and significance in {0, 1}"
            ),
            "invalid_labels": "excluded",
            "non_finite_values": "separate bucket",
            "buckets": list(BUCKETS),
            "json_parser": "ujson" if ujson is not None else "stdlib json",
        },
        "aggregate": asdict(all_result),
        "jobs": [asdict(item) for item in results],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "valid_injected_rows": all_result.valid_injected_rows,
                "invalid_label_rows": all_result.invalid_label_rows,
                "invalid_fault_rows": all_result.invalid_fault_rows,
                "is_sdc_mismatches": all_result.is_sdc_mismatches,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
