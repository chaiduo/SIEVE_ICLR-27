#!/usr/bin/env python3
"""Evaluate frozen 36D/K=28 SIEVE detectors in a 9x9 task-transfer matrix.

For source->target transfer, the source task contributes the frozen XGBoost
detector and its Calibration threshold. The target contributes only its
already materialized 36D/K=28 Final Test features, which were produced using
the target's clean-trained Mapping. Target Fit/Calibration fault labels are
never read. Target Final Test labels are used only after prediction for
offline scoring.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb

from detect_sdc.config import load_yaml
from detect_sdc.detector.xgboost import (
    binary_metrics,
    get_feature_columns,
    prepare_features,
    strict_feature_finite_mask,
)


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
METRIC_COLUMNS = (
    "support",
    "pred_total",
    "tp",
    "fp",
    "fn",
    "tn",
    "precision",
    "recall",
    "false_positive_rate",
    "f1",
)


@dataclass(frozen=True)
class TaskSpec:
    job: str
    model: str
    dataset: str
    feature_path: Path
    summary_path: Path
    evaluation_path: Path


@dataclass(frozen=True)
class SourceDetector:
    task: TaskSpec
    feature_columns: tuple[str, ...]
    threshold: float
    model_path: Path
    model_sha256: str
    summary: dict[str, Any]


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate frozen 36D/K=28 source detectors on all target Final "
            "Test feature sets."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=root / "configs/experiments/current.yaml",
    )
    parser.add_argument(
        "--feature-root",
        type=Path,
        default=root / "experiments/step_ablation_36d",
    )
    parser.add_argument(
        "--comparison-results-root",
        type=Path,
        default=root / "experiments/comparison_36d_k28/results",
        help=(
            "Current comparison evaluations used to validate the in-domain "
            "diagonal against the published Final Test cohort."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "experiments/transfer_36d_k28",
    )
    parser.add_argument(
        "--jobs",
        default=None,
        help="Comma-separated source/target jobs. Defaults to all nine tasks.",
    )
    parser.add_argument(
        "--sources",
        default=None,
        help="Optional comma-separated source jobs; defaults to --jobs.",
    )
    parser.add_argument(
        "--targets",
        default=None,
        help="Optional comma-separated target jobs; defaults to --jobs.",
    )
    parser.add_argument(
        "--bootstrap-replicates",
        type=int,
        default=10_000,
        help="Target-group bootstrap replicates for per-cell F1 intervals.",
    )
    parser.add_argument(
        "--bootstrap-seed",
        type=int,
        default=20260907,
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def parse_job_list(value: str | None, default: tuple[str, ...]) -> tuple[str, ...]:
    if value is None:
        return default
    names = tuple(item.strip() for item in value.split(",") if item.strip())
    if not names:
        raise ValueError("Job list cannot be empty")
    return names


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relationship(source: TaskSpec, target: TaskSpec) -> str:
    if source.job == target.job:
        return "in_domain"
    if source.model == target.model:
        return "same_model_cross_dataset"
    if source.dataset == target.dataset:
        return "cross_model_same_dataset"
    return "cross_model_cross_dataset"


def task_spec(
    *,
    job: str,
    config_jobs: dict[str, Any],
    feature_root: Path,
    comparison_results_root: Path,
) -> TaskSpec:
    job_config = config_jobs[job]
    root = feature_root / job / "k_28"
    feature_path = root / "features.csv"
    summary_path = root / "output/metrics_summary.json"
    evaluation_path = (
        comparison_results_root / job / "evaluation_strict_finite" / "metrics.json"
    )
    if not feature_path.is_file():
        raise FileNotFoundError(f"Missing target Final Test features: {feature_path}")
    if not summary_path.is_file():
        raise FileNotFoundError(f"Missing frozen detector summary: {summary_path}")
    if not evaluation_path.is_file():
        raise FileNotFoundError(
            f"Missing current comparison evaluation: {evaluation_path}"
        )
    return TaskSpec(
        job=job,
        model=str(job_config["model"]),
        dataset=str(job_config["dataset"]),
        feature_path=feature_path,
        summary_path=summary_path,
        evaluation_path=evaluation_path,
    )


def load_source_detector(task: TaskSpec) -> SourceDetector:
    summary = json.loads(task.summary_path.read_text(encoding="utf-8"))
    if int(summary.get("feature_count", -1)) != 36:
        raise ValueError(f"{task.job} is not a 36D detector artifact")
    feature_columns = tuple(summary["feature_columns"])
    if len(feature_columns) != 36:
        raise ValueError(f"{task.job} has {len(feature_columns)} detector features")
    calibration = summary.get("threshold_calibration", {})
    threshold = float(calibration["threshold"])
    model_path = Path(summary["model_path"]).resolve()
    if not model_path.is_file():
        raise FileNotFoundError(f"{task.job} frozen detector is missing: {model_path}")
    return SourceDetector(
        task=task,
        feature_columns=feature_columns,
        threshold=threshold,
        model_path=model_path,
        model_sha256=sha256_file(model_path),
        summary=summary,
    )


def load_target_test(task: TaskSpec) -> tuple[pd.DataFrame, tuple[str, ...]]:
    frame = pd.read_csv(task.feature_path)
    if "split" in frame.columns:
        observed = set(frame["split"].dropna().astype(str))
        if observed and observed != {"test"}:
            frame = frame.loc[frame["split"].astype(str).eq("test")].copy()
    if frame.empty:
        raise ValueError(f"{task.job} has no target Final Test rows")
    feature_columns = tuple(get_feature_columns(frame))
    if len(feature_columns) != 36:
        raise ValueError(f"{task.job} has {len(feature_columns)} target features")
    if frame["sample_uid"].astype(str).duplicated().any():
        raise ValueError(f"{task.job} Final Test sample_uids are not unique")
    return frame, feature_columns


def score(
    *,
    detector: xgb.Booster,
    frame: pd.DataFrame,
    feature_columns: tuple[str, ...],
    threshold: float,
) -> tuple[dict[str, int | float], np.ndarray]:
    probability = np.asarray(
        detector.inplace_predict(
            prepare_features(frame, list(feature_columns)),
            validate_features=False,
        ),
        dtype=float,
    )
    prediction = (probability > threshold).astype(int)
    probabilities = np.column_stack((1.0 - probability, probability))
    result = binary_metrics(
        frame["significant_sdc_target"].astype(int),
        prediction,
        probabilities,
    )["target_significant_sdc"]
    return {key: result[key] for key in METRIC_COLUMNS}, prediction


def bootstrap_f1_interval(
    *,
    frame: pd.DataFrame,
    prediction: np.ndarray,
    replicates: int,
    seed: int,
) -> dict[str, int | float]:
    if replicates < 1:
        raise ValueError("bootstrap_replicates must be positive")
    groups, group_values = pd.factorize(
        frame["semantic_group_id"].astype(str),
        sort=True,
    )
    if len(group_values) < 2:
        raise ValueError("Cluster bootstrap requires at least two target groups")
    target = frame["significant_sdc_target"].astype(int).to_numpy()
    predicted = np.asarray(prediction, dtype=int)
    group_count = len(group_values)
    tp = np.bincount(
        groups,
        weights=((target == 1) & (predicted == 1)).astype(int),
        minlength=group_count,
    )
    fp = np.bincount(
        groups,
        weights=((target == 0) & (predicted == 1)).astype(int),
        minlength=group_count,
    )
    fn = np.bincount(
        groups,
        weights=((target == 1) & (predicted == 0)).astype(int),
        minlength=group_count,
    )
    generator = np.random.default_rng(seed)
    samples: list[np.ndarray] = []
    batch_size = 500
    for start in range(0, replicates, batch_size):
        size = min(batch_size, replicates - start)
        sampled_groups = generator.integers(
            0,
            group_count,
            size=(size, group_count),
        )
        sampled_tp = tp[sampled_groups].sum(axis=1)
        sampled_fp = fp[sampled_groups].sum(axis=1)
        sampled_fn = fn[sampled_groups].sum(axis=1)
        denominator = 2.0 * sampled_tp + sampled_fp + sampled_fn
        samples.append(
            np.divide(
                2.0 * sampled_tp,
                denominator,
                out=np.zeros_like(denominator, dtype=float),
                where=denominator > 0,
            )
        )
    values = np.concatenate(samples)
    return {
        "bootstrap_replicates": replicates,
        "bootstrap_clusters": group_count,
        "f1_ci_low": float(np.quantile(values, 0.025)),
        "f1_ci_high": float(np.quantile(values, 0.975)),
    }


def validate_diagonal(
    *,
    source: SourceDetector,
    actual: dict[str, int | float],
) -> None:
    comparison = json.loads(
        source.task.evaluation_path.read_text(encoding="utf-8")
    )
    expected = comparison["methods"]["SIEVE"]["cohorts"]["full"]["metrics"]
    expected_threshold = float(
        comparison["methods"]["SIEVE"]["threshold_calibration"]["threshold"]
    )
    expected_metrics = {
        "precision": float(expected["significant_sdc_precision"]),
        "recall": float(expected["significant_sdc_recall"]),
        "false_positive_rate": float(expected["non_significant_fpr"]),
        "f1": float(expected["significant_sdc_f1"]),
    }
    for key, expected_value in expected_metrics.items():
        if not np.isclose(float(actual[key]), expected_value, atol=1e-12):
            raise AssertionError(
                f"{source.task.job} diagonal {key} mismatch: "
                f"{actual[key]} != {expected_value}"
            )
    if not np.isclose(source.threshold, expected_threshold, atol=1e-12):
        raise AssertionError(
            f"{source.task.job} threshold mismatch: "
            f"{source.threshold} != {expected_threshold}"
        )


def create_matrix(
    results: pd.DataFrame,
    *,
    cohort: str,
    metric: str,
    sources: tuple[str, ...],
    targets: tuple[str, ...],
) -> pd.DataFrame:
    frame = results.loc[results["cohort"].eq(cohort)]
    output = frame.pivot(
        index="source_job",
        columns="target_job",
        values=metric,
    ).loc[list(sources), list(targets)]
    output.index = [DISPLAY_NAMES[name] for name in sources]
    output.columns = [DISPLAY_NAMES[name] for name in targets]
    return output


def summarize_relationships(results: pd.DataFrame) -> pd.DataFrame:
    return (
        results.groupby(["cohort", "relationship"], as_index=False)
        .agg(
            cells=("f1", "size"),
            precision=("precision", "mean"),
            recall=("recall", "mean"),
            f1=("f1", "mean"),
            false_positive_rate=("false_positive_rate", "mean"),
        )
        .sort_values(["cohort", "relationship"])
    )


def summarize_sources(results: pd.DataFrame) -> pd.DataFrame:
    off_diagonal = results.loc[results["relationship"].ne("in_domain")]
    columns = [
        "source_job",
        "precision_full",
        "precision_strict_finite",
        "recall_full",
        "recall_strict_finite",
        "f1_full",
        "f1_strict_finite",
        "false_positive_rate_full",
        "false_positive_rate_strict_finite",
    ]
    if off_diagonal.empty:
        return pd.DataFrame(columns=columns)
    pivot = (
        off_diagonal.pivot_table(
            index="source_job",
            columns="cohort",
            values=["precision", "recall", "f1", "false_positive_rate"],
            aggfunc="mean",
        )
        .sort_index()
    )
    pivot.columns = [
        f"{metric}_{cohort}"
        for metric, cohort in pivot.columns.to_flat_index()
    ]
    return (
        pivot.reset_index()
        .reindex(columns=columns)
        .sort_values("f1_full", ascending=False)
    )


def mean_metrics(frame: pd.DataFrame) -> dict[str, float | None]:
    columns = ["precision", "recall", "f1", "false_positive_rate"]
    if frame.empty:
        return {column: None for column in columns}
    return {
        key: float(value)
        for key, value in frame.loc[:, columns].mean().to_dict().items()
    }


def main() -> int:
    args = parse_args()
    config = load_yaml(args.config.resolve())
    config_jobs = config["featurization"]["jobs"]
    all_jobs = tuple(config_jobs)
    jobs = parse_job_list(args.jobs, all_jobs)
    sources = parse_job_list(args.sources, jobs)
    targets = parse_job_list(args.targets, jobs)
    if args.bootstrap_replicates < 1:
        raise ValueError("--bootstrap-replicates must be positive")
    unknown = sorted((set(jobs) | set(sources) | set(targets)) - set(config_jobs))
    if unknown:
        raise ValueError(f"Unknown jobs: {unknown}")
    unknown_display = sorted((set(sources) | set(targets)) - set(DISPLAY_NAMES))
    if unknown_display:
        raise ValueError(f"Missing display names: {unknown_display}")

    output_dir = args.output_dir.resolve()
    metrics_path = output_dir / "transfer_metrics.csv"
    if metrics_path.exists() and not args.overwrite:
        raise FileExistsError(f"{metrics_path} exists; pass --overwrite to replace it")

    feature_root = args.feature_root.resolve()
    comparison_results_root = args.comparison_results_root.resolve()
    tasks = {
        job: task_spec(
            job=job,
            config_jobs=config_jobs,
            feature_root=feature_root,
            comparison_results_root=comparison_results_root,
        )
        for job in set(sources) | set(targets)
    }
    targets_by_job = {job: load_target_test(tasks[job]) for job in targets}
    sources_by_job = {job: load_source_detector(tasks[job]) for job in sources}

    result_rows: list[dict[str, Any]] = []
    source_rows: list[dict[str, Any]] = []
    diagonal_checked: list[str] = []
    for source_index, source_name in enumerate(sources, start=1):
        source = sources_by_job[source_name]
        detector = xgb.Booster()
        detector.load_model(source.model_path)
        source_rows.append(
            {
                "source_job": source.task.job,
                "source_model": source.task.model,
                "source_dataset": source.task.dataset,
                "threshold": source.threshold,
                "model_path": str(source.model_path),
                "model_sha256": source.model_sha256,
                "calibration_precision": source.summary["threshold_calibration"][
                    "calibration_precision"
                ],
                "calibration_recall": source.summary["threshold_calibration"][
                    "calibration_recall"
                ],
                "calibration_f1": source.summary["threshold_calibration"][
                    "calibration_f1"
                ],
                "calibration_fpr": source.summary["threshold_calibration"][
                    "calibration_fpr"
                ],
            }
        )
        print(f"[source {source_index}/{len(sources)}] {source_name}", flush=True)

        for target_name in targets:
            target = tasks[target_name]
            target_test, target_features = targets_by_job[target_name]
            if target_features != source.feature_columns:
                raise ValueError(
                    f"{source_name}->{target_name} feature schema mismatch"
                )
            strict_mask = strict_feature_finite_mask(
                target_test,
                list(target_features),
            )
            for cohort, frame in (
                ("full", target_test),
                ("strict_finite", target_test.loc[strict_mask].copy()),
            ):
                if frame.empty:
                    raise ValueError(f"{target_name} {cohort} cohort is empty")
                values, prediction = score(
                    detector=detector,
                    frame=frame,
                    feature_columns=source.feature_columns,
                    threshold=source.threshold,
                )
                cell_seed = int.from_bytes(
                    hashlib.sha256(
                        (
                            f"{args.bootstrap_seed}:{source_name}:"
                            f"{target_name}:{cohort}"
                        ).encode("utf-8")
                    ).digest()[:8],
                    byteorder="little",
                    signed=False,
                )
                bootstrap = bootstrap_f1_interval(
                    frame=frame,
                    prediction=prediction,
                    replicates=args.bootstrap_replicates,
                    seed=cell_seed,
                )
                if source_name == target_name and cohort == "full":
                    validate_diagonal(source=source, actual=values)
                    diagonal_checked.append(source_name)
                result_rows.append(
                    {
                        "source_job": source_name,
                        "source_model": source.task.model,
                        "source_dataset": source.task.dataset,
                        "target_job": target_name,
                        "target_model": target.model,
                        "target_dataset": target.dataset,
                        "relationship": relationship(source.task, target),
                        "cohort": cohort,
                        "threshold": source.threshold,
                        "target_rows": int(len(frame)),
                        "target_strict_finite_rows": int(strict_mask.sum()),
                        **values,
                        **bootstrap,
                    }
                )

    results = pd.DataFrame(result_rows).sort_values(
        ["cohort", "source_job", "target_job"]
    )
    expected_rows = len(sources) * len(targets) * 2
    if len(results) != expected_rows:
        raise AssertionError(f"Expected {expected_rows} rows, found {len(results)}")
    if set(sources) == set(targets) and tuple(sources) == tuple(targets):
        if sorted(diagonal_checked) != sorted(sources):
            raise AssertionError("Not all in-domain diagonal cells were validated")

    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(metrics_path, results)
    write_csv(output_dir / "source_detectors.csv", pd.DataFrame(source_rows))
    relationship_summary = summarize_relationships(results)
    source_summary = summarize_sources(results)
    write_csv(output_dir / "relationship_summary.csv", relationship_summary)
    write_csv(output_dir / "source_summary.csv", source_summary)

    matrices: dict[str, pd.DataFrame] = {}
    for cohort in ("full", "strict_finite"):
        for metric in ("f1", "precision", "recall", "false_positive_rate"):
            item = create_matrix(
                results,
                cohort=cohort,
                metric=metric,
                sources=sources,
                targets=targets,
            )
            matrices[f"{cohort}_{metric}"] = item
            item.to_csv(output_dir / f"{cohort}_{metric}_matrix.csv")

    off_diagonal = results.loc[results["relationship"].ne("in_domain")]
    summary = {
        "protocol": {
            "source": (
                "frozen 36D/K=28 detector and threshold selected on the "
                "source Fit/Calibration partitions"
            ),
            "target": (
                "target's own clean-trained Mapping materializes Final Test "
                "features; no target Fit/Calibration fault labels are read"
            ),
            "evaluation": (
                "target Final Test fault labels are used only after frozen "
                "prediction for offline scoring"
            ),
            "strict_finite": (
                "all 36 target Final Test features finite after numeric "
                "cleaning; NaN, +/-Inf, and values beyond float32 range excluded"
            ),
        },
        "feature_count": 36,
        "window_k": 28,
        "bootstrap_replicates": args.bootstrap_replicates,
        "bootstrap_seed": args.bootstrap_seed,
        "sources": list(sources),
        "targets": list(targets),
        "full_cell_count": len(sources) * len(targets),
        "scored_rows": int(results.loc[results["cohort"].eq("full"), "target_rows"].sum()),
        "off_diagonal_full_macro": mean_metrics(
            off_diagonal.loc[off_diagonal["cohort"].eq("full")]
        ),
        "off_diagonal_strict_finite_macro": mean_metrics(
            off_diagonal.loc[off_diagonal["cohort"].eq("strict_finite")]
        ),
        "diagonal_validated_sources": sorted(diagonal_checked),
        "relationship_summary": relationship_summary.to_dict(orient="records"),
        "artifact_paths": {
            "metrics": str(metrics_path),
            "source_detectors": str(output_dir / "source_detectors.csv"),
            "relationship_summary": str(output_dir / "relationship_summary.csv"),
            "source_summary": str(output_dir / "source_summary.csv"),
        },
    }
    write_json(output_dir / "summary.json", summary)
    write_json(
        output_dir / "protocol.json",
        {
            "protocol": summary["protocol"],
            "feature_root": str(feature_root),
            "source_detector_artifact": (
                "experiments/step_ablation_36d/{job}/k_28/output/"
                "significant_sdc_detector.ubj"
            ),
            "source_summary_artifact": (
                "experiments/step_ablation_36d/{job}/k_28/output/"
                "metrics_summary.json"
            ),
            "target_feature_artifact": (
                "experiments/step_ablation_36d/{job}/k_28/features.csv "
                "filtered to split=test"
            ),
            "diagonal_validation_artifact": (
                "experiments/comparison_36d_k28/results/{job}/"
                "evaluation_strict_finite/metrics.json"
            ),
        },
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
