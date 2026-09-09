#!/usr/bin/env python3

"""Run Fit-only component ablations for the frozen 36D/K=28 representation.

The script intentionally reads only the outer-Fit rows from the already
materialized 36D/K=28 feature files. It reuses the nested split manifests
created for configuration selection:
  * 70% of outer Fit for detector fitting,
  * 15% for threshold calibration,
  * 15% for ablation evaluation.

Neither outer Calibration nor Final Test rows are read. The complete 36D
configuration is rerun and checked against the original selection output
before leave-one-component-out results are accepted. Strict Finite is a
fixed selection-validation subset whose complete 36D/K=28 vector is finite;
all ablation variants use this same subset.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import sys
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from detect_sdc.config import load_yaml
from detect_sdc.detector.xgboost import (
    XGBoostConfig,
    add_significant_sdc_target,
    binary_metrics,
    calibrate_threshold_max_f1,
    get_feature_columns,
    prepare_features,
    train_binary_model,
)
from detect_sdc.splitting import split_by_group, validate_identity_columns


FULL_PAIRS = ((6, 7), (24, 25), (26, 27))
METRIC_GROUPS = ("cos_sim", "mean_diff", "std_diff", "l2_distance")
METRIC_KEYS = (
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
SEED = 20260907
EARLY_STOPPING_RATIO = 0.15


@dataclass(frozen=True)
class TaskSpec:
    job: str
    model: str
    dataset: str
    group_column: str = "semantic_group_id"


@dataclass(frozen=True)
class AblationSpec:
    name: str
    ablation_family: str
    removed_component: str | None


ABLATIONS = (
    AblationSpec("full_36d", "full", None),
    *(
        AblationSpec(
            f"without_pair_{left}_{right}",
            "layer_pair",
            f"({left},{right})",
        )
        for left, right in FULL_PAIRS
    ),
    *(
        AblationSpec(
            f"without_{metric}",
            "metric",
            metric,
        )
        for metric in METRIC_GROUPS
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Fit-only 36D/K=28 component ablations."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=REPOSITORY_ROOT / "configs/experiments/current.yaml",
    )
    parser.add_argument(
        "--feature-root",
        type=Path,
        default=REPOSITORY_ROOT / "experiments" / "step_ablation_36d",
    )
    parser.add_argument(
        "--split-root",
        type=Path,
        default=REPOSITORY_ROOT
        / "experiments"
        / "fit_only_configuration_selection"
        / "splits",
    )
    parser.add_argument(
        "--selection-root",
        type=Path,
        default=REPOSITORY_ROOT / "experiments" / "fit_only_configuration_selection",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            REPOSITORY_ROOT
            / "experiments"
            / "ablation_36d_k28_fit_only_strict_finite"
        ),
    )
    parser.add_argument(
        "--jobs",
        default=None,
        help="Comma-separated jobs; defaults to the nine configured jobs.",
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--xgb-n-jobs", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    if args.xgb_n_jobs < 1:
        parser.error("--xgb-n-jobs must be positive")
    return args


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def feature_path(feature_root: Path, job: str) -> Path:
    return feature_root / job / "k_28" / "features.csv"


def manifest_path(split_root: Path, job: str) -> Path:
    return split_root / f"{job}.json"


def expected_metrics_path(selection_root: Path, job: str) -> Path:
    return selection_root / "task_metrics" / f"{job}__36D.csv"


def output_path(output_dir: Path, job: str) -> Path:
    return output_dir / "task_metrics" / f"{job}.csv"


def detector_config(
    config_path: Path,
    *,
    model: str,
    n_jobs: int,
) -> XGBoostConfig:
    config = load_yaml(config_path)
    values = dict(config["detector"]["xgboost"]["common"])
    values.update(config["detector"]["xgboost"]["by_model"].get(model, {}))
    return replace(
        XGBoostConfig.from_mapping(values),
        random_state=SEED,
        n_jobs=n_jobs,
        verbose=False,
    )


def component_columns(
    columns: list[str],
    specification: AblationSpec,
) -> list[str]:
    if specification.name == "full_36d":
        selected = list(columns)
    elif specification.ablation_family == "layer_pair":
        if specification.removed_component is None:
            raise AssertionError("Layer-pair ablation is missing its component")
        pair = specification.removed_component.strip("()").replace(",", "_")
        selected = [
            column for column in columns if not column.endswith(f"_p{pair}")
        ]
    elif specification.ablation_family == "metric":
        if specification.removed_component is None:
            raise AssertionError("Metric ablation is missing its component")
        selected = [
            column
            for column in columns
            if not column.startswith(f"{specification.removed_component}_")
        ]
    else:
        raise ValueError(f"Unknown ablation family: {specification.ablation_family}")

    expected = {
        "full_36d": 36,
        "without_pair_6_7": 24,
        "without_pair_24_25": 24,
        "without_pair_26_27": 24,
        "without_cos_sim": 27,
        "without_mean_diff": 27,
        "without_std_diff": 27,
        "without_l2_distance": 27,
    }[specification.name]
    if len(selected) != expected:
        raise AssertionError(
            f"{specification.name} expected {expected} features, got {len(selected)}"
        )
    return selected


def validate_feature_schema(columns: list[str]) -> None:
    if len(columns) != 36:
        raise ValueError(f"Expected 36D source features, found {len(columns)}")

    for pair in FULL_PAIRS:
        suffix = f"_p{pair[0]}_{pair[1]}"
        observed = [column for column in columns if column.endswith(suffix)]
        if len(observed) != 12:
            raise ValueError(f"{pair} should contribute 12 features, got {len(observed)}")

    for metric in METRIC_GROUPS:
        observed = [column for column in columns if column.startswith(f"{metric}_")]
        if len(observed) != 9:
            raise ValueError(f"{metric} should contribute 9 features, got {len(observed)}")


def read_feature_frame(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Missing feature file: {path}")
    return add_significant_sdc_target(pd.read_csv(path))


def partition_fit(
    frame: pd.DataFrame,
    *,
    task: TaskSpec,
    manifest: Mapping[str, Any],
) -> dict[str, pd.DataFrame]:
    fit = frame.loc[frame["split"].eq("fit")].copy()
    validate_identity_columns(
        fit,
        group_column=task.group_column,
        sample_uid_column="sample_uid",
    )
    expected_uids = set(map(str, manifest["outer_fit_uids"]))
    observed_uids = set(fit["sample_uid"].astype(str))
    if observed_uids != expected_uids:
        raise ValueError(f"{task.job} outer-Fit identities differ from the manifest")

    split_groups = manifest["split_groups"]
    expected_groups = set().union(
        *(set(groups) for groups in split_groups.values())
    )
    observed_groups = set(fit[task.group_column].astype(str))
    if observed_groups != expected_groups:
        raise ValueError(f"{task.job} outer-Fit groups differ from the manifest")

    result = {
        name: fit.loc[
            fit[task.group_column].astype(str).isin(groups)
        ].copy()
        for name, groups in split_groups.items()
    }
    if sum(len(partition) for partition in result.values()) != len(fit):
        raise AssertionError(f"{task.job} nested partitions are not disjoint")
    return result


def metric_values(
    *,
    target: pd.Series,
    prediction: np.ndarray,
    probability: np.ndarray,
) -> dict[str, int | float]:
    probabilities = np.column_stack((1.0 - probability, probability))
    metrics = binary_metrics(
        target.astype(int),
        prediction,
        probabilities,
    )["target_significant_sdc"]
    return {key: metrics[key] for key in METRIC_KEYS}


def expected_full_results(
    *,
    selection_root: Path,
    task: TaskSpec,
) -> pd.DataFrame:
    path = expected_metrics_path(selection_root, task.job)
    if not path.is_file():
        raise FileNotFoundError(f"Missing selection metrics: {path}")
    expected = pd.read_csv(path)
    expected = expected.loc[
        expected["k"].eq(28)
        & expected["cohort"].eq("full")
    ].copy()
    if len(expected) != 1:
        raise ValueError(
            f"{task.job} expected one Full 36D/K=28 reference row, got "
            f"{len(expected)}"
        )
    return expected.set_index("cohort")


def verify_full_reproduction(
    rows: list[dict[str, Any]],
    *,
    expected: pd.DataFrame,
    task: TaskSpec,
) -> None:
    for row in rows:
        if row["cohort"] != "full":
            continue
        reference = expected.loc[row["cohort"]]
        for key in (
            "precision",
            "recall",
            "f1",
            "false_positive_rate",
            "tp",
            "fp",
            "fn",
            "tn",
        ):
            if not np.isclose(
                float(row[key]),
                float(reference[key]),
                rtol=0.0,
                atol=1e-12,
            ):
                raise AssertionError(
                    f"{task.job} full_36d does not reproduce {key} on "
                    f"{row['cohort']}: {row[key]} != {reference[key]}"
                )


def run_task(
    *,
    config_path: str,
    feature_root: str,
    split_root: str,
    selection_root: str,
    output_dir: str,
    task: TaskSpec,
    xgb_n_jobs: int,
    overwrite: bool,
) -> str:
    output = output_path(Path(output_dir), task.job)
    expected_row_count = len(ABLATIONS) * 2
    if output.is_file() and not overwrite:
        existing = pd.read_csv(output)
        if len(existing) == expected_row_count:
            return f"reuse {task.job}"

    frame = read_feature_frame(feature_path(Path(feature_root), task.job))
    feature_columns = get_feature_columns(frame)
    validate_feature_schema(feature_columns)

    manifest = json.loads(
        manifest_path(Path(split_root), task.job).read_text(encoding="utf-8")
    )
    partitions = partition_fit(frame, task=task, manifest=manifest)
    model_fit_pool = partitions["model_fit_pool"]
    early_stopping = split_by_group(
        model_fit_pool,
        group_column=task.group_column,
        holdout_ratio=EARLY_STOPPING_RATIO,
        random_state=SEED + 2,
    )
    threshold_calibration = partitions["threshold_calibration"]
    selection_validation = partitions["selection_validation"]
    strict_finite = prepare_features(
        selection_validation,
        feature_columns,
    ).notna().all(axis=1)
    if not strict_finite.any():
        raise ValueError(f"{task.job} strict Finite selection subset is empty")
    cohorts = {
        "full": np.ones(len(selection_validation), dtype=bool),
        "strict_finite": strict_finite.to_numpy(),
    }
    config = detector_config(
        Path(config_path),
        model=task.model,
        n_jobs=xgb_n_jobs,
    )
    rows: list[dict[str, Any]] = []

    for specification in ABLATIONS:
        selected = component_columns(feature_columns, specification)
        model, training = train_binary_model(
            prepare_features(early_stopping.train, selected),
            early_stopping.train["significant_sdc_target"].astype(int),
            prepare_features(early_stopping.holdout, selected),
            early_stopping.holdout["significant_sdc_target"].astype(int),
            config=config,
        )
        calibration_probability = model.predict_proba(
            prepare_features(threshold_calibration, selected)
        )[:, 1]
        threshold_info = calibrate_threshold_max_f1(
            calibration_probability,
            threshold_calibration["significant_sdc_target"].astype(int),
        )
        threshold = float(threshold_info["threshold"])
        probability = model.predict_proba(
            prepare_features(selection_validation, selected)
        )[:, 1]
        prediction = (probability > threshold).astype(int)
        ablation_rows: list[dict[str, Any]] = []

        for cohort, mask in cohorts.items():
            values = metric_values(
                target=selection_validation.loc[
                    mask, "significant_sdc_target"
                ],
                prediction=prediction[mask],
                probability=probability[mask],
            )
            row = {
                "job": task.job,
                "model": task.model,
                "dataset": task.dataset,
                "configuration": specification.name,
                "ablation_family": specification.ablation_family,
                "removed_component": specification.removed_component or "",
                "feature_count": len(selected),
                "cohort": cohort,
                "selection_threshold": threshold,
                "best_iteration": training["best_iteration"],
                "model_fit_rows": len(early_stopping.train),
                "early_stopping_rows": len(early_stopping.holdout),
                "threshold_calibration_rows": len(threshold_calibration),
                "selection_validation_rows": int(mask.sum()),
                "strict_finite_reference_feature_count": len(feature_columns),
                **values,
            }
            ablation_rows.append(row)
        if specification.name == "full_36d":
            verify_full_reproduction(
                ablation_rows,
                expected=expected_full_results(
                    selection_root=Path(selection_root),
                    task=task,
                ),
                task=task,
            )
        rows.extend(ablation_rows)

    result = pd.DataFrame(rows).sort_values(["configuration", "cohort"])
    write_csv(output, result)
    return f"completed {task.job}"


def summarize(
    *,
    output_dir: Path,
    tasks: tuple[TaskSpec, ...],
) -> None:
    paths = [output_path(output_dir, task.job) for task in tasks]
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} task-ablation files")
    detailed = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    expected = len(tasks) * len(ABLATIONS) * 2
    if len(detailed) != expected:
        raise ValueError(f"Expected {expected} rows, found {len(detailed)}")
    detailed["removed_component"] = detailed["removed_component"].fillna("")
    detailed = detailed.sort_values(["ablation_family", "configuration", "job", "cohort"])
    write_csv(output_dir / "detailed_metrics.csv", detailed)

    macro = (
        detailed.groupby(
            [
                "configuration",
                "ablation_family",
                "removed_component",
                "feature_count",
                "cohort",
            ],
            as_index=False,
        )
        .agg(
            tasks=("job", "size"),
            precision=("precision", "mean"),
            recall=("recall", "mean"),
            f1=("f1", "mean"),
            false_positive_rate=("false_positive_rate", "mean"),
        )
        .sort_values(["ablation_family", "configuration", "cohort"])
    )
    baseline = macro.loc[
        macro["configuration"].eq("full_36d"),
        ["cohort", "f1"],
    ].rename(columns={"f1": "full_f1"})
    macro = macro.merge(baseline, on="cohort", validate="many_to_one")
    macro["delta_f1_pp"] = 100.0 * (macro["f1"] - macro["full_f1"])
    macro["fpr_percent"] = 100.0 * macro["false_positive_rate"]
    write_csv(output_dir / "macro_metrics.csv", macro)


def main() -> int:
    args = parse_args()
    config_path = args.config.resolve()
    config = load_yaml(config_path)
    configured_jobs = config["featurization"]["jobs"]
    jobs = (
        tuple(item.strip() for item in args.jobs.split(",") if item.strip())
        if args.jobs
        else tuple(configured_jobs)
    )
    unknown = sorted(set(jobs) - set(configured_jobs))
    if unknown:
        raise ValueError(f"Unknown jobs: {unknown}")
    tasks = tuple(
        TaskSpec(
            job=job,
            model=str(configured_jobs[job]["model"]),
            dataset=str(configured_jobs[job]["dataset"]),
        )
        for job in jobs
    )

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.overwrite or not (output_dir / "protocol.json").is_file():
        write_json(
            output_dir / "protocol.json",
            {
                "protocol": (
                    "36D/K=28 component ablation on the outer-Fit-only nested "
                    "split used for configuration selection"
                ),
                "representation": {
                    "layer_pairs": [list(pair) for pair in FULL_PAIRS],
                    "window": "prefix K=28",
                    "metrics": list(METRIC_GROUPS),
                    "statistics": ["mean", "max", "min"],
                },
                "nested_split": {
                    "model_fit_pool": 0.70,
                    "threshold_calibration": 0.15,
                    "selection_validation": 0.15,
                    "early_stopping_within_model_fit_pool": EARLY_STOPPING_RATIO,
                    "seed": SEED,
                },
                "forbidden_partitions": ["outer calibration", "outer final test"],
                "cohorts": {
                    "full": "all selection-validation executions",
                    "strict_finite": (
                        "fixed selection-validation subset whose complete "
                        "36D/K=28 vector has 36 finite features after numeric "
                        "cleaning; thresholds remain calibrated on the complete "
                        "threshold-calibration partition"
                    ),
                },
                "ablations": [asdict(specification) for specification in ABLATIONS],
                "tasks": [asdict(task) for task in tasks],
            },
        )

    work = {
        task.job: task
        for task in tasks
    }
    completed = 0
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                run_task,
                config_path=str(config_path),
                feature_root=str(args.feature_root.resolve()),
                split_root=str(args.split_root.resolve()),
                selection_root=str(args.selection_root.resolve()),
                output_dir=str(output_dir),
                task=task,
                xgb_n_jobs=args.xgb_n_jobs,
                overwrite=args.overwrite,
            ): task.job
            for task in work.values()
        }
        for future in concurrent.futures.as_completed(futures):
            message = future.result()
            completed += 1
            print(f"[{completed}/{len(futures)}] {message}", flush=True)

    summarize(output_dir=output_dir, tasks=tasks)
    print(f"Wrote {output_dir / 'macro_metrics.csv'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
