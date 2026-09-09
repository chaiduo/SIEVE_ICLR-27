#!/usr/bin/env python3

"""Select the SIEVE profile and prefix window using outer-Fit data only.

Each task uses the same semantic-group split for every candidate:
  * selection validation: 15% of outer Fit,
  * threshold calibration: 15% of outer Fit,
  * model-fit pool: the remaining 70% of outer Fit.

The model-fit pool is further split group-wise for XGBoost early stopping. The
outer Calibration and Final Test partitions are intentionally never read.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

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


PROFILES = ("36D", "48D", "60D", "72D")
K_VALUES = (1, 2, 4, 8, 12, 16, 20, 24, 28, 32)
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
SELECTION_VALIDATION_RATIO = 0.15
THRESHOLD_CALIBRATION_RATIO = 0.15
MODEL_FIT_RATIO = 0.70
EARLY_STOPPING_RATIO = 0.15
SPLIT_REFERENCE_PROFILE = "72D"
SPLIT_REFERENCE_K = 2


@dataclass(frozen=True)
class TaskSpec:
    job: str
    model: str
    dataset: str
    group_column: str


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Run Fit-only SIEVE profile/window configuration selection."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=root / "configs/experiments/current.yaml",
    )
    parser.add_argument(
        "--feature-root",
        type=Path,
        default=root / "experiments",
        help="Directory containing step_ablation_{36d,48d,60d,72d}.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "experiments/fit_only_configuration_selection",
    )
    parser.add_argument(
        "--jobs",
        default=None,
        help="Comma-separated feature jobs. Defaults to all nine jobs.",
    )
    parser.add_argument(
        "--profiles",
        default=",".join(PROFILES),
        help="Comma-separated profiles from 36D,48D,60D,72D.",
    )
    parser.add_argument(
        "--k-values",
        default=",".join(map(str, K_VALUES)),
        help="Comma-separated prefix windows.",
    )
    parser.add_argument("--seed", type=int, default=20260907)
    parser.add_argument(
        "--workers",
        type=int,
        default=12,
        help="Concurrent job/profile tasks.",
    )
    parser.add_argument(
        "--xgb-n-jobs",
        type=int,
        default=8,
        help="Threads used by one XGBoost fit.",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    if args.xgb_n_jobs < 1:
        parser.error("--xgb-n-jobs must be positive")
    return args


def parse_profiles(value: str) -> tuple[str, ...]:
    profiles = tuple(item.strip() for item in value.split(",") if item.strip())
    unknown = sorted(set(profiles) - set(PROFILES))
    if unknown:
        raise ValueError(f"Unknown profiles: {unknown}")
    if not profiles:
        raise ValueError("At least one profile is required")
    return profiles


def parse_k_values(value: str) -> tuple[int, ...]:
    k_values = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    unknown = sorted(set(k_values) - set(K_VALUES))
    if unknown:
        raise ValueError(f"Unsupported K values: {unknown}")
    if not k_values:
        raise ValueError("At least one K value is required")
    return k_values


def profile_feature_path(
    feature_root: Path,
    *,
    profile: str,
    job: str,
    k: int,
) -> Path:
    return feature_root / f"step_ablation_{profile.lower()}" / job / f"k_{k}" / "features.csv"


def task_metric_path(output_dir: Path, task: TaskSpec, profile: str) -> Path:
    return output_dir / "task_metrics" / f"{task.job}__{profile}.csv"


def task_prediction_path(output_dir: Path, task: TaskSpec, profile: str) -> Path:
    return output_dir / "predictions" / f"{task.job}__{profile}.csv.gz"


def split_manifest_path(output_dir: Path, task: TaskSpec) -> Path:
    return output_dir / "splits" / f"{task.job}.json"


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def write_csv(path: Path, frame: pd.DataFrame, *, compression: str | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False, compression=compression)
    temporary.replace(path)


def read_frame(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Feature file does not exist: {path}")
    frame = add_significant_sdc_target(pd.read_csv(path))
    return frame


def profile_xgb_config(
    config: dict[str, Any],
    *,
    model: str,
    seed: int,
    n_jobs: int,
) -> XGBoostConfig:
    values = dict(config["detector"]["xgboost"]["common"])
    values.update(config["detector"]["xgboost"]["by_model"].get(model, {}))
    return replace(
        XGBoostConfig.from_mapping(values),
        random_state=seed,
        n_jobs=n_jobs,
        verbose=False,
    )


def group_values(frame: pd.DataFrame, group_column: str) -> set[str]:
    return set(frame[group_column].astype(str).tolist())


def build_split_manifest(
    *,
    feature_root: Path,
    output_dir: Path,
    task: TaskSpec,
    seed: int,
    overwrite: bool,
) -> dict[str, Any]:
    path = split_manifest_path(output_dir, task)
    if path.is_file() and not overwrite:
        return json.loads(path.read_text(encoding="utf-8"))

    split_reference_path = profile_feature_path(
        feature_root,
        profile=SPLIT_REFERENCE_PROFILE,
        job=task.job,
        k=SPLIT_REFERENCE_K,
    )
    split_reference = read_frame(split_reference_path)
    validate_identity_columns(
        split_reference,
        group_column=task.group_column,
        sample_uid_column="sample_uid",
    )
    fit = split_reference.loc[split_reference["split"].eq("fit")].copy()
    if fit.empty:
        raise ValueError(f"{task.job} has no outer-Fit rows")

    selection = split_by_group(
        fit,
        group_column=task.group_column,
        holdout_ratio=SELECTION_VALIDATION_RATIO,
        random_state=seed,
    )
    calibration_fraction = (
        THRESHOLD_CALIBRATION_RATIO / (1.0 - SELECTION_VALIDATION_RATIO)
    )
    calibration = split_by_group(
        selection.train,
        group_column=task.group_column,
        holdout_ratio=calibration_fraction,
        random_state=seed + 1,
    )
    split_groups = {
        "model_fit_pool": sorted(group_values(calibration.train, task.group_column)),
        "threshold_calibration": sorted(
            group_values(calibration.holdout, task.group_column)
        ),
        "selection_validation": sorted(
            group_values(selection.holdout, task.group_column)
        ),
    }
    all_groups = set().union(*(set(values) for values in split_groups.values()))
    if all_groups != group_values(fit, task.group_column):
        raise AssertionError(f"{task.job} inner split does not cover outer Fit groups")
    if sum(map(len, split_groups.values())) != len(all_groups):
        raise AssertionError(f"{task.job} inner split groups overlap")

    manifest = {
        "protocol": (
            "outer-Fit-only nested group split: 70% model-fit pool, "
            "15% threshold calibration, 15% configuration-selection validation"
        ),
        "job": task.job,
        "model": task.model,
        "dataset": task.dataset,
        "group_column": task.group_column,
        "seed": seed,
        "split_reference_feature_path": str(split_reference_path),
        "split_groups": split_groups,
        "outer_fit_uids": sorted(fit["sample_uid"].astype(str)),
        "row_counts": {
            "outer_fit": int(len(fit)),
            "model_fit_pool": int(len(calibration.train)),
            "threshold_calibration": int(len(calibration.holdout)),
            "selection_validation": int(len(selection.holdout)),
        },
        "group_counts": {
            "outer_fit": int(fit[task.group_column].nunique()),
            **{name: len(values) for name, values in split_groups.items()},
        },
    }
    write_json(path, manifest)
    return manifest


def partition_frame(
    frame: pd.DataFrame,
    *,
    task: TaskSpec,
    manifest: dict[str, Any],
) -> dict[str, pd.DataFrame]:
    fit = frame.loc[frame["split"].eq("fit")].copy()
    validate_identity_columns(
        fit,
        group_column=task.group_column,
        sample_uid_column="sample_uid",
    )
    groups = manifest["split_groups"]
    expected_groups = set().union(*(set(values) for values in groups.values()))
    actual_groups = group_values(fit, task.group_column)
    if actual_groups != expected_groups:
        missing = sorted(expected_groups - actual_groups)[:3]
        extra = sorted(actual_groups - expected_groups)[:3]
        raise ValueError(
            f"{task.job} feature identity differs from the canonical Fit split; "
            f"missing_groups={missing}, extra_groups={extra}"
        )
    expected_uids = set(manifest["outer_fit_uids"])
    actual_uids = set(fit["sample_uid"].astype(str))
    if actual_uids != expected_uids:
        missing = sorted(expected_uids - actual_uids)[:3]
        extra = sorted(actual_uids - expected_uids)[:3]
        raise ValueError(
            f"{task.job} feature rows differ from canonical outer Fit; "
            f"missing_uids={missing}, extra_uids={extra}"
        )

    result = {
        name: fit.loc[fit[task.group_column].astype(str).isin(values)].copy()
        for name, values in groups.items()
    }
    if sum(len(item) for item in result.values()) != len(fit):
        raise AssertionError(f"{task.job} split rows overlap or were dropped")
    return result


def metric_values(
    *,
    target: pd.Series,
    prediction: np.ndarray,
    probability: np.ndarray,
) -> dict[str, int | float]:
    probabilities = np.column_stack((1.0 - probability, probability))
    metrics = binary_metrics(target.astype(int), prediction, probabilities)[
        "target_significant_sdc"
    ]
    return {key: metrics[key] for key in METRIC_COLUMNS}


def run_task(
    *,
    feature_root: str,
    output_dir: str,
    config_path: str,
    task: TaskSpec,
    profile: str,
    k_values: tuple[int, ...],
    seed: int,
    xgb_n_jobs: int,
    overwrite: bool,
) -> str:
    root = Path(feature_root)
    destination = Path(output_dir)
    metrics_path = task_metric_path(destination, task, profile)
    predictions_path = task_prediction_path(destination, task, profile)
    expected_rows = len(k_values)
    if (
        metrics_path.is_file()
        and predictions_path.is_file()
        and not overwrite
        and len(pd.read_csv(metrics_path)) == expected_rows
    ):
        return f"reuse {task.job} {profile}"

    manifest = json.loads(
        split_manifest_path(destination, task).read_text(encoding="utf-8")
    )
    config = load_yaml(Path(config_path))
    xgb_config = profile_xgb_config(
        config,
        model=task.model,
        seed=seed,
        n_jobs=xgb_n_jobs,
    )
    metric_rows: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []

    for k in k_values:
        path = profile_feature_path(root, profile=profile, job=task.job, k=k)
        frame = read_frame(path)
        partitions = partition_frame(frame, task=task, manifest=manifest)
        features = get_feature_columns(frame)
        model_fit_pool = partitions["model_fit_pool"]
        early_stopping = split_by_group(
            model_fit_pool,
            group_column=task.group_column,
            holdout_ratio=EARLY_STOPPING_RATIO,
            random_state=seed + 2,
        )
        model, training = train_binary_model(
            prepare_features(early_stopping.train, features),
            early_stopping.train["significant_sdc_target"].astype(int),
            prepare_features(early_stopping.holdout, features),
            early_stopping.holdout["significant_sdc_target"].astype(int),
            config=xgb_config,
        )
        calibration = partitions["threshold_calibration"]
        calibration_probability = model.predict_proba(
            prepare_features(calibration, features)
        )[:, 1]
        threshold_info = calibrate_threshold_max_f1(
            calibration_probability,
            calibration["significant_sdc_target"].astype(int),
        )
        threshold = float(threshold_info["threshold"])

        validation = partitions["selection_validation"]
        probability = model.predict_proba(prepare_features(validation, features))[:, 1]
        prediction = (probability > threshold).astype(int)
        cohorts = {
            "full": np.ones(len(validation), dtype=bool),
        }
        for cohort, mask in cohorts.items():
            selected_target = validation.loc[mask, "significant_sdc_target"]
            selected_prediction = prediction[mask]
            selected_probability = probability[mask]
            values = metric_values(
                target=selected_target,
                prediction=selected_prediction,
                probability=selected_probability,
            )
            metric_rows.append(
                {
                    "job": task.job,
                    "model": task.model,
                    "dataset": task.dataset,
                    "profile": profile,
                    "k": k,
                    "feature_count": len(features),
                    "cohort": cohort,
                    "selection_threshold": threshold,
                    "calibration_precision": threshold_info["calibration_precision"],
                    "calibration_recall": threshold_info["calibration_recall"],
                    "calibration_f1": threshold_info["calibration_f1"],
                    "calibration_fpr": threshold_info["calibration_fpr"],
                    "best_iteration": training["best_iteration"],
                    "model_train_rows": len(early_stopping.train),
                    "early_stopping_rows": len(early_stopping.holdout),
                    "threshold_calibration_rows": len(calibration),
                    "selection_validation_rows": int(mask.sum()),
                    **values,
                }
            )
        prediction_frames.append(
            pd.DataFrame(
                {
                    "job": task.job,
                    "model": task.model,
                    "dataset": task.dataset,
                    "profile": profile,
                    "k": k,
                    "sample_uid": validation["sample_uid"].astype(str).to_numpy(),
                    "semantic_group_id": validation[task.group_column].astype(str).to_numpy(),
                    "significant_sdc_target": validation[
                        "significant_sdc_target"
                    ].astype(int).to_numpy(),
                    "probability": probability,
                    "threshold": threshold,
                    "prediction": prediction,
                }
            )
        )

    metrics = pd.DataFrame(metric_rows).sort_values(["k", "cohort"])
    predictions = pd.concat(prediction_frames, ignore_index=True)
    write_csv(metrics_path, metrics)
    write_csv(predictions_path, predictions, compression="gzip")
    return f"completed {task.job} {profile}"


def summarize(
    *,
    output_dir: Path,
    tasks: tuple[TaskSpec, ...],
    profiles: tuple[str, ...],
    k_values: tuple[int, ...],
) -> None:
    expected_task_files = [
        task_metric_path(output_dir, task, profile)
        for task in tasks
        for profile in profiles
    ]
    missing = [path for path in expected_task_files if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} task metrics files")

    detailed = pd.concat(
        [pd.read_csv(path) for path in expected_task_files],
        ignore_index=True,
    ).sort_values(["profile", "k", "job", "cohort"])
    expected_rows = len(tasks) * len(profiles) * len(k_values)
    if len(detailed) != expected_rows:
        raise ValueError(
            f"Expected {expected_rows} detailed rows, found {len(detailed)}"
        )
    write_csv(output_dir / "detailed_metrics.csv", detailed)

    macro = (
        detailed.groupby(
            ["profile", "k", "feature_count", "cohort"],
            as_index=False,
        )
        .agg(
            tasks=("job", "size"),
            precision=("precision", "mean"),
            recall=("recall", "mean"),
            f1=("f1", "mean"),
            false_positive_rate=("false_positive_rate", "mean"),
        )
        .sort_values(["cohort", "profile", "k"])
    )
    macro["fpr_percent"] = 100.0 * macro["false_positive_rate"]
    write_csv(output_dir / "macro_metrics.csv", macro)

    full = macro.loc[macro["cohort"].eq("full")].set_index(["profile", "k"])
    rows: list[dict[str, Any]] = []
    for profile, k in full.index:
        full_row = full.loc[(profile, k)]
        rows.append(
            {
                "profile": profile,
                "k": int(k),
                "feature_count": int(full_row["feature_count"]),
                "full_f1": float(full_row["f1"]),
                "full_recall": float(full_row["recall"]),
                "full_fpr": float(full_row["false_positive_rate"]),
            }
        )
    ranking = pd.DataFrame(rows).sort_values(
        [
            "full_f1",
            "full_recall",
            "k",
            "feature_count",
        ],
        ascending=[False, False, True, True],
    )
    ranking.insert(0, "rank", np.arange(1, len(ranking) + 1))
    write_csv(output_dir / "configuration_ranking.csv", ranking)
    winner = ranking.iloc[0].to_dict()
    write_json(
        output_dir / "selection.json",
        {
            "primary_selection_metric": "nine-task Full macro Significant-SDC F1",
            "tie_breakers": [
                "Full macro recall",
                "smaller K",
                "fewer features",
            ],
            "selected_configuration": winner,
            "ranking_path": str(output_dir / "configuration_ranking.csv"),
            "macro_metrics_path": str(output_dir / "macro_metrics.csv"),
        },
    )


def main() -> int:
    args = parse_args()
    config_path = args.config.resolve()
    config = load_yaml(config_path)
    all_jobs = config["featurization"]["jobs"]
    requested_jobs = (
        tuple(item.strip() for item in args.jobs.split(",") if item.strip())
        if args.jobs
        else tuple(all_jobs)
    )
    unknown_jobs = sorted(set(requested_jobs) - set(all_jobs))
    if unknown_jobs:
        raise ValueError(f"Unknown jobs: {unknown_jobs}")
    profiles = parse_profiles(args.profiles)
    k_values = parse_k_values(args.k_values)
    tasks = tuple(
        TaskSpec(
            job=job,
            model=str(all_jobs[job]["model"]),
            dataset=str(all_jobs[job]["dataset"]),
            group_column="semantic_group_id",
        )
        for job in requested_jobs
    )
    feature_root = args.feature_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    protocol_path = output_dir / "protocol.json"
    if not protocol_path.is_file() or args.overwrite:
        write_json(
            protocol_path,
            {
                "protocol": "outer-Fit-only configuration selection",
                "profiles": list(profiles),
                "k_values": list(k_values),
                "jobs": [asdict(task) for task in tasks],
                "seed": args.seed,
                "selection_validation_ratio_of_outer_fit": SELECTION_VALIDATION_RATIO,
                "threshold_calibration_ratio_of_outer_fit": THRESHOLD_CALIBRATION_RATIO,
                "model_fit_pool_ratio_of_outer_fit": MODEL_FIT_RATIO,
                "early_stopping_ratio_of_model_fit_pool": EARLY_STOPPING_RATIO,
                "split_reference": (
                    "72D/K=2 feature file used only to create shared "
                    "outer-Fit group manifests"
                ),
                "forbidden_partitions": ["outer calibration", "outer final test"],
                "selection_metric": "nine-task Full macro Significant-SDC F1",
                "feature_root": str(feature_root),
            },
        )

    for task in tasks:
        build_split_manifest(
            feature_root=feature_root,
            output_dir=output_dir,
            task=task,
            seed=args.seed,
            overwrite=args.overwrite,
        )

    work_items = [
        (task, profile)
        for task in tasks
        for profile in profiles
    ]
    completed = 0
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                run_task,
                feature_root=str(feature_root),
                output_dir=str(output_dir),
                config_path=str(config_path),
                task=task,
                profile=profile,
                k_values=k_values,
                seed=args.seed,
                xgb_n_jobs=args.xgb_n_jobs,
                overwrite=args.overwrite,
            ): (task, profile)
            for task, profile in work_items
        }
        for future in concurrent.futures.as_completed(futures):
            task, _profile = futures[future]
            message = future.result()
            completed += 1
            print(
                f"[{completed}/{len(work_items)}] {message}",
                flush=True,
            )

    summarize(
        output_dir=output_dir,
        tasks=tasks,
        profiles=profiles,
        k_values=k_values,
    )
    winner = json.loads((output_dir / "selection.json").read_text(encoding="utf-8"))
    print(json.dumps(winner, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
