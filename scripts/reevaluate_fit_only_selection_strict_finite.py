#!/usr/bin/env python3
"""Score saved Fit-only configuration predictions on a fixed strict-Finite cohort.

The configuration ranking remains Full-only. This script is a post-selection
diagnostic: every candidate is scored on the same subset, defined by the
frozen 36D/K=28 representation having all 36 cleaned features finite. It
reuses saved detector probabilities and thresholds, so it never retrains a
model, recalibrates a threshold, or reads outer Calibration/Final Test rows.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from detect_sdc.config import load_yaml
from detect_sdc.detector.xgboost import (
    binary_metrics,
    get_feature_columns,
    prepare_features,
)


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


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description=(
            "Re-evaluate saved Fit-only configuration predictions on the "
            "fixed strict 36D/K=28 Finite cohort."
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
        default=root / "experiments",
    )
    parser.add_argument(
        "--selection-root",
        type=Path,
        default=root / "experiments/fit_only_configuration_selection",
    )
    parser.add_argument(
        "--profiles",
        default=",".join(PROFILES),
    )
    parser.add_argument(
        "--k-values",
        default=",".join(map(str, K_VALUES)),
    )
    parser.add_argument(
        "--jobs",
        default=None,
        help="Comma-separated jobs. Defaults to all configured jobs.",
    )
    return parser.parse_args()


def parse_profiles(value: str) -> tuple[str, ...]:
    profiles = tuple(item.strip() for item in value.split(",") if item.strip())
    unknown = sorted(set(profiles) - set(PROFILES))
    if unknown or not profiles:
        raise ValueError(f"Invalid profiles: {unknown or 'empty'}")
    return profiles


def parse_k_values(value: str) -> tuple[int, ...]:
    k_values = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    unknown = sorted(set(k_values) - set(K_VALUES))
    if unknown or not k_values:
        raise ValueError(f"Invalid K values: {unknown or 'empty'}")
    return k_values


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


def strict_finite_uids(
    *,
    feature_root: Path,
    selection_root: Path,
    job: str,
) -> tuple[set[str], int]:
    feature_path = (
        feature_root / "step_ablation_36d" / job / "k_28" / "features.csv"
    )
    frame = pd.read_csv(feature_path)
    manifest_path = selection_root / "splits" / f"{job}.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    groups = set(manifest["split_groups"]["selection_validation"])
    validation = frame.loc[
        frame["split"].eq("fit")
        & frame["semantic_group_id"].astype(str).isin(groups)
    ].copy()
    expected_rows = int(manifest["row_counts"]["selection_validation"])
    if len(validation) != expected_rows:
        raise ValueError(
            f"{job} expected {expected_rows} selection-validation rows, "
            f"found {len(validation)}"
        )
    features = get_feature_columns(frame)
    if len(features) != 36:
        raise ValueError(f"{job} expected 36 strict-reference features")
    strict_mask = prepare_features(validation, features).notna().all(axis=1)
    uids = set(validation.loc[strict_mask, "sample_uid"].astype(str))
    if not uids:
        raise ValueError(f"{job} strict-Finite selection subset is empty")
    return uids, expected_rows


def metric_values(scores: pd.DataFrame) -> dict[str, int | float]:
    probability = scores["probability"].to_numpy()
    target = scores["significant_sdc_target"].astype(int)
    probabilities = pd.DataFrame(
        {
            "negative": 1.0 - probability,
            "positive": probability,
        }
    ).to_numpy()
    metrics = binary_metrics(
        target,
        scores["prediction"].astype(int).to_numpy(),
        probabilities,
    )["target_significant_sdc"]
    return {key: metrics[key] for key in METRIC_COLUMNS}


def main() -> int:
    args = parse_args()
    config = load_yaml(args.config.resolve())
    all_jobs = config["featurization"]["jobs"]
    jobs = (
        tuple(item.strip() for item in args.jobs.split(",") if item.strip())
        if args.jobs
        else tuple(all_jobs)
    )
    unknown_jobs = sorted(set(jobs) - set(all_jobs))
    if unknown_jobs:
        raise ValueError(f"Unknown jobs: {unknown_jobs}")
    profiles = parse_profiles(args.profiles)
    k_values = parse_k_values(args.k_values)
    feature_root = args.feature_root.resolve()
    selection_root = args.selection_root.resolve()

    strict_uids_by_job: dict[str, set[str]] = {}
    validation_rows_by_job: dict[str, int] = {}
    for job in jobs:
        strict_uids, validation_rows = strict_finite_uids(
            feature_root=feature_root,
            selection_root=selection_root,
            job=job,
        )
        strict_uids_by_job[job] = strict_uids
        validation_rows_by_job[job] = validation_rows

    rows: list[dict[str, Any]] = []
    for job in jobs:
        job_config = all_jobs[job]
        strict_uids = strict_uids_by_job[job]
        for profile in profiles:
            prediction_path = selection_root / "predictions" / f"{job}__{profile}.csv.gz"
            scores = pd.read_csv(prediction_path)
            for k in k_values:
                candidate = scores.loc[scores["k"].eq(k)].copy()
                candidate_uids = set(candidate["sample_uid"].astype(str))
                if len(candidate) != validation_rows_by_job[job]:
                    raise ValueError(
                        f"{job}/{profile}/K={k} expected "
                        f"{validation_rows_by_job[job]} score rows, "
                        f"found {len(candidate)}"
                    )
                if not strict_uids.issubset(candidate_uids):
                    raise ValueError(
                        f"{job}/{profile}/K={k} is missing strict-Finite score rows"
                    )
                strict_scores = candidate.loc[
                    candidate["sample_uid"].astype(str).isin(strict_uids)
                ].copy()
                values = metric_values(strict_scores)
                rows.append(
                    {
                        "job": job,
                        "model": str(job_config["model"]),
                        "dataset": str(job_config["dataset"]),
                        "profile": profile,
                        "k": k,
                        "feature_count": int(profile.removesuffix("D")),
                        "cohort": "strict_finite",
                        "selection_validation_rows": len(strict_scores),
                        "strict_reference_feature_count": 36,
                        **values,
                    }
                )

    detailed = pd.DataFrame(rows).sort_values(["profile", "k", "job"])
    expected_rows = len(jobs) * len(profiles) * len(k_values)
    if len(detailed) != expected_rows:
        raise AssertionError(f"Expected {expected_rows} rows, found {len(detailed)}")
    detailed_path = selection_root / "strict_finite_detailed_metrics.csv"
    write_csv(detailed_path, detailed)

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
        .sort_values(["profile", "k"])
    )
    macro["fpr_percent"] = 100.0 * macro["false_positive_rate"]
    macro_path = selection_root / "strict_finite_macro_metrics.csv"
    write_csv(macro_path, macro)
    write_json(
        selection_root / "strict_finite_protocol.json",
        {
            "purpose": (
                "post-selection diagnostic only; not used for configuration "
                "ranking or tie-breaking"
            ),
            "strict_finite_definition": (
                "fixed selection-validation subset whose complete final "
                "36D/K=28 vector has 36 finite cleaned features"
            ),
            "reference_feature_path_pattern": str(
                feature_root / "step_ablation_36d" / "{job}" / "k_28" / "features.csv"
            ),
            "prediction_source_pattern": str(
                selection_root / "predictions" / "{job}__{profile}.csv.gz"
            ),
            "training_or_recalibration": "none; reused saved probabilities and thresholds",
            "profiles": list(profiles),
            "k_values": list(k_values),
            "jobs": list(jobs),
            "detailed_metrics_path": str(detailed_path),
            "macro_metrics_path": str(macro_path),
        },
    )
    print(f"wrote {detailed_path}")
    print(f"wrote {macro_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
