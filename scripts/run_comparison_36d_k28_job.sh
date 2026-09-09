#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
    echo "usage: $0 GPU PYTHON JOB" >&2
    exit 2
fi

GPU="$1"
PYTHON="$2"
JOB="$3"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="$ROOT/compare_experiment/configs/detection_comparison_k28_36d.yaml"
RESULT="$ROOT/experiments/comparison_36d_k28/results/$JOB"
REFERENCE="$ROOT/experiments/telemetry_50/$JOB/labels.jsonl"
FEATURE_ROOT="$ROOT/experiments/step_ablation_36d/$JOB/k_28"
FINITE_REFERENCE="$FEATURE_ROOT/features.csv"
PROFILE="$RESULT/profiles.json"
RECORDS="$RESULT/score_records.jsonl"
VALIDATION="$RESULT/record_validation.json"
EVALUATION="$RESULT/evaluation_strict_finite"
LOG="$RESULT/job.log"

mkdir -p "$RESULT"
exec > >(tee -a "$LOG") 2>&1
export CUDA_VISIBLE_DEVICES="$GPU"
export GPU_PHYSICAL_ID="$GPU"
export PYTHONPATH="$ROOT/src:$ROOT"
export CUBLAS_WORKSPACE_CONFIG=":4096:8"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
cd "$ROOT"

echo "[$(date -Is)] start job=$JOB profile=36D K=28 gpu=$GPU"

if [[ ! -f "$PROFILE" ]]; then
    "$PYTHON" -m compare_experiment.profile_baselines \
        --job "$JOB" \
        --comparison-config "$CONFIG" \
        --device cuda:0 \
        --output "$PROFILE" \
        --overwrite
fi

if [[ ! -f "$RECORDS" ]]; then
    "$PYTHON" -m compare_experiment.collect_detection_data \
        --job "$JOB" \
        --comparison-config "$CONFIG" \
        --device cuda:0 \
        --telemetry-max-steps 28 \
        --profiles "$PROFILE" \
        --output "$RECORDS" \
        --skip-sieve-telemetry \
        --overwrite
fi

if [[ ! -f "$VALIDATION" ]]; then
    "$PYTHON" -m compare_experiment.validate_score_records \
        --candidate "$RECORDS" \
        --reference "$REFERENCE" \
        --output "$VALIDATION"
fi

if [[ ! -f "$EVALUATION/metrics.json" ]]; then
    "$PYTHON" -m compare_experiment.evaluate_results \
        --job "$JOB" \
        --comparison-config "$CONFIG" \
        --records "$RECORDS" \
        --detector-summary "$FEATURE_ROOT/output/metrics_summary.json" \
        --calibration-features "$FEATURE_ROOT/calibration.csv" \
        --test-features "$FEATURE_ROOT/features.csv" \
        --finite-cohort-features "$FINITE_REFERENCE" \
        --output-dir "$EVALUATION"
fi

if [[ "$JOB" == *_lingoqa ]]; then
    MODEL="${JOB%_lingoqa}"
    OUT="$ROOT/experiments/comparison_36d_k28/overhead_forward_r10/$MODEL"
    COMPLETE=false
    if [[ -f "$OUT/summary.json" ]]; then
        if jq -e \
            '.modes_completed | sort == ["drdna", "ranger", "sieve", "vanilla"]' \
            "$OUT/summary.json" >/dev/null \
            && [[ "$(jq -r '.warmup_samples' "$OUT/summary.json")" == "10" ]] \
            && [[ "$(jq -r '.repeats' "$OUT/summary.json")" == "10" ]] \
            && [[ "$(jq -r '.mode_order' "$OUT/summary.json")" == "forward" ]]; then
            COMPLETE=true
        fi
    fi
    if [[ "$COMPLETE" != true ]]; then
        rm -rf "$OUT"
        "$PYTHON" "$ROOT/scripts/benchmark_online_overhead.py" \
            --job "$JOB" \
            --comparison-config "$CONFIG" \
            --profiles "$PROFILE" \
            --detector-summary "$FEATURE_ROOT/output/metrics_summary.json" \
            --device cuda:0 \
            --samples 50 \
            --warmup-samples 10 \
            --repeats 10 \
            --online-steps 28 \
            --modes vanilla ranger drdna sieve \
            --mode-order forward \
            --output-root "$OUT"
    fi
fi

echo "[$(date -Is)] completed job=$JOB profile=36D K=28"
