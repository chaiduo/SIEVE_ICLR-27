#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="$ROOT/compare_experiment/configs/detection_comparison_k28_36d.yaml"
RESULT_ROOT="$ROOT/experiments/comparison_36d_k28/results"
PYTHON="$ROOT/Qwen2.5-VL-7B/.venv/bin/python"
JOBS=(
    qwen25_vl_earthvqa
    qwen25_vl_lingoqa
    qwen25_vl_vqav2
    internvl3_earthvqa
    internvl3_lingoqa
    internvl3_vqav2
    llava15_earthvqa
    llava15_lingoqa
    llava15_vqav2
)

export PYTHONPATH="$ROOT/src:$ROOT"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

run_job() {
    local job="$1"
    local detector_root="$ROOT/experiments/step_ablation_36d/$job/k_28"
    local finite_reference="$detector_root/features.csv"

    "$PYTHON" -m compare_experiment.evaluate_results \
        --job "$job" \
        --comparison-config "$CONFIG" \
        --records "$RESULT_ROOT/$job/score_records.jsonl" \
        --detector-summary "$detector_root/output/metrics_summary.json" \
        --calibration-features "$detector_root/calibration.csv" \
        --test-features "$detector_root/features.csv" \
        --finite-cohort-features "$finite_reference" \
        --output-dir "$RESULT_ROOT/$job/evaluation_strict_finite"
}

pids=()
for job in "${JOBS[@]}"; do
    run_job "$job" >"$RESULT_ROOT/$job/evaluation_strict_finite.log" 2>&1 &
    pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
    wait "$pid" || status=1
done
if [[ "$status" -ne 0 ]]; then
    echo "[strict-finite] at least one evaluation failed" >&2
    exit "$status"
fi

"$PYTHON" -m compare_experiment.summarize_results \
    --comparison-config "$CONFIG" \
    --evaluation-dir-name evaluation_strict_finite \
    --output-dir "$RESULT_ROOT/summary_strict_finite"

echo "[$(date -Is)] strict-Finite comparison re-evaluation completed"
