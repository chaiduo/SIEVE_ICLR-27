#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
mkdir -p experiments/comparison_36d_k28/results

run_job() {
    local gpu="$1" python="$2" job="$3"
    "$ROOT/scripts/run_comparison_36d_k28_job.sh" "$gpu" "$python" "$job"
}

run_job 0 "$ROOT/Qwen2.5-VL-7B/.venv/bin/python" qwen25_vl_earthvqa & P0=$!
run_job 1 "$ROOT/Qwen2.5-VL-7B/.venv/bin/python" qwen25_vl_lingoqa & P1=$!
run_job 2 "$ROOT/Qwen2.5-VL-7B/.venv/bin/python" qwen25_vl_vqav2 & P2=$!
run_job 3 "$ROOT/InternVL3-8B/.venv/bin/python" internvl3_earthvqa & P3=$!
run_job 4 "$ROOT/InternVL3-8B/.venv/bin/python" internvl3_lingoqa & P4=$!
run_job 5 "$ROOT/InternVL3-8B/.venv/bin/python" internvl3_vqav2 & P5=$!
(
    run_job 6 "$ROOT/llava-v1.5-7B/.venv/bin/python" llava15_vqav2
    run_job 6 "$ROOT/llava-v1.5-7B/.venv/bin/python" llava15_earthvqa
) & P6=$!
run_job 7 "$ROOT/llava-v1.5-7B/.venv/bin/python" llava15_lingoqa & P7=$!

status=0
for pid in "$P0" "$P1" "$P2" "$P3" "$P4" "$P5" "$P6" "$P7"; do
    wait "$pid" || status=1
done
if [[ "$status" -ne 0 ]]; then
    echo "[comparison-36d-k28] at least one job failed" >&2
    exit "$status"
fi

export PYTHONPATH="$ROOT/src:$ROOT"
PYTHON="$ROOT/Qwen2.5-VL-7B/.venv/bin/python"
CONFIG="$ROOT/compare_experiment/configs/detection_comparison_k28_36d.yaml"
"$PYTHON" -m compare_experiment.summarize_results \
    --comparison-config "$CONFIG" \
    --output-dir "$ROOT/experiments/comparison_36d_k28/results/summary"
"$PYTHON" "$ROOT/scripts/summarize_online_overhead.py" \
    --input-root "$ROOT/experiments/comparison_36d_k28/overhead_forward_r10" \
    --output-dir "$ROOT/experiments/comparison_36d_k28/overhead_forward_r10/combined" \
    --samples-per-model 50 \
    --warmup-samples 10 \
    --repeats 10 \
    --online-steps 28
echo "[$(date -Is)] all 36D/K=28 comparison jobs completed"
