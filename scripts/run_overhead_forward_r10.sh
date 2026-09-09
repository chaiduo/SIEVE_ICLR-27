#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

run_job() {
    local gpu="$1" python="$2" job="$3"
    "$ROOT/scripts/run_comparison_36d_k28_job.sh" "$gpu" "$python" "$job"
}

run_job 1 "$ROOT/Qwen2.5-VL-7B/.venv/bin/python" qwen25_vl_lingoqa &
P0=$!
run_job 4 "$ROOT/InternVL3-8B/.venv/bin/python" internvl3_lingoqa &
P1=$!
run_job 7 "$ROOT/llava-v1.5-7B/.venv/bin/python" llava15_lingoqa &
P2=$!

status=0
for pid in "$P0" "$P1" "$P2"; do
    wait "$pid" || status=1
done
if [[ "$status" -ne 0 ]]; then
    echo "[overhead-forward-r10] at least one benchmark failed" >&2
    exit "$status"
fi

export PYTHONPATH="$ROOT/src:$ROOT"
"$ROOT/Qwen2.5-VL-7B/.venv/bin/python" \
    "$ROOT/scripts/summarize_online_overhead.py" \
    --input-root "$ROOT/experiments/comparison_36d_k28/overhead_forward_r10" \
    --output-dir "$ROOT/experiments/comparison_36d_k28/overhead_forward_r10/combined" \
    --samples-per-model 50 \
    --warmup-samples 10 \
    --repeats 10 \
    --online-steps 28

echo "[$(date -Is)] forward-only overhead benchmark completed"
