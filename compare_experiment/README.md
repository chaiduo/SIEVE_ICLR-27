# Detection Method Comparison

This directory evaluates Ranger-style activation-range monitoring,
Dr.DNA-style distribution monitoring, and SIEVE on the same executions.

## Current Protocol

The frozen comparison uses the final SIEVE configuration:

- layer pairs `(6,7)`, `(24,25)`, and `(26,27)`;
- the first `min(28, T)` observed decoding steps;
- identical double-bit fault identities and Fit/Calibration/Final Test splits;
- method-specific F1-maximizing Calibration thresholds;
- Full and canonical-finite cohorts with image-cluster bootstrap confidence
  intervals.

Ranger-style and Dr.DNA-style are mechanism-matched PyTorch/VLM adaptations,
not reproductions of the original systems.

## Execution

The eight-GPU launcher profiles both baselines, collects nine 55,000-execution
campaigns, validates record identity against `telemetry_50`, evaluates all
methods, and summarizes forward-only LingoQA overhead measurements with 10
warmups and 10 repeats.

```bash
tmux new-session -d -s sieve_comparison_36d_k28 \
  'cd /data01/cd_workspace/Detect_SDC && bash scripts/run_comparison_36d_k28_all.sh'
```

The active configuration is
`compare_experiment/configs/detection_comparison_k28_36d.yaml`. Canonical
telemetry, clean profiles, and Predictor checkpoints reside under
`experiments/telemetry_50/<job>/`; results are written to
`experiments/comparison_36d_k28/`.
