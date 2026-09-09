# Figure Sources

Historical portable figure manifests were removed with the superseded 72D/K=4
and 48D experiments. Current paper figures are generated from:

| Figure family | Source |
|---|---|
| Predictor discrepancy patterns | `scripts/plot_sdc_cosine_by_layer_pair.py` |
| Configuration-selection matrix | `experiments/fit_only_configuration_selection/` |
| 36D window sensitivity | `scripts/plot_36d_window_ablation.py` |
| Component ablation | `scripts/plot_36d_k28_component_ablation.py` |
| Fault value deviation | `scripts/summarize_fault_deviation_buckets.py` |
| Method comparison and overhead | `experiments/comparison_36d_k28/` |

The raw 50-step telemetry, clean profiles, and Predictor checkpoints are local
under `experiments/telemetry_50/`. Compact figure source data will be exported
after the active 36D/K=28 method-comparison and overhead campaign completes.
