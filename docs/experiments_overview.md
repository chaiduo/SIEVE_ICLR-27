# Detect_SDC 实验总览

本文档仅索引当前 SIEVE 36D/`K=28` 实验。历史 72D/K=4、48D 和旧 telemetry
分析已清理，不得用于投稿主表。

## 实验矩阵

评估覆盖 Qwen2.5-VL-7B、InternVL3-8B、LLaVA-1.5-7B 与 EarthVQA、LingoQA、
VQAv2 的九个模型-数据集任务。每个任务包含 5,000 次无故障执行与 50,000 次
随机双比特故障注入。共享原始图像的样本整体划分为 Fit、Calibration 与 Final
Test（70%/15%/15%）。

## 当前配置

最终 SIEVE 配置为三组相邻层对：

```text
(6,7), (24,25), (26,27)
```

每组计算 CosSim、Mean Difference、Std Difference 和 L2 Distance，并对前
`min(28, T)` 个实际 decoding steps 进行 mean/max/min 聚合，得到 36 维表示。
40 组 feature-dimension/window 候选仅在 outer-Fit 内部选择；Calibration 与
Final Test 不参与选择。

## 当前产物

| 内容 | 位置 |
|---|---|
| 原始 telemetry、clean profile、Predictor checkpoint | `experiments/telemetry_50/<job>/` |
| 36D/K=28 特征与 Detector | `experiments/step_ablation_36d/<job>/k_28/` |
| Fit-only 配置选择 | `experiments/fit_only_configuration_selection/` |
| Layer-pair 和 metric 消融 | `experiments/ablation_36d_k28_fit_only/` |
| 附录统计 | `experiments/appendix_36d_k28/` |
| Ranger-style、Dr.DNA-style 与开销 | `experiments/comparison_36d_k28/` |

## 复现入口

验证配置：

```bash
PYTHONPATH=src python -m detect_sdc.cli config validate \
  configs/experiments/current.yaml
```

运行公平方法比较：

```bash
tmux new-session -d -s sieve_comparison_36d_k28 \
  'cd /data01/cd_workspace/Detect_SDC && bash scripts/run_comparison_36d_k28_all.sh'
```

方法对比、开销和置信区间的协议见
[`../compare_experiment/README.md`](../compare_experiment/README.md)，完整可复现
设置见 [`reproducibility.md`](reproducibility.md)。
