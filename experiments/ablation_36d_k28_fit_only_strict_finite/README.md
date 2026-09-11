# 36D/K=28 组件消融

[返回总手册](../README.md) | [配置选择协议](../fit_only_configuration_selection/README.md)

这是当前正式组件消融目录。路径保留 `strict_finite`，展示名称使用 Finite。

![Component ablation](component_ablation_36d_k28.png)

## 1. 实验问题

在固定 36D/K=28 配置中，单独删除某个 layer pair 或某类差异指标，性能如何变化？

每个变体重新训练 Detector、重新做内层阈值校准，**不是**训练一次后将某列置零，也不是仅重新预测。

## 2. 变体定义

| 机器 key | 删除组件 | 保留维度 |
| --- | --- | ---: |
| `full_36d` | 不删除，36D all components | 36 |
| `without_pair_6_7` | `(6,7)` 的全部 12 列 | 24 |
| `without_pair_24_25` | `(24,25)` 的全部 12 列 | 24 |
| `without_pair_26_27` | `(26,27)` 的全部 12 列 | 24 |
| `without_cos_sim` | 三个 pair 的 CosSim mean/max/min | 27 |
| `without_mean_diff` | 三个 pair 的 MeanDiff mean/max/min | 27 |
| `without_std_diff` | 三个 pair 的 StdDiff mean/max/min | 27 |
| `without_l2_distance` | 三个 pair 的 L2 mean/max/min | 27 |

`full_36d` 表示组件完整，不是 cohort。它在 Full 和 Finite 两个 cohort 上均有结果。图中的 “Full 36D” 标签也应按“全部组件”理解。

## 3. 数据划分与可比性

输入为 `step_ablation_36d/<job>/k_28/features.csv` 的 outer-Fit 行。复用配置选择的 `splits/<job>.json`：

- 70% model-fit pool，内部 15% early stopping；
- 15% threshold calibration；
- 15% selection validation，用于消融评分。

不使用 outer Calibration/Test 指标。seed 为 `20260907`，early-stop split 为 seed+2。

Finite mask 用完整 36D/K=28 的 selection-validation 表计算，所有八个变体完全共享。删除导致非有限的某一列后，不能把这条样本重新纳入 Finite。

训练与 calibration 仍保留完整分区；只有评分被分为 Full 和 Finite。共九任务 × 八变体 = 72 次 Detector 训练。

## 4. 当前结果

| 变体 | Full F1 | 相对完整配置变化 pp | Finite F1 |
| --- | ---: | ---: | ---: |
| 36D all components | 93.42% | 0.00 | 71.49% |
| w/o (6,7) | 92.26% | -1.17 | 68.82% |
| w/o (24,25) | 91.26% | -2.16 | 66.78% |
| w/o (26,27) | 90.95% | -2.47 | 65.86% |
| w/o CosSim | 91.47% | -1.96 | 64.80% |
| w/o MeanDiff | 89.68% | -3.74 | 65.29% |
| w/o StdDiff | 92.96% | -0.46 | 70.30% |
| w/o L2 | 92.02% | -1.40 | 68.99% |

来源：[macro_metrics.csv](macro_metrics.csv)。结论仅为“九任务宏平均上，移除任一组件降低 Full F1”；不意味着每个单任务必然下降。

93.42% 是 Fit-only 基线，不应该拿 Final Test 的 93.78% 作为计算这些下降值的分母或参照。

## 5. 文件说明

| 文件 | 内容 |
| --- | --- |
| [protocol.json](protocol.json) | 变体、pair、指标、窗口、内部 split、cohort |
| `task_metrics/<job>.csv` | 每任务 8×2=16 行 |
| [detailed_metrics.csv](detailed_metrics.csv) | 九任务共 144 行 |
| [macro_metrics.csv](macro_metrics.csv) | 8×2=16 行宏平均 |
| [component_ablation_paper_table.csv](component_ablation_paper_table.csv) | 展示用百分比及变化值 |
| PNG / PDF | 图 10 |

`delta_f1_pp = 100 × (变体 F1 - 同 cohort 完整配置 F1)`，是百分点。`strict_finite_reference_feature_count` 应始终为 36，而不是消融后的 24 或 27。

论文表可能在两个 panel 中重复完整 baseline，因此 paper table 行数不必等于 8。

## 6. 重新训练

需要完整 36D/K=28 features、配置选择 splits 和 `task_metrics/<job>__36D.csv`：

```bash
PYTHONPATH=src:. python scripts/run_36d_k28_component_ablation.py \
  --feature-root experiments/step_ablation_36d \
  --split-root experiments/fit_only_configuration_selection/splits \
  --selection-root experiments/fit_only_configuration_selection \
  --output-dir experiments/reproduced_component_ablation \
  --workers 4 --xgb-n-jobs 4
```

脚本首先重跑 `full_36d`，并与原配置选择 Full 的 Precision、Recall、F1、FPR、TP/FP/FN/TN 逐项比较，容差 `1e-12`。若失败，应检查输入/环境/seed，不要跳过校验去接受其他变体。

## 7. 重绘与复核

```bash
PYTHONPATH=src:. python scripts/plot_36d_k28_component_ablation.py
```

绘图脚本没有自定义输入参数，固定读取本目录，更新图及 paper table。若重跑结果在新目录，应先完成数据校验，再显式调整绘图来源；不要随意覆盖论文目录。

检查每任务 16 行、总共 144 行；Full baseline 复现；两 cohort 阈值相同；Finite 行集合固定。旧目录 [ablation_36d_k28_fit_only](../ablation_36d_k28_fit_only/README.md) 不能替代这里的 Finite 结果。
