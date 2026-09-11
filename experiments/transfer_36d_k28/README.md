# 9×9 Detector Transfer

[返回实验总手册](../README.md)

本实验测试固定源任务 Detector 和阈值在目标任务上的表现。**目标任务仍提供自身 fault-free Mapping，因此不是无目标数据的零样本迁移。**

![Transfer matrix](transfer_f1_matrix_wide_colorbar.png)

## 1. 每个单元的含义

矩阵行是 source，列是 target。每个单元的执行流程：

1. 从 source 的 36D/K=28 工件加载 Detector、特征列顺序、推理树范围及 source Calibration 阈值。
2. 加载 target 的 `features.csv` 并选择 `split=test`。
3. 用 target 自身 clean-trained Mapping 对应的 36D 特征作为输入，源 Detector 不重新拟合。
4. 使用冻结 source 阈值预测。
5. 最后用 target Test 标签进行离线评分。

不读取 target Fit/Calibration 故障标签做训练、调参或阈值选择。Target Test 标签必须用于计算 F1，因此“目标不提供故障监督”不能写成“实验完全不使用目标标签”。

## 2. 数据规模

九个 source × 九个 target = 81 个组合；每个组合分别评估 Full 和 Finite，共 162 行 cell metrics。

对角线九格是 in-domain；非对角线 72 格是 cross-task。`summary.json` 中 `scored_rows=671274` 是 74,586 个目标 Full 执行被九个源模型重复评分的总计，不是新增 67 万次注入。

Finite 由目标最终 36D/K=28 全部特征有限定义；source 改变时同一个 target 的 Finite 集合不变。

## 3. 结果

| 条件 | 单元数 | Full F1 | Full FPR |
| --- | ---: | ---: | ---: |
| in-domain | 9 | 93.78% | 0.11% |
| same model, cross dataset | 18 | 73.55% | 5.81% |
| cross model, same dataset | 18 | 75.24% | 3.19% |
| cross model, cross dataset | 36 | 72.53% | 3.45% |
| 全部 off-diagonal | 72 | 73.46% | 3.97% |

Off-diagonal Finite macro F1 为 28.91%。三类关系的格子数不同，总非对角 macro 应对 72 格等权平均，不能简单对三个关系均值再平均。

源阈值在目标上 FPR 明显升高，说明 calibration mismatch。该结果支持“量化跨任务迁移边界”，不支持“稳定通用迁移”。

## 4. 文件

| 文件 | 内容 |
| --- | --- |
| [protocol.json](protocol.json) | source/target 角色、路径及 Finite 定义 |
| [summary.json](summary.json) | 单元数、bootstrap、非对角汇总和对角验证列表 |
| [source_detectors.csv](source_detectors.csv) | 每个 source 的模型、阈值等信息 |
| [transfer_metrics.csv](transfer_metrics.csv) | 每个 source/target/cohort 的指标与区间 |
| [relationship_summary.csv](relationship_summary.csv) | 四种关系的 macro 指标 |
| [source_summary.csv](source_summary.csv) | 按 source 汇总的迁移表现 |
| `full_*_matrix.csv` | Full F1/Precision/Recall/FPR 矩阵 |
| `strict_finite_*_matrix.csv` | Finite 矩阵 |
| `transfer_f1_matrix_wide_colorbar.*` | 当前图 11 |

矩阵 CSV 和 metrics 保存比例，图中展示百分比。`styled`、`tight`、`balanced` 等图片是版式迭代，不是独立迁移实验。当前首选 `wide_colorbar`。

## 5. 对角线验证

脚本将 in-domain 单元与 `comparison_36d_k28/results/<job>/evaluation_strict_finite/metrics.json` 对齐。这个检查可以发现加载了旧 test 子集、不同 source 阈值、特征顺序或树数量的问题。

只训练新的 source Detector 后直接比较旧主表，不保证对角一致。要复核论文，应先使用保存的原模型。

## 6. 从已有特征重评估

需要本地九个 UBJ、各模型 summary、目标完整 feature CSV 以及正式比较 metrics：

```bash
PYTHONPATH=src:. python scripts/evaluate_detector_transfer_36d_k28.py \
  --feature-root experiments/step_ablation_36d \
  --comparison-results-root experiments/comparison_36d_k28/results \
  --output-dir experiments/reproduced_transfer \
  --bootstrap-replicates 10000 --bootstrap-seed 20260907
```

这是 CPU 模型评分和 bootstrap，不需重新注入。完整 GPU telemetry 或目标 feature 缺失时，才需要补齐上游。

单 source 调试可用 `--sources qwen25_vl_lingoqa`；`--targets` 同样接受逗号分隔的 job。子集实验不能标为完整 9×9。

## 7. Bootstrap

按 target `semantic_group_id` 进行 10,000 次重采样，保持组内执行一起出现。区间描述固定 source 模型下目标组采样的不确定性，不包含 source 重训练、随机 split 或新硬件变化。

每格 F1 CI 不能直接平均成严格的全矩阵 macro CI。图和 README 给出的 macro 主要是点估计。

## 8. 重绘

```bash
PYTHONPATH=src:. python scripts/plot_detector_transfer_36d_k28.py \
  --output-prefix experiments/transfer_36d_k28/transfer_f1_matrix_wide_colorbar
```

该命令使用已有 `transfer_metrics.csv`，不加载 Detector。当前版式为左右 Full/Finite，按背景亮度选深/浅字，色条宽度 5%。渲染尺寸和留白只影响可读性，不影响指标。

## 9. 验收

检查 81 Full + 81 Finite 单元、九个对角验证、source 阈值对所有 targets 不变、同 target 的 cohort 行数不变；不要将 target 的标签用于阈值优化后仍称为冻结阈值 transfer。
