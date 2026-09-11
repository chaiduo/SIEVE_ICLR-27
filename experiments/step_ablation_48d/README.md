# 48D 前缀窗口扫描

[返回总手册](../README.md) | [共享文件字段与重建 API](../step_ablation_36d/README.md)

## 1. 问题与配置

固定四个 layer pair，比较前缀窗口 K 对检测性能的影响。该目录保存了早期 Final Test 扫描结果，同时为后来的 Fit-only 配置选择提供特征。两种用途不能混淆。

| 项目 | 配置 |
| --- | --- |
| pair | `(6,7)`、`(22,23)`、`(25,26)`、`(26,27)` |
| metric | CosSim、MeanDiff、StdDiff、L2 |
| aggregation | mean / max / min |
| feature count | 4 × 4 × 3 = 48 |
| K | 1、2、4、8、12、16、20、24、28、32 |
| 来源 | `telemetry_50/<job>/labels.jsonl` |
| 任务 | 九个模型-数据集任务，每任务十个 K |

48D 不是在 36D 后面简单添加一个 pair：它不含 `(24,25)`，因此不能把维度间差异解释成纯粹“多了 12 个特征”的效果。

## 2. 文件与读取

- [summary.json](summary.json)：profile、K 列表、任务和历史 cohort 描述。
- [aggregate_results.csv](aggregate_results.csv)：按 K 汇总的 `macro_precision`、`macro_recall`、`macro_f1`、`macro_fpr`、`mean_steps_used`。
- `detailed_results.csv`：历史逐任务扫描汇总，本地可能存在但未被 Git 追踪。
- `<job>/k_<K>/features.csv`：包含身份和 split 的完整特征表。
- `<job>/k_<K>/{fit,calibration,test}.csv`：当时传入训练/评测的表。
- `<job>/k_<K>/output/metrics_summary.json`：冻结 Detector 的训练参数、阈值和评估切片。
- `<job>/k_<K>/output/significant_sdc_detector.ubj`：模型，通常仅在本地。

读取任意 `features.csv` 必须过滤 split。`aggregate_results.csv` 中比例为 0 到 1，不是已经乘 100 的百分比。`mean_steps_used` 是实际使用的步数，不保证等于 K。

## 3. 历史口径

旧 `summary.json` 的 cohort 是 **fixed K=2 non-all-NaN 48D Final Test cohort**。至少一列有效即可进入旧集合，不等于当前 36D/K=28 全列有限的 Finite。

即使历史扫描某个 K 的 F1 最高，也不能以它确定论文配置，因为这些指标来自 Final Test。正式选择使用 [Fit-only 排名](../fit_only_configuration_selection/configuration_ranking.csv)。

## 4. 如何复现

当前没有保留该扫描的独立批量 launcher。使用 [36D README 的单配置重建示例](../step_ablation_36d/README.md)，修改：

```python
pairs = ((6, 7), (22, 23), (25, 26), (26, 27))
out = root / "experiments/reproduced_48d" / job_name / f"k_{k}"
```

随后使用相同的特征 API 和 XGBoost API。该方式生成新 Full 分区实验，不会自动还原旧 cohort 汇总。不要将新结果直接覆盖到本目录后宣称复现了旧表。

已有原始特征时，只重新做 Fit-only 48D 验证的示例：

```bash
PYTHONPATH=src:. python scripts/run_fit_only_configuration_selection.py \
  --profiles 48D --k-values 1,2,4,8,12,16,20,24,28,32 \
  --workers 2 --xgb-n-jobs 4 \
  --output-dir experiments/reproduced_selection_48d
```

即使只跑 48D，这个脚本建立共享 split 时仍需要 `step_ablation_72d/<job>/k_2/features.csv`。单 profile 输出只用于调试或局部复核，不是 40 候选的最终排名。

## 5. 验收

检查 48 个特征、四个指定 pair、prefix 窗口、九任务完整性，以及和其余 profile 相同的 Fit 执行身份。跨维度比较时明确 pair 集合也发生了变化；不得将此结果标成“36D 加一个 layer pair”的嵌套消融。
