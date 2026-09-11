# 72D 前缀窗口扫描

[返回总手册](../README.md) | [共享字段与 API 复现](../step_ablation_36d/README.md)

## 1. 配置

| 项目 | 配置 |
| --- | --- |
| pair | `(6,7)`、`(22,23)`、`(23,24)`、`(24,25)`、`(25,26)`、`(26,27)` |
| metric | CosSim、MeanDiff、StdDiff、L2 |
| aggregation | mean / max / min |
| 维度 | 6 × 4 × 3 = 72 |
| K | 1、2、4、8、12、16、20、24、28、32 |
| 数据 | 九任务 telemetry-50 |

72D 是候选集合之一，不是当前最终部署配置。当前最终配置为 36D/K=28，依据 Fit-only 排名确定。

## 2. 该目录的三种用途

1. 历史 Final Test K 扫描，汇总于 [aggregate_results.csv](aggregate_results.csv)。
2. 提供 `features.csv` 给新的 Fit-only 40 配置评估。
3. `k_2/features.csv` 用于建立共享 outer-Fit UID/group manifest。

第三种用途只用身份和 Fit 组划分，**不意味着最终配置或 Finite 定义采用 72D/K=2**。该区分是理解 Fit-only 脚本的关键。

## 3. 文件

[summary.json](summary.json) 记录 profile、K 和历史 cohort。每个 `<job>/k_<K>/` 下有完整特征、分区 CSV、Detector UBJ 和 `output/metrics_summary.json`。根目录还可能留有旧论文段落 `section_4_3.xml`，它不是可执行脚本，也不是当前数值来源。

`features.csv` 带多个 split；`test.csv` 可能已经经过旧筛选。`test_full` 只表示当时 test 输入的全体，不能仅凭字段名判断与正式 Full 相同。

## 4. 历史 cohort

扫描 summary 写明使用 **fixed K=2 non-all-NaN 48D Final Test cohort**，而不是 72D 全列有限。这说明 profile 名、参考 cohort 和评测窗口是不同维度的信息。

这些旧扫描指标不能替代 [configuration_ranking.csv](../fit_only_configuration_selection/configuration_ranking.csv)，更不能据此重新选择 K。

如果需要比较固定 K=28 的四种维度，应从 Fit-only `macro_metrics.csv` 筛选 K=28；如果需要 Finite 诊断，应读独立的 `strict_finite_macro_metrics.csv`。

## 5. 重新运行

旧 step-sweep launcher 未保留。使用 [36D README 的 API 示例](../step_ablation_36d/README.md)，将参数替换为：

```python
pairs = ((6, 7), (22, 23), (23, 24), (24, 25), (25, 26), (26, 27))
out = root / "experiments/reproduced_72d" / job_name / f"k_{k}"
```

不要直接运行通用 `current.yaml` 后以为完成扫描：它当前默认只生成 prefix K=2 的特征，并且默认输出不在本目录。

用已有特征单独复核 72D Fit-only：

```bash
PYTHONPATH=src:. python scripts/run_fit_only_configuration_selection.py \
  --profiles 72D --k-values 1,2,4,8,12,16,20,24,28,32 \
  --workers 2 --xgb-n-jobs 4 \
  --output-dir experiments/reproduced_selection_72d
```

## 6. 验收

每个 pair 12 列，总计 72；K=2 参考文件包含所有 outer-Fit 身份；选择协议只使用 Fit 行；不覆盖旧测试结果或当时选用的 72D/K=4 模型。

旧三方法比较见 [comparison_72d_k4](../comparison_72d_k4/README.md)，正式结果见 [comparison_36d_k28](../comparison_36d_k28/README.md)。
