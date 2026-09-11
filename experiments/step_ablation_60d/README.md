# 60D 前缀窗口扫描

[返回总手册](../README.md) | [共享字段与重建流程](../step_ablation_36d/README.md)

## 1. 配置与研究问题

本目录考察五个 layer pair 在不同前缀 K 下的检测表现，并提供 Fit-only 选择所需的候选特征。

| 项目 | 配置 |
| --- | --- |
| pair | `(6,7)`、`(22,23)`、`(23,24)`、`(24,25)`、`(25,26)` |
| metric | `cos_sim`、`mean_diff`、`std_diff`、`l2_distance` |
| statistics | mean、max、min |
| 维度 | 5 × 4 × 3 = 60 |
| K | 1、2、4、8、12、16、20、24、28、32 |
| 任务 | 九任务，每任务十个窗口 |

**此 60D 不包含 `(26,27)`。** 它是 72D 去掉该 pair 的配置，不是给当前 36D 添加两个 pair。若 LLaVA/InternVL 的表现发生变化，不能只归因于输入维度变化。

## 2. 工件

[summary.json](summary.json) 保存 profile 与 cohort，[aggregate_results.csv](aggregate_results.csv) 保存按 K 的旧宏平均。

每个 `<job>/k_<K>/` 中保存完整 `features.csv`、分区 CSV 和 `output/`。`metrics_summary.json` 包含列顺序、训练参数、校准阈值、模型路径和当时的测试指标；UBJ 及大 CSV 通常只在本地。

`macro_f1` 等比例字段需要乘 100 后展示；`mean_steps_used` 不等于窗口上限。短输出不能被补成 K 步，也不能通过复制最后一帧制造新观测。

## 3. 历史结果限制

旧 cohort 由 48D/K=2 的“非全 NaN”参考固定，且评分在 Final Test 上进行。因此：

1. 这些表不能作为最终 36D/K=28 的选型依据。
2. 文件中的旧 finite 不等同于当前 Finite。
3. `output` 中的 `test_full` 仍需核对传入的 Test 行集合。
4. 如果要论文中的 60D 配置比较，应读 [Fit-only macro_metrics](../fit_only_configuration_selection/macro_metrics.csv) 并筛选 `profile == "60D"`。

## 4. 复现路径

当前树未保留旧完整扫描 launcher。[36D README](../step_ablation_36d/README.md) 给出了使用现有 FeatureJob 和 XGBoost API 的独立重建方式，修改为：

```python
pairs = ((6, 7), (22, 23), (23, 24), (24, 25), (25, 26))
out = root / "experiments/reproduced_60d" / job_name / f"k_{k}"
```

按十个 K 和九任务枚举可以生成新扫描。该流程不会自动复制旧 Test cohort，也不应覆盖原实验工件。

仅验证已有 60D 特征的 Fit-only 结果：

```bash
PYTHONPATH=src:. python scripts/run_fit_only_configuration_selection.py \
  --profiles 60D --k-values 1,2,4,8,12,16,20,24,28,32 \
  --workers 2 --xgb-n-jobs 4 \
  --output-dir experiments/reproduced_selection_60d
```

还需要 72D/K=2 参考文件以建立共享 Fit split。输出只是单 profile 的局部验证，不能声称完成所有候选的选择。

## 5. 检查清单

五对各贡献 12 列，总计 60；没有意外加入 `(26,27)`；prefix 不被默认 suffix 替代；所有候选的 Fit UID/group 一致。对新训练模型记录环境和 seed；对历史比较优先使用保存工件而不是重训练后覆盖。
