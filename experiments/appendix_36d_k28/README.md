# 历史附录汇总表

[返回实验总手册](../README.md)

本目录是早期附录表格的汇总导出，不是独立训练实验。**其中包含已废弃的 canonical-finite 字段，不能整表复制到当前论文。**

## 1. 文件与来源

| 文件 | 来源/内容 |
| --- | --- |
| [README.json](README.json) | 当时汇总的来源与标签/cohort 定义 |
| [label_and_cohort_statistics.csv](label_and_cohort_statistics.csv) | 九任务有效执行、类别和早期 Test cohort 数量 |
| [component_ablation_full_f1.csv](component_ablation_full_f1.csv) | 旧组件消融 Full 的逐任务表 |
| [component_ablation_canonical_finite_f1.csv](component_ablation_canonical_finite_f1.csv) | 旧 canonical-finite 消融 |

`README.json` 明确记录 canonical-finite 定义为 72D/K=2 表征至少一列有限，并指向旧 `ablation_36d_k28_fit_only/detailed_metrics.csv`。这不是当前 Finite。

## 2. 统计字段

`valid_executions` 为当时可用执行；`clean_executions` 为无注入执行；`injected_executions` 为可用故障执行。`non_sdc` 只指 injected Non-SDC，不包括 clean。

`significant_sdc_percent_of_injected` 的分母为可用 injected，不是所有 SDC。不能把该列用于图 2 的“SDC 中显著比例”。

`full_final_test`、`canonical_finite_final_test` 和 retention 是不同口径的计数，其中 canonical-finite 已废弃。即使某个 Full 计数仍相同，也应逐列核对来源。

## 3. 当前附录应改读哪里

- 当前 Full/Finite Test 数量：各任务 [comparison_36d_k28](../comparison_36d_k28/README.md) 的 `evaluation_strict_finite/metrics.json`。
- 当前组件表：[ablation_36d_k28_fit_only_strict_finite](../ablation_36d_k28_fit_only_strict_finite/README.md) 的 detailed/macro metrics。
- 当前配置选择：[fit_only_configuration_selection](../fit_only_configuration_selection/README.md)。
- 图 2 的有限标量 SDC 数量：[figure2](../figure2_significant_sdc_among_sdc/README.md)。

## 4. 更新/重建纪律

当前没有保留这一组附录导出表的独立生成脚本，不提供虚构入口。重建附录时应从明确的现行 CSV/JSON 字段用结构化解析器导出，记录新的来源、分母与定义。

不要把旧列 `canonical_finite_final_test` 简单重命名为 `finite_final_test` 而不重新计数。新表也应写入新的目录或版本，保持历史溯源。

## 5. 验收

附录每一列都标明统计总体、split、cohort、单位和来源。当前 Finite 的参考特征数必须为 36；主比较 Test 总数应复核为 Full 74,586、Finite 72,645，正例 2,720/779。
