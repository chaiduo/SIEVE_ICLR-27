# 历史组件消融存档

[返回总手册](../README.md) | [当前正式组件消融](../ablation_36d_k28_fit_only_strict_finite/README.md)

**本目录不作为当前 Finite 结果来源。** 当前正式结果在带 `strict_finite` 后缀的相邻目录中。

## 1. 保存了什么

这是完整 36D/K=28 与删除单个 layer pair/metric family 的早期 Fit-only 消融：

- 三个 pair：`(6,7)`、`(24,25)`、`(26,27)`；
- 四类 metric：CosSim、MeanDiff、StdDiff、L2；
- 保留 mean/max/min 聚合，窗口 K=28；
- 完整配置和七种删除变体；
- 复用 outer-Fit 内部 model-fit/calibration/validation 划分。

## 2. 文件

| 文件 | 含义 |
| --- | --- |
| [protocol.json](protocol.json) | 当时的 protocol 和 cohort 描述 |
| `task_metrics/<job>.csv` | 每任务消融结果 |
| [detailed_metrics.csv](detailed_metrics.csv) | 所有任务明细 |
| [macro_metrics.csv](macro_metrics.csv) | 变体宏平均 |
| [component_ablation_paper_table.csv](component_ablation_paper_table.csv) | 早期论文展示表 |
| PNG / PDF | 历史图 |

即使旧 Full 部分与新运行一致，也不代表旧 Finite 等价。旧筛选依据参考表征的可用性，而不是当前完整 36D 全列有限定义。

## 3. 与新版本的区别

新版本以完整 36D/K=28 的固定 Finite mask 评估每个变体，且保留全部训练/校准分区。旧 Finite 不能通过简单重命名列标题转换成新结果。

删除 pair 后维度为 24，删除指标族后为 27。任何版本都应区分组件完整的 `full_36d` 和评分 cohort `full`；两者不是同一概念。

## 4. 复现建议

当前 `scripts/run_36d_k28_component_ablation.py` 默认生成新目录的数据，不再是旧 Finite 的精确生成器。不要把输出目录指定为这里来覆盖历史记录。

正式重跑与参数见 [新 README](../ablation_36d_k28_fit_only_strict_finite/README.md)。若要审计旧协议，应先固定历史代码 revision、protocol、输入文件和 split，而不是用当前实现猜测旧结果。

## 5. 使用限制

不能将旧目录的 Finite、当前目录的 Full 以及不同版本的图混拼。论文、根 README 和附录优先引用新目录；本目录只服务历史追溯。
