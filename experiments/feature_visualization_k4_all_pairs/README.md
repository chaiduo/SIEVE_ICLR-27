# 历史图 7：全部相邻层对、prefix K=4

[返回总手册](../README.md) | [当前 all-steps 版本](../feature_visualization_all_steps_all_pairs/README.md)

## 1. 实验定位

这是前四个可观察 decoding steps 的 CosSim 诊断，保存于主文决定采用 all-steps 图之前。它不是最终 36D/K=28 Detector 的评分结果。

## 2. 协议

| 项目 | 值 |
| --- | --- |
| 来源 | 九任务 telemetry-50 labels |
| 分区 | Calibration only |
| 执行 | injected only |
| 类别 | injected Non-SDC、Non-significant SDC、Significant SDC |
| pair | 每模型全部相邻层对 |
| 窗口 | prefix K=4，短输出不足四步时使用实际步数 |
| 执行级统计 | 每 pair 在前至多四个观测中的有限 CosSim 均值 |
| bootstrap | semantic-group，10,000 次，seed=20260907 |

这张图不含 clean。无故障注入结果 Non-SDC 不能在图例中改写为 Clean。

## 3. 文件

- [protocol.json](protocol.json)：详细统计口径、输入 hash、每任务排除情况。
- [all9_pair_statistics.csv](all9_pair_statistics.csv)：九任务层对统计。
- [figure7_lingoqa_all_pairs_k4.csv](figure7_lingoqa_all_pairs_k4.csv)：LingoQA 三模型主图切片。
- `figure7_lingoqa_all_pairs_k4.png/.pdf`：历史图。
- [macro_pair_statistics.csv](macro_pair_statistics.csv)：跨任务统计。
- [non_finite_rates.csv](non_finite_rates.csv)：逐组无效数值率。

字段解释沿用 [all-steps README](../feature_visualization_all_steps_all_pairs/README.md)：`finite_rows` 是可用执行数，`semantic_groups` 是 bootstrap 的独立分组数，CI 不是原始值分布区间。

## 4. 与 all-steps 对比

输入标签和外层分区相同，但窗口不同。某样本前四步无有效值、后续步有有效值时，两版有限行数会不同，因此不能只比较均值而忽略有效样本数。

不要将 K=4 图的统计与 all-steps 图的 CI 拼成一张新图；它们的估计量不同。

## 5. 复现状态

当前没有保留这套九任务 Calibration group-bootstrap 图的独立批量生成器。现有单文件 CosSim 绘图脚本不包含全部同等协议，不能直接替代。

精确重建需要按本目录 protocol 重新实现批处理：取 Calibration、分类、截取前四个观察、执行内均值、group bootstrap、导出及绘图。全部步骤不需要新注入，但需要原始 labels。

现有图和 CSV 可直接查看；论文默认应引用 all-steps 版本。保留本目录用于研究窗口截断如何影响可解释信号，不将其混作新主结果。
