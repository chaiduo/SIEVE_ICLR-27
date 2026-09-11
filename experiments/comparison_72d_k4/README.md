# 历史 72D/K=4 方法比较

[返回总手册](../README.md) | [当前正式 36D/K=28 比较](../comparison_36d_k28/README.md)

本目录保存最终配置调整之前的 Ranger-style、Dr.DNA-style、SIEVE 比较及早期计时。**不作为当前论文主结果。**

## 1. 历史配置

| 项目 | 值 |
| --- | --- |
| layer pairs | `(6,7)`、`(22,23)`、`(23,24)`、`(24,25)`、`(25,26)`、`(26,27)` |
| unique monitored layers | 6、7、22、23、24、25、26、27 |
| 维度 | 72 |
| prefix K | 4 |
| 模型/数据 | 三模型 × 三数据集 |

72D/K=4 与当前 36D/K=28 的窗口、监控层及模型输入均不同。旧 profile 和标量 score records 不能直接移植为新配置的基线结果。

## 2. 目录

```text
comparison_72d_k4/
  results/<job>/
    profiles.json
    score_records.jsonl
    record_validation.json
    evaluation/
  results/summary/
  overhead_profiles/<job>/
  overhead/forward/
  overhead/reverse/
  overhead/combined/
  logs/
```

本地实际保存文件可能比 Git 多。profile/records/逐样本测量一般没有上传；汇总和部分 metrics 已追踪。

## 3. 如何读取

`results/summary/macro_average_metrics.csv` 是旧配置九任务宏平均。要知道具体分母，必须同时查看任务 `evaluation/metrics.json` 中的输入路径、cohort 和样本数。

旧 finite 或 canonical-finite 不能视为当前 36D 全列有限。旧计时的 repeats、warmup、forward/reverse 顺序也不能替代正式 forward-r10 设置。

## 4. 可复现范围

当前树已清理旧 72D/K=4 专用 launcher/config，不提供假想的一键命令。当前通用比较模块默认加载的是 `detection_comparison_k28_36d.yaml`。

如需做历史审计，必须先恢复对应 revision 的配置及其记录，再核对 source features、profile、monitored layers、K、Detector 和 split；只改输出路径不构成旧实验复现。

## 5. 不应作出的比较

- 不将旧 72D/K=4 F1 与新配置的 Finite F1 直接归因于维度变化。
- 不将旧 forward/reverse 测量混入新 forward-r10 均值。
- 不拿旧八层 profile 给新六层/K=28 campaign 打分。
- 不因文件已存在而让新 launcher 跳过必要的 profile/score 采集。

当前主结果、正确路径和重评分命令均见 [36D/K=28 README](../comparison_36d_k28/README.md)。
