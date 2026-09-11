# 最终 36D/K=28 方法比较

[返回总手册](../README.md) | [在线开销详情](overhead_forward_r10/README.md)

本目录保存 SIEVE、Ranger-style、Dr.DNA-style 的最终九任务比较。正式统计读取 `results/<job>/evaluation_strict_finite/` 和 `results/summary_strict_finite/`。

## 1. 主结果

| Cohort | Method | Precision | Recall | F1 | FPR |
| --- | --- | ---: | ---: | ---: | ---: |
| Full | Ranger-style | 82.33% | 87.40% | 83.46% | 0.95% |
| Full | Dr.DNA-style | 92.02% | 76.12% | 82.40% | 0.28% |
| Full | SIEVE | 96.38% | 91.49% | 93.78% | 0.11% |
| Finite | Ranger-style | 44.79% | 47.02% | 37.74% | 0.95% |
| Finite | Dr.DNA-style | 73.85% | 16.76% | 22.74% | 0.28% |
| Finite | SIEVE | 79.88% | 62.81% | 68.78% | 0.11% |

来源：[macro_average_metrics.csv](results/summary_strict_finite/macro_average_metrics.csv)。全部为九任务等权宏平均，Significant SDC 为正类。FPR 负类包括 clean、injected Non-SDC 和 Non-significant SDC。

Full 是主比较；Finite 是固定 36D/K=28 全列有限的条件诊断。三方法在同一 cohort 上比较，不能为每种方法分别过滤其“容易检测”的行。

## 2. 匹配了什么，没有匹配什么

| 项目 | 当前协议 |
| --- | --- |
| SIEVE pairs | `(6,7)`、`(24,25)`、`(26,27)` |
| 三方法监控层 | `6,7,24,25,26,27` |
| 窗口 | 前 `min(28,T)` 个实际 decoding steps |
| 外层 split | 相同 manifests |
| 评分身份 | 根据 `sample_uid` 对齐 |
| 阈值目标 | 全部 Calibration 正负样本上最大化 Significant-SDC F1 |
| Full / Finite | 共享最终 SIEVE 特征定义 |
| clean 数据预算 | 不相同，见下文 |

两种基线是 **configuration-matched adapted baselines**，不应声称完整复现了原 Ranger/Dr.DNA 系统。

基线 `profile_baselines.py` 从 outer-Fit 的 **`orig_id` 集合**确定性抽样，数量为 `ceil(N_fit_orig_ids × 0.2)`，最多 1,000 条。profile seed 为 42。这里是输入身份，不是精确的 20% semantic groups；实际成功 profiling 数量还要扣掉没有可观察 trace 的短输出。

SIEVE Predictor 使用 outer-Fit 的 clean Mapping telemetry。因此当前比较没有对齐 clean-data budget。全 profile 和 profile-size 敏感性没有执行，不能作为已完成实验描述。

Dr.DNA 配置包括 cohort size 64、bins 10、strike count 3、seed 42，三个 lambda 均为 1；完整值见 [比较 YAML](../../compare_experiment/configs/detection_comparison_k28_36d.yaml)。

## 3. 数据链

```text
outer-Fit clean inputs --> profiles.json
existing injection protocol --> score_records.jsonl
score_records + reference labels --> record_validation.json
saved SIEVE Detector + Calibration/Test features + baseline scores
  --> evaluation_strict_finite/metrics.json
  --> summary_strict_finite/macro_average_metrics.csv
```

`score_records.jsonl` 是基于当时 profile 的基线分数，不是任意 profile 都可重用的原始 activation 数据。更换 profile 后不能直接在旧标量分数上推导新 F1。

当前评估会加载冻结 SIEVE 模型重新对齐评分；三方法阈值均在相同 Calibration 上计算。Finite 只对测试行过滤，阈值不再校准。

## 4. 文件字典

| 路径 | 作用 |
| --- | --- |
| `results/<job>/profiles.json` | 两种基线的 clean profile 与样本身份 hash |
| `results/<job>/score_records.jsonl` | 每个执行的基线分数、身份和验证信息 |
| `results/<job>/record_validation.json` | 与 reference labels 的一致性验证结果 |
| `results/<job>/evaluation_strict_finite/metrics.json` | 当前 Full/Finite、阈值、bootstrap、per-run 指标 |
| `results/summary_strict_finite/` | 正式九任务汇总 |
| `results/<job>/evaluation/`、`results/summary/` | 可能保留旧评估口径，不默认使用 |
| `preview_sieve_evaluation/` | 比较 campaign 期间的 SIEVE 预览，非三方法最终主表 |
| `overhead/` | 早期 forward/reverse 开销 |
| `overhead_forward_r10/` | 正式开销 |

顶层 `metrics.json` 中 `finite_cohort` 记录参考特征数、参考 Test 行数、排除行数及实际匹配行数，优先用于检查口径。

宏平均 CSV 字段：

| 字段 | 含义 |
| --- | --- |
| `method` | 方法显示名 |
| `cohort` | `full` 或 `finite_only`；后者为当前 Finite |
| `jobs` | 汇总任务数，应为 9 |
| `sdc_recall` | 对全部 SDC 的召回，不是主正类召回 |
| `significant_sdc_recall` | 主召回率 |
| `non_significant_fpr` | 所有非 Significant-SDC 的误报率 |
| `significant_sdc_precision` / `significant_sdc_f1` | 主 Precision / F1 |

## 5. 重评分：无需重新注入

如果本地已有原始 score records、36D/K=28 features 和 UBJ，只重建当前 Finite 比较：

```bash
bash scripts/reevaluate_comparison_36d_k28_strict_finite.sh
```

该 wrapper 使用本机 Qwen 解释器并并行处理九任务，会更新已有 `evaluation_strict_finite` 和 `summary_strict_finite`，不是只读命令。

更安全的单任务新目录示例：

```bash
PYTHONPATH=src:. python -m compare_experiment.evaluate_results \
  --job qwen25_vl_lingoqa \
  --comparison-config compare_experiment/configs/detection_comparison_k28_36d.yaml \
  --records experiments/comparison_36d_k28/results/qwen25_vl_lingoqa/score_records.jsonl \
  --detector-summary experiments/step_ablation_36d/qwen25_vl_lingoqa/k_28/output/metrics_summary.json \
  --calibration-features experiments/step_ablation_36d/qwen25_vl_lingoqa/k_28/calibration.csv \
  --test-features experiments/step_ablation_36d/qwen25_vl_lingoqa/k_28/features.csv \
  --finite-cohort-features experiments/step_ablation_36d/qwen25_vl_lingoqa/k_28/features.csv \
  --output-dir experiments/reproduced_comparison/qwen25_vl_lingoqa
```

这里刻意使用完整 `features.csv`，由脚本筛选 `split=test`，而非历史 `test.csv`。summary 中的模型路径若仍指向旧目录，需要准备实际工件，不能用不存在的路径继续执行。

正式结果重新汇总：

```bash
PYTHONPATH=src:. python -m compare_experiment.summarize_results \
  --comparison-config compare_experiment/configs/detection_comparison_k28_36d.yaml \
  --evaluation-dir-name evaluation_strict_finite \
  --output-dir experiments/comparison_36d_k28/results/summary_strict_finite
```

**必须显式指定 `--evaluation-dir-name evaluation_strict_finite`。** 当前 all-job launcher 尾部的汇总没有该参数，不应将它默认生成的 `summary/` 直接视为当前论文表。

## 6. 重新采集基线信号

单 job launcher：

```bash
bash scripts/run_comparison_36d_k28_job.sh \
  0 Qwen2.5-VL-7B/.venv/bin/python qwen25_vl_lingoqa
```

全矩阵 launcher：

```bash
bash scripts/run_comparison_36d_k28_all.sh
```

这不是轻量离线评估，会在缺少 profile/records 时启动 VLM 推理。全矩阵默认使用物理 GPU 0 至 7，并使用三个本机 `.venv`；LLaVA 的 VQAv2 与 EarthVQA 在 GPU 6 顺序执行。

单 job 脚本按文件是否存在决定跳过阶段，不验证文件是否来自相同配置。修改 profile、K 或 layer 后不能原样保留旧 records 然后声称已重采集。

LingoQA job 还会触发 overhead；若现有 overhead 不满足完成判定，脚本会删除对应输出目录后重跑。运行前必须查看脚本路径，不能拿正式目录做试验。

## 7. 统计和验收

- 正式 Full 总评分行为 74,586；Finite 为 72,645。
- 正例分别 2,720 和 779；cohort support 很不相同。
- 逐任务 metrics 保存 10,000 次 group bootstrap CI，主表可以只展示点估计，但不能把行级 bootstrap 当成同一协议。
- 补充 FPR budgets 为 0.5%、1%、2%、5%，不代替 maximize-F1 的主 operating point。
- 检查 `record_validation.json`，不匹配时停止，不以重新解释标签掩盖差异。
- 确认三方法同一 UID 集合和同一负类定义，Finite 阈值与 Full 相同。
- Git 中通常只有小型 metrics/validation/summary；仅 clone 仓库无法直接重跑 score campaign。
