# Forward-r10 在线检测开销

[返回方法比较](../README.md) | [返回总手册](../../README.md)

本实验只回答固定 36D/K=28 检测器的在线代价，不重新评估 49.5 万次故障的检测效果。

## 1. 测量协议

| 参数 | 设定 |
| --- | --- |
| 数据集 | LingoQA |
| 模型 | Qwen、InternVL、LLaVA，各一个 job |
| 硬件 | NVIDIA H20 |
| batch size | 1 |
| 测量输入 | 固定 50 个样本 |
| warmup | 10 |
| repeats | 10 |
| 每模型/方法观测数 | 50 × 10 = 500 |
| mode order | No Detection -> Ranger-style -> Dr.DNA-style -> SIEVE |
| K | 28，短输出只处理实际可用 steps |

内部 mode key 为 `vanilla/ranger/drdna/sieve`；文档中把 `vanilla` 显示为 No Detection。`forward` 指方法执行顺序，不指模型仅运行 forward 而没有生成流程，也不是与反向传播作对比。

repeats 是同一批输入重复测量，不是新增 500 个独立样本。固定顺序可能受温度、频率和缓存漂移影响；当前结果不包含随机化顺序或多设备重复。

## 2. 当前结果

| 方法 | 三模型平均延迟开销 |
| --- | ---: |
| Ranger-style | 2.10% |
| Dr.DNA-style | 4.89% |
| SIEVE | 6.65% |

SIEVE 逐模型结果：

| 模型 | No Detection 平均 ms | SIEVE 平均 ms | 开销 |
| --- | ---: | ---: | ---: |
| Qwen2.5-VL-7B | 764.44 | 813.67 | 6.44% |
| InternVL3-8B | 984.34 | 1048.66 | 6.53% |
| LLaVA-1.5-7B | 363.84 | 389.22 | 6.98% |

来源：[deployment_summary.csv](combined/deployment_summary.csv) 与 [combined_summary.json](combined/combined_summary.json)。平均开销是逐模型百分比的平均，不是把三个模型延迟先合并后计算同一个比值。

```text
latency overhead (%) = (method mean latency / baseline mean latency - 1) × 100
throughput change (%) = (baseline mean latency / method mean latency - 1) × 100
```

开销 +6.65% 不等于吞吐下降恰好 6.65%，两者分母不同。开销也不是 TFLOPS，当前没有基于这张表测量算力吞吐。

## 3. 文件与字段

```text
overhead_forward_r10/
  qwen25_vl/{samples.csv,summary.json}
  internvl3/{samples.csv,summary.json}
  llava15/{samples.csv,summary.json}
  combined/
    combined_summary.json
    deployment_summary.csv
    component_breakdown.csv
```

Git 当前主要保留 `combined/`，逐样本测量通常仅在本地。

| 字段 | 含义 |
| --- | --- |
| `vanilla_latency_mean_ms` / `sieve_latency_mean_ms` | 无检测/有检测端到端平均延迟 |
| `end_to_end_overhead_percent` | 延迟相对增幅 |
| `throughput_change_percent` | 对应吞吐变化 |
| `detection_ready_mean_ms` / `p95` | 检测结果可用时刻的统计 |
| `detection_after_prefill_*` | 相对 Prefill 结束的检测等待时间 |
| `extra_peak_allocated_mb` | 额外峰值 allocated memory |
| `steps_processed` | 实际监测步数分布，不代表所有执行满 28 步 |

汇总文件可能保留 CI 字段，即使论文不展示它们。不要删除底层字段来制造“没有做 CI”的假象，也不要把这里的重复计时 CI 当成跨机器可泛化区间。

## 4. 独立 benchmark

需要已经冻结的 baseline profile、Predictor 和 Detector；只做计时应直接调用 benchmark，避免 comparison wrapper 补跑缺失的故障数据：

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src:. Qwen2.5-VL-7B/.venv/bin/python \
  scripts/benchmark_online_overhead.py \
  --job qwen25_vl_lingoqa \
  --comparison-config compare_experiment/configs/detection_comparison_k28_36d.yaml \
  --profiles experiments/comparison_36d_k28/results/qwen25_vl_lingoqa/profiles.json \
  --detector-summary experiments/step_ablation_36d/qwen25_vl_lingoqa/k_28/output/metrics_summary.json \
  --device cuda:0 --samples 50 --warmup-samples 10 --repeats 10 \
  --online-steps 28 --modes vanilla ranger drdna sieve --mode-order forward \
  --output-root experiments/reproduced_overhead/qwen25_vl
```

另外两个模型需使用其对应解释器和 job，分别输出 `internvl3`、`llava15` 子目录。只跑一个模型不能生成三模型最终宏平均。

```bash
PYTHONPATH=src:. python scripts/summarize_online_overhead.py \
  --input-root experiments/reproduced_overhead \
  --output-dir experiments/reproduced_overhead/combined \
  --samples-per-model 50 --warmup-samples 10 --repeats 10 --online-steps 28
```

`scripts/run_overhead_forward_r10.sh` 会调用完整 comparison job，默认 GPU 1/4/7；缺少 records 时可能先跑故障采集。它不是无条件“只测开销”的入口。

## 5. 复核边界

确认测量前 GPU 没有竞争进程；模型生成参数一致；50 个输入一致；warmup 不计入 500 次测量；所有方法使用相同窗口和监控层。旧 `../overhead/forward` 与 `reverse` 不混入本实验。

不要根据端到端平均延迟推导每一步固定增量，也不要把该结果直接外推到更大 batch、其他 GPU 或任意输出长度。
