# telemetry-50：采集、Predictor 与标签

[返回实验总手册](../README.md)

本目录是九任务的共享上游数据源，保存 clean/fault 执行、Judge 标签、投影 profile 和 Predictor 工件。后续 K 扫描、配置选择、方法比较、描述统计都依赖这里的执行身份。

## 1. 规模与目录

任务命名为 `<model>_<dataset>`。每任务计划 5,000 clean 和十轮、共 50,000 injected execution；九任务计划 495,000 次。实际可评分行数以派生表为准。

```text
telemetry_50/<job>/
  profile.json
  mapping_model.pt
  mapping_model.metadata.json
  injection.jsonl
  labels.jsonl
  labels_prometheus_parse_failed.jsonl   # 若该任务产生解析失败
  job.log
  mapping.jsonl                        # 采集/训练中间输入，当前未必仍保留
```

这些大文件未随 Git 上传。部分任务没有 `mapping.jsonl`，不能直接从 checkpoint 还原训练集。元数据中的旧 `artifacts/iclr_v2/...` 是历史来源路径，不代表当前文件仍在那里。

## 2. 采集协议

| 项目 | 设定 |
| --- | --- |
| 模型 | Qwen2.5-VL-7B、InternVL3-8B、LLaVA-1.5-7B |
| 数据 | EarthVQA、LingoQA、VQAv2 |
| 外层划分 | 固定 semantic-group Fit/Calibration/Test，70%/15%/15% |
| 故障时刻 | Prefill/forward step 0 |
| 可选模块 | 模型中符合实现条件的 `torch.nn.Linear` |
| 注入目标 | 随机模块输出中的随机标量 |
| 故障 | random 双 bit flip |
| 注入后保留 | 保留 Non-SDC，而非仅保留输出改变的执行 |
| telemetry | 保存采集窗口内的层间指标，支持离线前缀重聚合 |
| 最大采集窗口 | inject 阶段显式 `--telemetry-max-steps 50` |

`telemetry-max-steps` 是监测上限，不是生成长度强制值，也不是 `max_new_tokens` 的替代参数。短答案可只产生很少的 decode observations。现有图 7 协议记录的最大可见 step 数是 49；不能把目录名中的 50 当作每条记录都有 50 个 decoding steps。

## 3. Predictor

Predictor 输入来自 outer-Fit 的 fault-free Mapping telemetry。内部按 group 划分 70% Train、15% Validation、15% Test；Validation 负责 early stopping，Test 仅评估正常映射误差。

参考架构及训练配置：

| 参数 | 当前配置 |
| --- | --- |
| 投影维度 / `x_dim` | 64 |
| `hidden_dim` / `num_blocks` | 64 / 8 |
| `layer_emb_dim` | 16 |
| dropout | 0.1 |
| batch size | 2048 |
| learning rate | 0.0005 |
| weight decay | 0.0001 |
| 最大 epochs | 500 |
| cosine weight | 1.0 |
| early-stop patience | 10 |
| scheduler patience / factor | 5 / 0.5 |
| 最低 learning rate | 1e-6 |
| seed | 42 |

逐任务以 `mapping_model.metadata.json` 为准。`num_layers` 随模型变化，不能把 Qwen 的 28 直接复制给 LLaVA。

元数据中的 `metrics.mse`、`metrics.rmse`、`metrics.cosine_similarity` 是 Predictor Test 指标，不是 Detector 的 F1 或召回率。示例：Qwen/LingoQA 当前 metadata 的 MSE 为 0.145089、CosSim 为 0.776077。

元数据还包含 `mapping_data_sha256`、`checkpoint_sha256`、`split_assignment_sha256`，用于核验来源和身份。不要因目录移动而改写这些 hash。

## 4. 标签

| 类别 | 条件 |
| --- | --- |
| clean | `injected == 0` |
| injected Non-SDC | 注入后答案未改变 |
| Non-significant SDC | 答案改变且有效 significance 为 0/1 |
| Significant SDC | 答案改变且有效 significance 为 2 |

Judge 使用 [current.yaml](../../configs/experiments/current.yaml) 中的 Prometheus 配置。答案完全一致时按策略跳过 Judge 并标为未损坏。解析失败需要单独记录，不能静默填成负例。

`labels.jsonl` 同时承载执行身份、故障元数据、答案、telemetry 和标签；不是一个仅含标签 ID 的小表。读取它可能产生大量 I/O。

## 5. 运行入口

先配置模型、数据、Judge 和 split manifests，选择对应模型解释器。以下演示一个 Qwen job，不自动覆盖已有产物：

```bash
export PYTHONPATH=src:.
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false
PYTHON=Qwen2.5-VL-7B/.venv/bin/python
JOB=qwen25_vl_lingoqa

"$PYTHON" -m detect_sdc.cli run \
  --job "$JOB" --stage inject --device cuda:0 \
  --telemetry-max-steps 50 --dry-run
```

依赖顺序为 `profile -> collect_mapping -> train_mapping -> inject -> label`。已有注入数据不要为了画图重复运行：

```bash
for STAGE in profile collect_mapping train_mapping; do
  "$PYTHON" -m detect_sdc.cli run \
    --job "$JOB" --stage "$STAGE" --device cuda:0
done

"$PYTHON" -m detect_sdc.cli run \
  --job "$JOB" --stage inject --device cuda:0 --telemetry-max-steps 50

"$PYTHON" -m detect_sdc.cli run \
  --job "$JOB" --stage label --device cuda:0
```

该流程需要模型权重和 GPU，不是轻量检查命令。需要重跑时应先配置新的输出路径，而不是直接附加 `--overwrite` 覆盖论文数据。

## 6. 下游可以复用什么

- 保存范围内的 step/pair 指标可支持多个 K、多个 pair profile 的离线特征生成。
- 改 K 不要求重训 Predictor，但通常需要重训相应 Detector。
- 标签、split 和执行身份应保持固定。
- 现有 SIEVE telemetry 不包含可以任意重建 Ranger/Dr.DNA profile 分数所需的全部原始 activation。
- 更换故障模型、增加未保存的 step/layer 信号、改变投影或 Predictor，不保证能仅靠已有 JSONL 完成。

## 7. 验收与故障排查

检查日志和 metadata 中的模型、seed、split hash；检查每个输入是否有 clean 和预期 fault runs；按实际有效标签统计数量。

若缺少 telemetry，先确认输出是否过短或执行失败，不要强行补零。若阶段输出只有 `.tmp`，先看进程和文件是否增长。若 Judge 无法加载，核对其环境依赖，不要在 VLM 运行环境中盲目升级 Transformers。

相关实现：[mapping pipeline](../../src/detect_sdc/pipeline/mapping.py)、[Predictor training](../../src/detect_sdc/mapping/training.py)、[fault injector](../../src/detect_sdc/fault_injector.py)、[CLI](../../src/detect_sdc/cli.py)。
