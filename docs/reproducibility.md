# Detect_SDC ICLR-v2 Reproducibility Guide

本文档描述当前 36D/`K=28` 正式实验协议。旧版 85/15 `orig_id` 划分、
SDC-only fault retention、72D/K=4 和 replay comparison 已废弃。

## 1. 固定协议

- 3 个模型：Qwen2.5-VL-7B、InternVL3-8B、LLaVA-1.5-7B；
- 3 个数据集：EarthVQA、LingoQA、VQAv2；
- 每个数据集固定 5,000 个输入；
- semantic-group Fit/Calibration/Final-test 划分；
- Prefill activation double-bit flip；
- 正式主实验使用 `bit_policy: random`；
- 每个输入 10 个 fault runs，全部保留；
- 64 维正交投影；
- Mapping hidden width 64、8 个 residual blocks；
- 3 个层对 `(6,7)`、`(24,25)`、`(26,27)`，前 `min(28, T)` 个实际
  decoding steps，36 个特征；
- Calibration 使用全部正负样本选择 Significant-SDC F1 最大的阈值；
- Full 保留全部 Final Test 行；canonical-finite 使用固定的可比子集；
- Final test 只用于最终报告。

主配置：`configs/experiments/current.yaml`。

## 2. 参考环境

参考机器：

| Component | Reference |
|---|---|
| GPU | 8 × NVIDIA H20，单卡约 96 GiB |
| Driver | 580.173.02 |
| CUDA toolkit | 12.8 |
| Kernel | Linux 5.15.152.bsk.10-amd64 |
| Architecture | x86_64 |

三个模型使用独立环境：

| Model | Python | PyTorch | Transformers |
|---|---:|---:|---:|
| Qwen2.5-VL-7B | 3.12.13 | 2.11.0+cu128 | 5.13.1 |
| InternVL3-8B | 3.12.13 | 2.11.0+cu128 | 4.48.3 |
| LLaVA-1.5-7B | 3.10.20 | 2.1.2+cu121 | 4.37.2 |

完整依赖快照：

```text
reproducibility/environments/qwen25_vl.freeze.txt
reproducibility/environments/internvl3.freeze.txt
reproducibility/environments/llava15.freeze.txt
```

## 3. 外部输入

默认路径位于 `configs/models/*.yaml` 和 `configs/datasets/*.yaml`：

```text
/data01/cd_workspace/llm/Qwen2.5-VL-7B-Instruct
/data01/cd_workspace/llm/InternVL3-8B
/data01/cd_workspace/llm/llava-v1.5-7b
/data01/cd_workspace/llm/LLaVA
/data01/cd_workspace/llm/prometheus-7b-v2.0
/data01/cd_workspace/llm/EarthVQA
/data01/cd_workspace/llm/LingoQA
/data01/cd_workspace/llm/VQAv2
```

验证配置、模型 metadata、数据标注和 split manifests：

```bash
sha256sum -c reproducibility/reference_sha256.txt
```

## 4. 数据划分

三份 manifest 是所有模型共享的唯一划分来源：

```text
splits/earthvqa_seed42.json
splits/lingoqa_seed42.json
splits/vqav2_seed42.json
```

重新生成需要显式 `--overwrite`：

```bash
PYTHONPATH=src Qwen2.5-VL-7B/.venv/bin/python   -m detect_sdc.cli split --dataset earthvqa --overwrite
PYTHONPATH=src Qwen2.5-VL-7B/.venv/bin/python   -m detect_sdc.cli split --dataset lingoqa --overwrite
PYTHONPATH=src Qwen2.5-VL-7B/.venv/bin/python   -m detect_sdc.cli split --dataset vqav2 --overwrite
```

除非有意定义新实验协议，否则不得覆盖 manifest。划分单位为：

- EarthVQA：`image_filename`；
- LingoQA：`question_id`；
- VQAv2：`image_id`。

## 5. 配置校验

```bash
PYTHONPATH=src Qwen2.5-VL-7B/.venv/bin/python   -m detect_sdc.cli config validate configs/experiments/current.yaml
```

校验内容包括模型/数据路径、9 个 job、Mapping 维度和深度、统一 64×8
架构、训练参数、输出文件名以及 split manifest。

## 6. 正式 job

解释器分配：

```text
Qwen2.5-VL-7B/.venv/bin/python:
  qwen25_vl_earthvqa
  qwen25_vl_lingoqa
  qwen25_vl_vqav2

InternVL3-8B/.venv/bin/python:
  internvl3_earthvqa
  internvl3_lingoqa
  internvl3_vqav2

llava-v1.5-7B/.venv/bin/python:
  llava15_earthvqa
  llava15_lingoqa
  llava15_vqav2
```

单个 SIEVE job 的依赖顺序：

```bash
export PYTHONPATH=src:.
export CUDA_VISIBLE_DEVICES=2
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

PYTHON=Qwen2.5-VL-7B/.venv/bin/python
JOB=qwen25_vl_earthvqa

for STAGE in profile collect_mapping train_mapping inject label   featurize train_detector report; do
  "$PYTHON" -m detect_sdc.cli run     --job "$JOB" --stage "$STAGE" --device cuda:0
done
```

正式运行前先使用 `--dry-run`。`--overwrite` 会替换目标阶段产物，只能在归档
后使用。Injection 可从 run 边界恢复：

```bash
"$PYTHON" -m detect_sdc.cli run   --job "$JOB" --stage inject --device cuda:0   --resume-injection-from-run 4
```

## 7. 单次故障方法对比

正式对比不再 replay 历史故障。最终 36D/`K=28` 对比在每个 fault execution 中
采集 Ranger-style 与 Dr.DNA-style 信号，并与冻结的 SIEVE 特征、Detector 和
数据划分对齐：

```bash
tmux new-session -d -s sieve_comparison_36d_k28 \
  'cd /data01/cd_workspace/Detect_SDC && bash scripts/run_comparison_36d_k28_all.sh'
```

结果写入 `experiments/comparison_36d_k28/`。置信区间按原始样本簇
bootstrap，而不是按 fault row 独立重采样。

## 8. Artifact 布局

```text
experiments/telemetry_50/<job>/
├── profile.json
├── mapping_model.pt
├── injection.jsonl
└── labels.jsonl

experiments/step_ablation_36d/<job>/k_28/
├── features.csv
├── fit.csv
├── calibration.csv
├── test.csv
└── output/
    ├── metrics_summary.json
    └── significant_sdc_detector.ubj

experiments/comparison_36d_k28/
├── results/<job>/
│   ├── profiles.json
│   ├── score_records.jsonl
│   └── evaluation/
└── overhead/
```

大文件不进入 Git。归档时生成完整 SHA-256 manifest，并保留 split assignment
hash、Git commit、环境快照和硬件信息。

## 9. 配置选择与组件消融

配置选择在 outer-Fit 内部完成，比较 `36D/48D/60D/72D` 与
`K={1,2,4,8,12,16,20,24,28,32}`。最终 36D/`K=28` 组件消融移除一个 layer pair
或一种 discrepancy metric，且不读取 outer Calibration 或 Final Test：

```bash
PYTHONPATH=src Qwen2.5-VL-7B/.venv/bin/python \
  scripts/run_36d_k28_component_ablation.py
```

## 10. 在线开销

在线开销仅在最终 Detector 和阈值冻结后测量。三个 LingoQA 作业均使用 50 个
测量样本、10 个 warmup、10 次重复、固定正向 mode order 与 `K=28`：

```bash
PYTHONPATH=src Qwen2.5-VL-7B/.venv/bin/python \
  scripts/summarize_online_overhead.py \
  --input-root experiments/comparison_36d_k28/overhead_forward_r10 \
  --online-steps 28
```

## 11. 验收清单

1. `git diff --check` 通过。
2. 配置校验通过。
3. `sha256sum -c reproducibility/reference_sha256.txt` 通过。
4. 三份 split manifest 均为 5,000 个输入，跨集合 group overlap 为零。
5. Mapping 只包含 Fit 样本，内部 train/dev/test group overlap 为零。
6. 每个输入包含一个 clean 和十个完整 fault records。
7. Calibration 仅用于阈值，Final test 未参与训练或调参。
8. 三种方法的 test `sample_uid` 集合完全一致。
9. Judge parse failure 和人工验证结果已记录。
10. 单元测试通过：

```bash
PYTHONPATH=src Qwen2.5-VL-7B/.venv/bin/python   -m unittest discover tests
```
