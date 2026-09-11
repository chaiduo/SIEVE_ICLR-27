# 36D 前缀窗口扫描与最终 Detector 工件

[返回实验总手册](../README.md)

本目录有两个不同用途：历史上用于查看 K 扫描的 Final Test 表现；现在又作为正式 Fit-only 选型及 36D/K=28 比较的特征来源。**工件可以复用，不意味着旧汇总表可以作为正式配置选择依据。**

## 1. 特征规格

| 项目 | 值 |
| --- | --- |
| pair | `(6,7)`、`(24,25)`、`(26,27)` |
| metric | `cos_sim`、`mean_diff`、`std_diff`、`l2_distance` |
| 跨 step 聚合 | `mean`、`max`、`min` |
| 总维度 | `3 × 4 × 3 = 36` |
| K | `1,2,4,8,12,16,20,24,28,32` |
| 窗口 | prefix，前 `min(K, 实际可观察步数)` |

`last_k_steps` 是历史字段名，当前 `step_window="prefix"` 时含义是前 K 步，不是最后 K 步。

## 2. 文件结构

```text
step_ablation_36d/
  summary.json
  aggregate_results.csv
  detailed_results.csv
  <job>/
    k_<K>/
      features.csv
      fit.csv
      calibration.csv
      test.csv
      output/
        metrics_summary.json
        significant_sdc_detector.ubj
        significant_sdc_feature_importance.csv
        significant_sdc_binary_test_*_predictions.csv
        significant_sdc_binary_test_*_wrong_predictions.csv
```

| 文件 | 含义 |
| --- | --- |
| `features.csv` | 带 `split` 的合并特征表；不能整表作为 Test |
| `fit.csv` | 当时用于 Detector 训练的分区 |
| `calibration.csv` | 当时用于阈值选择的分区 |
| `test.csv` | 当时的评测输入，可能经过历史 cohort 筛选 |
| `metrics_summary.json` | 特征列、训练配置、阈值、树数量及旧评分切片 |
| `.ubj` | XGBoost 模型，迁移和三方法重评估依赖它 |
| `aggregate_results.csv` | 历史 K 扫描宏平均，不是 Fit-only 选型表 |

Git 主要保留 `summary.json`、`aggregate_results.csv` 和各任务/K 的 `output/metrics_summary.json`；多数 feature、prediction CSV 和 UBJ 仅存在于原实验机器。

## 3. 为什么不能直接用旧指标

[summary.json](summary.json) 明确写着旧扫描 cohort 是 **fixed K=2 non-all-NaN 48D Final Test cohort**。这是“固定 48D/K=2 中至少一列有效”的历史筛选，不是当前 Finite。

此外，`output/metrics_summary.json` 的 `test_full` 指的是当时传入的 `test.csv` 全部行，不自动等于今天正式定义的 Full。

具体例子：Qwen/LingoQA 的旧 `cohort_rows.test_full` 为 8,178，而正式比较重新从 `features.csv` 选 `split=test` 后评分 8,249 行。两者不同不是四舍五入造成的。

正式主结果应读 [comparison_36d_k28](../comparison_36d_k28/README.md)；正式选择应读 [fit_only_configuration_selection](../fit_only_configuration_selection/README.md)。

## 4. 字段字典

身份列：`orig_id`、`semantic_group_id`、`sample_uid`、`split`、`injected`、`run_index`。

标签列：`is_sdc`、`significance`、`label`、`significant_sdc_target`。

窗口列：`total_steps`、`last_k_steps`、`num_steps_used`。不同 K 的 `num_steps_used` 可以相同，因为实际输出已经结束。

特征名形式如 `cos_sim_mean_p6_7`、`mean_diff_max_p24_25`。`p6_7` 标识层对；前缀是指标族；中间是聚合统计。不要把 label 或 fault 元数据作为模型特征。

`metrics_summary.json` 重点看：

- `feature_columns` 和 `feature_count`：推理时的真实列顺序和数量。
- `input_csvs`：当时训练/校准/测试来源，路径可能已经搬迁。
- `fit_internal_split`：Fit 内 early-stop 的 group 划分。
- `threshold_calibration`：阈值、比较符、校准样本统计。
- `training.parameters`：XGBoost 参数。
- `inference_tree_count`：冻结模型用于推理的树范围。
- `metrics` 和 `cohort_rows`：必须结合当时 test 输入解释。

## 5. 从原始 telemetry 重建单个配置

当前仓库没有旧批量 step-sweep launcher。不要运行不存在的 `run_step_ablation.py`。可使用现有 `FeatureJob` API 明确构造新目录。

以下是 **新的 Full 分区重建示例**，不会复刻历史 48D/K=2 测试过滤，也不会覆盖本目录。需要本地 `labels.jsonl`：

```bash
PYTHONPATH=src:. python - <<'PY'
from dataclasses import replace
from pathlib import Path
import pandas as pd
from detect_sdc.features.jobs import load_feature_job, execute_feature_job

root = Path.cwd()
job_name = "qwen25_vl_lingoqa"
pairs = ((6, 7), (24, 25), (26, 27))
k = 28
out = root / "experiments/reproduced_36d" / job_name / f"k_{k}"
if out.exists():
    raise FileExistsError(f"Use a fresh output directory: {out}")
job = load_feature_job(
    root / "configs/experiments/current.yaml",
    job_name,
    repository_root=root,
)
job = replace(
    job,
    spec=replace(
        job.spec, selected_layer_pairs=pairs, distance_pairs=pairs,
        last_k_steps=k, step_window="prefix",
    ),
    fit_output=out / "fit.csv",
    calibration_output=out / "calibration.csv",
    test_output=out / "test.csv",
)
execute_feature_job(job)
frame = pd.concat(
    [pd.read_csv(out / f"{split}.csv") for split in ("fit", "calibration", "test")],
    ignore_index=True,
)
assert frame["sample_uid"].is_unique
frame.to_csv(out / "features.csv", index=False)
PY
```

示例先逐分区落盘，再合并完整表。不要将 `features.csv` 再交给只接受单个 Test 分区的训练 API。

## 6. 新目录训练 Detector

此步骤会生成新的 Detector 和测试结果，**不是恢复历史字节级相同的工件**。历史数据筛选、依赖版本不同可能产生差异：

```bash
PYTHONPATH=src:. python - <<'PY'
from pathlib import Path
from detect_sdc.config import load_yaml
from detect_sdc.detector.xgboost import XGBoostConfig, run_calibrated_xgboost

out = Path("experiments/reproduced_36d/qwen25_vl_lingoqa/k_28")
config = load_yaml(Path("configs/experiments/current.yaml"))
parameters = dict(config["detector"]["xgboost"]["common"])
parameters.update(config["detector"]["xgboost"]["by_model"]["qwen25_vl"])
if (out / "output").exists():
    raise FileExistsError("Detector output already exists")
run_calibrated_xgboost(
    out / "fit.csv", out / "calibration.csv", out / "test.csv", out / "output",
    config=XGBoostConfig.from_mapping(parameters),
)
PY
```

训练函数会清理指定输出目录中的相关工件，因此示例先检查目录不存在。对于正式历史结果复核，应优先使用保存的 UBJ 和比较评估脚本，不要先覆盖原模型。

要重建完整扫描，按九任务和十个 K 枚举上面的 API，并记录 profile、K、输入 hash、split hash、环境和新输出根目录。原始 JSONL 很大，反复扫描会增加 I/O 成本。

## 7. 关键检查

每个配置应有 36 个特征列；候选之间 outer-Fit 的 UID/group 集合一致；所有 split 的 group 交集为空；`prefix` 没有被默认 `suffix` 替代。

有限性过滤不能提前删除 Full 训练行。当前 Finite 必须来自固定完整 36D/K=28 向量，即使只观察 K=1 或删掉某个 pair，也不能用其自己的有效列重新定义 cohort。

相关实现：[feature extraction](../../src/detect_sdc/features/extraction.py)、[feature jobs](../../src/detect_sdc/features/jobs.py)、[XGBoost](../../src/detect_sdc/detector/xgboost.py)。
