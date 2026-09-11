# SIEVE 全实验手册

本文是 `experiments/` 的详细入口。根目录 [README](../README.md) 用于快速浏览结果；这里记录每项实验的目的、数据依赖、协议、执行方式、字段含义、结果边界和历史版本区别。

**阅读约定**

- 文档中的命令均从仓库根目录执行，不是从当前 README 所在目录执行。
- 展示名称统一为 **Full / Finite**；路径和机器字段中的 `strict_finite` 保留原样，以便定位已有产物。
- 结果取自当前落盘的 CSV/JSON。历史文件不会因为写了新文档而自动变成新协议结果。
- “可从汇总数据重绘”不等于“可从 Git clone 重跑推理”。模型权重、数据集、原始 JSONL 和多数 checkpoint 不在 Git 中。
- 本文提供执行说明，不代表所有命令已在本次文档修改中执行。特别是 GPU campaign、重训练及原始 JSONL 扫描没有重新运行。

## 1. 全目录索引

| 目录及详细说明 | 回答的问题 | 当前地位 |
| --- | --- | --- |
| [telemetry_50](telemetry_50/README.md) | 原始数据如何采集？Predictor 如何训练和评估？ | 下游共享数据源 |
| [step_ablation_36d](step_ablation_36d/README.md) | 固定三层对时，K 如何改变特征和检测性能？ | 特征/Detector 工件有效；旧扫描汇总不可用于选型 |
| [step_ablation_48d](step_ablation_48d/README.md) | 四层对的 K 扫描 | 历史扫描及 Fit-only 输入 |
| [step_ablation_60d](step_ablation_60d/README.md) | 五层对的 K 扫描 | 历史扫描及 Fit-only 输入 |
| [step_ablation_72d](step_ablation_72d/README.md) | 六层对的 K 扫描 | 历史扫描及 Fit-only 输入 |
| [fit_only_configuration_selection](fit_only_configuration_selection/README.md) | 不用 Final Test，如何从 40 个配置选出 36D/K=28？ | 正式配置选择、图 8/9 |
| [comparison_36d_k28](comparison_36d_k28/README.md) | 三种方法在最终配置下如何比较？ | 正式 Final Test 主结果 |
| [overhead_forward_r10](comparison_36d_k28/overhead_forward_r10/README.md) | 在线监测增加多少延迟？ | 正式开销实验 |
| [ablation_36d_k28_fit_only_strict_finite](ablation_36d_k28_fit_only_strict_finite/README.md) | 哪些层对和指标族贡献了性能？ | 正式组件消融、图 10 |
| [transfer_36d_k28](transfer_36d_k28/README.md) | 固定源 Detector/阈值能否迁移到目标任务？ | 9×9 Detector transfer、图 11 |
| [figure2_significant_sdc_among_sdc](figure2_significant_sdc_among_sdc/README.md) | 已发生的 SDC 中有多少是 Significant SDC？ | 图 2 描述统计 |
| [fault_deviation_buckets_telemetry50](fault_deviation_buckets_telemetry50/README.md) | 故障数值偏差与 SDC 严重程度如何关联？ | 三档偏差分桶 |
| [feature_visualization_all_steps_all_pairs](feature_visualization_all_steps_all_pairs/README.md) | 层间指标能否区分三种故障结果？ | Calibration 描述统计、图 7 |
| [feature_visualization_k4_all_pairs](feature_visualization_k4_all_pairs/README.md) | 前四步的 CosSim 分布是什么样？ | 历史窗口诊断 |
| [ablation_36d_k28_fit_only](ablation_36d_k28_fit_only/README.md) | 旧 cohort 下的组件消融是什么？ | 历史存档 |
| [comparison_72d_k4](comparison_72d_k4/README.md) | 旧 72D/K=4 方法比较是什么？ | 历史存档 |
| [appendix_36d_k28](appendix_36d_k28/README.md) | 早期附录表格由哪些数据汇总？ | 混合历史口径，不能整表照搬 |

`step_ablation_48d.log` 和 `step_ablation_pair_profiles.log` 是历史运行日志，不是独立实验，也不是指标来源。

## 2. 论文结果应从哪里取

| 用途 | 优先读取的文件 | 不应替代它的文件 |
| --- | --- | --- |
| 三方法主表 | [macro_average_metrics.csv](comparison_36d_k28/results/summary_strict_finite/macro_average_metrics.csv) | 72D/K=4 汇总、旧 `summary/` 的 Finite |
| 40 配置排名 | [configuration_ranking.csv](fit_only_configuration_selection/configuration_ranking.csv) | `step_ablation_*/aggregate_results.csv` |
| Fit-only Full | [macro_metrics.csv](fit_only_configuration_selection/macro_metrics.csv) | Final Test F1 |
| Fit-only Finite | [strict_finite_macro_metrics.csv](fit_only_configuration_selection/strict_finite_macro_metrics.csv) | 文件名含 `canonical_finite` 的旧表 |
| 组件消融 | [macro_metrics.csv](ablation_36d_k28_fit_only_strict_finite/macro_metrics.csv) | 不带 `strict_finite` 的历史消融目录 |
| 迁移 | [transfer_metrics.csv](transfer_36d_k28/transfer_metrics.csv) | 主表内域指标或图中四舍五入数字 |
| 开销 | [deployment_summary.csv](comparison_36d_k28/overhead_forward_r10/combined/deployment_summary.csv) | `overhead/forward`、`overhead/reverse` 旧测量 |
| 图 2 | [finite-value CSV](figure2_significant_sdc_among_sdc/significant_sdc_among_finite_value_sdc.csv) | 未过滤故障标量的 `among_all_sdc` |
| 图 7 | [LingoQA CosSim CSV](feature_visualization_all_steps_all_pairs/figure7_lingoqa_all_pairs_all.csv) | K=4 版本或四指标拼图 |

当前关键结果：

| 实验/集合 | Precision | Recall | F1 | FPR |
| --- | ---: | ---: | ---: | ---: |
| SIEVE，Final Test Full | 96.38% | 91.49% | 93.78% | 0.11% |
| Ranger-style，Final Test Full | 82.33% | 87.40% | 83.46% | 0.95% |
| Dr.DNA-style，Final Test Full | 92.02% | 76.12% | 82.40% | 0.28% |
| SIEVE，Final Test Finite | 79.88% | 62.81% | 68.78% | 0.11% |
| SIEVE，Fit-only 36D/K=28 Full | 99.17% | 88.66% | 93.42% | 0.03% |
| SIEVE，跨任务 72 个非对角单元 Full | 82.54% | 78.28% | 73.46% | 3.97% |

这些行对应不同分区或不同迁移条件，不能相互替代。特别是 93.42% 不是主测试 F1，68.78% 也不是重新在 Finite 上训练得到的 F1。

## 3. 数据依赖与执行顺序

```text
模型权重 + benchmark 输入 + 固定 split manifests
  |
  +-- outer-Fit clean profile / Mapping telemetry
  |     +-- Predictor Train / Validation / Test
  |     +-- mapping_model.pt
  |
  +-- clean + random double-bit fault executions
        +-- injection.jsonl
        +-- LLM Judge --> labels.jsonl
              |
              +-- step_ablation_{36d,48d,60d,72d}/<job>/k_<K>/features.csv
              |     |
              |     +-- outer-Fit rows --> Fit-only 40-config selection
              |     |                        +-- component ablation
              |     +-- final Detector artifacts --> detector transfer
              |     +-- 36D/K=28 features --> method comparison
              |
              +-- Figure 2 / deviation buckets (descriptive statistics)
              +-- Calibration rows --> Figure 7

outer-Fit clean inputs --> Ranger/Dr.DNA profiles
  +-- score campaign --> score_records.jsonl --> method comparison

frozen profiles + Predictor + Detector --> separate online-overhead benchmark
```

**复现分三级**

1. **查看或重绘**：只需要已提交的统计 CSV/JSON、绘图脚本和 Python 绘图库。
2. **重新评分/重新训练 Detector**：需要本地 `features.csv`、分区 CSV、UBJ 模型或保存的预测文件；不一定需要 GPU 推理。
3. **重新采集**：需要原始数据集、VLM、Judge、对应模型环境和 GPU。缺少原始信号时，汇总指标不能逆向还原。

| 修改内容 | 通常需要的最小工作 |
| --- | --- |
| 图片字号、颜色、间距 | 从汇总数据重绘 |
| 已有分数的 cohort 筛选或阈值重评估 | 保存的分数、目标标签、分区身份；不需重新注入 |
| telemetry 覆盖范围内改变 K 或层对 | 重建特征并训练 Detector；不需再次 VLM 推理 |
| 改 Predictor/投影 | 原有残差可能失效，需要重新采集或重新计算所需原始投影 |
| 改 Ranger/Dr.DNA profile | 新 clean profiling，且需要针对新 profile 重算故障执行分数 |
| 改 bit policy、注入位置分布 | 新的故障 campaign |

## 4. 九任务和身份字段

| 模型 key | 模型名 | job 后缀 |
| --- | --- | --- |
| `qwen25_vl` | Qwen2.5-VL-7B | `earthvqa` / `lingoqa` / `vqav2` |
| `internvl3` | InternVL3-8B | `earthvqa` / `lingoqa` / `vqav2` |
| `llava15` | LLaVA-1.5-7B | `earthvqa` / `lingoqa` / `vqav2` |

例如 `qwen25_vl_lingoqa` 表示一个完整的模型-数据集任务，不是单个样本。

| 字段 | 含义 | 使用限制 |
| --- | --- | --- |
| `orig_id` | 数据集输入身份 | 不等同于 semantic group |
| `semantic_group_id` | 分区及 bootstrap 的相关性单位 | EarthVQA/VQAv2 按图像，LingoQA 按问题 |
| `sample_uid` | 具体执行身份，区分 clean 和各次注入 | 跨方法、跨候选对齐的主键 |
| `run_index` | 故障运行轮次标识 | 不能把不同轮次当成独立图像 |
| `split` | `fit` / `calibration` / `test` | 每个派生表都必须显式检查 |
| `injected` | 是否执行了注入 | injected Non-SDC 不是 clean |
| `is_sdc` | 相对 clean 输出是否改变 | 改变并不意味着显著 |
| `significance` | Judge 输出的显著性级别 | 只接受有效的 0/1/2 |
| `significant_sdc_target` | 二分类监督目标 | Significant SDC 为 1，其余有效评分行为 0 |

每任务计划 5,000 clean + 50,000 injected，九任务计划合计 495,000 次执行。这是采集规模，**不是每一张表的分母**。标签解析失败、不可用 telemetry、分区筛选等都会改变有效行数。

## 5. 三层划分不要混淆

### 5.1 外层划分

固定的 [splits](../splits/) 将输入按 group 分成 Fit 70%、Calibration 15%、Final Test 15%。同一 group 的 clean 及注入执行不能跨分区。比例针对分组划分；行数不要求机械地等于总行数乘比例。

Fit 训练模型，Calibration 冻结阈值，Final Test 做最终评分。不得为了得到更好结果覆盖 manifest。

### 5.2 Predictor 内部划分

仅使用 outer-Fit 的 fault-free Mapping telemetry，再按 group 划分 Predictor Train/Validation/Test = 70%/15%/15%。

- Train 用于优化 Predictor。
- Validation 用于 early stopping。
- Predictor Test 用于 MSE/CosSim 等正常映射质量指标。

Predictor Test 不是 Detector Final Test，也不含故障监督。原始 telemetry 行数不等于独立输入数量，一个输入可贡献多个 step 和 layer pair。

### 5.3 配置选择内部划分

仅取 outer-Fit，再划分 70% model-fit pool、15% threshold calibration、15% selection validation。model-fit pool 内另取 15% group 做 XGBoost early stopping。

40 个候选共享身份和划分；按 selection-validation Full 的九任务 macro F1 排名。outer Calibration 和 outer Final Test 不参与候选排名。

这里的“不使用外层测试”指不将其用于训练、校准、选择指标；源 CSV 可能物理包含多个 split，脚本读取后立即过滤 `split == "fit"`。这不等于所有输入文件都只有 Fit 行。

## 6. Full、Finite 与另外两种“有限”

### 6.1 当前 Detector 的 Finite

固定最终 **36D/K=28** 表征，先执行和模型一致的数值预处理，再要求 36 列全部非缺失：

```python
from detect_sdc.detector.xgboost import get_feature_columns, prepare_features

columns = get_feature_columns(frame)
assert len(columns) == 36
finite_mask = prepare_features(frame, columns).notna().all(axis=1)
```

预处理包括数值转换、将 NaN/Inf 和超出 float32 范围的值视为缺失。XGBoost 可处理缺失特征，因此 Full 不需要删除这些行。

Finite 只改变评分子集，不改变训练集、阈值校准数据或候选排名。消融和配置诊断中的 Finite 也由完整 36D/K=28 参考定义，不可随候选删列而扩张。

当前 Final Test Full 为 74,586 行、2,720 个正例；Finite 为 72,645 行、779 个正例。减少的 1,941 行均为正例，因此 Finite 是强条件筛选，不能视为代表性测试集；两集合 FPR 相同也不意味着检测行为在所有情况下相同。

### 6.2 旧的 Finite

旧数据里存在“参考表征至少一列有效”的筛选，等价于只删除全 NaN 行。它不是当前的 Finite。旧参考也可能是 48D/K=2 或 72D/K=2，不能只靠名字判断，必须读取目录 README 和 `protocol.json` / `summary.json`。

### 6.3 故障标量有限与逐指标有限

图 2 和偏差分桶根据 `fault.before`、`fault.after` 是否有限筛选；分桶还检查差值是否有限。它们不使用 Detector 的 36D mask。

图 7 则逐 layer pair、逐 metric 筛选可用值，并报告 non-finite rate。一个样本可以进入某个 pair 的统计，却不进入另一个 pair。

`features.finite_only: true` 是**单个指标跨 step 聚合时**忽略无效值的设置，也不等于“只保留全部特征有限的样本”。

## 7. 指标和汇总规则

```text
Precision = TP / (TP + FP)
Recall    = TP / (TP + FN)
F1        = 2 TP / (2 TP + FP + FN)
FPR       = FP / (FP + TN)
```

正类是 Significant SDC。FPR 的负类包括 clean、injected Non-SDC、Non-significant SDC，不只是 clean。

- `sdc_recall` 针对所有 SDC，不等于 `significant_sdc_recall`。
- 九任务 macro：先计算各任务指标，再对九个任务等权平均。
- 不要先平均 Precision/Recall 再求 F1，也不要混淆宏平均与合并混淆矩阵的 micro 指标。
- 普通 CSV 指标通常为 `[0,1]`；`*_percent`、`*_pct` 已乘 100，`delta_f1_pp` 是百分点。
- 图 2 pooled share 和偏差分桶是计数比例，不是九任务 macro F1。
- 对阈值边界使用实现规定的 `score > threshold`，不要自行改成 `>=`。
- Bootstrap 按图像/问题 group 重采样，保留组内相关性，不把每次注入当成独立样本。

Full 的主比较结论是 SIEVE 比 Ranger-style/Dr.DNA-style 高 10.33/11.38 个 F1 百分点，不是相对增长 10.33%/11.38%。

## 8. 实验环境

```bash
# 轻量统计/Detector 训练环境，不包含完整 VLM 运行环境。
python -m pip install -e '.[train,dev]'
python -m pip install matplotlib
export PYTHONPATH=src:.
python -m detect_sdc.cli config validate configs/experiments/current.yaml
```

统计与绘图建议使用 Python 3.12 环境。VLM 推理沿用各自的依赖快照，不能将三个模型强行装进同一套 Transformers 环境：

| 模型 | 本机解释器示例 | 环境记录 |
| --- | --- | --- |
| Qwen | `Qwen2.5-VL-7B/.venv/bin/python` | [qwen25_vl.freeze.txt](../reproducibility/environments/qwen25_vl.freeze.txt) |
| InternVL | `InternVL3-8B/.venv/bin/python` | [internvl3.freeze.txt](../reproducibility/environments/internvl3.freeze.txt) |
| LLaVA | `llava-v1.5-7B/.venv/bin/python` | [llava15.freeze.txt](../reproducibility/environments/llava15.freeze.txt) |

模型和数据路径位于 [configs/models](../configs/models/) 与 [configs/datasets](../configs/datasets/)。Judge 配置位于 [current.yaml](../configs/experiments/current.yaml)。这些路径可能指向原实验服务器，clone 后需配置本地路径。

**重要：`current.yaml` 不是“最终 36D/K=28 一键配置”。** 它当前的通用 featurization 默认仍是 72D、prefix K=2。最终比较由 [detection_comparison_k28_36d.yaml](../compare_experiment/configs/detection_comparison_k28_36d.yaml) 配合现存 `step_ablation_36d/<job>/k_28/` 工件实现。仅验证 YAML 成功，不说明将运行最终论文配置。

## 9. 运行与复现纪律

1. 先核对所需输入是否存在，再选择“重绘”“重评分”“重训练”还是“重采集”。
2. 大型运行先单 job 验证，再启动矩阵；不要直接复制八卡 launcher 到单卡机器。
3. 正式产物不应作为试跑输出。支持 `--output-dir` 时写到新的实验目录。
4. `--overwrite`、部分脚本中的 `rm -rf` 或 Detector 输出清理会替换已有文件，必须先检查源码和目标路径。
5. “文件已存在所以跳过”不验证配置一致性。改变 K/profile/seed 后不能复用旧文件冒充新结果。
6. 多进程数乘 XGBoost 线程数是粗略 CPU 并发预算。例如 `12 workers × 8 threads` 可能过度占用机器。
7. 原子写入时 `.tmp` 可能持续增长，而目标文件直到完成才出现；不要以目标文件未出现判断进程卡死。
8. 改变数据、模型、Judge 或环境版本后，保留新的 SHA-256、Git revision、seed 和 split 记录。

## 10. Git 与本地数据

已提交的主要是图、聚合指标、部分逐任务 metrics、配置选择 split manifests 和压缩预测。原始 JSONL、绝大部分逐样本 feature/prediction、VLM 权重及模型 checkpoint 不在仓库。

```bash
# 当前仓库实际追踪的产物才是 clone 后能得到的文件。
git ls-files experiments

# 检查特定文件为什么未追踪。
git check-ignore -v experiments/telemetry_50/qwen25_vl_lingoqa/injection.jsonl
```

某些 JSON 仍记录旧绝对路径 `artifacts/iclr_v2/...`。这是采集时的 provenance，不应为了好看而篡改。运行时显式传入当前实际路径，并以内容 hash 判断是否同一份数据。

原始目录有的保存了 Mapping checkpoint 和 metadata，但已经没有 `mapping.jsonl`；因此“存在模型”不等于“能就地重新训练 Predictor”。

## 11. 已知边界

- 本轮没有补跑全 outer-Fit profile，也没有 20%/50%/100% profile-size 敏感性结果。
- Ranger-style/Dr.DNA-style 是监控层与窗口匹配的 adapted baselines，不是原论文完整系统复现，也不满足相同 clean 数据预算。
- 源码的 profile 子集按 outer-Fit **`orig_id`** 选择，不是精确选择 20% semantic groups。group 仍是外层隔离和 bootstrap 单位。
- 当前主故障模型是 Prefill random activation double-bit；不能外推到所有故障类型或 decode-stage 故障。
- 标签依赖 LLM Judge；这些文件不构成人工一致性审计。
- 迁移使用 target-specific clean Mapping，不是无需目标数据的零样本迁移。
- 固定 source threshold 的迁移 FPR 升高，说明不能宣称稳定通用迁移。
- 本目录没有恢复集成后的纠正率实验，也没有 bit-policy OOD 结果。检测率不能代替纠正率。

## 12. 快速排错

| 现象 | 优先核对 |
| --- | --- |
| 结果和根 README 不一致 | 是否读了旧 `summary`、旧 Finite、72D/K=4 或 step 扫描汇总 |
| `features.csv` 比 `test.csv` 大很多 | 前者是否包含所有 split；后者是否还带历史筛选 |
| `FileNotFoundError` 指向 UBJ/JSONL | 是否只有 Git 中的轻量工件，缺少本地原始数据 |
| 缺少 36 个特征或列名顺序不一致 | 是否错误使用 `current.yaml` 默认 72D/K=2 |
| 候选间 UID/group 不一致 | 特征生成或 split manifest 发生变化；不要静默取交集 |
| Finite F1 突然升高 | 是否又使用了“至少一列有效”，或删列后重定义 cohort |
| 图 2 分母与 Detector 不同 | 一个按 fault scalar 筛选，一个按 36D 表征筛选 |
| 更换 profile 后沿用旧分数 | 旧分数依赖旧 profile，无法直接用于新 profile |
| 迁移对角线校验失败 | 是否沿用了历史 `test.csv` 子集或不同阈值/树数量 |
| 重跑开销脚本却在采集故障 | wrapper 会先补齐缺失的 profile/score campaign；独立调用 benchmark 可隔离此步骤 |

## 13. 交付检查

- 记录每个指标的 partition、cohort、正类、分母、平均方式。
- 核对 Full/Finite 三方法使用相同执行身份。
- 核对配置排名只由 Full Fit-only 指标产生。
- 核对所有组件消融共享同一 36D/K=28 Finite mask。
- 核对迁移对角线复现最终内域比较。
- 核对开销只引用 forward-r10，不混合早期 reverse 运行。
- 保存源 CSV/JSON，不从图中读回精度较低的数字。
- 区分“命令已提供”“轻量检查通过”和“完整实验已重跑”。
