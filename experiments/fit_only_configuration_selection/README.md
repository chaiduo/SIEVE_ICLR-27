# Fit-only 配置选择：4×10 候选矩阵

[返回总手册](../README.md)

本实验回答：在不依赖 outer Final Test 表现的前提下，如何确定 layer-pair profile 和监测窗口 K？

**正式选择：36D/K=28。选择指标：九任务 Full macro Significant-SDC F1。**

![Configuration matrix](configuration_f1_matrix_final_layout.png)

## 1. 候选与层对

| profile | layer pairs |
| --- | --- |
| 36D | `(6,7)`、`(24,25)`、`(26,27)` |
| 48D | `(6,7)`、`(22,23)`、`(25,26)`、`(26,27)` |
| 60D | `(6,7)`、`(22,23)`、`(23,24)`、`(24,25)`、`(25,26)` |
| 72D | `(6,7)`、`(22,23)`、`(23,24)`、`(24,25)`、`(25,26)`、`(26,27)` |

K 为 `1,2,4,8,12,16,20,24,28,32`。共 40 个候选，九任务共 360 次 Detector 拟合。各 profile 的层对并非严格逐个添加，因此这是配置选择，不是纯维度的受控消融。

## 2. 三层内部划分

```text
outer-Fit
  70% model-fit pool
    85% model training
    15% early stopping
  15% threshold calibration
  15% selection validation
```

按 `semantic_group_id` 划分，同一候选和不同候选使用同一组身份。划分 seed 为 `20260907`；threshold calibration 的划分用 seed+1，early stopping 用 seed+2。

`72D/K=2` 源特征仅用于构造公共 Fit 身份清单。每个候选必须与 manifest 的 outer-Fit UID/group 集合完全一致，不允许静默取交集。

脚本会读取完整 feature CSV 后选择 `split == "fit"`；外层 Calibration/Test 行不进入训练、阈值或排名。

## 3. 训练和排序

XGBoost 参数继承 `current.yaml`，但本实验将 random state 改为选择 seed，并关闭 verbose。每个候选独立训练、独立在内层 calibration 上最大化 F1，之后冻结阈值评估 selection validation。

排名顺序：

1. Full macro F1 高者优先。
2. 同分时 Full macro recall 高者优先。
3. 再同分时 K 较小者优先。
4. 再同分时特征数较少者优先。

不使用 Finite 排名。不能看完 Finite 后再改变 tie-breaker。

## 4. 主要结果

| 冻结配置 | Full F1 | Finite F1 |
| --- | ---: | ---: |
| 36D/K=28 | 93.42% | 71.49% |

这里是内层 selection-validation 的宏平均，不是最终测试的 93.78%/68.78%。

![36D window diagnostics](window_ablation_36d_f1.png)

图 9 左右面板分别展示 Full、Finite，每个 K 点都标数值。虚线 K=28 来自 Full 排名，不是分别为两个面板选择最优点。

## 5. 文件字典

| 文件 | 内容 |
| --- | --- |
| [protocol.json](protocol.json) | 比例、seed、任务及禁止用于选择的外层分区 |
| `splits/<job>.json` | 共享分组、outer-Fit UID、各分区行/组数 |
| `task_metrics/<job>__<profile>.csv` | 每任务每 profile 的十个 K 指标 |
| `predictions/<job>__<profile>.csv.gz` | 内层验证概率、阈值和预测，用于后处理 |
| [detailed_metrics.csv](detailed_metrics.csv) | 当前文件 720 行，其中 Full 360 行，其余为旧 cohort |
| [macro_metrics.csv](macro_metrics.csv) | 当前文件 80 行，其中 Full 40 行，其余为旧 cohort |
| [configuration_ranking.csv](configuration_ranking.csv) | 排序结果 |
| [selection.json](selection.json) | 冻结配置及选择规则 |
| [strict_finite_protocol.json](strict_finite_protocol.json) | 固定 Finite 定义 |
| [strict_finite_detailed_metrics.csv](strict_finite_detailed_metrics.csv) | 360 行 Finite 后处理明细 |
| [strict_finite_macro_metrics.csv](strict_finite_macro_metrics.csv) | 40 行 Finite 宏平均 |

`selection_threshold` 是内层阈值，不是部署阈值。`support` 是正类数量，`selection_validation_rows` 才是该 cohort 的总评分行数。`calibration_*` 用于解释阈值选取，不是最终泛化指标。

现存 CSV 仍含 `canonical_finite` 历史行；读取上述两个主表必须显式筛选 `cohort == "full"`。当前脚本在全新目录运行只生成 Full，文件将分别为 360/40 行。不要为了凑成相同行数把旧 cohort 拼回去，当前 Finite 只从独立 `strict_finite_*` 文件读取。

## 6. Full 选择重跑

输入是四个 `step_ablation_*` 的完整特征表，不需要重新 VLM 推理。但只 clone Git 通常缺少这些表。

```bash
PYTHONPATH=src:. python scripts/run_fit_only_configuration_selection.py \
  --feature-root experiments \
  --output-dir experiments/reproduced_configuration_selection \
  --profiles 36D,48D,60D,72D \
  --k-values 1,2,4,8,12,16,20,24,28,32 \
  --seed 20260907 --workers 4 --xgb-n-jobs 4
```

CPU 并发大致为 workers×threads；这里选择 4×4，非原脚本默认 12×8。并发设置不等于额外实验 seed。

默认可复用已有 task metrics/predictions；若配置改变却使用同一路径，复用并不会自动验证所有超参数。正式重跑应使用新的输出目录。

## 7. Finite 后处理

固定最终 36D/K=28 内层验证集合的全部 36 特征有限行，对所有候选使用相同 UID 子集。复用已有概率和阈值，不重新训练、不重新校准：

```bash
PYTHONPATH=src:. python scripts/reevaluate_fit_only_selection_strict_finite.py \
  --feature-root experiments \
  --selection-root experiments/reproduced_configuration_selection
```

该命令需要 36D/K=28 完整特征、split manifests 和所有候选的 `.csv.gz` 预测。只拿一张宏平均表不能重新计算 Finite。

## 8. 重绘

只重绘图 8 可读已提交的两份宏平均表：

```bash
PYTHONPATH=src:. python scripts/plot_fit_only_configuration_matrix.py \
  --output-prefix experiments/fit_only_configuration_selection/configuration_f1_matrix_final_layout
```

图 9 脚本固定读取本目录，没有自定义 CLI 路径参数：

```bash
PYTHONPATH=src:. python scripts/plot_36d_window_ablation.py
```

以上绘图会更新对应图片及部分导出表；调试样式前注意保护版本。目录里的 `styled`、`spaced`、`final` 等是布局迭代，不代表不同训练实验。含 `canonical_finite` 的旧表不作为当前 Finite 来源。

## 9. 验收

完整运行应有九任务、四 profile、十 K；筛选后 Full 明细 360 行、宏平均 40 行。Finite 后处理也对应 360/40 行。现存带历史 cohort 文件的总行数较多，不能按总行数直接判失败。排名第一应在既有输入/环境下复现 36D/K=28。

若 Full 结果偏离，先检查特征 hash、split manifest、依赖、seed 和候选身份，不要通过读取 Test 来“纠正”排名。组件消融会再次验证完整 36D 的结果是否复现本实验。
