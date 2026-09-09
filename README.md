# SIEVE: Significant Silent Data Corruption Detection

This repository contains the code and experiment protocol for **SIEVE**, a
semantic-aware detector for significant silent data corruptions (SDCs) in
multimodal language models.

SIEVE monitors discrepancies between adjacent transformer layers, predicts the
fault-free representation with a lightweight mapping model, and uses an
XGBoost detector to identify corruptions that are likely to cause a significant
semantic failure. The repository is organized for reproducible evaluation
across three vision-language models and three multimodal benchmarks.

> The public repository contains source code, configuration, tests, and
> reproducible derived results. Model checkpoints, datasets, and large JSONL
> telemetry remain local and are intentionally kept outside Git.

## Highlights

- Significant-SDC detection rather than generic output-change detection.
- Group-aware `Fit` / `Calibration` / `Final Test` isolation.
- Prefill activation fault injection with configurable bit policies.
- Layer-aware residual mapping model for fault-free feature prediction.
- Fit-only configuration selection for the final 36D/`K=28` monitor.
- Complete 9-task evaluation matrix:
  Qwen2.5-VL, LLaVA-1.5, and InternVL3 on EarthVQA, LingoQA, and VQAv2.
- Online monitoring and cross-domain detector-transfer experiments.

## Repository Status

The current experiment track uses 50-step telemetry and selects its final
configuration strictly within the outer Fit partition:

```text
(6, 7), (24, 25), (26, 27), K=28
```

Raw telemetry, clean profiles, and Predictor checkpoints are stored below
`experiments/telemetry_50/<job>/`. Derived 36D features, Fit-only selection,
and component ablations are stored under `experiments/`.

## Method

The canonical experiment evaluates:

- **Models:** Qwen2.5-VL-7B, LLaVA-1.5-7B, and InternVL3-8B.
- **Datasets:** EarthVQA, LingoQA, and VQAv2.
- **Fault:** one Prefill activation double-bit flip on an eligible linear
  operation.
- **Fault policy:** `random` for the main campaign.
- **Runs:** one clean execution and ten fault runs per input.
- **Monitoring:** projected inter-layer discrepancies over decoding steps.
- **Features:** cosine similarity, mean difference, standard deviation
  difference, and L2 distance, aggregated with mean/max/min.
- **Target:** `significant_sdc_target`, assigned by the Prometheus judge when a
  corruption produces a severity-2 semantic failure.

The evaluation protocol is strictly separated:

```text
Fit            train Mapping and Detector
Calibration    freeze the operating threshold
Final Test     report metrics only
```

Splits are made by `semantic_group_id`, so related frames or questions cannot
cross the outer partition boundary:

- EarthVQA: image group
- LingoQA: question group
- VQAv2: image group

## Reported Results

The Fit-only-selected 36D/`K=28` detector reports the following nine-task
Final-Test macro averages:

| Cohort | Precision | Recall | F1 | FPR |
| --- | ---: | ---: | ---: | ---: |
| Full | 96.38% | 91.49% | 93.78% | 0.107% |
| Canonical-finite | 91.20% | 84.15% | 87.18% | 0.107% |

Configuration selection and ablation results are available in:

```text
experiments/fit_only_configuration_selection/
experiments/ablation_36d_k28_fit_only/
experiments/appendix_36d_k28/
```

## Installation

The pipeline requires Python 3.10 or newer. Install the package and the
CPU-side training dependencies with:

```bash
python -m pip install -e '.[train,dev]'
```

Model-specific environments are documented by the frozen dependency files:

```text
reproducibility/environments/qwen25_vl.freeze.txt
reproducibility/environments/internvl3.freeze.txt
reproducibility/environments/llava15.freeze.txt
```

The reference setup uses separate environments because the three model
adapters require different Transformers and model-runtime versions.

## Data and Model Preparation

The repository does not redistribute model checkpoints, datasets, or judge
weights. Set local paths in:

```text
configs/models/qwen25_vl.yaml
configs/models/llava15.yaml
configs/models/internvl3.yaml
configs/datasets/earthvqa.yaml
configs/datasets/lingoqa.yaml
configs/datasets/vqav2.yaml
```

The expected external inputs are:

```text
Qwen2.5-VL-7B-Instruct
InternVL3-8B
llava-v1.5-7b
LLaVA source tree
Prometheus judge checkpoint
EarthVQA, LingoQA, and VQAv2 data
```

Do not commit these inputs or generated JSONL artifacts.

## Quickstart

Validate the experiment configuration without loading model weights:

```bash
PYTHONPATH=src python -m detect_sdc.cli config validate \
  configs/experiments/current.yaml
```

Create or validate the shared semantic-group manifests:

```bash
PYTHONPATH=src python -m detect_sdc.cli split --dataset earthvqa
PYTHONPATH=src python -m detect_sdc.cli split --dataset lingoqa
PYTHONPATH=src python -m detect_sdc.cli split --dataset vqav2
```

Run a dry-run for one configured job:

```bash
PYTHONPATH=src python -m detect_sdc.cli run \
  --job qwen25_vl_earthvqa \
  --stage inject \
  --dry-run
```

## Full Pipeline

For one model/dataset job, the stages are executed in this order:

```text
profile
collect_mapping
train_mapping
inject
label
featurize
train_detector
report
```

Example:

```bash
export PYTHONPATH=src:.
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false

PYTHON=Qwen2.5-VL-7B/.venv/bin/python
JOB=qwen25_vl_earthvqa

for STAGE in profile collect_mapping train_mapping inject label \
    featurize train_detector report; do
  "$PYTHON" -m detect_sdc.cli run \
    --job "$JOB" \
    --stage "$STAGE" \
    --device cuda:0
done
```

Before a long GPU run, use `--dry-run` and verify the configured output paths.
Injection outputs are written atomically and can be resumed from a completed
fault-run boundary.

## Telemetry-Derived Studies

The 50-step telemetry campaign is complete. New feature configurations are
derived from `experiments/telemetry_50/` without repeating fault injection.
The final configuration uses the first `min(28, T)` observed decoding steps;
early EOS never forces generation to 28 tokens.

## Baseline Comparison

`compare_experiment/` evaluates Ranger-style, Dr.DNA-style, and SIEVE signals
on the same fault executions. This avoids comparing methods that observed
different injected samples.

```bash
tmux new-session -d -s sieve_comparison_36d_k28 \
  'cd /data01/cd_workspace/Detect_SDC && bash scripts/run_comparison_36d_k28_all.sh'
```

## Reproducibility

The detailed protocol and environment assumptions are documented in:

- [`docs/reproducibility.md`](docs/reproducibility.md)
- [`docs/xgboost_current_methods_and_results.md`](docs/xgboost_current_methods_and_results.md)
- [`reproducibility/reference_sha256.txt`](reproducibility/reference_sha256.txt)

The most important invariants are:

- zero overlap between outer Fit, Calibration, and Final-Test groups;
- Mapping and Detector training use Fit data only;
- Calibration labels are not used to train the Detector;
- Final Test is never used for feature or threshold selection;
- all clean and fault runs are retained;
- generated outputs use stable sample UIDs and atomic writes.

## Testing

Run the unit-test suite with:

```bash
python -m pytest
```

Or use the standard library test runner:

```bash
PYTHONPATH=src python -m unittest discover tests
```

## Citation

The repository name is now `SIEVE_ICLR-27`. Citation metadata will be added
when the associated paper record is public.

## License

No open-source license has been declared for this repository yet. Please
contact the authors before redistributing the code or generated artifacts.
