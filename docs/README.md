# Documentation

Concepts and design notes for TensorFoundry's narrowed scope: agent runtime,
evaluation, trace-to-learning, and benchmarking.

- `design_principles.md`: core principles for runtime, eval, data, and learning.
- `model_exchange.md`: specialist model registry/exchange contract and lifecycle.
- `pilot_readiness_sample.md`: example output from the production-pilot readiness check.
- `training_preflight_sample.json`: example `training_preflight.v0` evidence report.
- `training_run_sample.json`: example `training_run.v0` SFT execution evidence.
- `training_run_dpo_sample.json`: example `training_run.v0` DPO execution evidence.
- `release_candidate_sample.json`: example `release_candidate.v0` bundled gate evidence.
- `distillation_recipe_sample.json`: example `distillation_recipe.v0` gate recipe.
- `benchmark_matrix_sample.json`: example offline `benchmark_matrix.v0` config.
- `specialist_model_unit_sample.json`: example `specialist_model_unit.v0` manifest.
- `specialist_registry_index_sample.json`: example `specialist_registry_index.v0` index.
- `specs/specialist_model_unit.schema.json`: machine-readable manifest schema for exchange units.
- `../roadmap.md`: active workstreams and exit criteria.
- `../manifesto.md`: product philosophy and explicit non-goals.

## Pilot readiness

Run the offline deterministic readiness loop with:

```bash
tensorfoundry-pilot-check --mode dummy --work-dir results/pilot_check
```

The command produces a readiness report, validated trace/reward artifacts, and
validated SFT, preference, repair, and curriculum datasets. Hosted and local
provider checks are automated but optional unless explicitly required with
`--require-provider hosted`, `--require-provider local`, or
`--require-provider all`.

Dataset validation in the pilot loop runs as a hard quality gate. It checks
schema validity, deterministic split metadata, trace/reward provenance refs,
content hashes, duplicate payloads, and split leakage. The same strict path is
available through:

```bash
tensorfoundry-dataset-validate results/pilot_check/datasets/pilot.sft.jsonl \
  --kind sft --quality-gate --logs-root results/pilot_check/logs
```

For release checks, keep the last accepted `pilot_readiness.json` and compare
the current run against it:

```bash
tensorfoundry-pilot-check --mode dummy --work-dir results/pilot_current \
  --baseline results/pilot_baseline/pilot_readiness.json
```

Baseline mode emits `eval_regression.v0` as JSON and Markdown, and the readiness
command exits nonzero when the configured regression policy is violated.

## Release candidate gate

Run the complete deterministic release-candidate bundle with:

```bash
tensorfoundry-release-candidate-check \
  --work-dir results/release_candidate
```

The command writes `release_candidate.json` and `release_candidate.md`, then
stores child evidence under stable subdirectories for pilot readiness, mock
SFT/DPO training runs, distillation eval, benchmark matrix, specialist exchange
manifest, package check, smoke run, and registry index. Existing real training
evidence can replace generated mock evidence:

```bash
tensorfoundry-release-candidate-check \
  --work-dir results/release_candidate \
  --sft-run results/training/sft_training_run.json

tensorfoundry-release-candidate-check \
  --work-dir results/release_candidate \
  --dpo-run results/training/dpo_training_run.json
```

## Training readiness

Run a dry-run training preflight against pilot-generated datasets with:

```bash
tensorfoundry-pilot-check --mode dummy --training-preflight \
  --work-dir results/pilot_check
```

This writes `training_preflight.v0` evidence without loading models or launching
training. Standalone preflight is available with:

```bash
tensorfoundry-learn train --base-model dummy/base --sft --dpo \
  --sft-data results/pilot_check/datasets/pilot.sft.jsonl \
  --dpo-data results/pilot_check/datasets/pilot.dpo.jsonl \
  --dry-run --quality-gate --logs-root results/pilot_check/logs \
  --report-out results/pilot_check/training_preflight.json
```

Release environments can optionally run one bounded SFT smoke step and record
`training_run.v0`:

```bash
tensorfoundry-learn train --base-model hf/org/base --sft \
  --sft-data results/pilot_check/datasets/pilot.sft.jsonl \
  --sft-out results/training/sft_lora \
  --quality-gate --logs-root results/pilot_check/logs \
  --report-out results/pilot_check/training_preflight.json \
  --run-report-out results/training/sft_training_run.json \
  --smoke --max-steps 1
```

DPO smoke evidence is also optional. It requires the successful SFT
`training_run.v0` report through `--sft-run`, records parent lineage, and makes
the DPO adapter the final exchange artifact:

```bash
tensorfoundry-learn train --base-model hf/org/base --dpo \
  --dpo-data results/pilot_check/datasets/pilot.dpo.jsonl \
  --sft-run results/training/sft_training_run.json \
  --dpo-out results/training/dpo_lora \
  --quality-gate --logs-root results/pilot_check/logs \
  --report-out results/pilot_check/training_preflight.json \
  --run-report-out results/training/dpo_training_run.json \
  --smoke --max-steps 1
```

Then run the distillation eval gate:

```bash
tensorfoundry-distill-check \
  --recipe results/pilot_check/distillation_recipe.json \
  --work-dir results/distillation_gate
```

The gate emits `distillation_eval.v0` as JSON and Markdown, comparing the
candidate specialist model against the baseline/teacher model from the recipe.

Generate benchmark matrix and frontier evidence:

```bash
tensorfoundry-benchmark-matrix \
  --config docs/benchmark_matrix_sample.json \
  --work-dir results/benchmark_matrix
```

The matrix emits `benchmark_matrix.v0` JSON and Markdown with scorecards for
success, cost, latency, reliability, and effective-score frontier picks.

Build a local exchange unit from the generated evidence:

```bash
tensorfoundry-exchange build-unit \
  --training-preflight results/pilot_check/training_preflight.json \
  --training-run results/training/dpo_training_run.json \
  --distillation-eval results/distillation_gate/distillation_eval.json \
  --benchmark-matrix results/benchmark_matrix/benchmark_matrix.json \
  --out results/exchange/pilot-specialist.unit.json \
  --id pilot-specialist --name "Pilot Specialist" --version 0.1.0 --domain pilot \
  --model-family dummy --model-size 0B --model-format safetensors \
  --model-license Apache-2.0 --dataset-license CC-BY-4.0 \
  --usage-constraint "not for production decisions without review" \
  --failure-mode dummy_only \
  --failure-description "Dummy artifacts only prove exchange plumbing." \
  --failure-mitigation "Replace dummy refs before release." \
  --safetensors-ref hf://tensorfoundry/pilot-specialist/model.safetensors \
  --ollama-modelfile hf://tensorfoundry/pilot-specialist/Modelfile \
  --ollama-tag tensorfoundry/pilot-specialist:0.1.0
```

Attach packaging evidence, run the offline consumer smoke check, then validate
and index it:

```bash
tensorfoundry-exchange package-check \
  --manifest results/exchange/pilot-specialist.unit.json \
  --out results/exchange/specialist_package.json \
  --package-type auto --release-ready --update-manifest

tensorfoundry-exchange smoke-run \
  --manifest results/exchange/pilot-specialist.unit.json \
  --work-dir results/exchange/smoke --update-manifest

tensorfoundry-exchange validate \
  --manifest results/exchange/pilot-specialist.unit.json --release-ready
tensorfoundry-exchange index \
  --registry-dir results/exchange --out results/exchange/index.json --release-ready
```
