# Documentation

Concepts and design notes for TensorFoundry's narrowed scope: agent runtime,
evaluation, trace-to-learning, and benchmarking.

- `design_principles.md`: core principles for runtime, eval, data, and learning.
- `model_exchange.md`: specialist model registry/exchange contract and lifecycle.
- `pilot_readiness_sample.md`: example output from the production-pilot readiness check.
- `training_preflight_sample.json`: example `training_preflight.v0` evidence report.
- `distillation_recipe_sample.json`: example `distillation_recipe.v0` gate recipe.
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

Then run the distillation eval gate:

```bash
tensorfoundry-distill-check \
  --recipe results/pilot_check/distillation_recipe.json \
  --work-dir results/distillation_gate
```

The gate emits `distillation_eval.v0` as JSON and Markdown, comparing the
candidate specialist model against the baseline/teacher model from the recipe.
