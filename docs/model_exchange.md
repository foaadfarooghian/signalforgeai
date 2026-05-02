# Specialist Model Exchange

This document defines TensorFoundry's specialist model registry/exchange model.
The goal is to publish deployable specialist units, not generic checkpoints.

## Winning unit

The exchange publishes a **specialist model unit**. Each unit must include:

1. A small domain model
2. Its eval pack
3. Its trace/dataset lineage
4. Its hardware profile
5. Its failure modes
6. Its license and usage constraints
7. Ready-to-run artifacts (adapters, Safetensors/GGUF, Ollama packaging)

If any part is missing, the unit is incomplete and should not be listed as
production-ready.

## Why this shape

TensorFoundry optimizes for execution systems. A model alone is not enough.
The unit has to be tied to:

- Runtime evidence (traces)
- Evaluation evidence (task outcomes and failure profiles)
- Operational evidence (hardware/cost/latency characteristics)
- Legal evidence (license and usage constraints)

This makes specialist models comparable and safe to deploy.

## Standards alignment

The exchange should align with OpenTelemetry GenAI semantic conventions and MCP
tool semantics, so runtime traces, evaluation outcomes, and model artifacts can
be connected through shared IDs and attributes.

Minimum linkage expectations:

- `trace_id`/`span_id` references from eval and lineage metadata
- Stable run IDs for benchmark reports
- Tool invocation metadata linked to MCP contracts
- Model identifiers linked to packaged artifact digests

## Unit manifest (v0)

Each unit should provide a manifest validated against
`docs/specs/specialist_model_unit.schema.json`.

Example:

```json
{
  "schema_version": "specialist_model_unit.v0",
  "id": "nutrition-extractor-mini",
  "name": "Nutrition Extractor Mini",
  "version": "0.1.0",
  "domain": "nutrition",
  "model": {
    "family": "llama",
    "size": "0.5B",
    "format": "safetensors"
  },
  "eval_pack": {
    "suite_id": "nutrition_v1",
    "report_path": "results/nutrition_v1.summary.md",
    "score": 0.91
  },
  "lineage": {
    "trace_sources": ["logs/run_2026_03_08"],
    "dataset_sources": ["datasets/nutrition/sft_v1.jsonl"],
    "distillation_recipe": "learning_ops/recipes/nutrition_v1.yaml"
  },
  "hardware_profile": {
    "target": "cpu",
    "ram_gb": 8,
    "latency_ms_p50": 220
  },
  "failure_modes": [
    {
      "id": "unit_mismatch",
      "description": "Confuses mg and g in edge cases",
      "mitigation": "post-parse unit normalizer"
    }
  ],
  "license": {
    "model_license": "Apache-2.0",
    "dataset_license": "CC-BY-4.0"
  },
  "usage_constraints": [
    "not for medical diagnosis",
    "requires deterministic parser for final output"
  ],
  "artifacts": {
    "adapters": ["hf://org/nutrition-mini-lora"],
    "safetensors": ["s3://tf-models/nutrition-mini/model.safetensors"],
    "gguf": ["s3://tf-models/nutrition-mini/model-q4.gguf"],
    "ollama": {
      "modelfile": "artifacts/ollama/Modelfile",
      "tag": "tf/nutrition-mini:0.1.0"
    }
  }
}
```

## CLI workflow

Build a local exchange unit from release evidence:

```bash
tensorfoundry-exchange build-unit \
  --training-preflight results/pilot_check/training_preflight.json \
  --training-run results/training/training_run.json \
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

Attach package evidence, run a consumer smoke check, then validate and index
local units:

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

`--training-run` is optional. When supplied, `training_run.v0` output artifacts
and checksums are copied into the unit unless explicit artifact refs are passed.
`package-check` emits `specialist_package.v0` evidence for adapter, Safetensors,
GGUF, and Ollama references. `smoke-run` emits `specialist_smoke.v0` evidence
for the consumer-facing load/generate path. The default smoke mode is
deterministic dummy execution; Ollama and HF modes remain optional unless
explicitly required.

`--release-ready` is stricter than schema validation: it checks evidence links,
local artifact checksums, package evidence, usage constraints, failure modes,
and runnable artifact refs. URI artifact refs are accepted without network
access.

## Registry lifecycle

1. Build the specialist unit from training, distillation, and benchmark evidence.
2. Attach package evidence for runnable artifact refs.
3. Run the consumer smoke check for the intended load path.
4. Validate manifest schema, evidence links, and artifact integrity.
5. Publish with versioned metadata and immutable artifact references.
6. Track failure mode drift and reliability regressions over time.

## Publication gates (recommended)

- Task score above domain threshold
- Reliability threshold met on retry/tool metrics
- Cost/latency frontier better than baseline general model
- License and usage constraints explicitly reviewed
- Artifact integrity checksums present

## Early domain priorities

- Nutrition
- Auction houses
- Document-heavy verticals
- Compliance
- Support operations
- Telecom workflows
- Cataloguing
- Extraction
- Ranking
- Summarization

These domains are narrow enough for specialist distilled models to win on
cost, latency, and consistency when tasks are well-scoped.
