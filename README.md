<p align="center" style="margin: 0 0 1px 0;">
  <img src="docs/assets/logo.svg" alt="TensorFoundry logo" width="200">
</p>

<h1 align="center" style="margin: 0 0 10px 0;">
  <span style="color:#FF7A18;">Tensor</span><span style="color:#1F6FEB;">Foundry</span>
</h1>

<p align="center">
  <strong>Agent engineering and learning system for production workflows</strong><br/>
  Build runtime traces, evaluate failures, generate datasets, distill specialists, and benchmark trade-offs.
</p>

<p align="center">
  <img alt="status" src="https://img.shields.io/badge/status-active_development-blue" />
  <img alt="python" src="https://img.shields.io/badge/python-3.11%2B-purple" />
  <img alt="license" src="https://img.shields.io/badge/License-Apache%202.0-green.svg" />
  <img alt="ci" src="https://img.shields.io/github/actions/workflow/status/foaadfarooghian/tensorfoundry/ci.yml?branch=dev" />
</p>

---

## Project status

**TensorFoundry is in active, pre-1.0 development (`v0.x`).**

- APIs and schemas are still converging
- Breaking changes are expected while core runtime contracts are stabilized
- Best fit today: internal platforms, research, and production pilots with pinned versions

---

## Narrowed product scope

TensorFoundry is narrowing to five pillars:

1. **OTel/MCP-native runtime + artifact schema**
2. **Evaluation and failure analysis for multi-step, tool-using agents**
3. **Dataset generation from production traces**
4. **Distillation pipeline for specialist small models**
5. **Benchmarking cost, latency, and reliability across models and agent patterns**

This is a systems-first direction: execution artifacts and measurable outcomes come before model hype.

---

## Pillars in practice

### 1) OTel/MCP-native runtime + artifact schema
- Runtime events map cleanly to spans/events for distributed observability
- MCP tool calls are first-class execution units
- Shared artifact contracts for traces, rewards, eval verdicts, and dataset rows

### 2) Evaluation + failure analysis
- Suite-based evaluation for multi-step workflows
- Step-level and run-level scoring
- Failure taxonomy for tool errors, reasoning failures, recovery failures, and policy failures
- Regression diffing across runs, models, and orchestration patterns

### 3) Dataset generation from production traces
- Deterministic trace ETL into SFT, preference, repair, and critique datasets
- Provenance from dataset row back to trace/reward artifacts
- Data quality checks for schema validity, leakage risk, and label consistency

### 4) Distillation for specialist small models
- Teacher traces -> curated supervision -> student training/eval loops
- Focus on narrow specialist capabilities rather than general chat
- Reproducible train/eval pipelines for iterative deployment

### 5) Cost/latency/reliability benchmarking
- Comparable benchmark matrix across model providers and orchestration patterns
- Explicit trade-off reporting (quality vs cost vs latency vs failure rate)
- Reliability metrics for retries, tool success, and degraded-mode completion

---

## Specialist model exchange

TensorFoundry is expanding toward a specialist model registry/exchange where the
published unit is a **complete deployable package**, not only weights.

Each exchange unit includes:

- Small domain model
- Eval pack
- Trace/dataset lineage
- Hardware profile
- Failure modes
- License and usage constraints
- Ready-to-run artifacts (adapters, Safetensors/GGUF, Ollama packaging)

Use `tensorfoundry-exchange` to build and validate `specialist_model_unit.v0`
manifests from training, distillation, and benchmark evidence. See
`docs/model_exchange.md`, `docs/specialist_model_unit_sample.json`, and
`docs/specs/specialist_model_unit.schema.json` for the contract.

Good early domains:

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

---

## What is already in this repo

- Structured JSONL tracing with validation/inspection/diff tooling
- Evaluation harnesses and benchmark suites with reward artifacts
- Dataset export pipelines (SFT, preferences, repair pairs, curriculum)
- Learning/routing infrastructure and experimental SFT/DPO training utilities
- Multi-provider model abstraction (OpenAI, Ollama, HF, dummy)

---

## What TensorFoundry is not

- A chatbot framework
- A prompt library
- A no-code builder
- A model leaderboard without task context
- A fixed set of built-in agents

---

## Quickstart

```bash
git clone https://github.com/foaadfarooghian/tensorfoundry.git
cd tensorfoundry

python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Run a reference agent example:

```bash
python examples/quickstart_research_agent.py
```

Validate and inspect the latest trace:

```bash
python -m tensorfoundry.logging.validate logs/$(ls -t logs | head -n 1)
python -m tensorfoundry.logging.inspect logs/$(ls -t logs | head -n 1)
```

Run evaluation suites:

```bash
python -m tensorfoundry.evaluation.run src/tensorfoundry/evaluation/suites/quickstart.json
python -m tensorfoundry.evaluation.run src/tensorfoundry/evaluation/suites/research_quickstart.json
```

Run the deterministic production-pilot readiness loop:

```bash
tensorfoundry-pilot-check --mode dummy --work-dir results/pilot_check
```

This writes:
- `results/pilot_check/pilot_readiness.md`
- `results/pilot_check/pilot_readiness.json`
- `results/pilot_check/logs/` with `trace.v0` and `reward.v0` artifacts
- `results/pilot_check/datasets/manifest.json` plus SFT, preference, repair, and curriculum exports

Pilot datasets are strict-gated by default: exported rows include deterministic
split metadata, provenance links back to trace/reward artifacts, file hashes,
duplicate counts, and leakage checks. Validate an exported dataset directly with:

```bash
tensorfoundry-dataset-validate results/pilot_check/datasets/pilot.sft.jsonl \
  --kind sft --quality-gate --logs-root results/pilot_check/logs
```

Run a release regression gate by comparing against the last accepted readiness
artifact:

```bash
tensorfoundry-pilot-check --mode dummy --work-dir results/pilot_current \
  --baseline results/pilot_baseline/pilot_readiness.json
```

When `--baseline` is supplied, the command also writes:
- `results/pilot_current/eval_regression.md`
- `results/pilot_current/eval_regression.json`

The default deterministic gate allows no pass-rate drop, no mean-score drop, no
new failing cases, and no worse failure-mode movement.

Optional provider smoke checks can be required in configured environments:

```bash
tensorfoundry-pilot-check --require-provider hosted
tensorfoundry-pilot-check --require-provider local
```

Training remains experimental, but preflight evidence is available without loading models:

```bash
tensorfoundry-learn train --base-model dummy/base --sft --dpo \
  --sft-data results/pilot_check/datasets/pilot.sft.jsonl \
  --dpo-data results/pilot_check/datasets/pilot.dpo.jsonl \
  --dry-run --quality-gate --logs-root results/pilot_check/logs \
  --report-out results/pilot_check/training_preflight.json
```

Release environments with `[train]` dependencies installed can opt into a
bounded SFT smoke run and record `training_run.v0` evidence:

```bash
tensorfoundry-learn train --base-model hf/org/base --sft \
  --sft-data results/pilot_check/datasets/pilot.sft.jsonl \
  --sft-out results/training/sft_lora \
  --quality-gate --logs-root results/pilot_check/logs \
  --report-out results/pilot_check/training_preflight.json \
  --run-report-out results/training/sft_training_run.json \
  --smoke --max-steps 1
```

DPO evidence is opt-in and must point at successful parent SFT run evidence.
When a DPO run is present, pass its run report to exchange packaging so the
final adapter refs and checksums describe the preference-optimized artifact:

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

To have the pilot loop generate DPO-compatible preference data and attach the
preflight summary to readiness output:

```bash
tensorfoundry-pilot-check --mode dummy --training-preflight \
  --work-dir results/pilot_check
```

Run the evidence-only distillation eval gate from the generated recipe:

```bash
tensorfoundry-distill-check \
  --recipe results/pilot_check/distillation_recipe.json \
  --work-dir results/distillation_gate
```

This compares a candidate specialist model against a baseline/teacher model and
writes `distillation_eval.v0` JSON and Markdown evidence without requiring real
training.

Generate benchmark frontier evidence across suites and model ids:

```bash
tensorfoundry-benchmark-matrix \
  --config docs/benchmark_matrix_sample.json \
  --work-dir results/benchmark_matrix
```

This writes `benchmark_matrix.v0` JSON and Markdown with task success, cost per
successful task, latency p50/p95, reliability fields, mean effective score, and
frontier picks. Dummy rows are deterministic and mandatory; hosted/local/HF rows
skip unless their provider is explicitly required.

Build and index a local specialist exchange unit from the release evidence:

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

---

## Repository layout

```text
src/tensorfoundry/
├── agents/          # Reference agents used by eval suites
├── orchestration/   # Multi-step execution patterns
├── logging/         # Trace schema, emitter, validation, inspection
├── evaluation/      # Suites, harness, scoring, reporting
├── export/          # Trace -> dataset transformations
├── learning/        # Routing and learning loop primitives
└── training/        # Experimental SFT/DPO components
```

Top-level runtime assets:
- `logs/` -> execution traces and reward artifacts
- `datasets/` -> generated learning datasets
- `results/` -> evaluation outputs and summaries

See `manifesto.md` for principles and `roadmap.md` for the focused build plan.

---

## Branching & releases

- `prod` -> protected, tagged releases
- `dev` -> integration branch for ongoing work
