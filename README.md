<p align="center" style="margin: 0 0 1px 0;">
  <img src="docs/assets/logo.svg" alt="SignalForge AI logo" width="200">
</p>

<h1 align="center" style="margin: 0 0 10px 0;">
  <span style="color:#FF7A18;">Signal</span><span style="color:#1F6FEB;">Forge</span><span style="color:#124AA7;"> AI</span>
</h1>

<p align="center">
  <strong>Agent engineering and learning system for production workflows</strong><br/>
  Build runtime traces, evaluate failures, generate datasets, distill specialists, and benchmark trade-offs.
</p>

<p align="center">
  <img alt="status" src="https://img.shields.io/badge/status-active_development-blue" />
  <img alt="python" src="https://img.shields.io/badge/python-3.11%2B-purple" />
  <img alt="license" src="https://img.shields.io/badge/License-Apache%202.0-green.svg" />
  <img alt="ci" src="https://img.shields.io/github/actions/workflow/status/foaadfarooghian/signalforgeai/ci.yml?branch=prod" />
  <a href="https://pypi.org/project/signalforgeai/"><img alt="pypi" src="https://img.shields.io/pypi/v/signalforgeai" /></a>
</p>

---

## Project status

**SignalForge AI is in active, pre-1.0 development (`v0.x`).**

- APIs and schemas are still converging
- Breaking changes are expected while core runtime contracts are stabilized
- Best fit today: internal platforms, research, and production pilots with pinned versions

---

## Narrowed product scope

SignalForge AI is narrowing to five pillars:

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

SignalForge AI is expanding toward a specialist model registry/exchange where the
published unit is a **complete deployable package**, not only weights.

Each exchange unit includes:

- Small domain model
- Eval pack
- Trace/dataset lineage
- Hardware profile
- Failure modes
- License and usage constraints
- Ready-to-run artifacts (adapters, Safetensors/GGUF, Ollama packaging)

Use `signalforgeai-exchange` to build and validate `specialist_model_unit.v0`
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

## What SignalForge AI is not

- A chatbot framework
- A prompt library
- A no-code builder
- A model leaderboard without task context
- A fixed set of built-in agents

---

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install signalforgeai==0.4.0
```

Run the deterministic production-pilot readiness loop:

```bash
signalforgeai-pilot-check --mode dummy --work-dir results/pilot_check
```

This quickstart is offline-safe by default and uses the deterministic
`dummy_good` provider. It writes:
- `results/pilot_check/pilot_readiness.md`
- `results/pilot_check/pilot_readiness.json`
- `results/pilot_check/logs/` with `trace.v0` and `reward.v0` artifacts
- `results/pilot_check/datasets/manifest.json` plus SFT, preference, repair, and curriculum exports

Validate and inspect the latest trace:

```bash
python -m signalforgeai.logging.validate results/pilot_check/logs/$(ls -t results/pilot_check/logs | head -n 1)
python -m signalforgeai.logging.inspect results/pilot_check/logs/$(ls -t results/pilot_check/logs | head -n 1)
```

To run the repository examples, clone the source checkout and install editable:

```bash
git clone https://github.com/foaadfarooghian/signalforgeai.git
cd signalforgeai
pip install -e ".[dev]"
python examples/quickstart_research_agent.py
```

Run evaluation suites:

```bash
python -m signalforgeai.evaluation.run src/signalforgeai/evaluation/suites/quickstart.json
python -m signalforgeai.evaluation.run src/signalforgeai/evaluation/suites/research_quickstart.json
```

To run against a configured local or hosted model, set `SIGNALFORGEAI_MODEL_ID`
explicitly:

```bash
SIGNALFORGEAI_MODEL_ID=ollama:ministral-3:8b python examples/quickstart_research_agent.py
SIGNALFORGEAI_MODEL_ID=openai:gpt-5-mini python examples/quickstart_research_agent.py
```

Pilot datasets are strict-gated by default: exported rows include deterministic
split metadata, provenance links back to trace/reward artifacts, file hashes,
duplicate counts, and leakage checks. Validate an exported dataset directly with:

```bash
signalforgeai-dataset-validate results/pilot_check/datasets/pilot.sft.jsonl \
  --kind sft --quality-gate --logs-root results/pilot_check/logs
```

Run a release regression gate by comparing against the last accepted readiness
artifact:

```bash
signalforgeai-pilot-check --mode dummy --work-dir results/pilot_current \
  --baseline results/pilot_baseline/pilot_readiness.json
```

When `--baseline` is supplied, the command also writes:
- `results/pilot_current/eval_regression.md`
- `results/pilot_current/eval_regression.json`

The default deterministic gate allows no pass-rate drop, no mean-score drop, no
new failing cases, and no worse failure-mode movement.

Optional provider smoke checks can be required in configured environments:

```bash
signalforgeai-pilot-check --require-provider hosted
signalforgeai-pilot-check --require-provider local
```

Run the one-command offline release-candidate evidence gate:

```bash
signalforgeai-release-candidate-check \
  --work-dir results/release_candidate
```

This writes `release_candidate.v0` JSON and Markdown plus the full child
evidence bundle: pilot readiness, mock SFT/DPO `training_run.v0`, distillation
eval, benchmark matrix, specialist unit, package check, consumer smoke run, and
registry index.

Release reviewers can require non-mock training evidence. Use an SFT run as the
final artifact:

```bash
signalforgeai-release-candidate-check \
  --work-dir results/release_candidate \
  --sft-run results/training/sft_training_run.json \
  --final-training-stage sft \
  --require-real-training-evidence
```

Or use the final DPO run report:

```bash
signalforgeai-release-candidate-check \
  --work-dir results/release_candidate \
  --dpo-run results/training/dpo_training_run.json \
  --require-real-training-evidence
```

Linux release environments with `[train]` installed can let the release-candidate
gate run bounded SFT evidence and evaluate the trained adapter directly:

```bash
signalforgeai-release-candidate-check \
  --work-dir results/release_candidate_real \
  --run-training \
  --training-base-model hf/org/base \
  --training-max-steps 1 \
  --require-real-training-evidence
```

Add `--run-dpo` when the final candidate should be the DPO adapter. When
`--candidate-model-id` is omitted in this mode, SignalForge AI derives
`hf:<base>?adapter=<final-adapter-dir>` and uses it for distillation and
benchmark evidence.

Training remains experimental. In `v0.4.0`, the `[train]` extra and actual
SFT/DPO execution are Linux-only because the Torch/Triton/Unsloth dependency
stack is not portable across all supported core platforms. Preflight evidence
is still available without loading models:

```bash
signalforgeai-learn train --base-model dummy/base --sft --dpo \
  --sft-data results/pilot_check/datasets/pilot.sft.jsonl \
  --dpo-data results/pilot_check/datasets/pilot.dpo.jsonl \
  --dry-run --quality-gate --logs-root results/pilot_check/logs \
  --report-out results/pilot_check/training_preflight.json
```

Release environments with `[train]` dependencies installed can opt into a
bounded SFT smoke run and record `training_run.v0` evidence:

```bash
signalforgeai-learn train --base-model hf/org/base --sft \
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
signalforgeai-learn train --base-model hf/org/base --dpo \
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
signalforgeai-pilot-check --mode dummy --training-preflight \
  --work-dir results/pilot_check
```

Run the evidence-only distillation eval gate from the generated recipe:

```bash
signalforgeai-distill-check \
  --recipe results/pilot_check/distillation_recipe.json \
  --work-dir results/distillation_gate
```

This compares a candidate specialist model against a baseline/teacher model and
writes `distillation_eval.v0` JSON and Markdown evidence without requiring real
training.

Generate benchmark frontier evidence across suites and model ids:

```bash
signalforgeai-benchmark-matrix \
  --config docs/benchmark_matrix_sample.json \
  --work-dir results/benchmark_matrix
```

This writes `benchmark_matrix.v0` JSON and Markdown with task success, cost per
successful task, latency p50/p95, reliability fields, mean effective score, and
frontier picks. Dummy rows are deterministic and mandatory; hosted/local/HF rows
skip unless their provider is explicitly required.

Build and index a local specialist exchange unit from the release evidence:

```bash
signalforgeai-exchange build-unit \
  --training-preflight results/pilot_check/training_preflight.json \
  --training-run results/training/dpo_training_run.json \
  --distillation-eval results/distillation_gate/distillation_eval.json \
  --benchmark-matrix results/benchmark_matrix/benchmark_matrix.json \
  --out results/exchange/pilot-specialist.unit.json \
  --id pilot-specialist --name "Pilot Specialist" --version 0.4.0 --domain pilot \
  --model-family dummy --model-size 0B --model-format safetensors \
  --model-license Apache-2.0 --dataset-license CC-BY-4.0 \
  --usage-constraint "not for production decisions without review" \
  --failure-mode dummy_only \
  --failure-description "Dummy artifacts only prove exchange plumbing." \
  --failure-mitigation "Replace dummy refs before release." \
  --safetensors-ref hf://signalforgeai/pilot-specialist/model.safetensors \
  --ollama-modelfile hf://signalforgeai/pilot-specialist/Modelfile \
  --ollama-tag signalforgeai/pilot-specialist:0.4.0

signalforgeai-exchange package-check \
  --manifest results/exchange/pilot-specialist.unit.json \
  --out results/exchange/specialist_package.json \
  --package-type auto --release-ready --update-manifest

signalforgeai-exchange smoke-run \
  --manifest results/exchange/pilot-specialist.unit.json \
  --work-dir results/exchange/smoke --update-manifest

signalforgeai-exchange validate \
  --manifest results/exchange/pilot-specialist.unit.json --release-ready

signalforgeai-exchange index \
  --registry-dir results/exchange --out results/exchange/index.json --release-ready
```

---

## Repository layout

```text
src/signalforgeai/
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
