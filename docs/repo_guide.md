# TensorFoundry Repository Guide

This guide explains what the repository is for, how the components connect, and
where a new developer should start. It is based on the code and docs in this
repo, not on external assumptions.

## What This Repo Is Trying To Achieve

TensorFoundry is an opinionated framework for engineered AI agents. The core
goal is to treat agents as software systems with explicit state, tools, retries,
logging, and evaluation, so that behavior can be measured and improved over
time. The repo focuses on:

- OTel/MCP-aligned runtime semantics for multi-step agent execution.
- Structured, append-only artifacts (`trace.jsonl`, `reward.jsonl`) as source of truth.
- Evaluation and failure analysis for tool-using workflows.
- Dataset generation from production traces for supervised signals.
- Distillation and benchmarking loops for specialist small models.
- Specialist model exchange units with lineage, constraints, and runnable artifacts.

Project status: active development, with breaking changes expected before a
stable v1.0 release.

Read these files first to get the intent and scope:

- `README.md` for the product overview and quickstart.
- `manifesto.md` for philosophy and non-goals.
- `roadmap.md` for phased direction.
- `docs/design_principles.md` for architectural principles.

## Where To Start (New Developer Path)

1. Skim `README.md`, `manifesto.md`, and `roadmap.md`.
2. Run an example to see end-to-end logging:
   - `python examples/quickstart_research_agent.py`
3. Validate and inspect the generated trace:
   - `python -m tensorfoundry.logging.validate logs/<trace_id>.jsonl`
   - `python -m tensorfoundry.logging.inspect logs/<trace_id>.jsonl`
4. Run a minimal evaluation suite:
   - `python -m tensorfoundry.evaluation.run src/tensorfoundry/evaluation/suites/quickstart.json`
5. Export datasets from logs (optional, but shows the learning loop):
   - `tensorfoundry-learn export --logs-root logs --out-dir datasets/examples --sft --prefs --curriculum`

## Setup and Dependencies

The project is packaged via `pyproject.toml` and requires Python 3.11 or newer.
Use the dev extra for linting/tests, and the train extra only if you need local
model training.

Example setup:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Optional training stack:

```bash
# Linux-only in v0.4.0
pip install -e ".[train]"
```

## High-Level Architecture

At a high level, TensorFoundry is a loop:

```
Task
  -> Agent (planner/executor/critic) [src/tensorfoundry/agents]
  -> Runtime artifacts (trace/reward)  [src/tensorfoundry/logging]
  -> Eval + failure analysis            [src/tensorfoundry/evaluation]
  -> Dataset generation                 [src/tensorfoundry/export]
  -> Distillation/training              [src/tensorfoundry/training]
  -> Benchmark matrix + tradeoffs       [src/tensorfoundry/evaluation]
  -> Improved model/pattern choices
```

Execution artifacts are central: evaluation, dataset generation, and
distillation all consume the same trace-linked records.

## Repository Layout (Top-Level)

- `src/tensorfoundry/` - Core framework code (agents, logging, evaluation, etc).
- `examples/` - Runnable examples that exercise the core APIs.
- `docs/` - Design notes and documentation.
- `tests/` - Pytest coverage for logging, evaluation, and orchestration.
- `datasets/` - Example JSONL datasets (SFT / preference / teacher traces).
- `learning/` - Operational artifacts from local runs (e.g., bandit state).
- `learning_ops/` - Operational artifacts for learning (policies, scripts).
- `logs/` - Example run logs (trace.jsonl + reward.jsonl per run id).
- `results/` - Example evaluation outputs (results.json + summary.md).
- `pyproject.toml` - Packaging, dependencies, and CLI scripts.
- `.github/workflows/ci.yml` - CI pipeline definition.

## Core Packages and How They Connect

### Agents: Reference Implementations

Located in `src/tensorfoundry/agents/`.

These are reference templates showing the intended shape of an agent. Each
agent uses a `JsonlEmitter` to record events and typically uses a model
provider via `get_provider_for_model`.

- `src/tensorfoundry/agents/research_agent.py`
  - A simple plan -> search -> summarize workflow.
  - Logs planner/tool/model events.
  - Uses `TENSORFOUNDRY_MODEL_ID` (default `dummy_good`; set explicitly for Ollama/OpenAI/HF).
- `src/tensorfoundry/agents/decision_agent.py`
  - Produces a structured memo: constraints, options, tradeoffs, recommendation.
  - Emits a `model_called` event with metrics and output summary.
- `src/tensorfoundry/agents/refactor_agent.py`
  - Deterministic file find/replace patch workflow.
  - Logs file IO as tool events and supports dry-run mode.
- `src/tensorfoundry/agents/synth_agent.py`
  - Multi-step synthesis with citations and a strict JSON output contract.
  - Includes a critic pass that validates and repairs citations.

Agents are intentionally minimal; they demonstrate shape and trace emission
rather than full production behavior.

### Orchestration: Planner -> Executor -> Critic

Located in `src/tensorfoundry/orchestration/`.

- `src/tensorfoundry/orchestration/pec.py`
  - Defines the core orchestration loop.
  - Uses a `Planner`, `Executor`, and `Critic` protocol.
  - Emits `task_received`, `stage_completed`, `retry_requested`,
    and terminal events (`task_completed` / `task_failed`).
- `src/tensorfoundry/orchestration/adapters.py`
  - Adapter implementations that wrap `ResearchAgent` for planner/executor.

The orchestrator is the spine for multi-step agents. It keeps retries explicit
and stateful, with trace events for every transition.

### Logging: Traces Are The Dataset

Located in `src/tensorfoundry/logging/`.

Key files:

- `src/tensorfoundry/logging/emitter.py`
  - `JsonlEmitter` writes JSONL events to disk.
  - Validates stage values and supports streaming via context manager.
- `src/tensorfoundry/logging/events.py`
  - Canonical event shape and ID/timestamp helpers.
  - `sanitize_payload` truncates large payloads and preserves select keys.
- `src/tensorfoundry/logging/schema.md`
  - Minimal schema documentation for trace events.
- `src/tensorfoundry/logging/validate.py`
  - CLI and helpers to validate traces and reward logs.
  - Enforces terminal events and required keys.
- `src/tensorfoundry/logging/inspect.py`
  - CLI to summarize a trace (counts, stages, terminal status, timeline).
- `src/tensorfoundry/logging/diff.py`
  - CLI to diff two trace files and compare event counts + terminal outcomes.
- `src/tensorfoundry/logging/reward_validator.py`
  - Standalone validator for reward.jsonl rows.

Trace events are small, structured JSON objects. The logging layer is used by
agents and the orchestrator, and it is what powers evaluation and dataset
export.

Trace and reward contracts (important invariants):

- Trace event required keys:
  `trace_id`, `span_id`, `parent_span_id`, `timestamp`, `agent`, `stage`,
  `event_type`, `payload`, `metrics`, `outcome`.
- Allowed stages:
  `planner`, `executor`, `critic`, `tool`, `system`.
- Terminal events:
  `task_completed` or `task_failed` must include `outcome.status`.
- Reward schema:
  `reward.v0` rows include trace identity, suite/case, model, score, and
  optional cost/latency/token metrics.

### Evaluation: Suites, Scoring, Rewards

Located in `src/tensorfoundry/evaluation/`.

Key files:

- `src/tensorfoundry/evaluation/harness.py`
  - `run_suite` runs a suite of cases, writes traces, validates logs, and
    scores outcomes.
  - Produces `results/<suite>.results.json`, `<suite>.summary.md`, and
    `logs/<run_id>/reward.jsonl`.
  - Extracts trade-off metrics (cost, latency, tokens) from trace events.
  - Uses `get_agent_runner` to dispatch to the right agent.
- `src/tensorfoundry/evaluation/run.py`
  - CLI entrypoint. Supports routing policy and bandit-based model selection.
  - Policy selection (`--policy`) overrides bandits; bandits can be disabled
    with `--no-bandits`. `benchmark_v0_refactor` skips bandits by default.
- `src/tensorfoundry/evaluation/reward_schema.py`
  - Reward row schema (`reward.v0`) and JSONL serialization.
- `src/tensorfoundry/evaluation/reward_writer.py`
  - Writes reward rows to JSONL.
- `src/tensorfoundry/evaluation/diff.py`
  - Diff two evaluation result files.
- `src/tensorfoundry/evaluation/report_tradeoffs.py`
  - Summarize scores vs cost/latency from reward.jsonl files.

Suites and benchmarks:

- `src/tensorfoundry/evaluation/suites/`
  - Small quickstart suites for research and decision agents.
- `src/tensorfoundry/evaluation/benchmarks/v0/`
  - Outcome-focused benchmark suites for research, decision, and refactor.
- `src/tensorfoundry/evaluation/benchmarks/v1/suites/benchmark_v1_synth.json`
  - Large synthesis benchmark for `SynthAgent` with citation/contract checks.

Scoring logic summary:

- Default suites use status matching and optional keyword checks
  (`expect.contains_any`).
- Synth benchmark uses a rubric:
  schema validity, grounded citations, conflict/noise handling, and answer
  clarity/length.

### Models: Provider Abstraction and Pricing

Located in `src/tensorfoundry/models/`.

Key files:

- `src/tensorfoundry/models/base.py`
  - `ModelProvider` protocol: `generate(prompt, model_id, task_type)`.
- `src/tensorfoundry/models/types.py`
  - `ModelOutput` and `ModelMetrics` (latency/cost/extra).
- `src/tensorfoundry/models/registry.py`
  - `get_provider_for_model` chooses a provider by prefix:
    - `openai:` -> OpenAIProvider
    - `ollama:` -> OllamaProvider
    - `hf:` -> HFProvider
    - otherwise -> DummyProvider
  - CI guard: in CI, networked providers are replaced by dummy.
- Providers:
  - `src/tensorfoundry/models/providers/openai_provider.py`
    - Uses OpenAI Responses API and computes cost from pricing config.
  - `src/tensorfoundry/models/providers/ollama.py`
    - Sends HTTP requests to a local Ollama server.
  - `src/tensorfoundry/models/providers/hf.py`
    - Loads local HF models and optional LoRA adapters.
    - Supports `hf:<base>?adapter=/path/to/lora` model IDs.
  - `src/tensorfoundry/models/providers/dummy.py`
    - Deterministic responses for tests and CI.
- Pricing:
  - `src/tensorfoundry/models/pricing/config/openai_pricing.yaml`
  - `src/tensorfoundry/models/pricing/load_pricing.py`
  - `src/tensorfoundry/models/pricing/validate_openai_pricing.py`

Model providers are invoked directly by agents. The evaluation layer uses
`TENSORFOUNDRY_MODEL_ID` to control which provider is used during suites.

### Learning: Routing, Policies, Curriculum

Located in `src/tensorfoundry/learning/`.

Key files:

- `src/tensorfoundry/learning/bandits.py`
  - Thompson sampling bandits for suite-specific model routing.
  - Uses reward feedback to update alpha/beta for each model arm.
- `src/tensorfoundry/learning/routing_policy.py`
  - Static routing policy format (`routing.v0`).
- `src/tensorfoundry/learning/model_stats.py`
  - EWMA stats for cost, latency, tokens per suite and model.
- `src/tensorfoundry/learning/build_policy.py`
  - Aggregates reward.jsonl files to build a routing policy.
- `src/tensorfoundry/learning/curriculum.py`
  - Exports a curriculum dataset with difficulty buckets:
    `easy`, `repair`, `escalation`.
- `src/tensorfoundry/learning/learn.py`
  - Main CLI pipeline:
    - `tensorfoundry-learn export` -> datasets
    - `tensorfoundry-learn train` -> SFT/DPO training entrypoints

Learning consumes reward.jsonl and trace files. It does not directly call
models; it manipulates routing and datasets so that training can happen
offline and safely.

### Export: Turning Traces Into Training Data

Located in `src/tensorfoundry/export/`.

Key files:

- `src/tensorfoundry/export/extract.py`
  - Extracts prompts/responses from traces.
  - Prefers `model_called` events with `input_summary` values such as
    `critic_check` and `draft_answer`.
- `src/tensorfoundry/export/dataset.py`
  - Exports SFT JSONL (`sft.v0`) from reward logs + traces.
- `src/tensorfoundry/export/preferences.py`
  - Exports preference pairs (`prefs.v0` or `dpo.v0`), using
    effective reward = score - cost - latency penalties.
- `src/tensorfoundry/export/repairs.py`
  - Exports failure -> success pairs for repair training.

Export is the bridge from execution logs to training datasets.

### Training: SFT and DPO (Optional)

Located in `src/tensorfoundry/training/`.

- `src/tensorfoundry/training/sft_unsloth.py`
  - QLoRA SFT training using Unsloth + TRL.
- `src/tensorfoundry/training/dpo_trl.py`
  - DPO training on top of an SFT adapter.
- `src/tensorfoundry/training/collator_masked.py`
  - Masked chat collator for supervised training (prompt masked).

Training is optional. In `v0.4.0`, actual SFT/DPO execution and `[train]`
installs are Linux-only; use dry-run preflight on other platforms.

## Execution and Data Flow (Detailed)

### 1) Agent Run -> Trace

1. Agent receives a task and calls `JsonlEmitter.emit` to create
   `task_received` (system stage).
2. Planner/Executor/Critic steps emit structured events:
   - `plan_created`, `tool_called`, `tool_result`, `model_called`,
     `stage_completed`, `retry_requested`, etc.
3. A terminal event is emitted:
   - `task_completed` or `task_failed`.
4. The trace is written as JSONL in `logs/<trace_id>.jsonl`
   (or in `logs/<run_id>/<trace_id>.jsonl` during evaluation).

Trace validity is enforced by `src/tensorfoundry/logging/validate.py` and is
required for evaluation scoring to pass.

Newly emitted traces include `schema_version: "trace.v0"`. Legacy traces
without this field remain valid. New tool events are normalized with
`tool_name`, `tool_input`, `tool_output_summary`, `success`, and `error`.

### 2) Evaluation -> Results + Reward

1. `python -m tensorfoundry.evaluation.run <suite.json>` loads a suite.
2. For each case:
   - An agent is run with a `JsonlEmitter`.
   - Trace is validated.
   - `summarise_trace` extracts the terminal outcome.
   - Scores are computed (status + keyword checks, or synth-specific rubric).
3. Output files:
   - `results/<suite>.results.json`
   - `results/<suite>.summary.md`
   - `logs/<run_id>/reward.jsonl` (one row per case)

### 3) Learning -> Routing and Datasets

1. Bandits update per suite from reward.jsonl (see `bandits.py`).
2. Routing stats capture expected cost/latency to support constraints.
3. `tensorfoundry-learn export` generates:
   - SFT datasets from high-scoring traces.
   - Preference pairs comparing models on the same case.
   - Repair pairs (failed -> success).
   - Curriculum buckets for staged training.

### 4) Training -> Local Models

1. `tensorfoundry-learn train --sft --dpo` runs SFT and DPO jobs.
2. Outputs are saved under `artifacts/` (default).
3. `tensorfoundry-learn train --dry-run` validates datasets, output paths, and
   optional dependency availability without loading models.
4. `--quality-gate --logs-root ... --report-out ...` writes
   `training_preflight.v0` evidence with strict dataset validation, hashes,
   split summaries, dependency status, and a minimal `training_artifact.v0`
   manifest.
5. `--run-report-out ... --smoke --max-steps 1` records optional
   `training_run.v0` SFT evidence in release environments with `[train]`
   dependencies installed. It is not part of the mandatory offline gate.
6. DPO execution evidence is opt-in. DPO-only runs with `--run-report-out`
   require `--sft-run <successful training_run.v0>` so the preference step has
   auditable parent SFT lineage. Combined `--sft --dpo` runs record the
   same-command SFT output as the parent.

Example preflight:

```bash
tensorfoundry-learn train --base-model dummy/base --sft --dpo \
  --sft-data results/pilot_check/datasets/pilot.sft.jsonl \
  --dpo-data results/pilot_check/datasets/pilot.dpo.jsonl \
  --dry-run --quality-gate --logs-root results/pilot_check/logs \
  --report-out results/pilot_check/training_preflight.json
```

Example optional SFT smoke evidence:

```bash
tensorfoundry-learn train --base-model hf/org/base --sft \
  --sft-data results/pilot_check/datasets/pilot.sft.jsonl \
  --sft-out results/training/sft_lora \
  --quality-gate --logs-root results/pilot_check/logs \
  --report-out results/pilot_check/training_preflight.json \
  --run-report-out results/training/sft_training_run.json \
  --smoke --max-steps 1
```

Example optional DPO smoke evidence:

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

Distillation gate recipes use `distillation_recipe.v0`. They point at the
training preflight report, one eval suite, a baseline/teacher model, a candidate
specialist model, and release thresholds.

### 5) Pilot Readiness Check

Run the full deterministic offline loop:

```bash
tensorfoundry-pilot-check --mode dummy --work-dir results/pilot_check
```

Outputs:
- `results/pilot_check/pilot_readiness.md`
- `results/pilot_check/pilot_readiness.json`
- `results/pilot_check/eval_regression.md` when `--baseline` is supplied
- `results/pilot_check/eval_regression.json` when `--baseline` is supplied
- `results/pilot_check/training_preflight.json` when `--training-preflight` is supplied
- `results/pilot_check/distillation_recipe.json` when `--training-preflight` is supplied
- `results/pilot_check/logs/`
- `results/pilot_check/datasets/manifest.json`

The dataset manifest includes strict quality summaries: content hashes, split
counts, duplicate counts, provenance checks, missing refs, and quality issues.
Run the same quality gate outside pilot-check with:

```bash
tensorfoundry-dataset-validate results/pilot_check/datasets/pilot.sft.jsonl \
  --kind sft --quality-gate --logs-root results/pilot_check/logs
```

Compare the current pilot run with an accepted baseline artifact:

```bash
tensorfoundry-pilot-check --mode dummy --work-dir results/pilot_current \
  --baseline results/pilot_baseline/pilot_readiness.json
```

The default regression policy is strict for dummy runs: no pass-rate drop, no
mean-score drop, no new failing cases, and no worse failure-mode movement.

Provider smoke checks are tiered. Dummy is mandatory. Hosted and local checks
skip unless explicitly required:

```bash
tensorfoundry-pilot-check --require-provider hosted
tensorfoundry-pilot-check --require-provider local
tensorfoundry-pilot-check --require-provider all
```

Training preflight is optional and dry-run only inside pilot-check:

```bash
tensorfoundry-pilot-check --mode dummy --training-preflight \
  --work-dir results/pilot_check
```

That command generates an additional DPO-compatible preference export for the
training preflight report, records dependency status, and leaves real SFT/DPO
training opt-in through `tensorfoundry-learn train --smoke --max-steps 1`.

Run the evidence-only distillation gate from the generated recipe:

```bash
tensorfoundry-distill-check \
  --recipe results/pilot_check/distillation_recipe.json \
  --work-dir results/distillation_gate
```

The gate runs the configured suite for the baseline and candidate model, compares
cases by `(suite_name, case_id)`, enforces pass-rate, mean-score, and failure-mode
movement thresholds, and writes `distillation_eval.json` plus
`distillation_eval.md`.

Generate benchmark matrix frontier evidence after the distillation gate:

```bash
tensorfoundry-benchmark-matrix \
  --config docs/benchmark_matrix_sample.json \
  --work-dir results/benchmark_matrix
```

The matrix runner consumes `benchmark_matrix.v0` config, resolves packaged suite
aliases such as `decision_v0`, iterates `suite x model_id`, and writes
`benchmark_matrix.json` plus `benchmark_matrix.md`. It reports task success,
cost per successful task, latency p50/p95, retry/failure rates where available,
mean effective score, and named frontier picks. Dummy rows are mandatory for
offline release evidence; hosted, local, and HF rows skip unless passed with
`--require-provider`.

Build a local specialist exchange unit from pilot, distillation, and benchmark
evidence:

```bash
tensorfoundry-exchange build-unit \
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
  --safetensors-ref hf://tensorfoundry/pilot-specialist/model.safetensors \
  --ollama-modelfile hf://tensorfoundry/pilot-specialist/Modelfile \
  --ollama-tag tensorfoundry/pilot-specialist:0.4.0
```

Attach package evidence, run the consumer smoke check, then validate and index
release-ready manifests:

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

The exchange CLI performs schema validation, local checksum verification,
optional training run evidence mapping, package evidence generation, offline
consumer smoke evidence, evidence-link checks, duplicate `(id, version)`
rejection, and file-based
`specialist_registry_index.v0` generation. It does not upload or bundle weights.

Run the complete release-candidate bundle as one deterministic offline gate:

```bash
tensorfoundry-release-candidate-check \
  --work-dir results/release_candidate
```

That command orchestrates the existing library APIs directly and writes
`release_candidate.v0` evidence. It resets only its generated subdirectories,
then produces pilot readiness, mock SFT/DPO training run evidence, distillation
eval, benchmark matrix, specialist unit, package check, consumer smoke, and
registry index outputs.

Release environments can require non-mock training evidence. For SFT-only
evidence, make the SFT run the final packaged artifact:

```bash
tensorfoundry-release-candidate-check \
  --work-dir results/release_candidate \
  --sft-run results/training/sft_training_run.json \
  --final-training-stage sft \
  --require-real-training-evidence

tensorfoundry-release-candidate-check \
  --work-dir results/release_candidate \
  --dpo-run results/training/dpo_training_run.json \
  --require-real-training-evidence
```

Linux release environments with `[train]` installed can also run bounded SFT
evidence inside the release-candidate gate:

```bash
tensorfoundry-release-candidate-check \
  --work-dir results/release_candidate_real \
  --run-training \
  --training-base-model hf/org/base \
  --training-max-steps 1 \
  --require-real-training-evidence
```

This path calls the training runner directly, records non-mock
`training_run.v0`, derives `hf:<base>?adapter=<final-adapter-dir>` when no
candidate model id is supplied, and uses that adapter for distillation,
benchmark, exchange, package, smoke, and registry evidence. Add `--run-dpo` to
make the DPO adapter the final packaged candidate.

## Examples and Reference Workflows

Examples live in `examples/`:

- `examples/quickstart_research_agent.py`
  - Runs the `ResearchAgent` and prints a summary + trace id.
- `examples/quick_decision_agent.py`
  - Runs the `DecisionAgent` with options and constraints.
- `examples/quick_refactor_agent.py`
  - Demonstrates safe, dry-run refactoring.
- `examples/quick_orchestrator.py`
  - Demonstrates the PEC orchestrator.

## Operational Artifacts

These directories are intentionally separate from source code:

- `logs/`
  - Per-run directories: `logs/<run_id>/<trace_id>.jsonl`
  - Reward logs: `logs/<run_id>/reward.jsonl`
- `results/`
  - Evaluation outputs (.results.json + .summary.md)
- `datasets/`
  - Example JSONL datasets
- `learning/`
  - Local routing bandit state snapshots (for example `learning/bandits/routing_bandits_v0.json`)
- `learning_ops/`
  - Routing policy snapshots (for example `learning_ops/policies/routing_v0.json`)
  - `learning_ops/scripts/` contains helper scripts to prepare datasets

## Environment Variables (Common)

Model selection and routing:

- `TENSORFOUNDRY_MODEL_ID` - Current model id (`dummy_good` by default; set `openai:...`, `ollama:...`, or `hf:...` explicitly for provider runs).
- `TENSORFOUNDRY_CANDIDATE_MODELS` - Comma-separated candidates for bandit routing.
- `TENSORFOUNDRY_BANDIT_MIN_PULLS` - Minimum samples before exploitation.
- `TENSORFOUNDRY_MAX_COST_USD` - Hard cost constraint for routing.
- `TENSORFOUNDRY_MAX_LATENCY_S` - Hard latency constraint for routing.
- `TENSORFOUNDRY_UTILITY_LAMBDA_COST` - Cost penalty weight.
- `TENSORFOUNDRY_UTILITY_MU_LATENCY` - Latency penalty weight.

Provider configuration:

- `TENSORFOUNDRY_PROVIDER` - Set to `dummy` to force deterministic offline routing.
- `TENSORFOUNDRY_HOSTED_MODEL_ID` - Hosted provider smoke model, default `openai:gpt-5-mini`.
- `TENSORFOUNDRY_LOCAL_MODEL_ID` - Local provider smoke model, default `ollama:ministral-3:8b`.
- `TENSORFOUNDRY_OPENAI_PRICING_PATH` - Override OpenAI pricing config.
- `TENSORFOUNDRY_HF_DEVICE` - HF device string (default `cuda:0`).
- `TENSORFOUNDRY_HF_REQUIRE_FLASH` - Require flash attention.
- `TENSORFOUNDRY_HF_LOG_DEVICE_MAP` - Log HF device map details.

Dataset export:

- `TENSORFOUNDRY_SFT_STEP` - Which step to extract for SFT.
- `TENSORFOUNDRY_MIN_RESPONSE_CHARS` - Minimum response length.

Training (via `tensorfoundry-learn train`):

- `BASE_MODEL`, `SFT_DATASET`, `DPO_DATASET`, `OUT_DIR`, `SFT_DIR`
- `DATASET_NUM_PROC`, `INSTRUCTION_PART`, `RESPONSE_PART`, `MAX_STEPS`

## CLI Entry Points

Defined in `pyproject.toml`:

- `tensorfoundry-learn` -> `src/tensorfoundry/learning/learn.py`
- `tensorfoundry-pilot-check` -> `src/tensorfoundry/pilot_check.py`
- `tensorfoundry-otel-export` -> `src/tensorfoundry/logging/otel.py`
- `tensorfoundry-dataset-validate` -> `src/tensorfoundry/export/validate.py`
- `tensorfoundry-report-tradeoffs` -> `src/tensorfoundry/evaluation/report_tradeoffs.py`
- `tensorfoundry-pricing-validate` -> `src/tensorfoundry/models/pricing/validate_openai_pricing.py`
- `tensorfoundry-distill-check` -> `src/tensorfoundry/distillation/check.py`
- `tensorfoundry-benchmark-matrix` -> `src/tensorfoundry/evaluation/matrix.py`
- `tensorfoundry-exchange` -> `src/tensorfoundry/exchange/cli.py`

There are also module CLIs:

- `python -m tensorfoundry.logging.validate`
- `python -m tensorfoundry.logging.inspect`
- `python -m tensorfoundry.logging.diff`
- `python -m tensorfoundry.evaluation.run`
- `python -m tensorfoundry.evaluation.diff`

## CI and Quality Gates

Defined in `.github/workflows/ci.yml`:

- Lint: `ruff`
- Type check: `mypy src`
- Pricing config validation
- Pytest suite
- Deterministic pilot readiness check
- Benchmark v0 suites + log validation

CI uses dummy providers to avoid networked calls.

## Tests (What They Cover)

Located in `tests/`:

- `tests/test_orchestrator.py` - PEC orchestrator behavior and trace validity.
- `tests/test_validate.py` - Trace validation rules and errors.
- `tests/test_inspect.py` - Trace inspection summaries.
- `tests/test_evaluation.py` - Evaluation harness outputs.
- `tests/test_reward_schema.py` - Reward schema clamping/serialization.
- `tests/test_reward_writer.py` - Reward JSONL writing + validation.

## Branching and Releases

The repository currently uses:

- `dev` as the integration branch for ongoing work.
- `prod` as the stable release branch.

Check `README.md` for the latest release guidance.

## Extending The Repo (Practical Pointers)

### Add a New Agent

1. Create a new agent in `src/tensorfoundry/agents/`.
2. Ensure it emits schema-valid trace events (use `JsonlEmitter`).
3. Wire it into evaluation in `src/tensorfoundry/evaluation/harness.py`
   (`get_agent_runner`).
4. Add a suite JSON in `src/tensorfoundry/evaluation/benchmarks/` or
   `src/tensorfoundry/evaluation/suites/`.

### Add a New Model Provider

1. Implement the `ModelProvider` protocol in `src/tensorfoundry/models/providers/`.
2. Register it in `src/tensorfoundry/models/registry.py`.
3. Ensure `ModelMetrics` are populated (latency, cost, token usage if possible).

### Add a New Evaluation Suite

1. Add a JSON suite file with `suite_name`, `agent`, and `cases`.
2. Run with `python -m tensorfoundry.evaluation.run <suite.json>`.
3. Inspect and validate logs in `logs/<run_id>/`.

## Recommended Reading Order (Code)

If you want to understand the logic end-to-end, read in this order:

1. `src/tensorfoundry/logging/emitter.py`
2. `src/tensorfoundry/agents/research_agent.py`
3. `src/tensorfoundry/orchestration/pec.py`
4. `src/tensorfoundry/evaluation/harness.py`
5. `src/tensorfoundry/learning/bandits.py`
6. `src/tensorfoundry/export/dataset.py`
7. `src/tensorfoundry/training/sft_unsloth.py`

This ordering mirrors the execution -> evaluation -> learning loop.
