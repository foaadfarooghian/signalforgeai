# TensorFoundry Roadmap

This roadmap narrows TensorFoundry to an **agent engineering + learning system**
built around five pillars:

1. OTel/MCP-native runtime + artifact schema
2. Evaluation and failure analysis for multi-step/tool-using agents
3. Dataset generation from production traces
4. Distillation pipeline for specialist small models
5. Benchmarking cost/latency/reliability across models and agent patterns

The sequence is deliberate: stabilize runtime contracts first, then evaluation,
then learning loops, then model specialization. A specialist model exchange is
the distribution layer built on top of these pillars.

---

## Scope (explicit)

### In scope
- Trace-centric runtime for tool-using agents
- Measurable reliability and failure diagnostics
- Deterministic trace-to-dataset pipelines
- Reproducible specialist distillation
- Cross-model/pattern benchmark reporting

### Out of scope
- Prompt-only abstractions
- UI-first product work
- Generic chatbot features
- Leaderboards detached from task execution traces

---

## Workstream A — Runtime Contracts (OTel + MCP)

**Goal:** Make runtime telemetry and tool interfaces interoperable by default.

### Deliverables
- Canonical runtime artifact schema for:
  - `trace.jsonl`
  - `reward.jsonl`
  - eval verdict artifacts
  - dataset provenance metadata
- OTel mapping spec:
  - trace/span IDs, stages, events, attributes, errors
- MCP-native tool execution contract:
  - tool request/response envelopes
  - retries/timeouts/error classification

### Exit criteria
- Every run can be exported as valid OTel-compatible spans/events
- MCP tool calls are represented consistently in trace artifacts
- Schema versioning and migration rules are documented

---

## Workstream B — Evaluation and Failure Analysis

**Goal:** Make multi-step agent quality diagnosable, not just scorable.

### Deliverables
- Evaluation harness upgrades for multi-step + tool-heavy runs
- Failure taxonomy:
  - tool invocation failures
  - planning/reasoning failures
  - recovery policy failures
  - orchestration state failures
- Failure analysis reports:
  - where failure began
  - what recovery attempted
  - why final outcome failed/succeeded

### Exit criteria
- Eval output includes both score and structured failure diagnosis
- Regression diffing identifies failure mode movement, not only pass/fail deltas
- Runs are reproducible from stored artifacts

---

## Workstream C — Trace -> Dataset Generation

**Goal:** Convert production traces into high-quality supervised signals.

### Deliverables
- Deterministic ETL for:
  - SFT examples
  - preference pairs
  - repair trajectories
  - critique/rubric examples
- Provenance guarantees (row -> trace/span/case link)
- Data quality checks:
  - schema validity
  - duplicate filtering
  - leakage/split hygiene

### Exit criteria
- Dataset exports are reproducible from the same trace corpus
- Every dataset row is attributable to a source run
- Quality checks are enforced in CI or pre-release validation

---

## Workstream D — Specialist Distillation Pipeline

**Goal:** Train small specialist models from execution-derived supervision.

### Deliverables
- Teacher run capture and curation workflow
- Training preflight evidence:
  - strict SFT/DPO dataset checks
  - optional dependency and smoke-readiness status
  - minimal `training_artifact.v0` manifest for later exchange packaging
- SFT training run evidence:
  - opt-in `training_run.v0` execution report
  - output artifact refs and checksums for exchange packaging
- Evidence-only distillation eval gate:
  - baseline/teacher vs candidate specialist suite comparison
  - pass-rate, mean-score, and failure-mode movement thresholds
  - `distillation_eval.v0` release artifact
- Student training pipeline (SFT first, optional preference optimization)
- Distillation eval gate:
  - specialist benchmark pass thresholds
  - cost/latency improvement targets
- Packaging path for local/private deployment

### Exit criteria
- Training inputs can be validated reproducibly without loading models
- Opt-in SFT runs produce auditable training evidence without becoming a CI gate
- Candidate specialist models can be eval-gated before exchange packaging
- At least one specialist student model reaches benchmark quality gates
- Distilled model shows favorable cost/latency at acceptable reliability
- Training and evaluation runs are reproducible end-to-end

---

## Workstream E — Benchmark Matrix (Cost/Latency/Reliability)

**Goal:** Enable apples-to-apples comparisons across models and agent patterns.

**Status:** Active implementation after the distillation eval gate.

### Deliverables
- `benchmark_matrix.v0` matrix runner for:
  - model providers
  - orchestration patterns
  - tooling profiles
- Unified scorecard with:
  - task success
  - cost per successful task
  - latency distribution (p50/p95)
  - reliability (failure and retry rates)
- Frontier reports highlighting efficient operating points
- Offline deterministic dummy matrix as the required CI-compatible path

### Exit criteria
- Benchmark reports can be regenerated from versioned suites and artifacts
- Trade-off frontiers are visible by task family
- Release decisions can reference benchmark evidence directly

---

## Workstream F — Specialist Model Registry / Exchange

**Goal:** Publish specialist models as complete operational units.

**Status:** Packaging and consumer smoke checks in progress after the registry
MVP.

### Deliverables
- `specialist_model_unit.v0` manifest spec (domain model + eval + lineage + ops constraints)
- Machine-validated schema and release-ready checks for exchange entries
- `specialist_registry_index.v0` file-based registry index with immutable artifact references
- Packaging standards for:
  - adapters
  - Safetensors/GGUF weights
  - Ollama-ready bundles
- `specialist_package.v0` evidence for artifact checksums and bundle viability
- `specialist_smoke.v0` evidence for deterministic consumer load/generate checks
- Governance rules for license and usage-constraint disclosure

### Exit criteria
- Every listed model has a valid unit manifest
- Eval pack and benchmark evidence are linked in each listing
- Trace/dataset lineage is auditable for each published version
- Hardware profile and known failure modes are documented per unit
- Consumers can run listed artifacts through a documented smoke path without
  bespoke integration work

---

## Current baseline (already present in repo)

- Structured trace emission and validation
- Additive `trace.v0` emission with legacy trace compatibility
- Evaluation suites + reward artifact emission
- Trace export to SFT/preferences/repair datasets
- Production-pilot readiness check for eval -> reward -> dataset validation
- Provider readiness checks for dummy, hosted, and local model paths
- Artifact-baseline regression gating for pilot readiness checks
- Strict dataset quality gates for provenance, splits, duplicates, and leakage
- Training dry-run preflight for SFT/DPO inputs
- Evidence-only distillation eval gate for candidate specialists
- Learning/routing primitives and experimental training utilities
- Cost/latency-aware evaluation reporting

---

## Success criteria for this roadmap

TensorFoundry succeeds when:
- Production traces become the canonical source for eval and learning
- Reliability regressions are diagnosed by failure mode, not anecdotes
- Specialist models are trained and validated against real agent workloads
- Model/pattern choices are made from benchmark evidence, not intuition
