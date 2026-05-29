# SignalForge AI Roadmap

This roadmap narrows SignalForge AI to an **agent engineering + learning system**
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
- DPO/preference optimization run evidence:
  - DPO-only runs require successful parent SFT `training_run.v0` evidence
  - DPO run reports record parent lineage and final adapter refs
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

**Status:** Implemented for the offline/frontier evidence slice.

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

**Status:** File registry, package evidence, and consumer smoke evidence are
implemented for the offline release path.

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

## Completed Slice — Release Candidate Evidence Bundling

**Goal:** Prove the full pilot-to-exchange chain with one deterministic
offline gate.

### Deliverables
- `release_candidate.v0` JSON and Markdown evidence
- One command for pilot readiness, training evidence, distillation eval,
  benchmark matrix, exchange unit, package check, smoke run, and registry index
- Mock SFT/DPO adapter evidence for mandatory offline runs
- Optional override path for real SFT/DPO `training_run.v0` artifacts
- Strict reviewer mode that fails when final training evidence is mock-generated

### Exit criteria
- Fresh clones can produce a complete release-candidate evidence bundle offline
- Degraded candidates fail the gate with actionable child-gate issues
- Real training evidence can be substituted without changing downstream steps
- Release reviewers can require non-mock SFT or DPO training evidence explicitly

---

## Completed Phase — v0.5.0 Public Onboarding and Contract Clarity

**Goal:** Make the first public post-release milestone easy to evaluate by new
users and downstream contributors, without changing the `0.4.0` package
behavior or requiring hosted/local model providers.

### Deliverables
- Public contract inventory for current stable and experimental surfaces:
  package/import names, CLIs, environment variables, provider defaults, and
  artifact families
- First-pilot walkthrough from PyPI install to offline readiness artifacts,
  trace inspection, dataset validation, and provider-backed next steps
- Provider setup guidance that keeps `dummy_good` as the default and makes
  Ollama/OpenAI/HF opt-in
- Pilot-report interpretability issue for making first-user failures easier to
  understand
- `v0.5.0` release gate and acceptance checklist
- Explicit post-0.5 scope issue for real specialist training evidence

### Exit criteria
- Fresh public users can install the latest public package and complete the
  offline pilot walkthrough without networked model providers
- Public contracts identify what is stable enough to build against and what
  remains experimental before v1.0
- Existing offline defaults and `0.4.0` public interfaces remain unchanged
- GitHub milestone/issues and tracked docs tell the same 0.5.0 story
- Real SFT/DPO training evidence is scoped as follow-up work, not the main
  onboarding gate

---

## Completed Phase — v0.6.0 Real SFT Evidence

**Goal:** Prove that one Linux/HF-backed SFT smoke run can produce auditable
non-mock `training_run.v0` evidence and flow into the release-candidate chain,
while core CI remains offline and dummy-first.

### Deliverables
- Linux-only `[train]` SFT smoke path using `unsloth/tinyllama-chat-bnb-4bit`
  as the standard tiny release evidence model
- Low-risk smoke settings for `--smoke --max-steps 1`: small batch, short
  sequence length, deterministic seed, and automatic precision
- Additive adapter load/generate smoke evidence inside `training_run.v0`
- Release-candidate gate that accepts real SFT evidence with
  `--final-training-stage sft` and `--require-real-training-evidence`
- Manual v0.6 release evidence checklist for local checks, Linux evidence,
  TestPyPI, PyPI, GitHub Release, and milestone closure
- Explicit v0.7 scope for DPO lineage and meaningful quality thresholds

### Exit criteria
- Placeholder `dummy/base` cannot launch real training; users get a clear
  message to pass a Hugging Face base model
- A successful SFT smoke run records adapter refs, file checksums, and
  successful adapter smoke evidence
- The v0.6 manual release command produces an OK release-candidate bundle with
  `training_evidence.real_training_evidence == true`
- Core CI and first-run flows remain deterministic and provider-safe without
  `[train]`
- DPO and quality-improvement thresholds remain documented as post-v0.6 work

---

## Active Phase — v0.7.0 Real DPO Evidence

**Goal:** Prove that a Linux/HF-backed DPO smoke run can produce auditable
non-mock `training_run.v0` evidence after a real SFT parent run, while keeping
quality-improvement claims evidence-only instead of a hard release gate.

### Deliverables
- Linux-only `[train]` DPO smoke path from generated pilot DPO data and a real
  parent SFT `training_run.v0`
- Release-candidate gate that accepts real DPO evidence with
  `--run-training --run-dpo --final-training-stage dpo`
- Parent SFT enforcement for real DPO evidence:
  non-mock, successful, adapter refs, file checksums, and successful adapter
  smoke evidence
- DPO final adapter evidence:
  final adapter refs, file checksums, parent lineage, and successful adapter
  smoke evidence
- Manual v0.7 release evidence guide and checklist
- Explicit follow-up scope for hard model quality thresholds, deeper MCP runtime
  execution semantics, and critique/rubric dataset exports

### Exit criteria
- The v0.7 manual release command produces an OK release-candidate bundle with
  `training_evidence.final_stage == "dpo"`
- `training_evidence.real_training_evidence == true` and
  `training_evidence.parent_real_training_evidence == true`
- External DPO reports used with `--require-real-training-evidence` must satisfy
  the same non-mock DPO and parent SFT checks
- Core CI and public first-run flows remain deterministic and provider-safe
  without `[train]`
- Distillation and benchmark reports still surface score/cost/latency movement,
  but quality improvement is not required for this milestone

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
- Optional SFT and DPO `training_run.v0` evidence for exchange packaging
- Real SFT -> DPO evidence-first release gate with parent lineage checks
- Evidence-only distillation eval gate for candidate specialists
- Benchmark matrix and frontier reports
- Specialist exchange unit validation, package evidence, smoke evidence, and registry index
- One-command release-candidate evidence gate
- Optional non-mock training evidence requirement for release-candidate reviews
- Opt-in release-candidate SFT/DPO training execution and trained-adapter evaluation
- Learning/routing primitives and experimental training utilities
- Cost/latency-aware evaluation reporting

---

## Success criteria for this roadmap

SignalForge AI succeeds when:
- Production traces become the canonical source for eval and learning
- Reliability regressions are diagnosed by failure mode, not anecdotes
- Specialist models are trained and validated against real agent workloads
- Model/pattern choices are made from benchmark evidence, not intuition
