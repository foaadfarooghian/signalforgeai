# Changelog

All notable changes to **TensorFoundry** will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),  
and this project adheres to [Semantic Versioning](https://semver.org/).

---

## [Unreleased]

### Added
- `trace.v0` schema version emission for new runtime events while preserving legacy trace validation.
- Normalized tool event payload fields for new trace artifacts.
- `tensorfoundry-pilot-check` for deterministic production-pilot readiness validation.
- OTel JSON trace export via `tensorfoundry-otel-export`.
- Dataset export validation and manifest generation.
- Provider readiness checks for dummy, OpenAI, Ollama, and HF model IDs.
- `tensorfoundry-learn train --dry-run` for training input and dependency preflight.
- Optional reward diagnostics: `failure_mode`, `diagnosis`, and `artifact_refs`.

### Changed
- CI now runs offline with dummy providers by default and includes the pilot readiness check.
- Package data now includes benchmark suites and nested pricing config artifacts.
- Mypy is aligned to the project Python 3.11+ support floor.

### Planned
- Dataset export (trace + reward → SFT / preference data)
- Preference learning and distillation pipelines
- Research into small, continually learning agent-specialised models


---

## [0.3.0] — 2025-12-28

### Added
- **Real hosted and local model providers** behind the ModelProvider interface
- **Token usage, latency, and USD cost tracking** per model call
- **Config-driven pricing** for hosted providers
- **Cost- and latency-aware effective reward**
  - `effective = score − λ_cost · cost − μ_latency · latency`
- **Economics-aware Thompson Sampling routing**
  - Learned per benchmark suite
  - Fractional reward support
  - Minimum-pull exploration

### Changed
- Model routing decisions can now trade off quality vs cost vs latency
- Reward artifacts now carry full economic provenance
- Learning behaviour reflects real execution economics, not proxy metrics

### Architectural Notes
- Routing decisions are learned online, not statically configured
- Economic signals are first-class inputs to learning
- Agents remain model-agnostic; all optimisation happens at the routing layer

### Breaking Changes
- None  
  All changes are additive and backward-compatible.
---

## [0.2.0] — 2025-12-24

### Added
- **Closed-loop learning pipeline** completing *Execute → Evaluate → Learn*
- **ModelProvider abstraction** for all model calls
  - Keeps agents model-agnostic and reusable
- **DummyProvider** for deterministic testing and demos
- **Thompson-sampling bandits** for model routing
  - Learned per benchmark suite
  - Supports minimum-pull exploration to prevent early lock-in
- **Structured reward artifacts (`reward.jsonl`)**
  - Success, score, violations
  - Latency and cost (when available)
- **Reward-driven learning**
  - Bandits update directly from evaluation outputs
- **Latency and cost propagation**
  - Captured from `model_called` events
  - Flow end-to-end into learning decisions

### Changed
- Example agents now call models exclusively via the **ModelProvider interface**
- Learning logic is fully decoupled from agent implementations
- Deterministic refactor benchmarks are excluded from learning updates

### Architectural Notes
- Agents are treated strictly as **examples**, not special cases
- Learning happens at the orchestration/routing layer first
- No model weights are updated in this version (learning without retraining)

### Breaking Changes
- None  
  All changes are additive and backward-compatible.

---

## [0.1.0] — 2025-12-19

### Added
- Initial TensorFoundry execution framework
- Agent templates (Decision, Research, Refactor)
- Planner → Executor → Critic orchestration pattern
- Structured JSONL tracing with validation
- Evaluation harness with task-based benchmarks
- CI enforcement (linting, typing, tests, smoke runs)

---

## Versioning Notes

- **0.x releases** may introduce architectural changes as the platform matures
- Learning capabilities are introduced incrementally and guarded by benchmarks
- Backward compatibility is maintained wherever possible

---

## Philosophy

> Logs are the dataset.  
> Evaluation is the contract.  
> Learning must be measurable, reversible, and safe.

TensorFoundry prioritises **systemic intelligence** over ad-hoc model tuning.
