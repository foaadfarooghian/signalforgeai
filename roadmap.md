# TensorFoundry Roadmap

This roadmap describes how TensorFoundry evolves from open-source agent building blocks into a full system for **engineering, evaluating, and improving agentic intelligence**.

The focus is deliberate:
- Systems before models
- Execution before learning
- Reliability before scale

Dates are indicative. Milestones matter more than calendars.

---

## Phase 0 — Foundation (Now)

**Goal:** Establish TensorFoundry as a serious, opinionated open-source project for agent engineering.

### Deliverables
- Repository structure and governance
- Manifesto and design principles
- Clear scope and non-goals
- First runnable example

### Status
- Repo skeleton
- Manifesto
- Roadmap

### Exit criteria
- A new contributor understands what TensorFoundry is and is not
- Repo can be cloned and executed locally
- Direction is unambiguous

---

## Phase 1 — Agent Templates & Orchestration (Weeks 1–4)

**Goal:** Provide practical, production-oriented blueprints for building agents.

### Agent templates
- Research Agent (plan → tool use → synthesis)
- Codebase Refactor Agent (read → plan → patch → validate)
- Decision / Analysis Agent (assumptions → options → trade-offs)

Each agent includes:
- Explicit input/output schemas
- State representation
- Tool interfaces
- Failure modes

### Orchestration patterns
- Planner → Executor → Critic
- Retry with state mutation
- Cost-aware model routing
- Deterministic fallbacks

### Exit criteria
- Agents can be composed interchangeably with orchestration patterns
- Patterns are documented with trade-offs and failure cases
- At least one end-to-end agent demo runs reliably

---

## Phase 2 — Evaluation & Logging (Weeks 3–6)

**Goal:** Make agent behaviour measurable, comparable, and debuggable.

### Evaluation
- Task-based evaluation suites
- Metrics:
  - Task success
  - Cost per success
  - Latency
  - Robustness / variance
- Regression detection across versions

### Logging
- Canonical execution trace schema
- Structured JSONL logs capturing:
  - State transitions
  - Tool usage
  - Decisions and retries
  - Errors and outcomes
  - Cost and latency

### Exit criteria
- Every agent run emits valid structured logs
- Agents can be evaluated consistently across runs
- Failures are inspectable and reproducible

---

## Phase 3 — Benchmarks & Community (Weeks 6–10)

**Goal:** Define what “good” looks like for agentic systems.

### Benchmarks
- Public benchmark suite for agentic tasks
- Tasks across:
  - Research
  - Code modification
  - Operational decision-making
- Clear scoring rubrics (outcome-focused)

### Community
- Contribution guidelines for agents and patterns
- RFC process for major changes
- First external contributors

### Exit criteria
- Third parties can benchmark their agents against TensorFoundry tasks
- Contributions follow consistent interfaces and standards

---

## Phase 4 — Logs → Learning (Design-first)

**Goal:** Turn execution traces into training signal.

> This phase focuses on **infrastructure and interfaces**, not large-scale training.

### Learning pipelines
- Log → imitation dataset conversion
- Preference extraction from successful vs failed traces
- Curriculum construction from real executions

### Model-agnostic design
- No coupling to a single framework or provider
- Supports:
  - Teacher–student distillation
  - Offline fine-tuning
  - Continual learning loops

### Exit criteria
- Example datasets generated from real agent logs
- Clear training interfaces defined
- Small-scale experiments reproducible

---

## Phase 5 — Small Agent-Optimised Models (Future)

**Goal:** Train and deploy small, specialised models optimised for agent execution.

### Focus
- ~0.5B–1B parameter models
- Trained on:
  - Agent execution traces
  - Tool usage patterns
  - Recovery and retry behaviour

### Capabilities
- Planning and execution reliability
- Lower inference cost
- Private / local deployment

### Non-goals
- Competing with foundation models
- General-purpose chat capabilities

### Exit criteria
- Small models outperform larger models on defined agent benchmarks
- Demonstrable cost and reliability gains

---

## What success looks like

TensorFoundry succeeds if:
- Engineers use it to build real agents, not demos
- Agent behaviour is measurable and improvable
- Execution logs become a first-class training asset
- Small models reliably execute structured tasks

---

## What we deliberately avoid

- Prompt-only abstractions
- Model leaderboards without tasks
- Unmeasurable “intelligence”
- Vendor lock-in
- UI-first development

---

## Summary

TensorFoundry is built bottom-up:
1. **Agents**
2. **Orchestration**
3. **Evaluation**
4. **Logs**
5. **Learning**
6. **Models**

Each layer compounds the next.

This repository starts at the foundation.