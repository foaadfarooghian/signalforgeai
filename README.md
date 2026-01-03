<p align="center" style="margin: 0 0 1px 0;">
  <img src="docs/assets/logo.svg" alt="TensorFoundry logo" width="200">
</p>

<h1 align="center" style="margin: 0 0 10px 0;">
  <span style="color:#FF7A18;">Tensor</span><span style="color:#1F6FEB;">Foundry</span>
</h1>

<p align="center">
  <strong>An opinionated open-source framework for engineered AI agents</strong><br/>
  Build, orchestrate, observe, and evaluate agentic workflows — not chatbots.
</p>

<p align="center">
  <img alt="status" src="https://img.shields.io/badge/status-active_development-blue" />
  <img alt="python" src="https://img.shields.io/badge/python-3.10%2B-blue" />
  <img alt="license" src="https://img.shields.io/badge/License-Apache%202.0-green.svg" />
  <img alt="ci" src="https://img.shields.io/github/actions/workflow/status/foaadfarooghian/tensorfoundry/ci.yml?branch=dev" />
</p>

---

## Project status

**TensorFoundry is in active development.**

- Core abstractions and APIs are stabilising
- Logging, orchestration, and evaluation are production-oriented
- Expect **breaking changes** before the first stable release (`v1.0.0`)

This project is being built **in the open** with correctness, observability, and long-term maintainability as first-class goals.

If you’re evaluating TensorFoundry today:
- ✅ Suitable for experimentation and internal tools
- ⚠️ Not yet recommended for mission-critical production without pinning versions

---

## What is TensorFoundry?

**TensorFoundry** is an opinionated framework for building **agentic systems as engineered software**, not prompt demos.

It is designed for teams who care about:
- **execution**, not demos
- **observability**, not guesswork
- **evaluation**, not vibes

Agents in TensorFoundry are **stateful systems** with tools, retries, failure handling, and measurable outcomes.

TensorFoundry provides the **infrastructure and abstractions** needed to build these systems —  
*not* a fixed catalog of pre-built agents.

---

## What makes TensorFoundry different

Most agent frameworks optimise for prompts and single-run success.

TensorFoundry optimises for:

- **Execution** — tools, state, retries, failure handling
- **Orchestration** — planner → executor → critic patterns
- **Observability** — structured traces as first-class artefacts
- **Evaluation** — task-based scoring and regression detection
- **Provider flexibility** — OpenAI, Anthropic, Gemini, local models

If an agent can’t be logged, validated, and compared over time, it’s not production-ready.

### Cost-aware agent routing

TensorFoundry supports **economics-aware model routing**:

- token usage, latency, and USD cost captured per model call
- reward artifacts enriched with economic signals
- online learning (Thompson Sampling) routes models per suite
- configurable cost and latency penalties

This allows agents to choose *when* expensive intelligence is worth it — and when cheaper models are “good enough”.

---

## What TensorFoundry is not

TensorFoundry is **not**:
- a fixed set of pre-built agents
- a chatbot framework
- a prompt library
- a no-code agent builder

It is a **software framework** for engineering agentic systems with strong guarantees around
logging, evaluation, and reproducibility.

---

## What’s included (v1 scope)

TensorFoundry is a **framework**, not a catalog of agents.

The components below define the **core platform**, alongside a small set of
**reference implementations** that demonstrate how to use it.

### Reference agent templates (examples)

These agents are provided as **examples**, not limitations.
You are expected to build your own domain-specific agents on top of the framework.

- **ResearchAgent** — structured research with tools and synthesis
- **DecisionAgent** — decision memos with assumptions and trade-offs
- **RefactorAgent** — safe, deterministic code refactoring (dry-run by default)

### Orchestration

- Planner → Executor → Critic pattern
- Explicit retry logic with state mutation

### Logging & observability

- Canonical JSONL logging schema
- Safe-by-default payload sanitisation
- Trace validation (`validate`)
- Human-readable inspection (`inspect`)
- Trace diffing (`diff`)

### Evaluation

- Task-based evaluation harness
- Suite definitions in JSON
- Result summaries and pass rates
- Regression detection via diffing

### Learning & training (Phase 4 progress)

- Reward artifacts emitted per run (`reward.jsonl`)
- Logs → SFT dataset export (instruction/prompt)
- Logs → preference dataset export (utility + latency weighted)
- Logs → repair pairs (failed → repaired)
- Curriculum buckets from real executions (easy/repair/escalation)
- Bandit-based routing integrated into evaluation
- HF provider integrated into the evaluation loop
- Canonical learning pipeline (`tensorfoundry-learn`)
- Experimental teacher → student SFT/DPO via packaged training modules (`src/tensorfoundry/training/`)
- Measured local-model improvements (`results/benchmark_v1_synth.model_improvements.md`)
- Log-derived example datasets (`datasets/examples/`)

### Quality & CI

- Tests covering orchestration, logging, and evaluation
- CI enforcing schema correctness and evaluation success

---

## Quickstart (5 minutes)

```bash
git clone https://github.com/foaadfarooghian/tensorfoundry.git
cd tensorfoundry

python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Run a **reference research agent (example)**:

```bash
python examples/quick_research_agent.py
```

Validate and inspect the trace:

```bash
python -m tensorfoundry.logging.validate logs/$(ls -t logs | head -n 1)
python -m tensorfoundry.logging.inspect logs/$(ls -t logs | head -n 1)
```

Run evaluation suites:

```bash
python -m tensorfoundry.evaluation.run src/tensorfoundry/evaluation/suites/quickstart.json
python -m tensorfoundry.evaluation.run src/tensorfoundry/evaluation/suites/research_quickstart.json
```

These examples demonstrate TensorFoundry’s core primitives.
The same workflow applies to any custom agent you build.

---

## Building your own agents

TensorFoundry is designed for **custom, domain-specific agents**.

Typical use cases include:
- internal research assistants
- decision-support systems
- code analysis and refactoring tools
- evaluation-first LLM pipelines

A minimal agent is composed of:
- a planner
- an executor
- optional critics and tools

Documentation will expand as APIs stabilise.

---

## Core ideas

- Agents ≠ chatbots
- Logs are the dataset
- Evaluation > clever prompts
- Orchestration is the product

See `manifesto.md` for the full philosophy and `roadmap.md` for what’s coming next.

---

## Repository layout

```text
src/tensorfoundry/
├── agents/          # Reference agent templates (examples)
├── orchestration/   # Planner–Executor–Critic patterns
├── logging/         # Schema, emitter, validate, inspect, diff
├── evaluation/      # Harness, suites, results diff
├── examples/        # Runnable examples
└── docs/            # Design notes
```

Other top-level directories:
- `learning_ops/` — policies, bandit state, and experimental training utilities
- `datasets/` — generated JSONL datasets

---

## Branching & releases

- `prod` — protected, stable release branch
- `dev` — integration branch for ongoing work
