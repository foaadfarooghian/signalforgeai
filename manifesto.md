# SignalForge AI Manifesto

## We are not building chatbots

SignalForge AI exists to build **agentic systems** — systems that plan, act, observe, recover from failure, and improve over time.

A chatbot produces text.  
An agent produces **outcomes**.

Modern LLMs are powerful, but without structure they are brittle, expensive, and unfit for real work. SignalForge AI focuses on the missing layer: **how intelligence is operationalised**.

---

## Agents are software systems, not prompts

An agent is not a clever prompt.

A real agent has:
- Explicit **state**
- Access to **tools**
- The ability to **plan**
- Mechanisms for **verification and recovery**
- Measurable **success and failure**

If an agent cannot explain what it tried, why it failed, and what it will do differently next time, it is not production-ready.

SignalForge AI treats agents as **software systems**, subject to the same rigor as any other production service.

---

## Runtime is the product

Most agent frameworks focus on *how to talk to a model*.  
We focus on **how systems execute over time**.

Runtime engineering includes:
- Planner → Executor → Critic loops
- Cost-aware model routing
- Retry strategies that mutate state, not just prompts
- Deterministic fallbacks when intelligence fails
- OTel-native trace semantics for production observability
- MCP-native tool invocation contracts

This layer — not the model — is where reliability is built.

---

## Evaluation is non-negotiable

Language quality is not success.

SignalForge AI evaluates agents on:
- Task success
- Cost per successful outcome
- Latency
- Robustness under perturbation
- Regression across versions
- Failure-mode distribution across multi-step/tool-using runs

If you cannot measure whether an agent is improving, you are guessing.

Agents should fail loudly, measurably, and informatively.

---

## Logs are the dataset

Static datasets are a snapshot.  
Agent execution logs are a **living record of intelligence in action**.

Every agent run should produce structured traces and artifacts:
- Decisions made
- Tools used
- Failures encountered
- Retries attempted
- Costs incurred
- Outcomes achieved
- Eval verdicts and reward rows

These logs are not just for debugging — they are the foundation for:
- Imitation learning
- Preference learning
- Continual improvement
- Model specialization

SignalForge AI is built around the idea that **execution precedes learning**.

---

## Bigger models are not the answer

For agentic work, scale is often misapplied.

Most agent tasks are:
- Repetitive
- Tool-heavy
- Structured
- Cost-sensitive

These workloads favour:
- Smaller, specialised models
- Strong orchestration
- Explicit state machines
- Feedback loops

SignalForge AI prioritises **system intelligence** over raw parameter count.

---

## Small models will win at execution

Large models are excellent teachers.  
Small models are excellent workers.

By training on agent execution traces — not generic text — small models can:
- Learn domain-specific behaviour
- Execute plans reliably
- Operate at a fraction of the cost
- Run privately and locally

SignalForge AI is designed to support this transition, from foundation models to **agent-optimised models**.

---

## Open systems beat closed demos

SignalForge AI is open by design.

We believe:
- Interfaces should be inspectable
- Failures should be reproducible
- Benchmarks should be public
- Assumptions should be explicit

We are not building a black box.  
We are building a **commons for agent engineering**.

---

## What SignalForge AI is not

To be explicit, SignalForge AI is not:
- A prompt library
- A chatbot UI
- A no-code toy
- A model leaderboard
- A thin wrapper around a single LLM provider

If your system cannot survive model changes, cost constraints, or partial failures, it is not finished.

---

## Our direction

SignalForge AI is now deliberately narrowed to platform pillars:

1. **OTel/MCP-native runtime + artifact schema**
2. **Evaluation and failure analysis for multi-step/tool-using agents**
3. **Dataset generation from production traces**
4. **Distillation pipeline for specialist small models**
5. **Benchmarking cost, latency, and reliability across models and patterns**
6. **Specialist model registry/exchange with full operational metadata**

Execution order matters:
- Runtime contracts first
- Evaluation and failure diagnosis second
- Dataset and distillation loops third
- Benchmark-driven optimization throughout
- Exchange publication once units are benchmarked and auditable

---

## Our invitation

If you believe:
- Agents should be engineered, not prompted
- Evaluation matters more than demos
- Reliability beats cleverness
- Small, specialised systems will outperform monoliths

Then SignalForge AI is for you.

Build with us.
