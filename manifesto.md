# TensorFoundry Manifesto

## We are not building chatbots

TensorFoundry exists to build **agentic systems** — systems that plan, act, observe, recover from failure, and improve over time.

A chatbot produces text.  
An agent produces **outcomes**.

Modern LLMs are powerful, but without structure they are brittle, expensive, and unfit for real work. TensorFoundry focuses on the missing layer: **how intelligence is operationalised**.

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

TensorFoundry treats agents as **software systems**, subject to the same rigor as any other production service.

---

## Orchestration is the product

Most agent frameworks focus on *how to talk to a model*.  
We focus on **how systems behave over time**.

Orchestration includes:
- Planner → Executor → Critic loops
- Cost-aware model routing
- Retry strategies that mutate state, not just prompts
- Multi-agent coordination and debate
- Deterministic fallbacks when intelligence fails

This layer — not the model — is where reliability is built.

---

## Evaluation is non-negotiable

Language quality is not success.

TensorFoundry evaluates agents on:
- Task success
- Cost per successful outcome
- Latency
- Robustness under perturbation
- Regression across versions

If you cannot measure whether an agent is improving, you are guessing.

Agents should fail loudly, measurably, and informatively.

---

## Logs are the dataset

Static datasets are a snapshot.  
Agent execution logs are a **living record of intelligence in action**.

Every agent run should produce structured traces:
- Decisions made
- Tools used
- Failures encountered
- Retries attempted
- Costs incurred
- Outcomes achieved

These logs are not just for debugging — they are the foundation for:
- Imitation learning
- Preference learning
- Continual improvement
- Model specialization

TensorFoundry is built around the idea that **execution precedes learning**.

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

TensorFoundry prioritises **system intelligence** over raw parameter count.

---

## Small models will win at execution

Large models are excellent teachers.  
Small models are excellent workers.

By training on agent execution traces — not generic text — small models can:
- Learn domain-specific behaviour
- Execute plans reliably
- Operate at a fraction of the cost
- Run privately and locally

TensorFoundry is designed to support this transition, from foundation models to **agent-optimised models**.

---

## Open systems beat closed demos

TensorFoundry is open by design.

We believe:
- Interfaces should be inspectable
- Failures should be reproducible
- Benchmarks should be public
- Assumptions should be explicit

We are not building a black box.  
We are building a **commons for agent engineering**.

---

## What TensorFoundry is not

To be explicit, TensorFoundry is not:
- A prompt library
- A chatbot UI
- A no-code toy
- A model leaderboard
- A thin wrapper around a single LLM provider

If your system cannot survive model changes, cost constraints, or partial failures, it is not finished.

---

## Our direction

TensorFoundry will evolve across three layers:
1. **Agent templates** — practical, production-oriented blueprints
2. **Orchestration and evaluation** — the reliability layer
3. **Learning from execution** — turning logs into intelligence

This repository begins with the first two, and lays the foundation for the third.

---

## Our invitation

If you believe:
- Agents should be engineered, not prompted
- Evaluation matters more than demos
- Reliability beats cleverness
- Small, specialised systems will outperform monoliths

Then TensorFoundry is for you.

Build with us.