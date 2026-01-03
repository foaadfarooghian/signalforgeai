# TensorFoundry — Learning Ops

This directory contains the components that turn TensorFoundry into a **closed-loop learning system**.

Most runtime learning code lives in `src/tensorfoundry/learning/` (bandits, routing, stats).
This `learning_ops/` folder stores policies, bandit state, datasets, and operational learning utilities.

> **Execute → Evaluate → Learn**  
> Logs are the dataset. Evaluation is the signal. Learning is controlled, measurable, and reversible.

This is **not** a generic machine-learning training folder.  
It exists to support *agentic learning* grounded in real executions and benchmarks.

---

## Core Principles

1. **Agents learn from execution, not prompts**
   - Every agent run emits a structured trace
   - Traces are append-only and immutable
   - No hand-curated training examples

2. **Evaluation is the contract**
   - Learning is only allowed through explicit evaluation signals
   - Benchmarks gate all improvements
   - Regressions are first-class failures

3. **Learning starts without weight updates**
   - Policies, routing, and orchestration improve first
   - Model training comes later, behind strict gates

4. **Small, specialised models over large general ones**
   - Optimised for agentic workflows
   - Trained on high-signal traces
   - Designed for continual improvement

---

## Directory Structure

```text
learning_ops/
├── README.md
├── bandits/
│   └── routing_bandits_v0.json  # Bandit state snapshots
├── policies/
│   └── routing_v0.json        # Learned routing decisions
├── scripts/
│   ├── prepare_sft.py          # Build SFT JSONL from benchmark data
│   ├── prepare_dpo.py          # Build DPO JSONL preference pairs
│   ├── prepare_sft_chat.py     # Chat-format SFT JSONL
│   └── reports/
│       └── summarize_rewards.py    # Summarize reward stats from logs
└── datasets/                  # (future) training-ready datasets
```

Training scripts are **experimental** and may require extra dependencies (e.g. `unsloth`, `trl`).
Packaged training entrypoints live in `src/tensorfoundry/training/` and are exposed via `tensorfoundry-learn train`.

---

## Canonical pipeline (logs → datasets → training)

Export log-derived datasets:

```bash
tensorfoundry-learn export \
  --suite benchmark_v1_synth \
  --out-dir datasets/examples \
  --sft --prefs --repairs --curriculum \
  --prompt-source instruction \
  --prompt-normalize mask_json \
  --limit 50
```

Train a student model (optional deps required):

```bash
pip install -e ".[train]"

tensorfoundry-learn train \
  --base-model Qwen/Qwen2.5-3B-Instruct \
  --sft --dpo \
  --sft-data datasets/examples/benchmark_v1_synth.sft.jsonl \
  --dpo-data datasets/synth_train.dpo.ready.jsonl
```

Schema versions:
- `sft.v0` (instruction/prompt + response)
- `prefs.v0` / `dpo.v0` (preferences or DPO-ready)
- `repairs.v0` (failed → repaired pairs)
- `curriculum.v0` (difficulty buckets)

---

## What “Learning” Means in TensorFoundry

Learning happens in **stages**, each strictly safer than the next.

### Learn v0 — Policy Learning

No model weights are updated.

Learning improves:
- model selection (routing)
- agent configuration
- retry / escalation strategies
- cost–quality trade-offs

**Input**
- `logs/**/reward.jsonl`

**Output**
- `policies/routing_v0.json`

This stage is:
- fast
- reversible
- CI-guarded

---

### Learn v1 — Preference & Distillation (current)

Still conservative, but data now feeds models.

Learning produces:
- supervised fine-tuning datasets
- preference pairs
- teacher–student traces
- failure → repair pairs
- curriculum buckets for progressive difficulty

**Sources**
- benchmark-passing executions for positive examples
- high-confidence rewards
- repaired outcomes linked to failed traces

---

### Learn v2 — Continual Learning (research)

Experimental and gated.

Goals:
- avoid catastrophic forgetting
- support incremental updates
- specialise models per agent archetype

This stage is **not enabled by default** and will always require:
- benchmark parity
- rollback capability
- explicit opt-in

---

## Reward Signal

Learning is driven by explicit reward artifacts.

Each evaluation run produces:

```text
logs/<run_id>/reward.jsonl
```

Each reward entry includes:
- `suite_id`, `case_id`
- `agent_id`, `model_id`
- `overall_score` (0–1)
- `subscores`
- `violations`
- terminal outcome metadata

Rewards are:
- append-only
- versioned (`reward.v0`)
- validated in CI

---

## Routing Policy (Learn v0)

Routing policies select the best model or configuration per benchmark suite.

Example:

```json
{
  "version": "routing.v0",
  "default_model": "gpt-5-mini",
  "by_suite": {
    "research_quickstart": "gpt-5",
    "decision_quickstart": "gpt-5-mini",
    "refactor_quickstart": "gpt-5"
  }
}
```

Policies are:
- derived from reward history
- human-inspectable
- safe to deploy and revert

---

## What This Is *Not*

- ❌ A generic ML training pipeline
- ❌ A prompt-engineering playground
- ❌ A black-box RL system
- ❌ An auto-updating model loop

TensorFoundry prioritises **control, observability, and evidence** over automation.

---

## Roadmap (Learning/Training)

- [x] Reward schema (`reward.v0`)
- [x] CI-validated reward artifacts
- [x] Routing policy (Learn v0)
- [x] Bandit-based routing (Learn v0.1)
- [x] Dataset builders (SFT / preference)
- [x] Teacher → student QLoRA SFT + DPO (experimental)
- [x] Local model improvement measured (`results/benchmark_v1_synth.model_improvements.md`)
- [x] Curriculum construction from real executions
- [x] Canonical learn pipeline (`tensorfoundry-learn`)
- [ ] Specialist small models
- [ ] Continual learning experiments (opt-in)

---

## Design Philosophy

> *“We don’t ask models to be smart in isolation.  
> We ask systems to become better over time.”*

TensorFoundry learning is about **systemic intelligence**, not model mysticism.
