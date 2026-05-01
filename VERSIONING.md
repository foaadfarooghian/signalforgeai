# Versioning & Stability Policy

TensorFoundry follows **semantic versioning with research-grade guarantees**.

Version numbers communicate **which invariants contributors and users can rely on**, not whether the project is "finished" or production-ready.

---

## Current status

TensorFoundry is intentionally **pre-1.0**.

The project is an evolving agent-engineering system where:
- execution behaviour matters more than surface APIs
- learning is derived from real execution traces
- interfaces may evolve as research progresses

Pre-1.0 does **not** mean unstable or toy-grade. It means:
> correctness, observability, and reversibility take priority over frozen APIs.

Current pilot-readiness contracts are stabilized additively:
- new traces emit `schema_version: "trace.v0"`
- legacy traces without `schema_version` remain valid
- rewards remain `reward.v0`
- exported datasets remain `sft.v0`, `prefs.v0`, `dpo.v0`, `repairs.v0`, and `curriculum.v0`

---

## What version numbers mean

TensorFoundry uses the form:

```
vMAJOR.MINOR.PATCH
```

### MAJOR (`v1.0.0`, `v2.0.0`, …)

A MAJOR version is released **only when system interfaces are considered stable**.

`v1.0.0` will indicate:
- core agent and orchestration interfaces are stable
- logging, evaluation, and learning schemas are versioned and documented
- breaking changes are exceptional and deliberate

Importantly:
> `v1.0.0` does **not** imply trained models, benchmark leadership, or production guarantees.

It represents **interface and system stability**, not model quality.

---

### MINOR (`v0.x.0`)

MINOR versions represent **capability milestones**.

A MINOR version increments when TensorFoundry gains a new core capability, such as:
- new orchestration or routing mechanisms
- evaluation or benchmarking extensions
- learning infrastructure derived from execution traces
- dataset export or training interfaces

MINOR releases may include additive changes and limited breaking changes while pre-1.0.

---

### PATCH (`v0.x.y`)

PATCH versions are for:
- bug fixes
- documentation improvements
- internal refactors without behavioural changes

PATCH releases should not change external behaviour.

---

## Learning vs stability

Introducing learning capabilities **does not imply API stability**.

Learning features may evolve as:
- reward definitions improve
- dataset schemas mature
- training interfaces are refined

TensorFoundry intentionally separates:
- **learning principles** (stable)
- **learning implementations** (replaceable, experimental)

All learning is:
- system-level
- model-agnostic
- reversible

---

## Historical tags

Some early tags were created before this policy was formalised.

They should be interpreted as **capability snapshots**, not long-term stability guarantees.

From this point forward, version numbers follow the rules defined in this document.

---

## Guiding principles

> Logs are the dataset.  
> Evaluation is the contract.  
> Learning must be measurable, reversible, and safe.

Versioning exists to protect these invariants — not to signal hype.
