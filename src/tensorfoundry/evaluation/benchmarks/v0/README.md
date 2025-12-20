# TensorFoundry Benchmark v0

Benchmark v0 is a small, outcome-focused benchmark suite for **agentic systems**.

It is intentionally simple:
- task-based cases
- schema-valid traces required
- lightweight expectations (`status` + keyword checks)

This benchmark is designed to be:
- runnable locally in minutes
- CI-friendly
- extensible without breaking

---

### Reference agents vs user agents

The agents used in Benchmark v0 (ResearchAgent, DecisionAgent, RefactorAgent)
are **reference implementations**, not requirements.

Any agent can be benchmarked against these suites as long as it:
- emits TensorFoundry-compliant traces, and
- produces a terminal outcome.

The benchmark measures **agentic behaviour**, not specific agent classes.

## What it measures (v0)

Across three agent types:

### Research
- ability to produce a coherent short answer (placeholder today)
- consistent trace emission across planning/tool/model stages

### Decision
- structured recommendation style output
- constraints/options incorporated into the memo

### Refactor
- safe deterministic patch workflow (dry-run by default)
- file selection + patch proposal + validation

---

## How to run

From repo root:

```bash
python -m tensorfoundry.evaluation.run src/tensorfoundry/evaluation/benchmarks/v0/suites/research_v0.json
python -m tensorfoundry.evaluation.run src/tensorfoundry/evaluation/benchmarks/v0/suites/decision_v0.json
python -m tensorfoundry.evaluation.run src/tensorfoundry/evaluation/benchmarks/v0/suites/refactor_v0.json
```

## Outputs

- `results/<suite>.results.json`
- `results/<suite>.summary.md`
- `traces in logs/*.jsonl`

## Scoring (v0)

Each case is scored using:
- `expect.status` (primary)
- `expect.contains_any` keyword checks (secondary)

A case passes if:
- the terminal outcome.status matches, and
- (optionally) at least one expected keyword appears in the extracted result text

Trace validity is required:
- schema validation failures fail the case immediately

## How to add a benchmark case

Add a case to suite JSON file:

```json
{
  "id": "res_999",
  "task": "Your task here",
  "inputs": {},
  "expect": {
    "status": "success",
    "contains_any": ["keyword1", "keyword2"]
  }
}
```

Guidelines:
- keep tasks short and unambiguous
- keep expectations minimal (avoid brittle tests)
- prefer keyword checks that reflect intent, not exact phrasing

## Notes

Benchmark v0 is a foundation benchmark. It will evolve toward:
- Stronger rubrics
- Richer scoring
- Deterministic replay
- Regression tracking across releases