# First Pilot Walkthrough

This walkthrough starts from the public PyPI package and runs the deterministic
offline pilot path. It does not require OpenAI, Ollama, Hugging Face, GPUs, or
networked model calls.

## Install

Use Python 3.11 or newer:

```bash
python -m venv .venv
source .venv/bin/activate
pip install signalforgeai==0.4.0
```

## Run the offline pilot check

```bash
signalforgeai-pilot-check --mode dummy --work-dir results/pilot_check
```

The command uses the deterministic `dummy_good` provider and writes a complete
pilot bundle:

- `results/pilot_check/pilot_readiness.md`
- `results/pilot_check/pilot_readiness.json`
- `results/pilot_check/logs/`
- `results/pilot_check/datasets/manifest.json`
- SFT, preference, repair, and curriculum dataset exports

The check should exit successfully on a fresh environment. If it does not, the
Markdown readiness report is the first file to inspect.

## Inspect trace and reward artifacts

Find the latest run directory:

```bash
latest_run="$(ls -t results/pilot_check/logs | head -n 1)"
```

Validate and inspect the generated trace:

```bash
python -m signalforgeai.logging.validate "results/pilot_check/logs/${latest_run}"
python -m signalforgeai.logging.inspect "results/pilot_check/logs/${latest_run}"
```

The trace path should include `trace.v0` events, and the run directory should
include `reward.v0` rows for scored cases.

## Validate exported datasets

```bash
signalforgeai-dataset-validate results/pilot_check/datasets/pilot.sft.jsonl \
  --kind sft --quality-gate --logs-root results/pilot_check/logs
```

The dataset validator checks schema shape, deterministic split metadata,
provenance back to trace/reward artifacts, duplicate payloads, content hashes,
and split leakage.

## Add a regression baseline

Keep the last accepted readiness artifact and compare new pilot runs against it:

```bash
signalforgeai-pilot-check --mode dummy --work-dir results/pilot_current \
  --baseline results/pilot_baseline/pilot_readiness.json
```

Baseline mode emits `eval_regression.v0` JSON and Markdown. The default policy
allows no pass-rate drop, no mean-score drop, no new failing cases, and no worse
failure-mode movement.

## Move to provider-backed runs

Hosted and local providers are explicit configuration paths. Keep the offline
workflow working first, then require provider smoke checks in configured
environments:

```bash
SIGNALFORGEAI_HOSTED_MODEL_ID=openai:gpt-5-mini \
  signalforgeai-pilot-check --require-provider hosted --work-dir results/pilot_hosted

SIGNALFORGEAI_LOCAL_MODEL_ID=ollama:ministral-3:8b \
  signalforgeai-pilot-check --require-provider local --work-dir results/pilot_local
```

From a source checkout, examples can also run against explicit providers:

```bash
SIGNALFORGEAI_MODEL_ID=openai:gpt-5-mini python examples/quickstart_research_agent.py
SIGNALFORGEAI_MODEL_ID=ollama:ministral-3:8b python examples/quickstart_research_agent.py
```

Do not require provider-backed paths in public quickstarts unless the page is
specifically about OpenAI, Ollama, or Hugging Face setup.

## Next release gate

For a complete offline evidence bundle, run:

```bash
signalforgeai-release-candidate-check \
  --work-dir results/release_candidate
```

This emits `release_candidate.v0` evidence plus child artifacts for pilot
readiness, mock training evidence, distillation evaluation, benchmark matrix,
specialist exchange validation, package checks, smoke checks, and registry
indexing.
