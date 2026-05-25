# Provider Setup

SignalForge AI defaults to deterministic offline runs. Provider-backed execution
is opt-in and should be configured only after the offline pilot path is working.

## Offline dummy provider

The default public path uses `dummy_good` and does not require network access:

```bash
signalforgeai-pilot-check --mode dummy --work-dir results/pilot_check
```

To force dummy routing in a shell:

```bash
export SIGNALFORGEAI_PROVIDER=dummy
export SIGNALFORGEAI_MODEL_ID=dummy_good
```

## OpenAI hosted provider

Set an API key and choose an explicit hosted model:

```bash
export OPENAI_API_KEY=...
export SIGNALFORGEAI_MODEL_ID=openai:gpt-5-mini
```

Run a hosted provider smoke check only in configured environments:

```bash
SIGNALFORGEAI_HOSTED_MODEL_ID=openai:gpt-5-mini \
  signalforgeai-pilot-check --require-provider hosted --work-dir results/pilot_hosted
```

OpenAI pricing validation uses the bundled pricing config by default. Override it
only when validating a local pricing file:

```bash
SIGNALFORGEAI_OPENAI_PRICING_PATH=path/to/openai_pricing.yaml \
  signalforgeai-pricing-validate --require gpt-5-mini
```

## Ollama local provider

Start Ollama locally, pull the model, and choose an explicit local model id:

```bash
ollama pull ministral-3:8b
export SIGNALFORGEAI_MODEL_ID=ollama:ministral-3:8b
```

Run a local provider smoke check only when Ollama is reachable:

```bash
SIGNALFORGEAI_LOCAL_MODEL_ID=ollama:ministral-3:8b \
  signalforgeai-pilot-check --require-provider local --work-dir results/pilot_local
```

## Hugging Face provider

HF model ids use the `hf:` prefix. Adapter-backed ids can point at a local
adapter path:

```bash
export SIGNALFORGEAI_MODEL_ID=hf:org/base-model
export SIGNALFORGEAI_MODEL_ID="hf:org/base-model?adapter=results/training/sft_lora"
```

HF execution is experimental and may require the Linux-only `[train]` stack for
local model loading. Use offline training preflight on other platforms.

## Public quickstart rule

Public quickstarts must not require provider credentials. Keep `dummy_good` as
the default and document OpenAI, Ollama, and HF paths as explicit opt-in setup.
