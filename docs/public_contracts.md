# Public Contracts

SignalForge AI is pre-1.0 software. These contracts describe the public surface
that `v0.6.0` exposes for public pilots before any deeper behavior expansion.

## Stable enough for public pilots

- Distribution package: `signalforgeai`
- Import namespace: `signalforgeai`
- Supported Python versions: Python 3.11 or newer
- License: Apache-2.0
- Default offline model path: `dummy_good`
- Environment prefix: `SIGNALFORGEAI_*`
- CLI prefix: `signalforgeai-*`

The default public workflow must remain offline-safe. Examples, quickstarts, and
pilot checks should use deterministic dummy providers unless the user explicitly
sets a hosted or local provider.

## Installed CLIs

The package installs these console commands:

- `signalforgeai-pilot-check`
- `signalforgeai-release-candidate-check`
- `signalforgeai-learn`
- `signalforgeai-exchange`
- `signalforgeai-otel-export`
- `signalforgeai-dataset-validate`
- `signalforgeai-pricing-validate`
- `signalforgeai-report-tradeoffs`
- `signalforgeai-distill-check`
- `signalforgeai-benchmark-matrix`

CLI names are public entry points for `0.x` users. Arguments and artifact
details may still change before v1.0, but changes should be documented in the
changelog and release notes.

## Provider configuration

Provider-backed runs are opt-in:

- `SIGNALFORGEAI_MODEL_ID` selects a concrete model such as
  `openai:gpt-5-mini`, `ollama:ministral-3:8b`, or an HF model id.
- `SIGNALFORGEAI_PROVIDER=dummy` forces deterministic offline routing.
- `SIGNALFORGEAI_HOSTED_MODEL_ID` controls hosted provider smoke checks.
- `SIGNALFORGEAI_LOCAL_MODEL_ID` controls local provider smoke checks.
- `SIGNALFORGEAI_OPENAI_PRICING_PATH` overrides the bundled OpenAI pricing
  config used by pricing validation.

Docs and examples should not require OpenAI, Ollama, or Hugging Face credentials
unless the page is explicitly about that provider path.

See `docs/provider_setup.md` for provider-specific setup commands and smoke
checks.

## Artifact families

Current public artifacts use additive `*.v0` names. They are suitable for pilots
and review gates, but are not yet v1-stable schemas:

- `trace.v0`
- `reward.v0`
- `eval_regression.v0`
- `training_preflight.v0`
- `training_run.v0`
- `adapter_smoke.v0` nested inside real `training_run.v0` evidence
- `distillation_recipe.v0`
- `distillation_eval.v0`
- `benchmark_matrix.v0`
- `specialist_model_unit.v0`
- `specialist_package.v0`
- `specialist_smoke.v0`
- `specialist_registry_index.v0`
- `release_candidate.v0`

Schema changes should remain additive where practical during `0.x`. Breaking
artifact changes should be called out directly in release notes.

## Experimental surfaces

These areas are intentionally experimental in `v0.6.0` and should be treated as
pre-v1 work:

- Real SFT/DPO training execution
- `[train]` optional dependency stack
- Specialist exchange packaging beyond local file-backed evidence
- Bandit routing policy tuning and learning-state persistence
- Provider-backed benchmark comparability across hosted/local/HF models

Training execution and the `[train]` extra are Linux-only for `v0.6.0`.
Non-Linux environments can still run training preflight and offline release
candidate checks.

## Version posture

`signalforgeai==0.6.0` is the real SFT evidence release. Users should pin the
package version for pilots, and maintainers should keep public docs clear about
whether an interface is stable enough for pilots or experimental before v1.0.
