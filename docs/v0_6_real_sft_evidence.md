# v0.6.0 Real SFT Evidence

`v0.6.0` proves that SignalForge AI can produce auditable non-mock SFT evidence
from the existing release-candidate chain. Core CI stays offline and dummy-first;
real training is a manual Linux/HF release gate.

## Standard smoke target

- Base model: `unsloth/tinyllama-chat-bnb-4bit`
- Platform: Linux
- Install: `pip install 'signalforgeai[train]==0.6.0'`
- Scope: SFT only
- Evidence: `training_run.v0` with adapter refs, file checksums, and successful
  `adapter_smoke.v0`

The one-step smoke is a load/generate proof, not a model quality claim. DPO and
quality-improvement thresholds are v0.7 scope.

## Manual release evidence command

```bash
signalforgeai-release-candidate-check \
  --work-dir results/v0_6_real_sft \
  --run-training \
  --final-training-stage sft \
  --candidate-model-id dummy_good \
  --training-base-model unsloth/tinyllama-chat-bnb-4bit \
  --training-max-steps 1 \
  --require-real-training-evidence
```

The explicit `--candidate-model-id dummy_good` keeps downstream distillation and
benchmark gates deterministic while the training gate proves the real adapter
artifact path.

## Acceptance checks

The release evidence bundle must satisfy:

- `release_candidate.json` has `ok == true`
- `training_evidence.real_training_evidence == true`
- `training_evidence.final_stage == "sft"`
- `training_evidence.adapter_smoke.ok == true`
- `training/sft_training_run.json` has `ok == true`
- `training/sft_training_run.json` has `training_stage == "sft"`
- `training/sft_training_run.json` has non-empty `artifact_refs.adapters`
- `training/sft_training_run.json` has non-empty `file_checksums`

## Non-goals

- Real DPO evidence is not required for v0.6.0.
- Adapter quality improvement is not required for v0.6.0.
- GPU/self-hosted CI is not required for v0.6.0.
- Offline examples and default release-candidate checks must not require hosted,
  local, or HF providers.
