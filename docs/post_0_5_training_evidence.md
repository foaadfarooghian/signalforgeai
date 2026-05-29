# Post-0.5 Real Training Evidence Scope

Real SFT/DPO training evidence remains visible after `v0.5.0`, but it is not
part of the `v0.5.0` onboarding acceptance gate. The mandatory public path stays
offline and deterministic.

`v0.6.0` took the SFT portion of this scope as a manual Linux/HF release gate.
`v0.7.0` takes the DPO evidence portion as a manual Linux/HF release gate.
See `v0_7_real_dpo_evidence.md` for the current command and acceptance checks.
Quality-improvement thresholds remain post-v0.7 scope.

## Intended follow-up path

Release environments with Linux and `[train]` installed can opt into real SFT
evidence through the release-candidate gate:

```bash
signalforgeai-release-candidate-check \
  --work-dir results/release_candidate_real \
  --run-training \
  --training-base-model hf/org/base \
  --training-max-steps 1 \
  --require-real-training-evidence
```

Add DPO only after SFT evidence succeeds:

```bash
signalforgeai-release-candidate-check \
  --work-dir results/release_candidate_real \
  --run-training \
  --run-dpo \
  --training-base-model hf/org/base \
  --training-max-steps 1 \
  --require-real-training-evidence
```

## Constraints

- Actual SFT/DPO execution and `[train]` dependencies are Linux-only.
- HF model loading requires the model id, local cache, adapter path, and hardware
  profile to be valid for the release environment.
- Provider/HF failures should fail the opt-in evidence gate clearly.
- Non-Linux environments should use dry-run training preflight and offline
  release-candidate evidence.

## Evidence flow

When `--run-training` succeeds, the gate records non-mock `training_run.v0`
evidence. If no `--candidate-model-id` is supplied, the candidate model id is
derived as:

```text
hf:<base>?adapter=<final-adapter-dir>
```

That candidate then flows into:

- `distillation_eval.v0` comparison evidence
- `benchmark_matrix.v0` frontier evidence
- `specialist_model_unit.v0` exchange manifest
- `specialist_package.v0` package evidence
- `specialist_smoke.v0` consumer smoke evidence
- `specialist_registry_index.v0` file registry evidence

## Boundary for v0.5.0

`v0.5.0` documents this path but does not make real training a required release
gate. The next implementation milestone should decide the minimum Linux/HF
environment, runtime budget, and pass/fail thresholds for non-mock specialist
candidate evidence.
