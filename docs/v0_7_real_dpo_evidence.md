# v0.7.0 Real DPO Evidence

`v0.7.0` proves that SignalForge AI can produce auditable non-mock DPO evidence
after a real SFT parent run. Core CI stays offline and dummy-first; real
training remains a manual Linux/HF release gate.

## Standard smoke target

- Base model: `unsloth/tinyllama-chat-bnb-4bit`
- Platform: Linux
- Install: `pip install 'signalforgeai[train]==0.7.0'`
- Scope: SFT parent plus DPO final adapter evidence
- Evidence:
  - parent SFT `training_run.v0`
  - final DPO `training_run.v0`
  - DPO `dpo_parent_run` lineage
  - adapter refs, file checksums, and successful `adapter_smoke.v0`

The one-step DPO smoke is an evidence and lineage proof, not a model quality
claim. Distillation and benchmark reports still show quality, cost, and latency
movement, but hard quality-improvement thresholds remain follow-up work.

## Manual release evidence command

```bash
signalforgeai-release-candidate-check \
  --work-dir results/v0_7_real_dpo \
  --run-training \
  --run-dpo \
  --final-training-stage dpo \
  --candidate-model-id dummy_good \
  --training-base-model unsloth/tinyllama-chat-bnb-4bit \
  --training-max-steps 1 \
  --require-real-training-evidence
```

The explicit `--candidate-model-id dummy_good` keeps downstream distillation and
benchmark gates deterministic while the training gate proves the real SFT -> DPO
adapter evidence path.

## Acceptance checks

The release evidence bundle must satisfy:

- `release_candidate.json` has `ok == true`
- `training_evidence.final_stage == "dpo"`
- `training_evidence.real_training_evidence == true`
- `training_evidence.parent_real_training_evidence == true`
- `training_evidence.adapter_smoke.ok == true`
- `training_evidence.dpo_parent_run.training_stage == "sft"`
- `training/sft_training_run.json` has `ok == true`
- `training/sft_training_run.json` has `training_stage == "sft"`
- `training/dpo_training_run.json` has `ok == true`
- `training/dpo_training_run.json` has `training_stage == "dpo"`
- `training/dpo_training_run.json` has non-empty `final_adapter_refs`
- `training/dpo_training_run.json` has non-empty `file_checksums`

## Non-goals

- Adapter quality improvement is not required for v0.7.0.
- GPU/self-hosted CI is not required for v0.7.0.
- Offline examples and default release-candidate checks must not require hosted,
  local, or HF providers.
- MCP runtime hardening and critique/rubric dataset exports remain separate
  roadmap work.
