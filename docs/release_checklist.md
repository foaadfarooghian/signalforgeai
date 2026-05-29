# v0.7.0 Release Checklist

This checklist is the release gate for `signalforgeai==0.7.0`. It keeps public
onboarding offline-safe while adding a manual Linux/HF evidence gate for real
SFT -> DPO lineage.

## Local release checks

Run from a clean checkout:

```bash
uv run --extra dev python -m pytest -q
uv run --extra dev ruff check .
uv run --extra dev mypy src
uv run --extra dev signalforgeai-pilot-check --mode dummy --work-dir /tmp/signalforgeai-pilot-check
uv run --extra dev signalforgeai-release-candidate-check --work-dir /tmp/signalforgeai-release-candidate
```

Build and smoke the wheel:

```bash
rm -rf dist
uv build
python -m venv /tmp/signalforgeai-wheel-smoke
/tmp/signalforgeai-wheel-smoke/bin/python -m pip install dist/signalforgeai-0.7.0-py3-none-any.whl
/tmp/signalforgeai-wheel-smoke/bin/python -c "import signalforgeai; assert signalforgeai.__version__ == '0.7.0'"
/tmp/signalforgeai-wheel-smoke/bin/signalforgeai-pilot-check --mode dummy --work-dir /tmp/signalforgeai-wheel-pilot
```

## Manual Linux DPO evidence gate

Run in a Linux environment with the `[train]` extra installed and enough local
HF/GPU capacity for the tiny smoke target:

```bash
python -m venv /tmp/signalforgeai-train-smoke
source /tmp/signalforgeai-train-smoke/bin/activate
python -m pip install 'dist/signalforgeai-0.7.0-py3-none-any.whl[train]'

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

Acceptance checks:

```bash
python - <<'PY'
import json
from pathlib import Path

root = Path("results/v0_7_real_dpo")
release = json.loads((root / "release_candidate.json").read_text())
sft = json.loads((root / "training" / "sft_training_run.json").read_text())
dpo = json.loads((root / "training" / "dpo_training_run.json").read_text())

assert release["ok"] is True
assert release["training_evidence"]["final_stage"] == "dpo"
assert release["training_evidence"]["real_training_evidence"] is True
assert release["training_evidence"]["parent_real_training_evidence"] is True
assert release["training_evidence"]["adapter_smoke"]["ok"] is True
assert sft["ok"] is True and sft["training_stage"] == "sft"
assert dpo["ok"] is True and dpo["training_stage"] == "dpo"
assert dpo["dpo_parent_run"]["training_stage"] == "sft"
assert dpo["final_adapter_refs"]
assert dpo["file_checksums"]
print("v0.7 DPO evidence OK")
PY
```

## Publishing checks

Verify package metadata before publishing:

```bash
python -m twine check dist/*
```

After TestPyPI:

```bash
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple signalforgeai==0.7.0
python -c "import signalforgeai; assert signalforgeai.__version__ == '0.7.0'"
signalforgeai-pilot-check --mode dummy --work-dir /tmp/signalforgeai-testpypi-smoke
```

After PyPI:

```bash
pip install signalforgeai==0.7.0
python -c "import signalforgeai; assert signalforgeai.__version__ == '0.7.0'"
signalforgeai-pilot-check --mode dummy --work-dir /tmp/signalforgeai-pypi-smoke
```

## Release notes

Mention:

- Offline quickstarts still default to deterministic `dummy_good`.
- v0.7 proves real SFT -> DPO evidence and parent lineage, not quality gains.
- `[train]` and actual SFT/DPO execution remain Linux-only and experimental.
- Hard specialist quality thresholds, MCP runtime hardening, and critique/rubric
  datasets remain follow-up roadmap work.
