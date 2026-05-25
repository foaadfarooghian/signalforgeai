# v0.5.0 Release Checklist

This checklist is the release gate for `signalforgeai==0.5.0`. It is a
docs/gates-only release: no public API, CLI name, schema, artifact contract, or
provider default changes are included.

## Local release gate

Run from a clean release branch before merging:

```bash
uv lock --check
uv run --extra dev ruff check .
uv run --extra dev mypy src
uv run --extra dev pytest -q
```

Run public offline flows:

```bash
for f in examples/quick*.py; do uv run --extra dev python "$f"; done
uv run --extra dev signalforgeai-pilot-check --mode dummy --work-dir /tmp/signalforgeai-pilot-check
uv run --extra dev signalforgeai-release-candidate-check --work-dir /tmp/signalforgeai-release-candidate
```

Build and smoke the package:

```bash
python -m build
twine check dist/*
python -m venv /tmp/signalforgeai-wheel-smoke
/tmp/signalforgeai-wheel-smoke/bin/python -m pip install --upgrade pip
/tmp/signalforgeai-wheel-smoke/bin/python -m pip install dist/*.whl
/tmp/signalforgeai-wheel-smoke/bin/python -c "import signalforgeai; assert signalforgeai.__version__ == '0.5.0'"
/tmp/signalforgeai-wheel-smoke/bin/signalforgeai-pricing-validate --require gpt-5-mini
```

## Merge gate

- PR CI must pass on Python 3.11 and 3.12.
- Merge to `dev`, wait for `dev` CI to pass.
- Fast-forward or merge `dev` to `prod`, wait for `prod` CI to pass.
- Confirm `origin/dev` and `origin/prod` point to the same release commit.

## Publish gate

Before tagging:

```bash
curl -s -o /dev/null -w '%{http_code}\n' https://pypi.org/pypi/signalforgeai/0.5.0/json
curl -s -o /dev/null -w '%{http_code}\n' https://test.pypi.org/pypi/signalforgeai/0.5.0/json
```

Both commands should print `404` before publishing.

Tag and build:

```bash
git tag -a v0.5.0 -m "SignalForge AI 0.5.0"
git push origin v0.5.0
```

The tag push should build and smoke artifacts only.

Publish TestPyPI first:

```bash
gh workflow run Release --repo foaadfarooghian/signalforgeai --ref v0.5.0 -f target=testpypi
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple signalforgeai==0.5.0
python -c "import signalforgeai; assert signalforgeai.__version__ == '0.5.0'"
signalforgeai-pricing-validate --require gpt-5-mini
```

Publish PyPI after TestPyPI smoke passes:

```bash
gh workflow run Release --repo foaadfarooghian/signalforgeai --ref v0.5.0 -f target=pypi
pip install signalforgeai==0.5.0
python -c "import signalforgeai; assert signalforgeai.__version__ == '0.5.0'"
signalforgeai-pilot-check --mode dummy --work-dir /tmp/signalforgeai-pypi-smoke
```

## Release notes

The GitHub Release should include:

- Pre-1.0 active-development caveat.
- Install command: `pip install signalforgeai==0.5.0`.
- Public onboarding and contract-clarity highlights.
- Note that examples and quickstarts default to offline `dummy_good`.
- Note that `[train]` remains Linux-only and experimental.
- Links to successful release workflow runs.
