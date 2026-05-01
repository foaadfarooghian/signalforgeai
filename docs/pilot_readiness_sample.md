# TensorFoundry Pilot Readiness

- OK: `true`
- Work dir: `results/pilot_check`
- Suites run: `2`
- Dataset manifest: `results/pilot_check/datasets/manifest.json`

## Providers

| provider | model | ok | required | skipped | reason |
|---|---|---:|---:|---:|---|
| dummy | `dummy_good` | true | true | false |  |
| openai | `openai:gpt-5-mini` | false | false | true | OPENAI_API_KEY is not set |
| ollama | `ollama:ministral-3:8b` | false | false | true | Ollama is not reachable |

## Datasets

| kind | rows | ok | path |
|---|---:|---:|---|
| sft | 8 | true | `results/pilot_check/datasets/pilot.sft.jsonl` |
| prefs | 4 | true | `results/pilot_check/datasets/pilot.prefs.jsonl` |
| repairs | 4 | true | `results/pilot_check/datasets/pilot.repairs.jsonl` |
| curriculum | 8 | true | `results/pilot_check/datasets/pilot.curriculum.jsonl` |

## Dataset Quality

| kind | sha256 | splits | duplicates | provenance | missing refs | quality issues |
|---|---|---|---:|---:|---:|---:|
| sft | `503de10ca51c` | test:4, train:4 | 0 | true | 0 | 0 |
| prefs | `52409c4fcf78` | test:2, train:2 | 0 | true | 0 | 0 |
| repairs | `3957d0299232` | test:2, train:2 | 0 | true | 0 | 0 |
| curriculum | `bbc60bce79cd` | test:4, train:4 | 0 | true | 0 | 0 |

## Regression Gate

- OK: `true`
- Report: `results/pilot_check/eval_regression.md`
- Pass-rate delta: `+0.00%`
- Mean-score delta: `+0.0000`
- Regressed cases: `0`
- New failing cases: `0`
- Worse failure modes: `0`
