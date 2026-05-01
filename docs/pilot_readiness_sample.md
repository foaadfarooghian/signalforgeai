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

## Regression Gate

- OK: `true`
- Report: `results/pilot_check/eval_regression.md`
- Pass-rate delta: `+0.00%`
- Mean-score delta: `+0.0000`
- Regressed cases: `0`
- New failing cases: `0`
- Worse failure modes: `0`
