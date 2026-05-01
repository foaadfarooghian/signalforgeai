# Documentation

Concepts and design notes for TensorFoundry's narrowed scope: agent runtime,
evaluation, trace-to-learning, and benchmarking.

- `design_principles.md`: core principles for runtime, eval, data, and learning.
- `model_exchange.md`: specialist model registry/exchange contract and lifecycle.
- `pilot_readiness_sample.md`: example output from the production-pilot readiness check.
- `specs/specialist_model_unit.schema.json`: machine-readable manifest schema for exchange units.
- `../roadmap.md`: active workstreams and exit criteria.
- `../manifesto.md`: product philosophy and explicit non-goals.

## Pilot readiness

Run the offline deterministic readiness loop with:

```bash
tensorfoundry-pilot-check --mode dummy --work-dir results/pilot_check
```

The command produces a readiness report, validated trace/reward artifacts, and
validated SFT, preference, repair, and curriculum datasets. Hosted and local
provider checks are automated but optional unless explicitly required with
`--require-provider hosted`, `--require-provider local`, or
`--require-provider all`.
