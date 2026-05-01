# Design principles

1. **Runtime-native observability**: execution should map cleanly to OTel traces and events.
2. **MCP-first tool semantics**: tool calls should follow consistent request/response/error contracts.
3. **Outcomes over text**: evaluate agents by task success, cost, latency, and reliability.
4. **Failure analysis, not only scoring**: every failed run should produce actionable diagnosis.
5. **Logs are the dataset**: execution artifacts must be structured, versioned, and replayable.
6. **Deterministic data pipelines**: dataset generation from traces must be reproducible and auditable.
7. **Distill specialists, not generalists**: small models are trained for narrow, high-value execution tasks.
8. **Benchmark for trade-offs**: model and pattern choices must be justified by cost/latency/reliability evidence.
9. **Ship complete model units**: registry entries must include eval pack, lineage, hardware profile, failure modes, license/constraints, and runnable artifacts.
