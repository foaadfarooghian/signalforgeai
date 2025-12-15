# Design principles

1. **Outcomes over text**: evaluate agents by task success, cost, and latency.
2. **State is first-class**: every agent run has explicit state and transitions.
3. **Logs are the dataset**: execution traces must be structured and replayable.
4. **Tooling realism**: agents should use tools the way production systems do.
5. **Composable interfaces**: swap agents/patterns/models without rewiring code.
