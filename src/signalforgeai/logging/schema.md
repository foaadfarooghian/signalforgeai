# SignalForge AI Logging Schema

This document defines the **canonical execution trace schema** for SignalForge AI agents.

The purpose of this schema is to ensure that:
- Agent behaviour is observable and debuggable
- Outcomes are measurable and reproducible
- Execution traces can be converted into learning signals later

This schema is **model-agnostic**, **framework-agnostic**, and **provider-agnostic**.

---

## Design principles

1. **Logs are first-class**
   - Logging is not an afterthought or debug aid
   - Every agent run must emit structured logs

2. **Execution over text**
   - We log decisions, actions, and outcomes — not just prompts and responses

3. **Append-only**
   - Logs are immutable event streams
   - State is reconstructed, not overwritten

4. **Human-readable, machine-parseable**
   - JSONL format
   - Clear field names
   - No binary blobs

5. **Learning-ready**
   - Logs should be convertible into:
     - (state → action) pairs
     - preference comparisons
     - curriculum signals

---

## Log format

- **Format:** JSON Lines (`.jsonl`)
- **One event per line**
- **Ordered by timestamp**

Each line represents a **single event** in an agent run.

---

## Top-level fields (required)

```json
{
  "schema_version": "trace.v0",
  "trace_id": "uuid",
  "span_id": "uuid",
  "parent_span_id": "uuid | null",
  "timestamp": "ISO-8601",
  "agent": {
    "name": "string",
    "version": "string"
  },
  "stage": "planner | executor | critic | tool | system",
  "event_type": "string",
  "payload": {},
  "metrics": {},
  "outcome": {}
}
```

`schema_version` is emitted for new traces. Legacy traces without this field
remain valid during the v0 compatibility window.

## Tool event payloads

New `trace.v0` tool events normalize payloads to include:

```json
{
  "tool_name": "string",
  "tool_input": {},
  "tool_output_summary": "string",
  "success": "boolean | null",
  "error": "string | object | null"
}
```

## OTel export

Trace JSONL files can be exported to OpenTelemetry-compatible JSON:

```bash
signalforgeai-otel-export logs/<run_id>/<trace_id>.jsonl
```
