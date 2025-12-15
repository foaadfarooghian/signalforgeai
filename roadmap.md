# Roadmap

## Milestone 0: Foundations
- Repository scaffolding, packaging (`pyproject.toml`), lint/test harness
- Branch protections for `prod`, CI gating for `exp`, optional `dev` for daily work
- Basic observability hooks (logging interfaces, structured events)

## Milestone 1: Agent templates
- Curated library of opinionated agent templates (researcher, planner, evaluator)
- Template registry with metadata (tools, capabilities, dependencies)
- Examples and docs for wiring templates to user stacks (APIs, tools, MCP servers)

## Milestone 2: Orchestration
- Multi-agent workflow builder with shared state and memory abstraction
- Provider-flexible LLM adapters (OpenAI, Anthropic, Gemini, local)
- Guardrails and permissions for tool use; deterministic replay hooks

## Milestone 3: Evaluation and QA
- Offline and online evaluation harnesses with pluggable metrics
- Golden tasks and regression suites for core templates
- CI-integrated eval gates before promoting from `exp` to `prod`

## Milestone 4: Operations
- Observability pipeline (traces, logs, metrics) with dashboards
- Rollout and rollback strategy for workflows; versioned artifacts
- Deployment playbooks and runbooks for SRE/ops
