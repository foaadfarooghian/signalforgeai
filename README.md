# TensorFoundry

TensorFoundry is an opinionated platform for composing agentic workflows instead of generic chatbots. It helps teams:

- Discover pre-built agent templates
- Compose multi-agent workflows
- Plug agents into their own stack (APIs, tools, MCP servers, storage, evaluation)
- Operate across multiple LLM providers (OpenAI, Anthropic, Gemini, local)

## What makes it different

- Optimized for execution: tools, state, memory, and evaluation are first-class
- Built for multi-agent orchestration over ad-hoc prompts
- Provider-flexible: swap or mix LLM backends without rewriting workflows

## Repository layout

- `src/tensorfoundry`: core library and modules (agents, orchestration, evaluation, logging)
- `examples`: runnable samples, including a quickstart research agent
- `docs`: design notes and principles
- `manifesto.md`: product philosophy
- `roadmap.md`: near-term delivery plan

## Branching and CI/CD (baseline proposal)

- `prod`: protected, deployable branch for production releases
- `exp`: integration branch for experimental features gated by CI
- `dev` (optional): shared collaboration branch for day-to-day work
- Feature work: short-lived branches -> PRs into `exp` (or `dev`), promoted to `prod` after checks
- CI: lint/test on every PR; promotion jobs handle versioning, artifacts, and deploys

See `roadmap.md` for the initial milestones.
