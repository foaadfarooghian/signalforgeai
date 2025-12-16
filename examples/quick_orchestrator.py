"""Runner demonstrating the Planner→Executor→Critic (PEC) orchestrator."""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running the example without installing the package
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.append(str(SRC_PATH))

from tensorfoundry.agents.research_agent import ResearchAgent
from tensorfoundry.logging.emitter import JsonlEmitter
from tensorfoundry.orchestration.adapters import ResearchExecutor, ResearchPlanner, SimpleCritic
from tensorfoundry.orchestration.pec import PECOrchestrator


def main() -> None:
    logs_dir = PROJECT_ROOT / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    # Create emitter with placeholder file, then set per-trace path
    emitter = JsonlEmitter(
        logs_dir / "temp.jsonl",
        agent_name="orchestrator",
        agent_version="0.1.0",
        default_stage="system",
    )

    trace_id = emitter.start_trace()
    emitter.file_path = logs_dir / f"{trace_id}.jsonl"

    with emitter:
        # The agent emits tool/model events; the orchestrator emits system spine events.
        agent = ResearchAgent(emitter=emitter)

        planner = ResearchPlanner(agent=agent)
        executor = ResearchExecutor(agent=agent)
        critic = SimpleCritic(min_len=10)

        orch = PECOrchestrator(
            planner=planner,
            executor=executor,
            critic=critic,
            emitter=emitter,
            max_retries=2,
        )

        state = orch.run("state of the art in agentic workflows")

    print("Trace:", trace_id)
    print("Result:", state.get("result"))
    print("Attempts:", state.get("attempt"))
    if state.get("constraints"):
        print("Constraints:", state["constraints"])
    if state.get("errors"):
        print("Errors:", state["errors"])

    print("\nLog file:", logs_dir / f"{trace_id}.jsonl")
    print("\nValidate with:")
    print(f"  python -m tensorfoundry.logging.validate logs/{trace_id}.jsonl")


if __name__ == "__main__":
    main()