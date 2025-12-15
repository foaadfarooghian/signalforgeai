"""Runner for the ResearchAgent example."""
from __future__ import annotations

import sys
from pathlib import Path

# Allow running the example without installing the package
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.append(str(SRC_PATH))

from tensorfoundry.agents import ResearchAgent
from tensorfoundry.logging import JsonlEmitter


def main() -> None:
    """Instantiate the research agent and run a single query."""
    logs_dir = PROJECT_ROOT / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    emitter = JsonlEmitter(
        logs_dir / "temp.jsonl",
        agent_name="research_agent",
        agent_version="0.1.0",
        default_stage="executor",
    )

    trace_id = emitter.start_trace()
    emitter.file_path = logs_dir / f"{trace_id}.jsonl"

    with emitter:
        agent = ResearchAgent(emitter=emitter)
        result = agent.run("state of the art in agentic workflows")

    print("Trace:", result["trace_id"])
    print("\nSummary:\n", result["summary"])
    print("\nJournal:")
    for entry in result["journal"]:
        print("-", entry)


if __name__ == "__main__":
    main()
