"""Runner for the RefactorAgent example."""
from __future__ import annotations

import sys
from pathlib import Path

# Allow running the example without installing the package
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.append(str(SRC_PATH))

from tensorfoundry.agents.refactor_agent import RefactorAgent
from tensorfoundry.logging.emitter import JsonlEmitter


def main() -> None:
    logs_dir = PROJECT_ROOT / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    emitter = JsonlEmitter(
        logs_dir / "temp.jsonl",
        agent_name="refactor_agent",
        agent_version="0.1.0",
        default_stage="system",
    )

    trace_id = emitter.start_trace()
    emitter.file_path = logs_dir / f"{trace_id}.jsonl"

    # Safe default: dry-run only.
    with emitter:
        agent = RefactorAgent(emitter=emitter)
        result = agent.run(
            "Replace a placeholder string in a Python file (dry-run).",
            repo_root=PROJECT_ROOT,
            # Provide a file to make it deterministic for demo:
            target_file="src/tensorfoundry/agents/research_agent.py",
            find="Summarise findings",
            replace="Summarize findings",
            dry_run=True,
        )

    print("Trace:", result["trace_id"])
    print("Target file:", result["patch"].file)
    print("Patch result:", result["patch_result"])
    print("\nLog file:", logs_dir / f"{trace_id}.jsonl")
    print("\nValidate with:")
    print(f"  python -m tensorfoundry.logging.validate logs/{trace_id}.jsonl")
    print("\nInspect with:")
    print(f"  python -m tensorfoundry.logging.inspect logs/{trace_id}.jsonl")


if __name__ == "__main__":
    main()