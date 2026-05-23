"""Runner for the DecisionAgent example."""
from __future__ import annotations

import sys
from pathlib import Path

# Allow running the example without installing the package
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.append(str(SRC_PATH))


def main() -> None:
    from signalforgeai.agents.decision_agent import DecisionAgent
    from signalforgeai.logging.emitter import JsonlEmitter

    logs_dir = PROJECT_ROOT / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    emitter = JsonlEmitter(
        logs_dir / "temp.jsonl",
        agent_name="decision_agent",
        agent_version="0.4.0",
        default_stage="system",
    )

    trace_id = emitter.start_trace()
    emitter.file_path = logs_dir / f"{trace_id}.jsonl"

    with emitter:
        agent = DecisionAgent(emitter=emitter)
        result = agent.run(
            "What should SignalForge AI build next after logging + validation?",
            constraints=[
                "Hardware not ready for training yet",
                "Prefer OSS artefacts that build credibility",
            ],
            options=[
                "Build evaluation harness v1",
                "Build trace diff tool",
                "Build a codebase refactor agent",
            ],
        )

    memo = result["memo"]
    print("Trace:", result["trace_id"])
    print("\nRecommendation:", memo["recommendation"])
    print("\nNext steps:")
    for s in memo["next_steps"]:
        print("-", s)

    print("\nLog file:", logs_dir / f"{trace_id}.jsonl")
    print("\nValidate with:")
    print(f"  python -m signalforgeai.logging.validate logs/{trace_id}.jsonl")
    print("\nInspect with:")
    print(f"  python -m signalforgeai.logging.inspect logs/{trace_id}.jsonl")


if __name__ == "__main__":
    main()
