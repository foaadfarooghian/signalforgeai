from signalforgeai.evaluation.reward_schema import RewardV0
import json

def test_reward_jsonl_roundtrip():
    r = RewardV0(
        version="reward.v0",
        trace_id="t1",
        run_id="r1",
        suite_id="s1",
        case_id="task1",
        agent_id="refactor",
        model_id="gpt-x",
        commit_sha="abc123",
        success=True,
        overall_score=0.9,
        subscores={"format": 1.0},
        violations=[],
        failure_mode="expectation_failed",
        diagnosis={"notes": ["missing keyword"]},
        artifact_refs={"trace": "logs/t1.jsonl"},
    )
    obj = json.loads(r.to_jsonl())
    assert obj["version"] == "reward.v0"
    assert 0.0 <= obj["overall_score"] <= 1.0
    assert obj["failure_mode"] == "expectation_failed"
    assert obj["diagnosis"]["notes"] == ["missing keyword"]
    assert obj["artifact_refs"]["trace"] == "logs/t1.jsonl"
