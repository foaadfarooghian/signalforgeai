from tensorfoundry.evaluation.reward_schema import RewardV0
import json

def test_reward_jsonl_roundtrip():
    r = RewardV0(
        version="reward.v0",
        trace_id="t1",
        run_id="r1",
        suite_id="s1",
        task_id="task1",
        agent_id="refactor",
        model_id="gpt-x",
        commit_sha="abc123",
        success=True,
        overall_score=0.9,
        subscores={"format": 1.0},
        violations=[],
    )
    obj = json.loads(r.to_jsonl())
    assert obj["version"] == "reward.v0"
    assert 0.0 <= obj["overall_score"] <= 1.0