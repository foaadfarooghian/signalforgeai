from tensorfoundry.evaluation.reward_schema import RewardV0
from tensorfoundry.evaluation.reward_writer import write_rewards_jsonl
from tensorfoundry.logging.reward_validator import validate_reward_jsonl

def test_write_and_validate(tmp_path):
    p = tmp_path / "reward.jsonl"
    rewards = [
        RewardV0(
            version="reward.v0",
            trace_id="t1",
            run_id="r1",
            suite_id="s1",
            case_id="task1",
            agent_id="decision",
            model_id="m1",
            commit_sha="abc",
            success=False,
            overall_score=0.2,
            subscores={},
            violations=["failed_expectation"],
        )
    ]
    write_rewards_jsonl(p, rewards)
    validate_reward_jsonl(p)