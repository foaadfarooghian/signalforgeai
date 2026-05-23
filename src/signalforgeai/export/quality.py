"""Dataset quality helpers for split, hashing, and provenance checks."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


SPLIT_POLICY_VERSION = "case_hash_split.v0"
DEFAULT_SPLIT_POLICY: Dict[str, Any] = {
    "version": SPLIT_POLICY_VERSION,
    "train": 0.8,
    "validation": 0.1,
    "test": 0.1,
}
VALID_SPLITS = {"train", "validation", "test"}


def split_key_for_case(suite_id: Any, case_id: Any) -> str:
    """Return the stable split key for a suite/case identity."""
    return f"{str(suite_id or 'unknown')}:{str(case_id or 'unknown')}"


def split_for_key(split_key: str, policy: Optional[Dict[str, Any]] = None) -> str:
    """Assign a deterministic split from a stable key."""
    policy = policy or DEFAULT_SPLIT_POLICY
    digest = hashlib.sha256(split_key.encode("utf-8")).hexdigest()
    bucket = int(digest[:8], 16) / 0xFFFFFFFF
    train = float(policy.get("train", DEFAULT_SPLIT_POLICY["train"]))
    validation = float(policy.get("validation", DEFAULT_SPLIT_POLICY["validation"]))
    if bucket < train:
        return "train"
    if bucket < train + validation:
        return "validation"
    return "test"


def split_meta(suite_id: Any, case_id: Any) -> Dict[str, Any]:
    """Return additive split metadata for a dataset row."""
    key = split_key_for_case(suite_id, case_id)
    return {
        "split": split_for_key(key),
        "split_key": key,
        "split_policy": dict(DEFAULT_SPLIT_POLICY),
    }


def attach_split_meta(meta: Dict[str, Any], *, suite_id: Any, case_id: Any) -> None:
    """Mutate a metadata object with deterministic split fields."""
    meta.update(split_meta(suite_id, case_id))


def content_sha256(path: Path) -> str:
    """Hash full dataset file bytes."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def row_payload_hash(kind: str, row: Dict[str, Any]) -> str:
    """Hash only the training payload for duplicate detection."""
    payload = _training_payload(kind, row)
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def provenance_inputs(row: Dict[str, Any]) -> Dict[str, str]:
    """Extract normalized provenance input references from a row."""
    meta = row.get("meta")
    if not isinstance(meta, dict):
        return {}
    provenance = meta.get("provenance")
    if not isinstance(provenance, dict):
        return {}
    inputs = provenance.get("inputs")
    if not isinstance(inputs, dict):
        return {}
    return {str(k): str(v) for k, v in inputs.items() if v is not None}


def required_provenance_keys(kind: str) -> List[str]:
    """Return provenance input keys required by strict validation."""
    if kind in {"sft", "curriculum"}:
        return ["trace_path", "reward_jsonl"]
    if kind in {"prefs", "dpo"}:
        return ["trace_a_path", "trace_b_path", "reward_a_jsonl", "reward_b_jsonl"]
    if kind == "repairs":
        return [
            "trace_success_path",
            "trace_failure_path",
            "reward_success_jsonl",
            "reward_failure_jsonl",
        ]
    return []


def resolve_artifact_ref(value: str, *, logs_root: Optional[Path]) -> Path:
    """Resolve an artifact reference against common runtime roots."""
    path = Path(value)
    if path.is_absolute() or logs_root is None:
        return path
    candidates = [path, logs_root / path, logs_root.parent / path]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _training_payload(kind: str, row: Dict[str, Any]) -> Dict[str, Any]:
    if kind == "sft":
        return {"prompt": _norm(row.get("prompt")), "response": _norm(row.get("response"))}
    if kind == "prefs":
        return {
            "prompt": _norm(row.get("prompt")),
            "response_a": _norm(row.get("response_a")),
            "response_b": _norm(row.get("response_b")),
            "preferred": row.get("preferred"),
        }
    if kind == "dpo":
        return {
            "prompt": _norm(row.get("prompt")),
            "chosen": _norm(row.get("chosen")),
            "rejected": _norm(row.get("rejected")),
        }
    if kind == "repairs":
        return {
            "prompt": _norm(row.get("prompt")),
            "failed": _norm(row.get("failed")),
            "repaired": _norm(row.get("repaired")),
        }
    if kind == "curriculum":
        return {
            "prompt": _norm(row.get("prompt")),
            "response": _norm(row.get("response")),
            "bucket": row.get("bucket"),
            "difficulty": row.get("difficulty"),
        }
    return dict(row)


def _norm(value: Any) -> str:
    return str(value or "").strip()
