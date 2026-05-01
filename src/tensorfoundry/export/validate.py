"""Dataset validators for TensorFoundry export artifacts."""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


@dataclass(frozen=True)
class DatasetValidationResult:
    """Validation summary for a dataset JSONL artifact."""
    path: str
    kind: str
    rows: int
    issues: List[str]

    @property
    def ok(self) -> bool:
        return not self.issues and self.rows > 0

    def to_dict(self) -> Dict[str, Any]:
        return {"path": self.path, "kind": self.kind, "rows": self.rows, "ok": self.ok, "issues": self.issues}


def _iter_jsonl(path: Path) -> Iterable[tuple[int, Dict[str, Any]]]:
    for idx, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        obj = json.loads(line)
        if not isinstance(obj, dict):
            raise ValueError(f"line {idx}: row must be an object")
        yield idx, obj


def _detect_kind(row: Dict[str, Any], fallback: str) -> str:
    version = str(row.get("version") or "")
    if version.startswith("sft."):
        return "sft"
    if version.startswith("prefs."):
        return "prefs"
    if version.startswith("dpo."):
        return "dpo"
    if version.startswith("repairs."):
        return "repairs"
    if version.startswith("curriculum."):
        return "curriculum"
    return fallback


def _require(row: Dict[str, Any], keys: tuple[str, ...]) -> List[str]:
    return [f"missing key: {k}" for k in keys if k not in row]


def _validate_row(kind: str, row: Dict[str, Any]) -> List[str]:
    issues: List[str] = []
    if kind == "sft":
        issues.extend(_require(row, ("version", "instruction", "prompt", "response", "meta")))
        if not isinstance(row.get("prompt"), str) or not row.get("prompt", "").strip():
            issues.append("prompt must be a non-empty string")
        if not isinstance(row.get("response"), str) or not row.get("response", "").strip():
            issues.append("response must be a non-empty string")
    elif kind == "prefs":
        issues.extend(_require(row, ("version", "prompt", "response_a", "response_b", "preferred", "meta")))
        if row.get("preferred") not in {"a", "b"}:
            issues.append("preferred must be 'a' or 'b'")
    elif kind == "dpo":
        issues.extend(_require(row, ("version", "prompt", "chosen", "rejected", "meta")))
        if not isinstance(row.get("chosen"), str) or not row.get("chosen", "").strip():
            issues.append("chosen must be a non-empty string")
        if not isinstance(row.get("rejected"), str) or not row.get("rejected", "").strip():
            issues.append("rejected must be a non-empty string")
    elif kind == "repairs":
        issues.extend(_require(row, ("version", "prompt", "failed", "repaired", "meta")))
    elif kind == "curriculum":
        issues.extend(
            _require(
                row,
                ("version", "trace_id", "suite_id", "case_id", "model_id", "bucket", "difficulty", "prompt", "response"),
            )
        )
        if row.get("bucket") not in {"easy", "repair", "escalation"}:
            issues.append("bucket must be one of easy, repair, escalation")
    else:
        issues.append(f"unknown dataset kind: {kind}")

    meta = row.get("meta")
    if "meta" in row and not isinstance(meta, dict):
        issues.append("meta must be an object")
    return issues


def validate_dataset_jsonl(path: str | Path, *, kind: str = "auto") -> DatasetValidationResult:
    """Validate a TensorFoundry dataset export JSONL file."""
    path = Path(path)
    issues: List[str] = []
    rows = 0
    resolved_kind = kind
    if not path.exists():
        return DatasetValidationResult(str(path), kind, 0, [f"file not found: {path}"])
    try:
        for line_no, row in _iter_jsonl(path):
            rows += 1
            if resolved_kind == "auto":
                resolved_kind = _detect_kind(row, "unknown")
            for issue in _validate_row(resolved_kind, row):
                issues.append(f"line {line_no}: {issue}")
    except Exception as exc:
        issues.append(str(exc))
    if rows == 0:
        issues.append("dataset has no rows")
    return DatasetValidationResult(str(path), resolved_kind, rows, issues)


def write_dataset_manifest(
    results: List[DatasetValidationResult],
    out_path: str | Path,
) -> Path:
    """Write a manifest for dataset validation outputs."""
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": "dataset_manifest.v0",
        "ok": all(r.ok for r in results),
        "datasets": [r.to_dict() for r in results],
    }
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return out


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Validate TensorFoundry dataset JSONL exports.")
    parser.add_argument("path", type=str, help="Dataset JSONL path")
    parser.add_argument("--kind", type=str, default="auto", choices=["auto", "sft", "prefs", "dpo", "repairs", "curriculum"])
    args = parser.parse_args(argv)
    result = validate_dataset_jsonl(args.path, kind=args.kind)
    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
