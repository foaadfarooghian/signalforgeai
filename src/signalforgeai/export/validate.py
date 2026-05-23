"""Dataset validators for SignalForge AI export artifacts."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from signalforgeai.export.quality import (
    VALID_SPLITS,
    content_sha256,
    provenance_inputs,
    required_provenance_keys,
    resolve_artifact_ref,
    row_payload_hash,
)


@dataclass(frozen=True)
class DatasetValidationResult:
    """Validation summary for a dataset JSONL artifact."""
    path: str
    kind: str
    rows: int
    issues: List[str]
    content_sha256: Optional[str] = None
    split_counts: Dict[str, int] = field(default_factory=dict)
    duplicate_count: int = 0
    provenance_checked: bool = False
    missing_artifact_refs: List[str] = field(default_factory=list)
    quality_issues: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.issues and not self.quality_issues and self.rows > 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "path": self.path,
            "kind": self.kind,
            "rows": self.rows,
            "ok": self.ok,
            "issues": self.issues,
            "content_sha256": self.content_sha256,
            "split_counts": self.split_counts,
            "duplicate_count": self.duplicate_count,
            "provenance_checked": self.provenance_checked,
            "missing_artifact_refs": self.missing_artifact_refs,
            "quality_issues": self.quality_issues,
        }


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


def validate_dataset_jsonl(
    path: str | Path,
    *,
    kind: str = "auto",
    logs_root: str | Path | None = None,
    require_provenance: bool = False,
    require_splits: bool = False,
    allow_duplicates: bool = True,
    allow_leakage: bool = True,
) -> DatasetValidationResult:
    """Validate a SignalForge AI dataset export JSONL file."""
    path = Path(path)
    issues: List[str] = []
    quality_issues: List[str] = []
    missing_artifact_refs: List[str] = []
    split_counts: Dict[str, int] = {}
    rows = 0
    resolved_kind = kind
    if not path.exists():
        return DatasetValidationResult(str(path), kind, 0, [f"file not found: {path}"])
    file_hash = content_sha256(path)
    row_items: List[tuple[int, Dict[str, Any]]] = []
    try:
        for line_no, row in _iter_jsonl(path):
            rows += 1
            if resolved_kind == "auto":
                resolved_kind = _detect_kind(row, "unknown")
            row_items.append((line_no, row))
            for issue in _validate_row(resolved_kind, row):
                issues.append(f"line {line_no}: {issue}")
    except Exception as exc:
        issues.append(str(exc))
    if rows == 0:
        issues.append("dataset has no rows")

    quality_enabled = (
        require_provenance
        or require_splits
        or not allow_duplicates
        or not allow_leakage
    )
    duplicate_count = 0
    if quality_enabled and row_items:
        quality = _validate_quality(
            resolved_kind,
            row_items,
            logs_root=Path(logs_root) if logs_root is not None else None,
            require_provenance=require_provenance,
            require_splits=require_splits,
            allow_duplicates=allow_duplicates,
            allow_leakage=allow_leakage,
        )
        quality_issues = quality["quality_issues"]
        missing_artifact_refs = quality["missing_artifact_refs"]
        split_counts = quality["split_counts"]
        duplicate_count = int(quality["duplicate_count"])

    return DatasetValidationResult(
        str(path),
        resolved_kind,
        rows,
        issues,
        content_sha256=file_hash,
        split_counts=split_counts,
        duplicate_count=duplicate_count,
        provenance_checked=require_provenance,
        missing_artifact_refs=missing_artifact_refs,
        quality_issues=quality_issues,
    )


def _validate_quality(
    kind: str,
    rows: List[tuple[int, Dict[str, Any]]],
    *,
    logs_root: Optional[Path],
    require_provenance: bool,
    require_splits: bool,
    allow_duplicates: bool,
    allow_leakage: bool,
) -> Dict[str, Any]:
    quality_issues: List[str] = []
    missing_artifact_refs: List[str] = []
    split_counts: Counter[str] = Counter()
    hash_counts: Counter[str] = Counter()
    split_by_key: Dict[str, str] = {}
    split_by_prompt: Dict[str, str] = {}

    for line_no, row in rows:
        row_hash = row_payload_hash(kind, row)
        hash_counts[row_hash] += 1

        meta = _dict_field(row, "meta")
        split = str(meta.get("split") or "")
        split_key = str(meta.get("split_key") or "")
        prompt = _row_prompt(row)

        if split:
            split_counts[split] += 1
        if require_splits:
            if split not in VALID_SPLITS:
                quality_issues.append(f"line {line_no}: missing or invalid meta.split")
            if not split_key:
                quality_issues.append(f"line {line_no}: missing meta.split_key")
            if not isinstance(meta.get("split_policy"), dict):
                quality_issues.append(f"line {line_no}: missing meta.split_policy")

        if not allow_leakage and split in VALID_SPLITS:
            if split_key:
                _record_leakage(
                    split_by_key,
                    split_key,
                    split,
                    f"line {line_no}: split_key {split_key!r}",
                    quality_issues,
                )
            if prompt:
                _record_leakage(
                    split_by_prompt,
                    prompt,
                    split,
                    f"line {line_no}: prompt",
                    quality_issues,
                )

        if require_provenance:
            inputs = provenance_inputs(row)
            selected_keys = _select_provenance_keys(kind, inputs)
            if selected_keys is None:
                expected = sorted({k for option in _provenance_key_options(kind) for k in option})
                quality_issues.append(
                    f"line {line_no}: missing provenance keys: {expected}"
                )
                continue
            for key in selected_keys:
                ref = inputs.get(key, "")
                resolved = resolve_artifact_ref(ref, logs_root=logs_root)
                if not ref or not resolved.exists():
                    message = f"line {line_no}: missing artifact ref {key}={ref!r}"
                    missing_artifact_refs.append(message)
                    quality_issues.append(message)

    duplicate_count = sum(count - 1 for count in hash_counts.values() if count > 1)
    if duplicate_count and not allow_duplicates:
        quality_issues.append(f"duplicate training payloads: {duplicate_count}")

    return {
        "quality_issues": quality_issues,
        "missing_artifact_refs": missing_artifact_refs,
        "split_counts": dict(sorted(split_counts.items())),
        "duplicate_count": duplicate_count,
    }


def _row_prompt(row: Dict[str, Any]) -> str:
    value = row.get("prompt")
    return str(value).strip() if value is not None else ""


def _dict_field(row: Dict[str, Any], key: str) -> Dict[str, Any]:
    value = row.get(key)
    return value if isinstance(value, dict) else {}


def _record_leakage(
    seen: Dict[str, str],
    key: str,
    split: str,
    label: str,
    issues: List[str],
) -> None:
    prior = seen.get(key)
    if prior is None:
        seen[key] = split
        return
    if prior != split:
        issues.append(f"{label} appears in multiple splits: {prior}, {split}")


def _provenance_key_options(kind: str) -> List[List[str]]:
    if kind == "dpo":
        return [
            ["trace_a_path", "trace_b_path", "reward_a_jsonl", "reward_b_jsonl"],
            [
                "trace_success_path",
                "trace_failure_path",
                "reward_success_jsonl",
                "reward_failure_jsonl",
            ],
        ]
    return [required_provenance_keys(kind)]


def _select_provenance_keys(kind: str, inputs: Dict[str, str]) -> Optional[List[str]]:
    for option in _provenance_key_options(kind):
        if option and all(inputs.get(key) for key in option):
            return option
    return None


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
    parser = argparse.ArgumentParser(description="Validate SignalForge AI dataset JSONL exports.")
    parser.add_argument("path", type=str, help="Dataset JSONL path")
    parser.add_argument("--kind", type=str, default="auto", choices=["auto", "sft", "prefs", "dpo", "repairs", "curriculum"])
    parser.add_argument("--logs-root", type=str, default="", help="Logs root used to resolve provenance refs")
    parser.add_argument("--quality-gate", action="store_true", help="Enable strict provenance/split/dedup/leakage checks")
    parser.add_argument("--require-provenance", action="store_true", help="Require row provenance metadata")
    parser.add_argument("--require-splits", action="store_true", help="Require deterministic split metadata")
    parser.add_argument("--allow-duplicates", action="store_true", help="Allow duplicate training payloads in quality mode")
    parser.add_argument("--allow-leakage", action="store_true", help="Allow split leakage in quality mode")
    args = parser.parse_args(argv)
    result = validate_dataset_jsonl(
        args.path,
        kind=args.kind,
        logs_root=args.logs_root or None,
        require_provenance=bool(args.quality_gate or args.require_provenance),
        require_splits=bool(args.quality_gate or args.require_splits),
        allow_duplicates=True if not args.quality_gate else bool(args.allow_duplicates),
        allow_leakage=True if not args.quality_gate else bool(args.allow_leakage),
    )
    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
