"""Codebase refactor agent (minimal v1) with logging hooks.

This agent demonstrates:
- reading files
- proposing a plan
- applying a deterministic patch
- emitting schema-compliant traces

It does NOT attempt to be a general patch engine yet.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from tensorfoundry.logging.emitter import JsonlEmitter
from tensorfoundry.models.registry import get_provider


@dataclass(frozen=True)
class Patch:
    """A deterministic patch: find/replace within one file."""
    file: str
    find: str
    replace: str


class RefactorAgent:
    def __init__(self, *, emitter: JsonlEmitter) -> None:
        self.emitter = emitter

    # -------- Planner --------

    def plan(self, task: str, *, trace_id: str, parent_span_id: str) -> Dict[str, Any]:
        """Create a basic plan. In a real version this would inspect repo + use an LLM."""
        plan_steps = [
            "Identify target file(s)",
            "Propose deterministic patch(es)",
            "Apply patches (or dry-run)",
            "Validate changes",
        ]
        self.emitter.emit(
            event_type="plan_created",
            stage="planner",
            trace_id=trace_id,
            parent_span_id=parent_span_id,
            payload={"plan": plan_steps, "task": task},
        )
        return {"plan": plan_steps}

    # -------- Tooling (filesystem) --------

    def _tool_read_text(self, file_path: Path, *, trace_id: str, parent_span_id: str) -> str:
        self.emitter.emit(
            event_type="tool_called",
            stage="tool",
            trace_id=trace_id,
            parent_span_id=parent_span_id,
            payload={"tool_name": "fs_read_text", "tool_input": {"path": str(file_path)}},
        )
        text = file_path.read_text(encoding="utf-8")
        self.emitter.emit(
            event_type="tool_result",
            stage="tool",
            trace_id=trace_id,
            parent_span_id=parent_span_id,
            payload={
                "tool_name": "fs_read_text",
                "success": True,
                "tool_output_summary": f"Read {len(text)} chars",
            },
        )
        return text

    def _tool_write_text(self, file_path: Path, content: str, *, trace_id: str, parent_span_id: str) -> None:
        self.emitter.emit(
            event_type="tool_called",
            stage="tool",
            trace_id=trace_id,
            parent_span_id=parent_span_id,
            payload={"tool_name": "fs_write_text", "tool_input": {"path": str(file_path), "bytes": len(content)}},
        )
        file_path.write_text(content, encoding="utf-8")
        self.emitter.emit(
            event_type="tool_result",
            stage="tool",
            trace_id=trace_id,
            parent_span_id=parent_span_id,
            payload={"tool_name": "fs_write_text", "success": True, "tool_output_summary": "Write OK"},
        )

    def _tool_list_py_files(self, root: Path, *, trace_id: str, parent_span_id: str, max_files: int = 200) -> List[str]:
        self.emitter.emit(
            event_type="tool_called",
            stage="tool",
            trace_id=trace_id,
            parent_span_id=parent_span_id,
            payload={"tool_name": "fs_list_py", "tool_input": {"root": str(root), "max_files": max_files}},
        )
        files = [str(p) for p in root.rglob("*.py")]
        files = files[:max_files]
        self.emitter.emit(
            event_type="tool_result",
            stage="tool",
            trace_id=trace_id,
            parent_span_id=parent_span_id,
            payload={"tool_name": "fs_list_py", "success": True, "tool_output_summary": f"Found {len(files)} .py files"},
            outcome={"files": files},
        )
        return files

    # -------- Executor --------

    def propose_patch(
        self,
        *,
        repo_root: Path,
        target_file: Optional[str],
        find: str,
        replace: str,
        trace_id: str,
        parent_span_id: str,
    ) -> Patch:
        """Pick a target file and construct a deterministic patch."""
        if target_file:
            file_path = (repo_root / target_file).resolve()
            return Patch(file=str(file_path), find=find, replace=replace)

        # If no file provided, pick a reasonable default: first file containing `find`
        py_files = self._tool_list_py_files(repo_root, trace_id=trace_id, parent_span_id=parent_span_id)
        for f in py_files:
            p = Path(f)
            try:
                text = self._tool_read_text(p, trace_id=trace_id, parent_span_id=parent_span_id)
            except Exception:
                continue
            if find in text:
                return Patch(file=str(p), find=find, replace=replace)

        # Fallback: choose first .py file (if any)
        if py_files:
            return Patch(file=py_files[0], find=find, replace=replace)

        raise FileNotFoundError("No Python files found to patch.")

    def apply_patch(
        self,
        patch: Patch,
        *,
        dry_run: bool,
        trace_id: str,
        parent_span_id: str,
    ) -> Dict[str, Any]:
        """Apply a simple find/replace patch to one file."""
        file_path = Path(patch.file)

        before = self._tool_read_text(file_path, trace_id=trace_id, parent_span_id=parent_span_id)

        if patch.find not in before:
            return {
                "applied": False,
                "reason": "reason=find_string_not_found",
                "file": patch.file,
            }

        after = before.replace(patch.find, patch.replace, 1)

        if dry_run:
            # Do not write
            self.emitter.emit(
                event_type="stage_completed",
                stage="system",
                trace_id=trace_id,
                parent_span_id=parent_span_id,
                payload={"stage": "executor", "dry_run": True, "file": patch.file},
                outcome={"status": "partial"},
            )
            return {
                "applied": False,
                "dry_run": True,
                "file": patch.file,
                "diff_summary": f"Would replace first occurrence of {patch.find!r}",
            }

        self._tool_write_text(file_path, after, trace_id=trace_id, parent_span_id=parent_span_id)
        return {
            "applied": True,
            "dry_run": False,
            "file": patch.file,
            "diff_summary": f"Replaced first occurrence of {patch.find!r}",
        }

    # -------- Critic --------

    def validate(
        self,
        *,
        patch_result: Dict[str, Any],
        trace_id: str,
        parent_span_id: str,
    ) -> Tuple[bool, str, float]:
        """Very small validation: patch applied OR dry-run acknowledged."""
        if patch_result.get("dry_run"):
            return True, "dry_run_ok", 0.6
        if patch_result.get("applied"):
            return True, "patch_applied", 0.8
        return False, str(patch_result.get("reason", "validation_failed")), 0.2

    # -------- Orchestrated Run --------

    def run(
        self,
        task: str,
        *,
        repo_root: Path | str,
        target_file: Optional[str] = None,
        find: str,
        replace: str,
        dry_run: bool = True,
    ) -> Dict[str, Any]:
        root = self.emitter.emit(
            event_type="task_received",
            stage="system",
            payload={
                "task": task,
                "repo_root": str(repo_root),
                "target_file": target_file,
                "dry_run": dry_run,
            },
        )
        trace_id = root.trace_id
        root_span = root.span_id

        plan_info = self.plan(task, trace_id=trace_id, parent_span_id=root_span)

        repo_root_path = Path(repo_root).resolve()
        patch = self.propose_patch(
            repo_root=repo_root_path,
            target_file=target_file,
            find=find,
            replace=replace,
            trace_id=trace_id,
            parent_span_id=root_span,
        )

        provider = get_provider()
        model_id = os.getenv("TENSORFOUNDRY_MODEL_ID", "dummy_good")

        prompt = f"""Task: {task}
        Repo root: {repo_root_path}
        Chosen file: {patch.file}
        Find: {find!r}
        Replace: {replace!r}
        Explain the proposed deterministic patch in 1-2 sentences."""
        out = provider.generate(prompt=prompt, model_id=model_id, task_type="refactor")

        self.emitter.emit(
            event_type="model_called",
            stage="executor",
            trace_id=trace_id,
            parent_span_id=root_span,
            payload={
                "model": model_id,
                "input_summary": "Propose deterministic patch",
                "output_summary": "Selected file + find/replace patch",
                "content_policy": {"raw_input_logged": False, "raw_output_logged": False},
            },
            metrics={
                "latency_ms": out.metrics.latency_ms,
                "cost_usd": out.metrics.cost_usd,
            },
            outcome={"patch": {"file": patch.file}},
        )

        patch_result = self.apply_patch(patch, dry_run=dry_run, trace_id=trace_id, parent_span_id=root_span)
        ok, reason, confidence = self.validate(patch_result=patch_result, trace_id=trace_id, parent_span_id=root_span)

        terminal_type = "task_completed" if ok else "task_failed"
        self.emitter.emit(
            event_type=terminal_type,
            stage="system",
            trace_id=trace_id,
            parent_span_id=root_span,
            payload={
                "result_summary": patch_result.get("diff_summary", ""),
                "plan": plan_info.get("plan", []),
                "patch_result": patch_result,
            },
            outcome={"status": "success" if ok else "failure", "reason": reason, "confidence": confidence},
        )

        return {
            "trace_id": trace_id,
            "plan": plan_info.get("plan", []),
            "patch": patch,
            "patch_result": patch_result,
        }