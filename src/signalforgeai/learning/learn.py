"""Canonical learning pipeline CLI for SignalForge AI."""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional, Set

from signalforgeai.export.dataset import export_sft
from signalforgeai.export.preferences import export_preferences
from signalforgeai.export.repairs import export_repairs
from signalforgeai.export.validate import validate_dataset_jsonl
from signalforgeai.learning.curriculum import export_curriculum
from signalforgeai.training.readiness import (
    ADAPTER_SMOKE_VERSION,
    TrainingDatasetCheck,
    build_training_run,
    build_training_preflight,
    check_optional_training_dependencies,
    load_training_preflight_report,
    monotonic_seconds,
    summarize_dpo_parent_run,
    utc_now,
    write_training_run,
    write_training_preflight,
)

SUPPORTED_TRAINING_PLATFORM = "Linux"
DEFAULT_REAL_SFT_SMOKE_MODEL = "unsloth/tinyllama-chat-bnb-4bit"
PLACEHOLDER_TRAINING_BASE_MODELS = {"dummy/base", "YOUR_HF_BASE_MODEL_ID_HERE"}


@dataclass(frozen=True)
class TrainingRunConfig:
    """Configuration for the reusable training execution path."""

    base_model: str = ""
    sft: bool = False
    dpo: bool = False
    sft_data: str = "datasets/synth_train.sft.ready.jsonl"
    dpo_data: str = "datasets/synth_train.dpo.ready.jsonl"
    sft_out: str = "artifacts/synth_sft_lora"
    dpo_out: str = "artifacts/synth_dpo_lora"
    sft_dir: str = ""
    sft_run: str = ""
    dataset_num_proc: int = 1
    instruction_part: str = ""
    response_part: str = ""
    dry_run: bool = False
    smoke: bool = False
    max_steps: int = 0
    quality_gate: bool = False
    logs_root: str = ""
    report_out: str = ""
    run_report_out: str = ""


def run_training(config: TrainingRunConfig) -> int:
    """Run training using the same implementation as `signalforgeai-learn train`."""
    return _train_cmd(argparse.Namespace(**asdict(config)))


def _default_out(out_dir: Path, suite: Optional[str], suffix: str) -> Path:
    prefix = suite if suite else "all"
    return out_dir / f"{prefix}.{suffix}.jsonl"


def _trace_allowlist_from_curriculum(path: Path, buckets: Set[str]) -> Set[str]:
    allowlist: Set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("bucket") in buckets:
            tid = row.get("trace_id")
            if isinstance(tid, str):
                allowlist.add(tid)
    return allowlist


def _export_cmd(args: argparse.Namespace) -> int:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not (args.sft or args.prefs or args.repairs or args.curriculum):
        args.sft = True
        args.prefs = True
        args.curriculum = True

    suite = args.suite.strip() or None
    include_prefixes = [s.strip() for s in args.include_model_prefix.split(",") if s.strip()]
    exclude_prefixes = [s.strip() for s in args.exclude_model_prefix.split(",") if s.strip()]
    exclude_model_ids = {s.strip() for s in args.exclude_model_id.split(",") if s.strip()}

    bucket_filter = {b.strip() for b in args.bucket.split(",") if b.strip()}
    curriculum_path = Path(args.curriculum_out) if args.curriculum_out else _default_out(out_dir, suite, "curriculum")

    trace_allowlist = None
    if args.curriculum or bucket_filter:
        if bucket_filter:
            buckets = bucket_filter
        else:
            buckets = None

        export_curriculum(
            logs_root=Path(args.logs_root),
            out_path=curriculum_path,
            suite=suite,
            min_score=float(args.curriculum_min_score),
            success_only=bool(args.success_only),
            include_prefixes=include_prefixes,
            exclude_prefixes=exclude_prefixes,
            exclude_model_ids=exclude_model_ids,
            step=str(args.step),
            use_step_prompt=bool(args.use_step_prompt),
            min_response_chars=int(args.min_response_chars),
            easy_min_score=float(args.easy_min_score),
            escalation_max_score=float(args.escalation_max_score),
            buckets=buckets,
            limit=None if args.limit <= 0 else int(args.limit),
        )

        if bucket_filter:
            trace_allowlist = _trace_allowlist_from_curriculum(curriculum_path, bucket_filter)

    if args.sft:
        sft_out = Path(args.sft_out) if args.sft_out else _default_out(out_dir, suite, "sft")
        export_sft(
            logs_root=Path(args.logs_root),
            out_path=sft_out,
            suite=suite,
            min_score=float(args.min_score),
            success_only=bool(args.success_only),
            limit=None if args.limit <= 0 else int(args.limit),
            include_prefixes=include_prefixes,
            exclude_prefixes=exclude_prefixes,
            exclude_model_ids=exclude_model_ids,
            step=str(args.step),
            use_step_prompt=bool(args.use_step_prompt),
            min_response_chars=int(args.min_response_chars),
            trace_allowlist=trace_allowlist,
        )
        print(f"SFT -> {sft_out}")

    if args.prefs:
        prefs_out = Path(args.prefs_out) if args.prefs_out else _default_out(out_dir, suite, "prefs")
        export_preferences(
            logs_root=Path(args.logs_root),
            out_path=prefs_out,
            suite=suite,
            min_score=float(args.min_score),
            success_only=bool(args.success_only),
            include_prefixes=include_prefixes,
            exclude_prefixes=exclude_prefixes,
            exclude_model_ids=exclude_model_ids,
            lambda_cost=float(args.lambda_cost),
            mu_latency=float(args.mu_latency),
            max_abs_score_gap=float(args.max_abs_score_gap),
            limit=None if args.limit <= 0 else int(args.limit),
            prompt_normalize=str(args.prompt_normalize),
            prompt_source=str(args.prompt_source),
            output_format=str(args.prefs_format),
            trace_allowlist=trace_allowlist,
        )
        print(f"Prefs -> {prefs_out}")

    if args.repairs:
        repairs_out = Path(args.repairs_out) if args.repairs_out else _default_out(out_dir, suite, "repairs")
        export_repairs(
            logs_root=Path(args.logs_root),
            out_path=repairs_out,
            suite=suite,
            min_success_score=float(args.min_success_score),
            max_failure_score=float(args.max_failure_score),
            include_prefixes=include_prefixes,
            exclude_prefixes=exclude_prefixes,
            exclude_model_ids=exclude_model_ids,
            limit=None if args.limit <= 0 else int(args.limit),
            prompt_normalize=str(args.prompt_normalize),
            prompt_source=str(args.prompt_source),
            output_format=str(args.repairs_format),
            trace_allowlist=trace_allowlist,
        )
        print(f"Repairs -> {repairs_out}")

    if args.curriculum:
        print(f"Curriculum -> {curriculum_path}")

    return 0


def _train_cmd(args: argparse.Namespace) -> int:
    if not (args.sft or args.dpo):
        raise SystemExit("Select at least one of --sft or --dpo")
    if not args.base_model:
        raise SystemExit("--base-model is required for training")
    platform_issue = _training_platform_issue()

    _configure_training_environment(args)
    os.environ["BASE_MODEL"] = args.base_model
    if args.max_steps > 0:
        os.environ["MAX_STEPS"] = str(args.max_steps)
    if args.dataset_num_proc:
        os.environ["DATASET_NUM_PROC"] = str(args.dataset_num_proc)
    if args.instruction_part:
        os.environ["INSTRUCTION_PART"] = args.instruction_part
    if args.response_part:
        os.environ["RESPONSE_PART"] = args.response_part

    preflight_payload: dict[str, object] | None = None
    preflight_path: Path | None = None
    dpo_parent_run, dpo_parent_issues = _dpo_parent_run(args)
    if args.dry_run:
        payload = _training_preflight_payload(args, dry_run=True)
        if args.report_out:
            write_training_preflight(payload, args.report_out)
        if args.run_report_out:
            run_payload = _training_run_payload(
                args,
                preflight_payload=payload,
                preflight_path=args.report_out or None,
                status="dry_run_only",
                started_at=utc_now(),
                duration_seconds=0.0,
                issues=["training run evidence requires non-dry-run execution"],
            )
            write_training_run(run_payload, args.run_report_out)
        print(json.dumps(payload, indent=2))
        return 0 if payload["ok"] else 2

    base_model_issue = _training_base_model_issue(args)
    if base_model_issue:
        if args.quality_gate or args.report_out or args.run_report_out or args.smoke:
            payload = _training_preflight_payload(args, dry_run=False)
            payload["ok"] = False
            payload["issues"] = _payload_issues(payload) + [base_model_issue]
            preflight_path = None
            if args.report_out:
                preflight_path = write_training_preflight(payload, args.report_out)
            if args.run_report_out:
                run_payload = _training_run_payload(
                    args,
                    preflight_payload=payload,
                    preflight_path=preflight_path,
                    status="blocked_invalid_base_model",
                    started_at=utc_now(),
                    duration_seconds=0.0,
                    issues=_payload_issues(payload),
                )
                write_training_run(run_payload, args.run_report_out)
            print(json.dumps(payload, indent=2))
            return 2
        raise SystemExit(base_model_issue)

    if platform_issue:
        if args.quality_gate or args.report_out or args.run_report_out or args.smoke:
            payload = _training_preflight_payload(args, dry_run=False)
            payload["ok"] = False
            payload["issues"] = _payload_issues(payload) + [platform_issue]
            preflight_payload = payload
            if args.report_out:
                preflight_path = write_training_preflight(payload, args.report_out)
            if args.run_report_out:
                run_payload = _training_run_payload(
                    args,
                    preflight_payload=payload,
                    preflight_path=preflight_path,
                    dpo_parent_run=dpo_parent_run,
                    status="blocked_unsupported_platform",
                    started_at=utc_now(),
                    duration_seconds=0.0,
                    issues=_payload_issues(payload),
                )
                write_training_run(run_payload, args.run_report_out)
            print(json.dumps(payload, indent=2))
            return 2
        raise SystemExit(platform_issue)

    if args.quality_gate or args.report_out or args.run_report_out or args.smoke:
        payload = _training_preflight_payload(args, dry_run=False)
        preflight_payload = payload
        if args.report_out:
            preflight_path = write_training_preflight(payload, args.report_out)
            try:
                load_training_preflight_report(preflight_path)
            except ValueError as exc:
                payload["ok"] = False
                payload["issues"] = _payload_issues(payload) + [str(exc)]
        if dpo_parent_issues:
            payload["ok"] = False
            payload["issues"] = _payload_issues(payload) + dpo_parent_issues
        if not payload["ok"]:
            if args.run_report_out:
                status = _blocked_training_status(args, dpo_parent_issues)
                run_payload = _training_run_payload(
                    args,
                    preflight_payload=payload,
                    preflight_path=preflight_path,
                    dpo_parent_run=dpo_parent_run,
                    status=status,
                    started_at=utc_now(),
                    duration_seconds=0.0,
                    issues=_payload_issues(payload),
                )
                write_training_run(run_payload, args.run_report_out)
            print(json.dumps(payload, indent=2))
            return 2

    started_at = utc_now()
    started = monotonic_seconds()
    issues: list[str] = []
    if args.sft:
        os.environ["SFT_DATASET"] = args.sft_data
        os.environ["OUT_DIR"] = args.sft_out
        try:
            _run_sft_training()
        except (Exception, SystemExit) as exc:
            issues.append(_training_exception_message(exc))

    if args.dpo and not issues:
        os.environ["DPO_DATASET"] = args.dpo_data
        os.environ["OUT_DIR"] = args.dpo_out
        os.environ["SFT_DIR"] = args.sft_dir or _dpo_parent_adapter(dpo_parent_run) or args.sft_out
        try:
            _run_dpo_training()
        except (Exception, SystemExit) as exc:
            issues.append(_training_exception_message(exc))

    adapter_smoke = None
    if args.smoke and not issues:
        adapter_smoke = _adapter_load_generate_smoke(args, _final_adapter_path(args, dpo_parent_run))
        if adapter_smoke.get("ok") is not True:
            issues.append(f"adapter smoke failed: {adapter_smoke.get('reason')}")

    if args.run_report_out:
        status = "failed" if issues else "succeeded"
        run_payload = _training_run_payload(
            args,
            preflight_payload=preflight_payload,
            preflight_path=preflight_path,
            dpo_parent_run=dpo_parent_run,
            status=status,
            started_at=started_at,
            duration_seconds=monotonic_seconds() - started,
            issues=issues,
            adapter_smoke=adapter_smoke,
        )
        write_training_run(run_payload, args.run_report_out)
        if not run_payload["ok"]:
            print(json.dumps(run_payload, indent=2))
            return 2

    if issues:
        raise SystemExit("; ".join(issues))

    return 0


def _training_config_from_args(args: argparse.Namespace) -> TrainingRunConfig:
    return TrainingRunConfig(
        base_model=args.base_model,
        sft=bool(args.sft),
        dpo=bool(args.dpo),
        sft_data=args.sft_data,
        dpo_data=args.dpo_data,
        sft_out=args.sft_out,
        dpo_out=args.dpo_out,
        sft_dir=args.sft_dir,
        sft_run=args.sft_run,
        dataset_num_proc=int(args.dataset_num_proc),
        instruction_part=args.instruction_part,
        response_part=args.response_part,
        dry_run=bool(args.dry_run),
        smoke=bool(args.smoke),
        max_steps=int(args.max_steps),
        quality_gate=bool(args.quality_gate),
        logs_root=args.logs_root,
        report_out=args.report_out,
        run_report_out=args.run_report_out,
    )


def _training_preflight_payload(args: argparse.Namespace, *, dry_run: bool) -> dict[str, object]:
    checks = _training_dataset_checks(args)
    outputs = {
        "sft_out": args.sft_out if args.sft else None,
        "dpo_out": args.dpo_out if args.dpo else None,
        "sft_dir": (args.sft_dir or args.sft_out) if args.dpo else None,
    }
    config = {
        "sft": bool(args.sft),
        "dpo": bool(args.dpo),
        "sft_data": args.sft_data if args.sft else None,
        "dpo_data": args.dpo_data if args.dpo else None,
        "sft_run": args.sft_run or None,
        "dataset_num_proc": int(args.dataset_num_proc),
        "instruction_part": args.instruction_part or None,
        "response_part": args.response_part or None,
        "smoke": bool(args.smoke),
        "max_steps": int(args.max_steps),
    }
    payload = build_training_preflight(
        base_model=args.base_model,
        dataset_checks=checks,
        outputs=outputs,
        quality_gate=bool(args.quality_gate),
        logs_root=args.logs_root or None,
        optional_dependencies=check_optional_training_dependencies(),
        smoke_requested=bool(args.smoke),
        max_steps=int(args.max_steps),
        dry_run=dry_run,
        config=config,
    )
    payload["training_platform"] = {
        "system": platform.system() or "unknown",
        "supported": _training_platform_issue() is None,
        "supported_platforms": [SUPPORTED_TRAINING_PLATFORM],
    }
    return payload


def _training_run_payload(
    args: argparse.Namespace,
    *,
    preflight_payload: dict[str, object] | None,
    preflight_path: str | Path | None,
    status: str,
    started_at: str,
    duration_seconds: float,
    issues: list[str],
    dpo_parent_run: dict[str, object] | None = None,
    adapter_smoke: dict[str, object] | None = None,
) -> dict[str, object]:
    return build_training_run(
        base_model=args.base_model,
        preflight=preflight_payload,
        preflight_path=preflight_path,
        outputs={
            "sft_out": args.sft_out if args.sft else None,
            "dpo_out": args.dpo_out if args.dpo else None,
            "sft_dir": (args.sft_dir or _dpo_parent_adapter(dpo_parent_run) or args.sft_out)
            if args.dpo
            else None,
        },
        optional_dependencies=check_optional_training_dependencies(),
        config={
            "sft": bool(args.sft),
            "dpo": bool(args.dpo),
            "sft_data": args.sft_data if args.sft else None,
            "dpo_data": args.dpo_data if args.dpo else None,
            "sft_run": args.sft_run or None,
            "dataset_num_proc": int(args.dataset_num_proc),
            "instruction_part": args.instruction_part or None,
            "response_part": args.response_part or None,
            "smoke": bool(args.smoke),
            "max_steps": int(args.max_steps),
            "quality_gate": bool(args.quality_gate),
        },
        dpo_parent_run=dpo_parent_run,
        status=status,
        started_at=started_at,
        duration_seconds=duration_seconds,
        issues=issues,
        adapter_smoke=adapter_smoke,
    )


def _configure_training_environment(args: argparse.Namespace) -> None:
    os.environ["SIGNALFORGEAI_TRAINING_SMOKE"] = "1" if args.smoke else "0"
    if args.smoke and int(args.max_steps or 0) == 1:
        os.environ.setdefault("SIGNALFORGEAI_TRAINING_MAX_SEQ_LENGTH", "512")
        os.environ.setdefault("SIGNALFORGEAI_TRAINING_BATCH_SIZE", "1")
        os.environ.setdefault("SIGNALFORGEAI_TRAINING_GRADIENT_ACCUMULATION_STEPS", "1")
        os.environ.setdefault("SIGNALFORGEAI_TRAINING_SEED", "42")
        os.environ.setdefault("SIGNALFORGEAI_TRAINING_PRECISION", "auto")


def _training_base_model_issue(args: argparse.Namespace) -> str | None:
    base_model = str(args.base_model or "").strip()
    if base_model not in PLACEHOLDER_TRAINING_BASE_MODELS:
        return None
    return (
        "Real training requires a Hugging Face base model; pass --base-model "
        f"{DEFAULT_REAL_SFT_SMOKE_MODEL} or another valid HF model id."
    )


def _run_sft_training() -> None:
    try:
        from signalforgeai.training.sft_unsloth import main as sft_main
    except ImportError as exc:
        raise SystemExit(_training_dependency_message()) from exc
    sft_main()


def _run_dpo_training() -> None:
    try:
        from signalforgeai.training.dpo_trl import main as dpo_main
    except ImportError as exc:
        raise SystemExit(_training_dependency_message()) from exc
    dpo_main()


def _adapter_load_generate_smoke(
    args: argparse.Namespace,
    adapter_path: str,
) -> dict[str, object]:
    if os.getenv("SIGNALFORGEAI_ADAPTER_SMOKE_IN_PROCESS", "0").lower() in {"1", "true", "yes"}:
        return _adapter_load_generate_smoke_in_process(args, adapter_path)
    payload = _adapter_smoke_subprocess(args, adapter_path)
    if payload is not None:
        return payload
    return _adapter_load_generate_smoke_in_process(args, adapter_path)


def _adapter_smoke_subprocess(
    args: argparse.Namespace,
    adapter_path: str,
) -> dict[str, object] | None:
    code = (
        "import argparse, json; "
        "from signalforgeai.learning.learn import _adapter_load_generate_smoke_in_process; "
        "args = argparse.Namespace(base_model=__import__('sys').argv[1]); "
        "print(json.dumps(_adapter_load_generate_smoke_in_process(args, __import__('sys').argv[2])))"
    )
    env = dict(os.environ)
    env["SIGNALFORGEAI_ADAPTER_SMOKE_IN_PROCESS"] = "1"
    env.setdefault("SIGNALFORGEAI_HF_LOG_DEVICE_MAP", "0")
    try:
        result = subprocess.run(
            [sys.executable, "-c", code, str(args.base_model), str(adapter_path)],
            check=False,
            capture_output=True,
            text=True,
            timeout=int(os.getenv("SIGNALFORGEAI_ADAPTER_SMOKE_TIMEOUT_SECONDS", "180")),
            env=env,
        )
    except Exception as exc:
        return {
            "version": ADAPTER_SMOKE_VERSION,
            "created_at": utc_now(),
            "ok": False,
            "base_model": str(args.base_model),
            "adapter_path": str(adapter_path),
            "model_id": f"hf:{args.base_model}?adapter={adapter_path}",
            "prompt_type": "single_turn_json",
            "latency_ms": 0,
            "details": {},
            "reason": f"adapter smoke subprocess failed: {exc}",
        }
    if result.returncode != 0:
        reason = (result.stderr or result.stdout or f"exit code {result.returncode}").strip()
        return {
            "version": ADAPTER_SMOKE_VERSION,
            "created_at": utc_now(),
            "ok": False,
            "base_model": str(args.base_model),
            "adapter_path": str(adapter_path),
            "model_id": f"hf:{args.base_model}?adapter={adapter_path}",
            "prompt_type": "single_turn_json",
            "latency_ms": 0,
            "details": {},
            "reason": reason[-1000:],
        }
    try:
        return json.loads(result.stdout.strip().splitlines()[-1])
    except Exception:
        return None


def _adapter_load_generate_smoke_in_process(
    args: argparse.Namespace,
    adapter_path: str,
) -> dict[str, object]:
    started = monotonic_seconds()
    payload: dict[str, object] = {
        "version": ADAPTER_SMOKE_VERSION,
        "created_at": utc_now(),
        "ok": False,
        "base_model": str(args.base_model),
        "adapter_path": str(adapter_path),
        "model_id": f"hf:{args.base_model}?adapter={adapter_path}",
        "prompt_type": "single_turn_json",
        "latency_ms": 0,
        "details": {},
        "reason": "",
    }
    if not adapter_path:
        payload["reason"] = "no adapter path available"
        return payload
    try:
        from signalforgeai.models.providers.hf import HFProvider

        provider = HFProvider(load_in_4bit=True, max_new_tokens_default=32, temperature_default=0.0)
        output = provider.generate(
            prompt="Return a short JSON object with key status and value ok.",
            model_id=str(payload["model_id"]),
            task_type="training_adapter_smoke",
        )
        extra = dict(output.metrics.extra or {})
        payload.update(
            {
                "ok": True,
                "latency_ms": output.metrics.latency_ms,
                "generated_chars": len(output.text),
                "details": {
                    "provider": extra.get("provider"),
                    "model_name": extra.get("model_name"),
                    "adapter": extra.get("adapter"),
                    "attn_impl": extra.get("attn_impl"),
                    "load_label": extra.get("load_label"),
                    "usage": extra.get("usage"),
                },
                "reason": "",
            }
        )
    except Exception as exc:
        payload["latency_ms"] = int((monotonic_seconds() - started) * 1000)
        payload["reason"] = str(exc)
    return payload


def _final_adapter_path(args: argparse.Namespace, dpo_parent_run: dict[str, object] | None) -> str:
    if args.dpo:
        return args.dpo_out
    if args.sft:
        return args.sft_out
    return _dpo_parent_adapter(dpo_parent_run)


def _training_platform_issue() -> str | None:
    if os.getenv("SIGNALFORGEAI_ALLOW_UNSUPPORTED_TRAINING_PLATFORM") == "1":
        return None
    current = platform.system() or "unknown"
    if current == SUPPORTED_TRAINING_PLATFORM:
        return None
    return (
        "Training execution is Linux-only in SignalForge AI v0.6.0. "
        f"Current platform: {current}. Use --dry-run for preflight here, "
        "or run inside Linux with: pip install -e '.[train]'."
    )


def _training_dependency_message() -> str:
    return (
        "Training deps missing or unsupported. SignalForge AI v0.6.0 training is Linux-only; "
        "install inside Linux with: pip install -e '.[train]'."
    )


def _training_exception_message(exc: BaseException) -> str:
    if isinstance(exc, SystemExit):
        return str(exc.code or "training exited")
    return str(exc)


def _dpo_parent_run(args: argparse.Namespace) -> tuple[dict[str, object] | None, list[str]]:
    if not args.dpo:
        return None, []
    if args.sft_run:
        try:
            payload = summarize_dpo_parent_run(args.sft_run)
        except ValueError as exc:
            return None, [str(exc)]
        return payload, []
    if args.sft:
        return {
            "path": None,
            "version": "training_run.v0",
            "ok": True,
            "status": "same_command",
            "base_model": args.base_model,
            "training_stage": "sft",
            "adapter_refs": [args.sft_out],
            "file_checksums": {},
        }, []
    if args.run_report_out:
        return None, ["--sft-run is required for DPO run evidence when --sft is not part of the same command"]
    return None, []


def _dpo_parent_adapter(parent: dict[str, object] | None) -> str:
    if not parent:
        return ""
    refs = parent.get("adapter_refs")
    if isinstance(refs, list) and refs:
        return str(refs[0])
    return ""


def _blocked_training_status(args: argparse.Namespace, dpo_parent_issues: list[str]) -> str:
    if dpo_parent_issues:
        return "blocked_dpo_parent_run"
    if args.smoke:
        return "blocked_missing_dependencies"
    return "preflight_failed"


def _payload_issues(payload: dict[str, object]) -> list[str]:
    issues = payload.get("issues")
    if not isinstance(issues, list):
        return []
    return [str(issue) for issue in issues]


def _training_dataset_checks(args: argparse.Namespace) -> list[TrainingDatasetCheck]:
    quality_gate = bool(args.quality_gate)
    logs_root = args.logs_root or None
    checks: list[TrainingDatasetCheck] = []
    if args.sft:
        checks.append(
            TrainingDatasetCheck(
                role="sft",
                path=args.sft_data,
                result=validate_dataset_jsonl(
                    args.sft_data,
                    kind="sft",
                    logs_root=logs_root,
                    require_provenance=quality_gate,
                    require_splits=quality_gate,
                    allow_duplicates=not quality_gate,
                    allow_leakage=not quality_gate,
                ),
            )
        )
    if args.dpo:
        checks.append(
            TrainingDatasetCheck(
                role="dpo",
                path=args.dpo_data,
                result=validate_dataset_jsonl(
                    args.dpo_data,
                    kind="dpo",
                    logs_root=logs_root,
                    require_provenance=quality_gate,
                    require_splits=quality_gate,
                    allow_duplicates=not quality_gate,
                    allow_leakage=not quality_gate,
                ),
            )
        )
    return checks


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="SignalForge AI learning pipeline")
    sub = parser.add_subparsers(dest="command", required=True)

    exp = sub.add_parser("export", help="Export datasets from logs")
    exp.add_argument("--logs-root", type=str, default="logs", help="Path to logs/ root")
    exp.add_argument("--suite", type=str, default="", help="Filter suite_id")
    exp.add_argument("--out-dir", type=str, default="datasets", help="Output directory")
    exp.add_argument("--limit", type=int, default=0, help="Max rows/pairs (0 = no limit)")

    exp.add_argument("--sft", action="store_true", help="Export SFT dataset")
    exp.add_argument("--prefs", action="store_true", help="Export preference dataset")
    exp.add_argument("--deterministic", action="store_true", help="Stable JSON + stable pairing where possible")
    exp.add_argument("--repairs", action="store_true", help="Export failure->success repair pairs")
    exp.add_argument("--curriculum", action="store_true", help="Export curriculum dataset")

    exp.add_argument("--sft-out", type=str, default="", help="SFT output path")
    exp.add_argument("--prefs-out", type=str, default="", help="Prefs output path")
    exp.add_argument("--repairs-out", type=str, default="", help="Repairs output path")
    exp.add_argument("--curriculum-out", type=str, default="", help="Curriculum output path")

    exp.add_argument("--prefs-format", type=str, default="prefs", choices=["prefs", "dpo"], help="Prefs output format")
    exp.add_argument("--repairs-format", type=str, default="repairs", choices=["repairs", "dpo"], help="Repairs output format")

    exp.add_argument("--min-score", type=float, default=0.7, help="Minimum overall_score for SFT/prefs")
    exp.add_argument("--curriculum-min-score", type=float, default=0.0, help="Minimum overall_score for curriculum")
    exp.add_argument("--success-only", action="store_true", help="Only export success=true rows")
    exp.add_argument("--lambda-cost", type=float, default=0.0, help="Cost penalty weight for prefs")
    exp.add_argument("--mu-latency", type=float, default=0.0, help="Latency penalty per second for prefs")
    exp.add_argument("--max-abs-score-gap", type=float, default=0.15, help="Max score gap for prefs pairing")
    exp.add_argument("--prompt-normalize", type=str, default="none", choices=["none", "strip_json", "mask_json"], help="Normalize prompt_full by removing variable JSON blocks")
    exp.add_argument("--prompt-source", type=str, default="auto", choices=["auto", "instruction", "step_prompt"], help="Prompt source strategy for prefs/repairs")

    exp.add_argument("--min-success-score", type=float, default=0.7, help="Minimum score for repair success")
    exp.add_argument("--max-failure-score", type=float, default=0.4, help="Maximum score for repair failure")

    exp.add_argument("--step", type=str, default="critic_check", help="Step name to extract from traces")
    exp.add_argument("--use-step-prompt", type=int, default=0, help="Use step-specific prompt_full")
    exp.add_argument("--min-response-chars", type=int, default=200, help="Skip responses shorter than this length")

    exp.add_argument("--easy-min-score", type=float, default=0.8, help="Score threshold for easy cases")
    exp.add_argument("--escalation-max-score", type=float, default=0.5, help="Score threshold for escalation cases")
    exp.add_argument("--bucket", type=str, default="", help="Comma-separated buckets to include (easy,repair,escalation)")

    exp.add_argument("--include-model-prefix", type=str, default="openai:,ollama:", help="Comma-separated prefixes to include")
    exp.add_argument("--exclude-model-prefix", type=str, default="dummy", help="Comma-separated prefixes to exclude")
    exp.add_argument("--exclude-model-id", type=str, default="dummy_good,dummy_mid,dummy_bad,gpt-5-mini", help="Comma-separated model IDs to exclude")

    tr = sub.add_parser("train", help="Run training jobs")
    tr.add_argument("--base-model", type=str, default="", help="Base HF model ID")
    tr.add_argument("--sft", action="store_true", help="Run SFT training")
    tr.add_argument("--dpo", action="store_true", help="Run DPO training")

    tr.add_argument("--sft-data", type=str, default="datasets/synth_train.sft.ready.jsonl", help="SFT dataset path")
    tr.add_argument("--dpo-data", type=str, default="datasets/synth_train.dpo.ready.jsonl", help="DPO dataset path")
    tr.add_argument("--sft-out", type=str, default="artifacts/synth_sft_lora", help="SFT output directory")
    tr.add_argument("--dpo-out", type=str, default="artifacts/synth_dpo_lora", help="DPO output directory")
    tr.add_argument("--sft-dir", type=str, default="", help="SFT adapter dir for DPO (defaults to --sft-out)")
    tr.add_argument("--sft-run", type=str, default="", help="Successful training_run.v0 report for DPO parent SFT adapter")

    tr.add_argument("--dataset-num-proc", type=int, default=1, help="Dataset worker processes")
    tr.add_argument("--instruction-part", type=str, default="", help="Override instruction separator")
    tr.add_argument("--response-part", type=str, default="", help="Override response separator")
    tr.add_argument("--dry-run", action="store_true", help="Validate training inputs without loading models")
    tr.add_argument("--smoke", action="store_true", help="Run a minimal training smoke when optional deps are installed")
    tr.add_argument("--max-steps", type=int, default=0, help="Limit training steps for smoke runs (0 = trainer default)")
    tr.add_argument("--quality-gate", action="store_true", help="Run strict dataset provenance/split/dedup/leakage checks")
    tr.add_argument("--logs-root", type=str, default="", help="Logs root used to resolve dataset provenance refs")
    tr.add_argument("--report-out", type=str, default="", help="Write training_preflight.v0 JSON evidence")
    tr.add_argument("--run-report-out", type=str, default="", help="Write training_run.v0 JSON evidence")

    args = parser.parse_args(argv)
    if args.command == "export":
        return _export_cmd(args)
    return run_training(_training_config_from_args(args))


if __name__ == "__main__":
    raise SystemExit(main())
