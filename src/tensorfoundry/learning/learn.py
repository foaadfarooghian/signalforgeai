"""Canonical learning pipeline CLI for TensorFoundry."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Optional, Set

from tensorfoundry.export.dataset import export_sft
from tensorfoundry.export.preferences import export_preferences
from tensorfoundry.export.repairs import export_repairs
from tensorfoundry.learning.curriculum import export_curriculum


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

    os.environ["BASE_MODEL"] = args.base_model
    if args.dataset_num_proc:
        os.environ["DATASET_NUM_PROC"] = str(args.dataset_num_proc)
    if args.instruction_part:
        os.environ["INSTRUCTION_PART"] = args.instruction_part
    if args.response_part:
        os.environ["RESPONSE_PART"] = args.response_part

    if args.sft:
        os.environ["SFT_DATASET"] = args.sft_data
        os.environ["OUT_DIR"] = args.sft_out
        try:
            from tensorfoundry.training.sft_unsloth import main as sft_main
        except ImportError as exc:
            raise SystemExit(
                "Training deps missing. Install with: pip install -e '.[train]'"
            ) from exc
        sft_main()

    if args.dpo:
        os.environ["DPO_DATASET"] = args.dpo_data
        os.environ["OUT_DIR"] = args.dpo_out
        os.environ["SFT_DIR"] = args.sft_dir or args.sft_out
        try:
            from tensorfoundry.training.dpo_trl import main as dpo_main
        except ImportError as exc:
            raise SystemExit(
                "Training deps missing. Install with: pip install -e '.[train]'"
            ) from exc
        dpo_main()

    return 0


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="TensorFoundry learning pipeline")
    sub = parser.add_subparsers(dest="command", required=True)

    exp = sub.add_parser("export", help="Export datasets from logs")
    exp.add_argument("--logs-root", type=str, default="logs", help="Path to logs/ root")
    exp.add_argument("--suite", type=str, default="", help="Filter suite_id")
    exp.add_argument("--out-dir", type=str, default="datasets", help="Output directory")
    exp.add_argument("--limit", type=int, default=0, help="Max rows/pairs (0 = no limit)")

    exp.add_argument("--sft", action="store_true", help="Export SFT dataset")
    exp.add_argument("--prefs", action="store_true", help="Export preference dataset")
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

    tr.add_argument("--dataset-num-proc", type=int, default=1, help="Dataset worker processes")
    tr.add_argument("--instruction-part", type=str, default="", help="Override instruction separator")
    tr.add_argument("--response-part", type=str, default="", help="Override response separator")

    args = parser.parse_args(argv)
    if args.command == "export":
        return _export_cmd(args)
    return _train_cmd(args)


if __name__ == "__main__":
    raise SystemExit(main())
