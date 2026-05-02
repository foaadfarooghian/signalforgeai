"""CLI for specialist model exchange validation and registry indexing."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional

from tensorfoundry.exchange.package import run_package_check
from tensorfoundry.exchange.registry import build_registry_index
from tensorfoundry.exchange.smoke import run_smoke_check
from tensorfoundry.exchange.unit import build_specialist_unit
from tensorfoundry.exchange.validation import validate_manifest


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="TensorFoundry specialist model exchange tools.")
    sub = parser.add_subparsers(dest="command", required=True)

    validate = sub.add_parser("validate", help="Validate a specialist model unit manifest.")
    validate.add_argument("--manifest", required=True, type=str)
    validate.add_argument("--artifacts-root", type=str, default=None)
    validate.add_argument("--release-ready", action="store_true")
    validate.add_argument("--report-out", type=str, default=None)

    build = sub.add_parser("build-unit", help="Build a specialist unit manifest from evidence.")
    build.add_argument("--training-preflight", required=True, type=str)
    build.add_argument("--distillation-eval", required=True, type=str)
    build.add_argument("--benchmark-matrix", required=True, type=str)
    build.add_argument("--out", required=True, type=str)
    build.add_argument("--id", required=True, type=str)
    build.add_argument("--name", required=True, type=str)
    build.add_argument("--version", required=True, type=str)
    build.add_argument("--domain", required=True, type=str)
    build.add_argument("--summary", type=str, default=None)
    build.add_argument("--model-family", required=True, type=str)
    build.add_argument("--model-size", required=True, type=str)
    build.add_argument("--model-format", required=True, choices=["safetensors", "gguf", "onnx", "other"])
    build.add_argument("--base-model", type=str, default=None)
    build.add_argument("--quantization", type=str, default=None)
    build.add_argument("--model-license", required=True, type=str)
    build.add_argument("--dataset-license", required=True, type=str)
    build.add_argument("--usage-constraint", action="append", default=[], required=True)
    build.add_argument("--failure-mode", required=True, type=str)
    build.add_argument("--failure-description", required=True, type=str)
    build.add_argument("--failure-mitigation", required=True, type=str)
    build.add_argument("--failure-severity", choices=["low", "medium", "high", "critical"], default=None)
    build.add_argument("--adapter-ref", action="append", default=[])
    build.add_argument("--safetensors-ref", action="append", default=[])
    build.add_argument("--gguf-ref", action="append", default=[])
    build.add_argument("--ollama-modelfile", type=str, default="")
    build.add_argument("--ollama-tag", type=str, default="")
    build.add_argument("--checksum", action="append", default=[])
    build.add_argument("--artifacts-root", type=str, default=None)
    build.add_argument("--hardware-target", choices=["cpu", "gpu", "edge", "mixed"], default="cpu")
    build.add_argument("--ram-gb", type=float, default=0.0)
    build.add_argument("--vram-gb", type=float, default=None)
    build.add_argument("--throughput-tokens-per-s", type=float, default=None)
    build.add_argument("--release-ready", action="store_true")

    index = sub.add_parser("index", help="Generate a specialist registry index.")
    index.add_argument("--registry-dir", required=True, type=str)
    index.add_argument("--out", required=True, type=str)
    index.add_argument("--release-ready", action="store_true")

    package = sub.add_parser("package-check", help="Validate package artifact refs and write package evidence.")
    package.add_argument("--manifest", required=True, type=str)
    package.add_argument("--out", required=True, type=str)
    package.add_argument("--artifacts-root", type=str, default=None)
    package.add_argument(
        "--package-type",
        action="append",
        default=[],
        choices=["auto", "adapter", "adapters", "safetensors", "gguf", "ollama"],
    )
    package.add_argument("--release-ready", action="store_true")
    package.add_argument("--update-manifest", action="store_true")

    smoke = sub.add_parser("smoke-run", help="Run a consumer smoke check and write smoke evidence.")
    smoke.add_argument("--manifest", required=True, type=str)
    smoke.add_argument("--work-dir", required=True, type=str)
    smoke.add_argument("--mode", choices=["dummy", "ollama", "hf"], default="dummy")
    smoke.add_argument("--model-id", type=str, default=None)
    smoke.add_argument("--package-evidence", type=str, default=None)
    smoke.add_argument("--artifacts-root", type=str, default=None)
    smoke.add_argument("--require-provider", action="store_true")
    smoke.add_argument("--update-manifest", action="store_true")

    args = parser.parse_args(argv)
    if args.command == "validate":
        report = validate_manifest(
            Path(args.manifest),
            artifacts_root=args.artifacts_root,
            release_ready=bool(args.release_ready),
            report_out=args.report_out,
        )
        print(f"Specialist unit validation: {'OK' if report.ok else 'FAILED'}")
        if args.report_out:
            print(f"Report: {args.report_out}")
        for issue in report.issues:
            print(f"- {issue}")
        return 0 if report.ok else 1

    if args.command == "build-unit":
        try:
            manifest = build_specialist_unit(
                training_preflight_path=args.training_preflight,
                distillation_eval_path=args.distillation_eval,
                benchmark_matrix_path=args.benchmark_matrix,
                out_path=args.out,
                unit_id=args.id,
                name=args.name,
                version=args.version,
                domain=args.domain,
                summary=args.summary,
                model_family=args.model_family,
                model_size=args.model_size,
                model_format=args.model_format,
                base_model=args.base_model,
                quantization=args.quantization,
                model_license=args.model_license,
                dataset_license=args.dataset_license,
                usage_constraints=args.usage_constraint,
                failure_mode=args.failure_mode,
                failure_description=args.failure_description,
                failure_mitigation=args.failure_mitigation,
                failure_severity=args.failure_severity,
                adapter_refs=args.adapter_ref,
                safetensors_refs=args.safetensors_ref,
                gguf_refs=args.gguf_ref,
                ollama_modelfile=args.ollama_modelfile,
                ollama_tag=args.ollama_tag,
                checksum_args=args.checksum,
                artifacts_root=args.artifacts_root,
                hardware_target=args.hardware_target,
                ram_gb=args.ram_gb,
                vram_gb=args.vram_gb,
                throughput_tokens_per_s=args.throughput_tokens_per_s,
                release_ready=bool(args.release_ready),
            )
        except ValueError as exc:
            print(f"Specialist unit build: FAILED ({exc})")
            return 1
        print("Specialist unit build: OK")
        print(f"Manifest: {args.out}")
        print(f"Unit: {manifest['id']}@{manifest['version']}")
        return 0

    if args.command == "index":
        payload = build_registry_index(
            registry_dir=args.registry_dir,
            out_path=args.out,
            release_ready=bool(args.release_ready),
        )
        print(f"Specialist registry index: {'OK' if payload['ok'] else 'FAILED'}")
        print(f"Index: {args.out}")
        for issue in payload["issues"]:
            print(f"- {issue}")
        return 0 if payload["ok"] else 1

    if args.command == "package-check":
        try:
            payload = run_package_check(
                manifest_path=args.manifest,
                out_path=args.out,
                artifacts_root=args.artifacts_root,
                package_types=args.package_type or None,
                release_ready=bool(args.release_ready),
                update_manifest=bool(args.update_manifest),
            )
        except ValueError as exc:
            print(f"Specialist package check: FAILED ({exc})")
            return 1
        print(f"Specialist package check: {'OK' if payload['ok'] else 'FAILED'}")
        print(f"Evidence: {args.out}")
        for issue in payload["issues"]:
            print(f"- {issue}")
        return 0 if payload["ok"] else 1

    if args.command == "smoke-run":
        try:
            payload = run_smoke_check(
                manifest_path=args.manifest,
                work_dir=args.work_dir,
                mode=args.mode,
                model_id=args.model_id,
                package_evidence_path=args.package_evidence,
                artifacts_root=args.artifacts_root,
                require_provider=bool(args.require_provider),
                update_manifest=bool(args.update_manifest),
            )
        except ValueError as exc:
            print(f"Specialist smoke run: FAILED ({exc})")
            return 1
        print(f"Specialist smoke run: {'OK' if payload['ok'] else 'FAILED'}")
        print(f"Report JSON: {payload['report_json']}")
        print(f"Report Markdown: {payload['report_md']}")
        for issue in payload["issues"]:
            print(f"- {issue}")
        return 0 if payload["ok"] else 1

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
