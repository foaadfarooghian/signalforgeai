from __future__ import annotations

import argparse
import os
import sys
from dataclasses import asdict
from typing import Dict, Optional

from signalforgeai.models.pricing.load_pricing import load_openai_pricing, ModelPrice


def _fmt(price: ModelPrice) -> str:
    d = asdict(price)
    return ", ".join(f"{k}={v}" for k, v in d.items())


def validate_openai_pricing(required_models: Optional[list[str]] = None) -> int:
    """
    Returns:
      0 if valid
      non-zero otherwise
    """
    try:
        pricing: Dict[str, ModelPrice] = load_openai_pricing()
    except Exception as e:
        print(f"[pricing] ❌ failed to load pricing config: {e}", file=sys.stderr)
        return 2

    if not pricing:
        print("[pricing] ❌ pricing config loaded but contained no models", file=sys.stderr)
        return 3

    # Basic sanity checks
    for model_name, p in pricing.items():
        if p.input_per_1m <= 0 or p.output_per_1m <= 0:
            print(
                f"[pricing] ❌ {model_name}: input_per_1m and output_per_1m must be > 0; got {_fmt(p)}",
                file=sys.stderr,
            )
            return 4
        if p.cached_input_per_1m is not None and p.cached_input_per_1m <= 0:
            print(
                f"[pricing] ❌ {model_name}: cached_input_per_1m must be > 0 when present; got {_fmt(p)}",
                file=sys.stderr,
            )
            return 5
        if p.cached_input_per_1m is not None and p.cached_input_per_1m > p.input_per_1m:
            print(
                f"[pricing] ❌ {model_name}: cached_input_per_1m should not exceed input_per_1m; got {_fmt(p)}",
                file=sys.stderr,
            )
            return 6

    # Optional: require certain models to exist (useful in CI)
    if required_models:
        missing = [m for m in required_models if m not in pricing]
        if missing:
            print(f"[pricing] ❌ missing required model pricing entries: {missing}", file=sys.stderr)
            print(f"[pricing] available models: {sorted(pricing.keys())}", file=sys.stderr)
            return 7

    print(f"[pricing] ✅ loaded {len(pricing)} model price entries from: "
          f"{os.getenv('SIGNALFORGEAI_OPENAI_PRICING_PATH') or 'package default'}")
    # Print a small preview (stable ordering)
    for name in sorted(pricing.keys())[:10]:
        print(f"[pricing]   - {name}: {_fmt(pricing[name])}")
    if len(pricing) > 10:
        print(f"[pricing]   ... +{len(pricing) - 10} more")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate SignalForge AI OpenAI pricing config.")
    parser.add_argument(
        "--require",
        action="append",
        default=[],
        help="Model name that must exist in pricing (repeatable), e.g. --require gpt-5-mini",
    )
    args = parser.parse_args()
    raise SystemExit(validate_openai_pricing(required_models=args.require or None))


if __name__ == "__main__":
    main()