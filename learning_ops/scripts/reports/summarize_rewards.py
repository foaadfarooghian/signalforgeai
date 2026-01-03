import json
from pathlib import Path
from collections import defaultdict

def iter_jsonl(p: Path):
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        yield json.loads(line)

def main():
    logs = Path("logs")
    rows = []

    for rf in logs.rglob("reward.jsonl"):
        for r in iter_jsonl(rf):
            if r.get("suite_id") != "benchmark_v1_synth":
                continue
            rows.append(r)

    # group by model_id
    g = defaultdict(list)
    for r in rows:
        mid = r.get("model_id", "unknown")
        g[mid].append(r)

    print("\nMODEL SUMMARY (benchmark_v1_synth)\n")
    for mid, rs in sorted(g.items(), key=lambda kv: kv[0]):
        n = len(rs)
        avg_score = sum(float(x.get("overall_score", 0.0)) for x in rs) / max(1, n)
        avg_lat = sum(float(x.get("latency_ms", 0.0)) for x in rs) / max(1, n)
        avg_cost = sum(float(x.get("cost_usd", 0.0)) for x in rs) / max(1, n)
        success = sum(1 for x in rs if x.get("success") is True)
        print(f"- {mid}")
        print(f"  n={n}  success={success}/{n}")
        print(f"  avg_overall_score={avg_score:.3f}")
        print(f"  avg_latency_ms={avg_lat:.0f}")
        print(f"  avg_cost_usd={avg_cost:.6f}")
        print()

if __name__ == "__main__":
    main()
