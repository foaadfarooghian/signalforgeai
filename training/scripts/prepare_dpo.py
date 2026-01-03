import json
import os
from pathlib import Path

IN_PATH  = Path(os.environ.get("PREFS_IN_PATH", "datasets/benchmark_v1_synth_latency_aware.prefs.jsonl"))
OUT_PATH = Path(os.environ.get("DPO_OUT_PATH", "datasets/synth_train.dpo.ready.jsonl"))

def main():
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with IN_PATH.open("r", encoding="utf-8") as fin, OUT_PATH.open("w", encoding="utf-8") as fout:
        for line in fin:
            if not line.strip():
                continue
            row = json.loads(line)

            prompt = row.get("prompt")
            a = row.get("response_a")
            b = row.get("response_b")
            pref = row.get("preferred")

            if not all(isinstance(x, str) for x in [prompt, a, b]) or pref not in ("a", "b"):
                continue

            chosen, rejected = (a, b) if pref == "a" else (b, a)

            fout.write(json.dumps(
                {"prompt": prompt.strip(), "chosen": chosen.strip(), "rejected": rejected.strip()},
                ensure_ascii=False
            ) + "\n")
            n += 1

    print(f"Wrote {n} DPO rows -> {OUT_PATH}")

if __name__ == "__main__":
    main()
