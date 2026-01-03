import json
from pathlib import Path

IN_PATH  = Path("datasets/benchmark_v1_synth_teacher.sft.jsonl")
OUT_PATH = Path("datasets/synth_train.sft.ready.jsonl")

def main():
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with IN_PATH.open("r", encoding="utf-8") as fin, OUT_PATH.open("w", encoding="utf-8") as fout:
        for line in fin:
            if not line.strip():
                continue
            row = json.loads(line)
            # Support either {instruction,response} or {prompt,response}
            prompt = row.get("instruction") or row.get("prompt")
            resp   = row.get("response")
            if not isinstance(prompt, str) or not isinstance(resp, str):
                continue

            # Optional: wrap prompt to reduce distribution shift
            prompt = prompt.strip()
            resp = resp.strip()

            fout.write(json.dumps({"prompt": prompt, "response": resp}, ensure_ascii=False) + "\n")
            n += 1

    print(f"Wrote {n} SFT rows -> {OUT_PATH}")

if __name__ == "__main__":
    main()
