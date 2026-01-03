import json
from pathlib import Path
from datasets import load_dataset
from transformers import AutoTokenizer

BASE_MODEL = "Qwen/Qwen2.5-3B-Instruct"  # change if needed

IN_PATH  = Path("datasets/synth_train.sft.ready.jsonl")      # your prompt_full+response file
OUT_PATH = Path("datasets/synth_critic.chattext.jsonl") # output

def main():
    tok = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with IN_PATH.open("r", encoding="utf-8") as fin, OUT_PATH.open("w", encoding="utf-8") as fout:
        for line in fin:
            if not line.strip():
                continue
            row = json.loads(line)
            prompt = (row.get("prompt") or "").strip()
            resp   = (row.get("response") or "").strip()
            if not prompt or not resp:
                continue

            messages = [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": resp},
            ]

            text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            fout.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
            n += 1

    print(f"Wrote {n} rows -> {OUT_PATH}")

if __name__ == "__main__":
    main()
