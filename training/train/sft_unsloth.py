import os
from datasets import load_dataset
from unsloth import FastLanguageModel
from unsloth.chat_templates import train_on_responses_only
from trl import SFTTrainer
from transformers import AutoTokenizer, TrainingArguments
import psutil
import builtins
BASE_MODEL = os.environ.get("BASE_MODEL", "YOUR_HF_BASE_MODEL_ID_HERE")  # e.g. ministral 3B HF id
OUT_DIR    = os.environ.get("OUT_DIR", "artifacts/synth_sft_lora")

builtins.psutil = psutil

_FORMAT_TOKENIZER = None


def _get_format_tokenizer():
    global _FORMAT_TOKENIZER
    if _FORMAT_TOKENIZER is None:
        tok = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        _FORMAT_TOKENIZER = tok
    return _FORMAT_TOKENIZER


def format_fn(examples):
    prompts = examples["prompt"]
    responses = examples["response"]
    tokenizer = _get_format_tokenizer()

    pairs = zip(prompts, responses) if isinstance(prompts, list) else [(prompts, responses)]
    texts = []
    for p, r in pairs:
        messages = [
            {"role": "user", "content": p.strip()},
            {"role": "assistant", "content": r.strip()},
        ]
        texts.append(
            tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
        )
    return texts

def main():
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = BASE_MODEL,
        max_seq_length = 2048,
        dtype = None,
        load_in_4bit = True,   # QLoRA
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
        lora_alpha=16,
        lora_dropout=0.0,
        bias="none",
        use_gradient_checkpointing=True,
        random_state=42,
    )

    data_path = os.environ.get("SFT_DATASET", "datasets/synth_train.sft.ready.jsonl")
    ds = load_dataset("json", data_files=data_path, split="train")

    args = TrainingArguments(
        output_dir=OUT_DIR,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=8,   # effective batch 32
        learning_rate=5e-5,
        num_train_epochs=2,
        warmup_ratio=0.03,
        logging_steps=10,
        save_steps=200,
        save_total_limit=2,
        bf16=True,
        fp16=False,
        optim="paged_adamw_8bit",
        weight_decay=0.0,
        lr_scheduler_type="cosine",
        max_grad_norm=1.0,
        report_to="none",
    )
    # Avoid Unsloth's psutil default branch by setting a fixed dataset worker count.
    args.dataset_num_proc = int(os.environ.get("DATASET_NUM_PROC", "1"))

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=ds,
        args=args,
        formatting_func=format_fn,
        dataset_text_field="text",
        max_seq_length=2048,
        packing=False
    )
    try:
        trainer = train_on_responses_only(trainer, num_proc=1)
    except ValueError:
        instruction_part = os.environ.get("INSTRUCTION_PART", "<|im_start|>user\n")
        response_part = os.environ.get("RESPONSE_PART", "<|im_start|>assistant\n")
        trainer = train_on_responses_only(
            trainer,
            instruction_part=instruction_part,
            response_part=response_part,
            num_proc=1,
        )

    trainer.train()
    trainer.save_model(OUT_DIR)
    tokenizer.save_pretrained(OUT_DIR)

    print("SFT done:", OUT_DIR)

if __name__ == "__main__":
    main()
