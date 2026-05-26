import os
import datasets
from unsloth import FastLanguageModel
from unsloth.chat_templates import train_on_responses_only
from trl import SFTTrainer
from transformers import AutoTokenizer, TrainingArguments
import psutil
import builtins

BASE_MODEL = os.environ.get("BASE_MODEL", "YOUR_HF_BASE_MODEL_ID_HERE")  # e.g. ministral 3B HF id
OUT_DIR = os.environ.get("OUT_DIR", "artifacts/synth_sft_lora")
SMOKE = os.environ.get("SIGNALFORGEAI_TRAINING_SMOKE", "0").lower() in {"1", "true", "yes"}
MAX_SEQ_LENGTH = int(
    os.environ.get("SIGNALFORGEAI_TRAINING_MAX_SEQ_LENGTH", "512" if SMOKE else "2048")
)
BATCH_SIZE = int(os.environ.get("SIGNALFORGEAI_TRAINING_BATCH_SIZE", "1" if SMOKE else "4"))
GRADIENT_ACCUMULATION_STEPS = int(
    os.environ.get("SIGNALFORGEAI_TRAINING_GRADIENT_ACCUMULATION_STEPS", "1" if SMOKE else "8")
)
SEED = int(os.environ.get("SIGNALFORGEAI_TRAINING_SEED", "42"))

setattr(builtins, "psutil", psutil)
load_dataset = getattr(datasets, "load_dataset")

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


def _precision_flags():
    precision = os.environ.get("SIGNALFORGEAI_TRAINING_PRECISION", "auto").strip().lower()
    if precision == "bf16":
        return True, False
    if precision == "fp16":
        return False, True
    if precision == "none":
        return False, False
    try:
        import torch

        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            return True, False
        if torch.cuda.is_available():
            return False, True
    except Exception:
        return False, False
    return False, False


def main():
    bf16, fp16 = _precision_flags()
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,
        load_in_4bit=True,  # QLoRA
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=16,
        lora_dropout=0.0,
        bias="none",
        use_gradient_checkpointing=True,
        random_state=SEED,
    )

    data_path = os.environ.get("SFT_DATASET", "datasets/synth_train.sft.ready.jsonl")
    ds = load_dataset("json", data_files=data_path, split="train")

    args = TrainingArguments(
        output_dir=OUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=5e-5,
        num_train_epochs=1 if SMOKE else 2,
        warmup_ratio=0.03,
        logging_steps=1 if SMOKE else 10,
        save_steps=1 if SMOKE else 200,
        save_total_limit=2,
        bf16=bf16,
        fp16=fp16,
        optim="paged_adamw_8bit",
        weight_decay=0.0,
        lr_scheduler_type="cosine",
        max_grad_norm=1.0,
        report_to="none",
        seed=SEED,
        data_seed=SEED,
        max_steps=int(os.environ.get("MAX_STEPS", "0")) if os.environ.get("MAX_STEPS") else -1,
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
        max_seq_length=MAX_SEQ_LENGTH,
        packing=False,
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
