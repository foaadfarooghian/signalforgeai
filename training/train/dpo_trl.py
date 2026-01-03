import os
from datasets import load_dataset
from unsloth import FastLanguageModel
from peft import PeftModel
from trl import DPOTrainer, DPOConfig

BASE_MODEL = os.environ.get("BASE_MODEL", "YOUR_HF_BASE_MODEL_ID_HERE")
SFT_DIR    = os.environ.get("SFT_DIR", "artifacts/synth_sft_lora")
OUT_DIR    = os.environ.get("OUT_DIR", "artifacts/synth_dpo_lora")
DATA_PATH  = os.environ.get("DPO_DATASET", "datasets/synth_train.dpo.ready.jsonl")

def main():
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL,
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=True,
    )

    # Load the SFT adapter onto the base model
    model = PeftModel.from_pretrained(model, SFT_DIR, is_trainable=True)

    ds = load_dataset("json", data_files=DATA_PATH, split="train")

    cfg = DPOConfig(
        output_dir=OUT_DIR,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=16,  # effective batch 32
        learning_rate=1e-5,
        num_train_epochs=1,
        bf16=True,
        logging_steps=10,
        save_steps=200,
        save_total_limit=2,
        optim="paged_adamw_8bit",
        report_to="none",
        beta=0.1,  # DPO strength
        max_length=2048,
        max_prompt_length=1024,
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=None,  # TRL will create ref from initial weights unless you pass one
        args=cfg,
        train_dataset=ds,
        tokenizer=tokenizer,
    )

    trainer.train()
    trainer.save_model(OUT_DIR)
    tokenizer.save_pretrained(OUT_DIR)

    print("DPO done:", OUT_DIR)

if __name__ == "__main__":
    main()
