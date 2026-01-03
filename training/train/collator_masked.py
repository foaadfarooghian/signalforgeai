from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List

import torch

@dataclass
class MaskedChatCollator:
    tokenizer: Any
    max_length: int = 2048

    def __call__(self, batch: List[Dict[str, str]]) -> Dict[str, torch.Tensor]:
        input_ids_list = []
        attn_list = []
        labels_list = []

        for ex in batch:
            prompt = ex["prompt"].strip()
            response = ex["response"].strip()

            # Prompt-only ids (this defines what to mask)
            prompt_ids = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                add_generation_prompt=True,
                return_tensors="pt",
            )[0]

            # Full conversation ids (what we train on)
            full_ids = self.tokenizer.apply_chat_template(
                [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": response},
                ],
                add_generation_prompt=False,
                return_tensors="pt",
            )[0]

            # Truncate (keep the end, but ensure prompt masking still valid)
            if full_ids.shape[0] > self.max_length:
                full_ids = full_ids[-self.max_length:]

            # Build attention
            attn = torch.ones_like(full_ids)

            # Labels = full_ids, but mask prompt portion
            labels = full_ids.clone()
            prompt_len = min(prompt_ids.shape[0], labels.shape[0])
            labels[:prompt_len] = -100

            input_ids_list.append(full_ids)
            attn_list.append(attn)
            labels_list.append(labels)

        # Pad to max in batch
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids_list,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id,
        )
        attention_mask = torch.nn.utils.rnn.pad_sequence(
            attn_list,
            batch_first=True,
            padding_value=0,
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels_list,
            batch_first=True,
            padding_value=-100,
        )

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
