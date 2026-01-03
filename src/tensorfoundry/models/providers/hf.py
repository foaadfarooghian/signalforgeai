# tensorfoundry/models/providers/hf.py
from __future__ import annotations

import os
import time
from typing import Any, Dict, Optional, Tuple
from urllib.parse import parse_qs, urlsplit

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tensorfoundry.models.types import ModelMetrics, ModelOutput

try:
    from peft import PeftModel
except Exception:
    PeftModel = None


def _parse_hf_model_id(model_id: str) -> Tuple[str, Optional[str]]:
    """
    model_id forms:
      - hf:<base>
      - hf:<base>?adapter=/path/to/lora
    """
    assert model_id.startswith("hf:"), model_id
    raw = model_id[len("hf:") :]

    parts = urlsplit("x://" + raw)  # fake scheme so urlsplit parses query
    base = (parts.netloc + parts.path).lstrip("/")
    q = parse_qs(parts.query)
    adapter = q.get("adapter", [None])[0]
    return base, adapter


class HFProvider:
    def __init__(
        self,
        *,
        load_in_4bit: bool = False,
        max_new_tokens_default: int = 512,
        temperature_default: float = 0.2,
    ) -> None:
        self.load_in_4bit = load_in_4bit
        self.max_new_tokens_default = max_new_tokens_default
        self.temperature_default = temperature_default
        self._cache: Dict[Tuple[str, Optional[str]], Tuple[Any, Any]] = {}

    def _load(self, base_model: str, adapter: Optional[str]) -> Tuple[Any, Any]:
        key = (base_model, adapter)
        if key in self._cache:
            return self._cache[key]

        require_flash = os.getenv("TENSORFOUNDRY_HF_REQUIRE_FLASH", "0").lower() in {"1", "true", "yes"}
        tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model_kwargs: Dict[str, Any] = {}
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            device = os.getenv("TENSORFOUNDRY_HF_DEVICE", "cuda:0")
            if self.load_in_4bit:
                model_kwargs.update(dict(load_in_4bit=True))
            else:
                model_kwargs.update(dict(torch_dtype=torch.bfloat16))
        else:
            model_kwargs.update(dict(device_map={"": "cpu"}))

        if torch.cuda.is_available():
            attempts: list[tuple[str, Dict[str, Any], str]] = []
            full_gpu_kwargs = dict(model_kwargs, device_map={"": device})
            auto_kwargs = dict(model_kwargs, device_map="auto")
            attempts.append(("cuda_full_flash", dict(full_gpu_kwargs, attn_implementation="flash_attention_2"), "flash_attention_2"))
            attempts.append(("cuda_full", full_gpu_kwargs, "default"))
            attempts.append(("auto_flash", dict(auto_kwargs, attn_implementation="flash_attention_2"), "flash_attention_2"))
            attempts.append(("auto", auto_kwargs, "default"))
            if require_flash:
                attempts = [a for a in attempts if a[2] == "flash_attention_2"]
        else:
            attempts = [("cpu", model_kwargs, "default")]

        last_err: Optional[Exception] = None
        model: Any = None
        for label, kwargs, attn_impl in attempts:
            try:
                model = AutoModelForCausalLM.from_pretrained(base_model, **kwargs)
                setattr(model, "_tf_attn_impl", attn_impl)
                setattr(model, "_tf_load_label", label)
                if os.getenv("TENSORFOUNDRY_HF_LOG_DEVICE_MAP", "1") != "0":
                    cfg_attn = getattr(model.config, "attn_implementation", None) or getattr(model.config, "_attn_implementation", None)
                    device_map = getattr(model, "hf_device_map", None)
                    if device_map:
                        print(f"HFProvider: device_map={device_map} (load={label} attn={cfg_attn or attn_impl})")
                    else:
                        try:
                            dev = next(model.parameters()).device
                            print(f"HFProvider: device={dev} (load={label} attn={cfg_attn or attn_impl})")
                        except StopIteration:
                            print(f"HFProvider: device=unknown (load={label} attn={cfg_attn or attn_impl})")
                if require_flash:
                    cfg_attn = getattr(model.config, "attn_implementation", None) or getattr(model.config, "_attn_implementation", None)
                    if cfg_attn not in (None, "flash_attention_2"):
                        raise RuntimeError(f"HFProvider: expected flash_attention_2, got {cfg_attn!r}")
                break
            except Exception as exc:
                last_err = exc
                continue
        else:
            if last_err is not None:
                if require_flash:
                    raise RuntimeError("HFProvider: flash_attention_2 requested but no flash-compatible load succeeded.") from last_err
                raise last_err

        if model is None:
            raise RuntimeError("HFProvider: failed to load model.")

        model.eval()

        if adapter:
            if PeftModel is None:
                raise RuntimeError("peft is not installed but adapter= was provided.")
            model = PeftModel.from_pretrained(model, adapter)
            model.eval()

        self._cache[key] = (model, tokenizer)
        return model, tokenizer

    def generate(self, *, prompt: str, model_id: str, task_type: Optional[str] = None) -> ModelOutput:
        base, adapter = _parse_hf_model_id(model_id)
        model, tokenizer = self._load(base, adapter)

        t0 = time.time()

        # ✅ IMPORTANT for instruct models (Qwen etc.)
        messages = [{"role": "user", "content": prompt}]
        enc = tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
            add_generation_prompt=True,
            return_dict=True,
        )

        enc = {k: v.to(model.device) for k, v in enc.items()}

        max_new_tokens = 256 if "synth" in (task_type or "") else self.max_new_tokens_default
        temperature = self.temperature_default

        gen_kwargs = {
            "input_ids": enc["input_ids"],
            "attention_mask": enc.get("attention_mask"),
            "max_new_tokens": max_new_tokens,
            "pad_token_id": tokenizer.eos_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }

        do_sample = bool(self.temperature_default and self.temperature_default > 0.0)
        gen_kwargs["do_sample"] = do_sample
        if do_sample:
            gen_kwargs["temperature"] = temperature
            gen_kwargs["top_p"] = 0.95

        with torch.no_grad():
            out_ids = model.generate(**gen_kwargs)

        gen_ids = out_ids[0][enc["input_ids"].shape[-1]:]
        text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
        last = text.rfind("}")
        if last != -1:
            text = text[: last + 1]

        latency_ms = int((time.time() - t0) * 1000)

        input_tokens = int(enc["input_ids"].shape[-1])
        output_tokens = int(gen_ids.shape[-1])

        return ModelOutput(
            text=text,
            metrics=ModelMetrics(
                latency_ms=latency_ms,
                cost_usd=0.0,
                extra={
                    "provider": "hf",
                    "model_name": base,
                    "adapter": adapter,
                    "attn_impl": getattr(model.config, "attn_implementation", None)
                    or getattr(model.config, "_attn_implementation", None)
                    or getattr(model, "_tf_attn_impl", None),
                    "load_label": getattr(model, "_tf_load_label", None),
                    "usage": {
                        "input_tokens": input_tokens,
                        "output_tokens": output_tokens,
                        "total_tokens": input_tokens + output_tokens,
                    },
                    "task_type": task_type,
                },
            ),
        )
