from __future__ import annotations
from typing import Protocol, Optional
from .types import ModelOutput

class ModelProvider(Protocol):
    def generate(
        self,
        *,
        prompt: str,
        model_id: str,
        task_type: Optional[str] = None,
    ) -> ModelOutput:
        ...