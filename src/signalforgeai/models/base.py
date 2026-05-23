"""Model provider protocol used by agents and evaluation."""
from __future__ import annotations
from typing import Protocol, Optional
from .types import ModelOutput

class ModelProvider(Protocol):
    """Protocol for model backends that return structured output + metrics."""
    def generate(
        self,
        *,
        prompt: str,
        model_id: str,
        task_type: Optional[str] = None,
    ) -> ModelOutput:
        """Generate a model response for a prompt and task type."""
        ...
