"""Reusable helpers for the CodeSwitching MTEB evaluation bundle (local port).

Kept API-compatible with the original
``codeswitch_eval_code_package_20260415/newly_updated/codeswitch_eval`` so the
legacy shims ``run_two_stage_eval`` / ``run_minilm_humaneval_smoke`` continue to
work unchanged.
"""

from .models import (
    QWEN3_PROMPTS,
    load_embedding_model,
    load_model_ST,
    qwen3_instruction_template,
    qwen_model_kwargs,
    smart_load_model,
)
from .tasks import TASK_REGISTRY, available_task_groups, resolve_tasks

__all__ = [
    "QWEN3_PROMPTS",
    "TASK_REGISTRY",
    "available_task_groups",
    "load_embedding_model",
    "load_model_ST",
    "qwen3_instruction_template",
    "qwen_model_kwargs",
    "resolve_tasks",
    "smart_load_model",
]
