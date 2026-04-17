"""Backward-compatible imports for two-stage retrieval/reranking scripts.

Re-exports the same symbols the original
``newly_updated/run_two_stage_eval.py`` exposed, so any caller that did
``from run_two_stage_eval import smart_load_model`` keeps working.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Make ``codeswitch_eval`` importable when this module is executed / imported
# from outside the ``reranker_tasks/`` directory.
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from codeswitch_eval.models import (  # noqa: E402
    QWEN3_PROMPTS,
    load_model_ST,
    qwen3_instruction_template,
    qwen_model_kwargs,
    smart_load_model,
)
from codeswitch_eval.paths import EVAL_ROOT as REPO_ROOT  # noqa: E402
from codeswitch_eval.tasks import TASK_REGISTRY, resolve_tasks  # noqa: E402

ATTN_IMPLEMENTATION = os.environ.get("CODESWITCH_ATTN_IMPLEMENTATION", "sdpa").strip()

__all__ = [
    "ATTN_IMPLEMENTATION",
    "QWEN3_PROMPTS",
    "REPO_ROOT",
    "TASK_REGISTRY",
    "load_model_ST",
    "qwen3_instruction_template",
    "qwen_model_kwargs",
    "resolve_tasks",
    "smart_load_model",
]
