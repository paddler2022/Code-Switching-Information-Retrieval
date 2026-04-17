"""Filesystem layout helpers for the runnable eval bundle.

Simplified relative to the original package: because our CSR-L task classes
load data from HuggingFace (by ``language=`` kwarg) rather than from local
JSONL files, we no longer need ``FIXED_DATASET_ROOT`` or the fixed dataset
plumbing.

The package still resolves its own root via ``CODESWITCH_EVAL_ROOT`` for
parity; by default it points at the repo root (the parent of
``reranker_tasks/``).
"""

from __future__ import annotations

import os
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parent           # reranker_tasks/codeswitch_eval/
RUN_ROOT = PACKAGE_ROOT.parent                            # reranker_tasks/
REPO_ROOT = RUN_ROOT.parent                               # <repo>/
EVAL_ROOT = Path(os.environ.get("CODESWITCH_EVAL_ROOT", REPO_ROOT)).resolve()


def require_existing_path(path: Path, label: str) -> Path:
    """Return path if present, otherwise fail with a useful message."""
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")
    return path


def ensure_output_dir(path: str | Path) -> Path:
    output_dir = Path(path)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir
