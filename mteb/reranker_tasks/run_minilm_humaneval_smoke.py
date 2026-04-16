#!/usr/bin/env python
"""Backward-compatible smoke-test entrypoint (delegates to codeswitch_eval.run_embedding)."""

from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from codeswitch_eval.run_embedding import main  # noqa: E402


if __name__ == "__main__":
    main()
