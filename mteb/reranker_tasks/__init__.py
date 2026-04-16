"""reranker_tasks — ported from codeswitch_eval_code_package_20260415/newly_updated.

Provides the workflow layer (runner, scoring, task registry, embedding CLI,
two-stage cross-encoder reranking) on top of the task classes that already live
in mteb/. Task loading is rewritten to use our CSR-L classes with the
``language=`` kwarg instead of the legacy ``query_file=`` / ``instruction_file=``
JSONL loaders.
"""
