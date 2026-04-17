#!/usr/bin/env bash
# Environment setup for reranker_tasks (ported, trimmed).
#
# Unlike the original cluster-specific script, this version only:
#   1. Adds reranker_tasks/ and the repo root to PYTHONPATH so local imports
#      of `codeswitch_eval.*` and the repo-level `mteb/` package work from
#      any cwd.
#   2. Sets CODESWITCH_ATTN_IMPLEMENTATION=sdpa by default (safer than
#      flash_attention_2 on older GPUs).
#   3. Leaves HF_HOME alone unless the caller already set it.
#
# Source from inside `reranker_tasks/`:
#     source ./codeswitch_env.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

export PYTHONNOUSERSITE=1
export PYTHONPATH="${SCRIPT_DIR}:${REPO_ROOT}:${PYTHONPATH:-}"
export CODESWITCH_ATTN_IMPLEMENTATION="${CODESWITCH_ATTN_IMPLEMENTATION:-sdpa}"
export CODESWITCH_EVAL_ROOT="${CODESWITCH_EVAL_ROOT:-${REPO_ROOT}}"

echo "reranker_tasks env ready"
echo "  SCRIPT_DIR=${SCRIPT_DIR}"
echo "  REPO_ROOT=${REPO_ROOT}"
echo "  CODESWITCH_ATTN_IMPLEMENTATION=${CODESWITCH_ATTN_IMPLEMENTATION}"
