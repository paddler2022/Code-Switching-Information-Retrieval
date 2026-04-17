# crossencoder_tasks — Runbook

Port of `codeswitch_eval_code_package_20260415/newly_updated/` into this repo.
The workflow layer is the same; the only substantive change is task loading:

- **Their code** instantiated 6 `*CodeSwitching` task classes with
  `query_file=` / `instruction_file=` kwargs pointing at local JSONL files
  under `CodeSwitching_Dataset_fixed/`.
- **Our code** uses the 6 `*CSRL` classes (renamed, and a superset is
  additionally exposed under `*CodeSwitching` for 9-language CS-MTEB — not
  used here) with a single `language=` kwarg; data is pulled from HuggingFace.

Task groups (`TASK_REGISTRY`) keep the original labels for parity:

| Group              | Count | Contents                                                        |
| ------------------ | ----- | --------------------------------------------------------------- |
| `Original`         | 4     | Core17 / News21 / Robust04 IR + `HumanEvalRetrieval`            |
| `OG_Retrieval`     | 2     | `TRECCOVID` + `Touche2020v3Retrieval`                           |
| `Fixed_Chinese`    | 4     | `*CSRL(language="zh")` — HumanEval + Core17/News21/Robust04 IR  |
| `Fixed_Japanese`   | 4     | same four tasks, `language="ja"`                                |
| `Fixed_R_Chinese`  | 2     | `TRECCOVIDCSRL(zh)` + `Touche2020v3RetrievalCSRL(zh)`           |
| `Fixed_R_Japanese` | 2     | same two tasks, `language="ja"`                                 |

## Environment

```bash
cd <repo_root>/reranker_tasks
source ./codeswitch_env.sh       # adds reranker_tasks/ + repo root to PYTHONPATH
```

The env script is self-locating, so moving the repo folder does not break it.
`codeswitch_env.sh` sets:

- `PYTHONNOUSERSITE=1`
- `PYTHONPATH=reranker_tasks/:<repo_root>/:...`
- `CODESWITCH_ATTN_IMPLEMENTATION=sdpa` (safer than `flash_attention_2` on old GPUs)
- `CODESWITCH_EVAL_ROOT=<repo_root>` (override to relocate)

## Smallest smoke test

`all-MiniLM-L12-v2` on HumanEval (Original + zh CSR-L + ja CSR-L), ~3 tasks.

```bash
cd <repo_root>/reranker_tasks
python -m codeswitch_eval.run_embedding --suite humaneval_smoke
```

Legacy entrypoint works the same:

```bash
python run_minilm_humaneval_smoke.py
```

Output goes to `smoke_minilm_humaneval/results_summary.json`. Expected scores
for `all-MiniLM-L12-v2`:

```text
HumanEval Orig   ≈ 70
HumanEval CSR-L  ≈ 62   (mean of zh and ja)
```

## Custom embedding runs

Only HumanEval across Original / zh / ja:

```bash
python -m codeswitch_eval.run_embedding \
  --task_groups Original Fixed_Chinese Fixed_Japanese \
  --only_task_name HumanEval \
  --model sentence-transformers/all-MiniLM-L12-v2 \
  --batch_size 64 \
  --output_dir results_minilm_humaneval
```

Qwen3 embedding through the MTEB backend:

```bash
python -m codeswitch_eval.run_embedding \
  --task_groups Original Fixed_Chinese Fixed_Japanese OG_Retrieval Fixed_R_Chinese Fixed_R_Japanese \
  --model Qwen/Qwen3-Embedding-0.6B \
  --backend mteb \
  --batch_size 4 \
  --output_dir results_qwen3_embedding_06b
```

Keep `--overwrite_strategy always` (the default) when running multiple
language variants together. `validate_cache_safety()` will raise if it detects
same-named variants on a non-`always` strategy.

## Cross-encoder reranking

Run language variants in separate prediction directories to avoid any chance
of overwrite, even though our CSR-L task names already carry the `_{lang}`
suffix.

Stage 1 (dense retrieval with Qwen3-Embedding-0.6B) — three separate dirs:

```bash
python run_crossencoder_reranking.py \
  --stage 1 \
  --tasks Original OG_Retrieval \
  --predictions_dir ./predictions_qwen3_embedding_06b/original \
  --batch_size 4

python run_crossencoder_reranking.py \
  --stage 1 \
  --tasks Fixed_Chinese Fixed_R_Chinese \
  --predictions_dir ./predictions_qwen3_embedding_06b/chinese \
  --batch_size 4

python run_crossencoder_reranking.py \
  --stage 1 \
  --tasks Fixed_Japanese Fixed_R_Japanese \
  --predictions_dir ./predictions_qwen3_embedding_06b/japanese \
  --batch_size 4
```

Stage 2 (cross-encoder reranking) for the Chinese split:

```bash
python run_crossencoder_reranking.py \
  --stage 2 \
  --reranker qwen3-reranker-0.6b \
  --tasks Fixed_Chinese Fixed_R_Chinese \
  --predictions_dir ./predictions_qwen3_embedding_06b/chinese \
  --output_dir ./results_crossencoder_qwen3_06b/chinese \
  --top_k 100 \
  --batch_size 1 \
  --reranker_batch_size 1
```

Available rerankers: `bge-reranker-v2-m3`, `jina-reranker-v3`,
`qwen3-reranker-{0.6b,4b,8b}`.

## Query instruction rule

Queries are evaluated with the task instruction prepended.

- BGE / Jina cross-encoder: `Instruct: <instruction>\nQuery: <query>`
- Qwen3 reranker (chat pair format):
  `<Instruct>: <instruction>\n<Query>: <query>\n<Document>: <document>`

Instruction source order:

1. Per-query `instruction` column from the MTEB batch, if present.
2. `codeswitch_eval.models.QWEN3_PROMPTS[task_metadata.name]`
   (CSR-L entries are expanded with `_{lang}` suffix).
3. `DEFAULT_INSTRUCTION`.

## Result table

```bash
python build_original_vs_codeswitching_table.py \
  --root ./results_crossencoder_qwen3_06b \
  --output ./retrieval_and_followir_original_vs_csrl.csv
```

One row per model, one column for Original + one column each for `cs_zh` /
`cs_ja` per task, plus FollowIR macro-average p-MRR per language.

## Files

```text
reranker_tasks/
├── RUN_PACKAGE.md                         (this file)
├── codeswitch_env.sh                      (minimal env setup)
├── codeswitch_eval/
│   ├── __init__.py
│   ├── paths.py                           (PACKAGE_ROOT / RUN_ROOT / REPO_ROOT / EVAL_ROOT)
│   ├── models.py                          (QWEN3_PROMPTS + load helpers)
│   ├── tasks.py                           (TASK_REGISTRY with 6 groups, HuggingFace loading)
│   ├── scoring.py                         (score_to_percent, mean/compare, write_json)
│   ├── runner.py                          (EmbeddingRunConfig, validate_cache_safety)
│   └── run_embedding.py                   (CLI: --suite humaneval_smoke | custom)
├── run_crossencoder_reranking.py          (Stage 1 + Stage 2)
├── run_two_stage_eval.py                  (backward-compat re-exports)
├── run_minilm_humaneval_smoke.py          (backward-compat shim)
└── build_original_vs_codeswitching_table.py
```
