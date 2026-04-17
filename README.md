# Code-Switching Information Retrieval

Benchmarks, analysis, and the limits of current retrievers on code-switched queries.

Originally a fork of [MTEB](https://github.com/embeddings-benchmark/mteb) extended with code-switching task variants and the accompanying evaluation pipelines used in *Code-Switching Information Retrieval: Benchmarks, Analysis, and the Limits of Current Retrievers*.

## What this repo does

Evaluates embedding and reranking models across three task families built on top of the MTEB framework:

| Family | Coverage | Description |
|---|---|---|
| **CS-MTEB** | 9 languages × 13 tasks | Code-switched variants where English queries/documents are mixed with a second language (zh / ja / de / es / ko / fr / it / pt / nl). Hosted on HuggingFace under `UTokyo-Yokoya-Lab/*_CS-MTEB`. |
| **CSR-L** | zh / ja × 6 tasks | A curated, fixed-dataset superset focused on retrieval + FollowIR, used as the primary reranker benchmark. Hosted under `UTokyo-Yokoya-Lab/*-CSR-L`. |
| **Original** | 13 English baselines | Non-CS counterparts (ArguAna, TRECCOVID, Touche2020, HumanEval, Core17/News21/Robust04 FollowIR, plus STS/classification/clustering/reranking). Serves as the reference point for measuring CS degradation. |

Two evaluation pipelines are provided:

- **Bi-Encoder** (`mteb/playground.py`) — dense retrieval / classification / clustering / reranking through the MTEB evaluator with injected instruction prompts (Qwen3 / E5 / etc.).
- **Crossencoder** (`mteb/reranker_tasks/`) — Stage 1 dense retrieval produces candidate sets; Stage 2 reranks with a cross-encoder (BGE / Jina / Qwen3-Reranker).

## Repository layout

```
.
├── README.md
├── mteb/                             # Modified MTEB source + entrypoints
│   ├── mteb/                         #   The MTEB package (patched)
│   │   └── tasks/...                 #   CS / CSR-L task classes live here
│   ├── playground.py                 #   Single-stage evaluation CLI
│   ├── smoke_test.py                 #   Offline sanity checks (no network)
│   ├── multiple_run.sh               #   Convenience: CS-MTEB across 9 langs
│   ├── reranker_tasks/               #   Two-stage (embed + rerank) pipeline
│   │   ├── RUN_PACKAGE.md            #     Full runbook for this pipeline
│   │   ├── codeswitch_env.sh         #     Env setup (PYTHONPATH, attn, ...)
│   │   ├── codeswitch_eval/          #     Runner package
│   │   ├── run_crossencoder_reranking.py
│   │   └── build_original_vs_codeswitching_table.py
│   └── Model_Align_Embedding/        #   Cross-lingual embedding alignment
└── scripts/                          # One-off maintenance scripts
    └── dedup_trec_covid_es.py        #   Dedup helper for one HF dataset config
```

## Environment setup

Tested on Python 3.12, torch is 2.9.1+cu13.00. However, in lower torch and cuda version you can also build a runable environment. 

### 1. Create conda env

```bash
conda create -n codeswitching python=3.12 -y
conda activate codeswitching
```

### 2. Install dependencies

From the repo root:

```bash
pip install -r requirements.txt
```

You can also prepare the environment step by step, first:
```bash
pip install mteb
```
Then search for the [Flash Attention v2.8.3](https://github.com/Dao-AILab/flash-attention/releases/tag/v2.8.3) that is suitable for your environment and wget it and pip install.
## How to run

### For Bi-Encoder mteb tasks (`mteb/playground.py`)

Runs MTEB-style evaluation on a chosen model. All three task families are supported. Below is an example on how to run a CS-MTEB zh_en tasks: 

```bash
cd mteb
python playground.py \
  --model_path <hf_model_name> \
  --batch_size <batch_size> \
  --tasks CS-MTEB_zh \
  --evaluation_output_dir <eval_results_dir> \
  --output_subfolder_name <subfolder_name>
```

Supported `--tasks` values:

| Value | What it runs |
|---|---|
| `CS-MTEB_{lang}` | All 13 CS-MTEB tasks for a single language (`lang` ∈ zh, ja, de, es, ko, fr, it, pt, nl) |
| `CSR-L_{lang}` | All 6 CSR-L tasks for `lang` ∈ zh, ja |
| `Original` | All 13 English original tasks |
| `Original_retrieval` / `Original_ir` / `Original_other` | Subsets of Original by task type |

Multiple values can be passed (e.g. `--tasks CS-MTEB_zh CSR-L_zh`).

The helper `mteb/multiple_run.sh` loops over all 9 CS-MTEB languages + 2 CSR-L languages for Qwen3-Embedding-0.6B.

### For crossencoder_tasks — two-stage embedding + cross-encoder rerank

Full runbook in `mteb/reranker_tasks/RUN_PACKAGE.md`. Summary:

```bash
cd mteb/reranker_tasks
source ./codeswitch_env.sh            # PYTHONPATH, attention impl, env vars

# Smoke: all-MiniLM on HumanEval (Orig + CSR-L zh/ja)
python -m codeswitch_eval.run_embedding --suite humaneval_smoke

# Stage 1: dense retrieval → predictions_{dir}
python run_crossencoder_reranking.py --stage 1 \
  --tasks Fixed_Chinese Fixed_R_Chinese \
  --predictions_dir ./predictions_qwen3_embedding_06b/chinese \
  --batch_size 4

# Stage 2: cross-encoder rerank
python run_crossencoder_reranking.py --stage 2 \
  --reranker qwen3-reranker-0.6b \
  --tasks Fixed_Chinese Fixed_R_Chinese \
  --predictions_dir ./predictions_qwen3_embedding_06b/chinese \
  --output_dir ./results_crossencoder_qwen3_06b/chinese \
  --top_k 100 --batch_size 1 --reranker_batch_size 1

# Summary table across Original vs CSR-L
python build_original_vs_codeswitching_table.py \
  --root ./results_crossencoder_qwen3_06b \
  --output ./retrieval_and_followir_original_vs_csrl.csv
```

Available rerankers: `bge-reranker-v2-m3`, `jina-reranker-v3`, `qwen3-reranker-{0.6b,4b,8b}`.

### Output layout

- `playground.py` writes MTEB-format JSON per task into
  `<evaluation_output_dir>/<YYYYMMDD>/<output_subfolder_name>/<model_name>/<TaskName>.json`.
- CrossEncoder tasks: writes Stage 1 predictions under `predictions_dir/` and Stage 2 final scores under `output_dir/`.

## License

See `LICENSE` and `NOTICE` (inherits from upstream MTEB where applicable).
