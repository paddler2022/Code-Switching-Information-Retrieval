"""
Cross-Encoder Reranking Evaluation Script (Two-Stage).

Stage 1: Use Qwen3-Embedding-0.6B (or any embedding model) for full corpus
         retrieval and save predictions.
Stage 2: Use a cross-encoder reranker (BGE / Jina / Qwen3) to rerank the
         top-k documents from Stage 1.

Ported from codeswitch_eval_code_package_20260415/newly_updated with imports
rewritten to the local ``codeswitch_eval`` package.

Usage:
    python run_crossencoder_reranking.py --stage 1 --tasks Original Fixed_Chinese

    python run_crossencoder_reranking.py --stage 2 \\
        --reranker qwen3-reranker-0.6b \\
        --tasks Fixed_Chinese Fixed_R_Chinese \\
        --predictions_dir ./predictions/chinese \\
        --output_dir ./results_crossencoder/chinese \\
        --top_k 100

    python run_crossencoder_reranking.py --stage both --reranker bge-reranker-v2-m3 \\
        --tasks Original
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any

# Ensure ``codeswitch_eval`` is importable regardless of cwd.
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import torch
from torch.utils.data import DataLoader

import mteb
from mteb.abstasks.task_metadata import TaskMetadata
from mteb.models.model_meta import ModelMeta
from mteb.types import Array, BatchedInput, PromptType

from codeswitch_eval.models import (
    DEFAULT_INSTRUCTION,
    QWEN3_PROMPTS,
    qwen3_instruction_template,
    qwen_model_kwargs,
    smart_load_model,
)
from codeswitch_eval.tasks import TASK_REGISTRY, resolve_tasks

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")


# The module-level instruction table for cross-encoder prompts. Keys are
# runtime ``task_metadata.name``; see ``codeswitch_eval.models.QWEN3_PROMPTS``
# for how CSR-L tasks get the ``_{lang}`` suffix.
TASK_INSTRUCTIONS: dict[str, str] = dict(QWEN3_PROMPTS)


def _as_list(value: Any) -> list:
    if isinstance(value, str):
        return [value]
    return list(value)


def _default_instruction(task_metadata: TaskMetadata) -> str:
    return TASK_INSTRUCTIONS.get(task_metadata.name, DEFAULT_INSTRUCTION)


def _extract_queries_and_instructions(
    inputs1: DataLoader[BatchedInput],
    task_metadata: TaskMetadata,
) -> tuple[list[str], list[str]]:
    """Extract raw query text and per-query instruction from MTEB batches."""
    queries: list[str] = []
    instructions: list[str] = []
    fallback_instruction = _default_instruction(task_metadata)

    for batch in inputs1:
        batch_queries = _as_list(batch["query"]) if "query" in batch else _as_list(batch["text"])
        batch_texts = _as_list(batch["text"]) if "text" in batch else []
        batch_instructions = (
            _as_list(batch["instruction"])
            if "instruction" in batch
            else [fallback_instruction] * len(batch_queries)
        )

        for idx, query_text in enumerate(batch_queries):
            if (query_text is None or query_text == "") and idx < len(batch_texts):
                query_text = batch_texts[idx]

            query_instruction = (
                batch_instructions[idx]
                if idx < len(batch_instructions)
                else fallback_instruction
            )
            if not isinstance(query_instruction, str) or not query_instruction.strip():
                query_instruction = fallback_instruction

            queries.append(query_text)
            instructions.append(query_instruction)

    return queries, instructions


def _extract_instruction_prefixed_queries(
    inputs1: DataLoader[BatchedInput],
    task_metadata: TaskMetadata,
) -> list[str]:
    queries, instructions = _extract_queries_and_instructions(inputs1, task_metadata)
    return [
        qwen3_instruction_template(instruction, query)
        for query, instruction in zip(queries, instructions)
    ]


# ========================================================================
# Cross-Encoder Reranker Wrappers
# ========================================================================

class BGERerankerV2M3:
    """BGE Reranker v2-m3 wrapper implementing CrossEncoderProtocol."""

    def __init__(self, batch_size: int = 32):
        from FlagEmbedding import FlagReranker
        self.model = FlagReranker("BAAI/bge-reranker-v2-m3", use_fp16=True)
        self.batch_size = batch_size
        self._mteb_model_meta = ModelMeta(
            loader=None,
            name="BAAI/bge-reranker-v2-m3",
            revision="953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e",
            release_date="2024-06-24",
            languages=None,
            open_weights=True,
            framework=["PyTorch"],
            similarity_fn_name=None,
            n_parameters=None,
            memory_usage_mb=None,
            max_tokens=None,
            embed_dim=None,
            license=None,
            public_training_code=None,
            public_training_data=None,
            use_instructions=None,
            training_datasets=None,
            is_cross_encoder=True,
        )

    @property
    def mteb_model_meta(self) -> ModelMeta:
        return self._mteb_model_meta

    @torch.inference_mode()
    def predict(
        self,
        inputs1: DataLoader[BatchedInput],
        inputs2: DataLoader[BatchedInput],
        *,
        task_metadata: TaskMetadata,
        hf_split: str,
        hf_subset: str,
        prompt_type: PromptType | None = None,
        **kwargs: Any,
    ) -> Array:
        queries = _extract_instruction_prefixed_queries(inputs1, task_metadata)
        passages = [text for batch in inputs2 for text in batch["text"]]
        if len(queries) != len(passages):
            raise ValueError(
                f"Query/document pair mismatch: {len(queries)} queries vs {len(passages)} documents."
            )

        sentence_pairs = list(zip(queries, passages))
        scores = self.model.compute_score(sentence_pairs, normalize=True)
        if isinstance(scores, float):
            scores = [scores]
        return scores


class JinaRerankerV3:
    """Jina Reranker v3 wrapper implementing CrossEncoderProtocol.

    jina-reranker-v3 uses AutoModel with trust_remote_code and provides a
    ``.rerank()`` method. We adapt it to ``CrossEncoderProtocol.predict()`` by
    scoring (query, doc) pairs.
    """

    def __init__(self, batch_size: int = 32):
        from transformers import AutoModel, AutoConfig
        config = AutoConfig.from_pretrained("jinaai/jina-reranker-v3", trust_remote_code=True)
        config.tie_word_embeddings = False
        self.model = AutoModel.from_pretrained(
            "jinaai/jina-reranker-v3",
            config=config,
            torch_dtype="auto",
            trust_remote_code=True,
        )
        self.model.eval()
        if torch.cuda.is_available():
            self.model = self.model.to("cuda")
        self.batch_size = batch_size
        self._mteb_model_meta = ModelMeta(
            loader=None,
            name="jinaai/jina-reranker-v3",
            revision=None,
            release_date="2025-01-01",
            languages=None,
            open_weights=True,
            framework=["PyTorch"],
            similarity_fn_name=None,
            n_parameters=None,
            memory_usage_mb=None,
            max_tokens=None,
            embed_dim=None,
            license=None,
            public_training_code=None,
            public_training_data=None,
            use_instructions=None,
            training_datasets=None,
            is_cross_encoder=True,
        )

    @property
    def mteb_model_meta(self) -> ModelMeta:
        return self._mteb_model_meta

    @torch.inference_mode()
    def predict(
        self,
        inputs1: DataLoader[BatchedInput],
        inputs2: DataLoader[BatchedInput],
        *,
        task_metadata: TaskMetadata,
        hf_split: str,
        hf_subset: str,
        prompt_type: PromptType | None = None,
        **kwargs: Any,
    ) -> Array:
        queries = _extract_instruction_prefixed_queries(inputs1, task_metadata)
        passages = [text for batch in inputs2 for text in batch["text"]]
        if len(queries) != len(passages):
            raise ValueError(
                f"Query/document pair mismatch: {len(queries)} queries vs {len(passages)} documents."
            )

        from collections import OrderedDict
        query_to_indices: OrderedDict[str, list[int]] = OrderedDict()
        for i, q in enumerate(queries):
            query_to_indices.setdefault(q, []).append(i)

        all_scores = [0.0] * len(queries)
        for query, indices in query_to_indices.items():
            docs = [passages[i] for i in indices]
            results = self.model.rerank(query, docs)
            index_to_score = {r["index"]: r["relevance_score"] for r in results}
            for local_idx, global_idx in enumerate(indices):
                all_scores[global_idx] = index_to_score[local_idx]

        return all_scores


class Qwen3Reranker:
    """Qwen3-Reranker wrapper implementing CrossEncoderProtocol.

    Uses AutoModelForCausalLM with yes/no token log-probability scoring.
    Works for 0.6B / 4B / 8B variants.
    """

    def __init__(self, model_name: str = "Qwen/Qwen3-Reranker-0.6B", batch_size: int = 32):
        from transformers import AutoTokenizer, AutoModelForCausalLM

        self.model_name = model_name
        self.batch_size = batch_size

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            **qwen_model_kwargs(torch.float16),
        ).cuda().eval()

        self.token_true_id = self.tokenizer.convert_tokens_to_ids("yes")
        self.token_false_id = self.tokenizer.convert_tokens_to_ids("no")
        self.max_length = 8192

        prefix = (
            '<|im_start|>system\n'
            'Judge whether the Document meets the requirements based on the Query and the Instruct provided. '
            'Note that the answer can only be "yes" or "no".<|im_end|>\n'
            '<|im_start|>user\n'
        )
        suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        self.prefix_tokens = self.tokenizer.encode(prefix, add_special_tokens=False)
        self.suffix_tokens = self.tokenizer.encode(suffix, add_special_tokens=False)

        self._mteb_model_meta = ModelMeta(
            loader=None,
            name=model_name,
            revision=None,
            release_date="2025-01-01",
            languages=None,
            open_weights=True,
            framework=["PyTorch"],
            similarity_fn_name=None,
            n_parameters=None,
            memory_usage_mb=None,
            max_tokens=None,
            embed_dim=None,
            license=None,
            public_training_code=None,
            public_training_data=None,
            use_instructions=None,
            training_datasets=None,
            is_cross_encoder=True,
        )

    @property
    def mteb_model_meta(self) -> ModelMeta:
        return self._mteb_model_meta

    def _format_pair(self, instruction: str, query: str, document: str) -> str:
        return f"<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {document}"

    def _process_inputs(self, texts: list[str]) -> dict:
        inputs = self.tokenizer(
            texts,
            padding=False,
            truncation="longest_first",
            return_attention_mask=False,
            max_length=self.max_length - len(self.prefix_tokens) - len(self.suffix_tokens),
        )
        for i, ids in enumerate(inputs["input_ids"]):
            inputs["input_ids"][i] = self.prefix_tokens + ids + self.suffix_tokens
        inputs = self.tokenizer.pad(inputs, padding=True, return_tensors="pt", max_length=self.max_length)
        for key in inputs:
            inputs[key] = inputs[key].to(self.model.device)
        return inputs

    @torch.inference_mode()
    def predict(
        self,
        inputs1: DataLoader[BatchedInput],
        inputs2: DataLoader[BatchedInput],
        *,
        task_metadata: TaskMetadata,
        hf_split: str,
        hf_subset: str,
        prompt_type: PromptType | None = None,
        **kwargs: Any,
    ) -> Array:
        queries: list[str] = []
        query_instructions: list[str | None] = []
        for batch in inputs1:
            batch_queries = list(batch["query"]) if "query" in batch else list(batch["text"])
            batch_instructions = (
                list(batch["instruction"])
                if "instruction" in batch
                else [None] * len(batch_queries)
            )
            for idx, (query_text, query_instruction) in enumerate(zip(batch_queries, batch_instructions)):
                if (query_text is None or query_text == "") and "text" in batch:
                    query_text = batch["text"][idx]
                queries.append(query_text)
                query_instructions.append(query_instruction)
        passages = [text for batch in inputs2 for text in batch["text"]]
        if len(queries) != len(passages):
            raise ValueError(
                f"Query/document pair mismatch: {len(queries)} queries vs {len(passages)} documents."
            )

        default_instruction = TASK_INSTRUCTIONS.get(task_metadata.name, DEFAULT_INSTRUCTION)

        texts: list[str] = []
        for query_text, query_instruction, document_text in zip(queries, query_instructions, passages):
            instruction = (
                query_instruction
                if isinstance(query_instruction, str) and query_instruction.strip()
                else default_instruction
            )
            texts.append(self._format_pair(instruction, query_text, document_text))

        all_scores: list[float] = []
        for start in range(0, len(texts), self.batch_size):
            batch_texts = texts[start:start + self.batch_size]
            inputs = self._process_inputs(batch_texts)
            logits = self.model(**inputs).logits[:, -1, :]
            true_logits = logits[:, self.token_true_id]
            false_logits = logits[:, self.token_false_id]
            stacked = torch.stack([false_logits, true_logits], dim=1)
            log_probs = torch.nn.functional.log_softmax(stacked, dim=1)
            scores = log_probs[:, 1].exp().cpu().tolist()
            all_scores.extend(scores)

        return all_scores


# ========================================================================
# Reranker registry
# ========================================================================

RERANKER_REGISTRY = {
    "bge-reranker-v2-m3": lambda bs: BGERerankerV2M3(batch_size=bs),
    "jina-reranker-v3": lambda bs: JinaRerankerV3(batch_size=bs),
    "qwen3-reranker-0.6b": lambda bs: Qwen3Reranker("Qwen/Qwen3-Reranker-0.6B", batch_size=bs),
    "qwen3-reranker-4b": lambda bs: Qwen3Reranker("Qwen/Qwen3-Reranker-4B", batch_size=bs),
    "qwen3-reranker-8b": lambda bs: Qwen3Reranker("Qwen/Qwen3-Reranker-8B", batch_size=bs),
}


def _validate_unique_task_names(tasks: list[Any], context: str) -> None:
    """Prevent same-named variants from sharing cache/prediction files."""
    task_name_counts = Counter(task.metadata.name for task in tasks)
    duplicate_names = sorted(name for name, count in task_name_counts.items() if count > 1)
    if not duplicate_names:
        return

    raise ValueError(
        f"Unsafe duplicate task names for {context}: {duplicate_names}. "
        "Run Chinese and Japanese groups in separate commands with separate "
        "predictions/output directories. MTEB prediction/result filenames are keyed by "
        "task name, so same-named language variants can overwrite or reuse each other."
    )


# ========================================================================
# Stage 1: Full corpus retrieval
# ========================================================================

def run_stage1(retriever_model, task_names: list[str], predictions_dir: str, batch_size: int):
    """Full corpus dense retrieval with the retriever model."""
    predictions_path = Path(predictions_dir)
    predictions_path.mkdir(parents=True, exist_ok=True)

    print("\nResolving tasks for Stage 1...")
    tasks = resolve_tasks(task_names)

    if not tasks:
        print("[WARN] No tasks to run!")
        return
    _validate_unique_task_names(tasks, "Stage 1 prediction writing")

    for task in tasks:
        task_name = task.metadata.name
        print(f"\n{'=' * 60}")
        print(f"[Stage 1] Full retrieval: {task_name}")
        print(f"{'=' * 60}")

        task.load_data()
        task.convert_v1_dataset_format_to_v2()

        # Clear top_ranked to force full corpus search
        for subset in task.dataset:
            for split in task.dataset[subset]:
                task.dataset[subset][split]["top_ranked"] = None
                print(f"  Cleared top_ranked for {subset}/{split}")

        result = mteb.evaluate(
            retriever_model,
            task,
            prediction_folder=predictions_path,
            encode_kwargs={"batch_size": batch_size, "show_progress_bar": True},
            overwrite_strategy="always",
        )

        pred_file = task._predictions_path(predictions_path)
        if pred_file.exists():
            print(f"  [OK] Predictions saved: {pred_file}")
        else:
            print(f"  [WARN] Prediction file not found: {pred_file}")

        for tr in result.task_results:
            print(f"  Stage 1 scores ({tr.task_name}): {tr.get_score()}")


# ========================================================================
# Stage 2: Cross-encoder reranking
# ========================================================================

def run_stage2(
    reranker,
    reranker_name: str,
    task_names: list[str],
    predictions_dir: str,
    output_dir: str,
    top_k: int,
    batch_size: int,
    stage2_predictions_dir: str | None = None,
):
    """Cross-encoder reranking on top-k documents from Stage 1."""
    print("\nResolving tasks for Stage 2...")
    tasks = resolve_tasks(task_names)

    if not tasks:
        print("[WARN] No tasks to run!")
        return
    _validate_unique_task_names(tasks, "Stage 2 reranking")

    ready_tasks: list[Any] = []
    for task in tasks:
        task_name = task.metadata.name
        print(f"\n  Preparing {task_name}...")

        task.load_data()
        task.convert_v1_dataset_format_to_v2()

        try:
            task.convert_to_reranking(predictions_dir, top_k=top_k)
        except FileNotFoundError as e:
            print(f"  [SKIP] No predictions for {task_name}: {e}")
            print("  Run Stage 1 first.")
            continue

        print(f"  [OK] {task_name}: loaded predictions, will rerank top-{top_k} documents")
        ready_tasks.append(task)

    if not ready_tasks:
        print("[WARN] No tasks ready for reranking!")
        return

    import warnings
    from mteb.deprecated_evaluator import MTEB
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        evaluation = MTEB(tasks=ready_tasks)

    if stage2_predictions_dir:
        prediction_output_dir = Path(stage2_predictions_dir)
    else:
        prediction_output_dir = Path(output_dir) / f"{reranker_name}_predictions"
    prediction_output_dir.mkdir(parents=True, exist_ok=True)
    print(f"  Stage 2 reranked predictions will be saved to: {prediction_output_dir}")

    results = evaluation.run(
        reranker,
        encode_kwargs={"batch_size": batch_size, "show_progress_bar": True},
        output_folder=output_dir,
        overwrite_results=True,
        model_name=reranker_name,
        prediction_folder=prediction_output_dir,
    )

    for tr in results:
        print(f"  Stage 2 scores ({tr.task_name}): {tr.get_score()}")


# ========================================================================
# Main
# ========================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Two-Stage Cross-Encoder Reranking: Stage 1 (dense retrieval) + Stage 2 (cross-encoder reranking)"
    )
    parser.add_argument(
        "--stage", type=str, choices=["1", "2", "both"], required=True,
        help="Which stage to run: 1 / 2 / both",
    )
    parser.add_argument(
        "--reranker", type=str, default=None,
        choices=list(RERANKER_REGISTRY.keys()),
        help="Cross-encoder reranker name (required for stage=2/both)",
    )
    parser.add_argument(
        "--retriever_model_path", type=str, default="Qwen/Qwen3-Embedding-0.6B",
        help="Stage 1 retriever model path (default: Qwen/Qwen3-Embedding-0.6B)",
    )
    parser.add_argument(
        "--tasks", nargs="+", type=str, required=True,
        help="Task group names, e.g.: Original Fixed_Chinese Fixed_R_Chinese",
    )
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size (default: 4)")
    parser.add_argument(
        "--reranker_batch_size", type=int, default=4,
        help="Reranker internal batch size for Qwen3-Reranker (default: 4)",
    )
    parser.add_argument(
        "--predictions_dir", type=str, default="./predictions",
        help="Directory to save/load Stage 1 predictions (default: ./predictions)",
    )
    parser.add_argument(
        "--output_dir", type=str, default="./results_crossencoder",
        help="Stage 2 results output directory (default: ./results_crossencoder)",
    )
    parser.add_argument(
        "--stage2_predictions_dir", type=str, default=None,
        help=(
            "Directory to save Stage 2 reranked predictions as *_predictions.json. "
            "Default: <output_dir>/<reranker_name>_predictions"
        ),
    )
    parser.add_argument("--top_k", type=int, default=100, help="Top-k documents for Stage 2 (default: 100)")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.stage in ("2", "both") and not args.reranker:
        raise ValueError("--reranker is required for stage=2 or stage=both")

    print(f"Task groups: {args.tasks}")
    print(f"Available task groups: {sorted(TASK_REGISTRY.keys())}")
    print(f"Available rerankers: {list(RERANKER_REGISTRY.keys())}")
    _validate_unique_task_names(resolve_tasks(args.tasks), "requested task groups")

    if args.stage in ("1", "both"):
        print(f"\n{'#' * 60}")
        print(f"# Stage 1: Full corpus retrieval with {args.retriever_model_path}")
        print(f"{'#' * 60}")

        retriever_model = smart_load_model(args.retriever_model_path)
        run_stage1(retriever_model, args.tasks, args.predictions_dir, args.batch_size)

        del retriever_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if args.stage in ("2", "both"):
        print(f"\n{'#' * 60}")
        print(f"# Stage 2: Cross-encoder reranking with {args.reranker}")
        print(f"{'#' * 60}")

        reranker = RERANKER_REGISTRY[args.reranker](args.reranker_batch_size)
        run_stage2(
            reranker,
            args.reranker,
            args.tasks,
            args.predictions_dir,
            args.output_dir,
            args.top_k,
            args.batch_size,
            args.stage2_predictions_dir,
        )

        del reranker
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("\nDone!")


if __name__ == "__main__":
    main()
