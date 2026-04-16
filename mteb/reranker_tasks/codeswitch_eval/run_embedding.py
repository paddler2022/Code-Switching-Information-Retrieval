"""CLI for reusable embedding-model evaluation runs.

Run from ``reranker_tasks/``:

    python -m codeswitch_eval.run_embedding --suite humaneval_smoke
    python -m codeswitch_eval.run_embedding --task_groups Fixed_Chinese --batch_size 4
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .models import load_embedding_model
from .runner import AggregateSpec, EmbeddingRunConfig, run_embedding_suite, validate_cache_safety
from .tasks import (
    available_task_groups,
    filter_task_runs,
    humaneval_smoke_task_runs,
    resolve_task_runs,
)


DEFAULT_SMOKE_MODEL = "sentence-transformers/all-MiniLM-L12-v2"
HUMANEVAL_MINILM_EXPECTED = {
    "HumanEval Orig": 70.08,
    "HumanEval CSR-L": 62.18,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CodeSwitching embedding evaluations.")
    parser.add_argument("--model", default=DEFAULT_SMOKE_MODEL, help="Embedding model name or path.")
    parser.add_argument(
        "--backend",
        default="sentence-transformers",
        choices=["sentence-transformers", "mteb"],
        help="Model loading backend.",
    )
    parser.add_argument("--device", default="auto", help="Device for SentenceTransformer backend.")
    parser.add_argument("--torch_dtype", default="float16", help="Torch dtype for Qwen ST models.")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument(
        "--overwrite_strategy",
        default="always",
        choices=["always", "only-missing", "never", "only-cache"],
        help="MTEB cache overwrite strategy.",
    )
    parser.add_argument(
        "--suite",
        default="humaneval_smoke",
        choices=["humaneval_smoke", "custom"],
        help="Predefined suite. Use custom with --task_groups.",
    )
    parser.add_argument(
        "--task_groups",
        nargs="+",
        default=None,
        help=f"Task groups for custom runs. Available: {available_task_groups()}",
    )
    parser.add_argument(
        "--only_task_name",
        nargs="+",
        default=None,
        help="Optional case-insensitive substrings to filter task metadata names.",
    )
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--continue_on_error", action="store_true")
    parser.add_argument("--no_progress", action="store_true")
    return parser.parse_args()


def _build_run_plan(args: argparse.Namespace):
    if args.task_groups:
        task_runs = resolve_task_runs(args.task_groups, strict=True)
        task_runs = filter_task_runs(task_runs, args.only_task_name)
        aggregates: list[AggregateSpec] = []
        expected: dict[str, float] = {}
        default_output = "results_embedding_eval"
    elif args.suite == "humaneval_smoke":
        task_runs = humaneval_smoke_task_runs()
        aggregates = [
            AggregateSpec(
                "HumanEval CSR-L",
                ["HumanEval Chinese CSR", "HumanEval Japanese CSR"],
            )
        ]
        expected = HUMANEVAL_MINILM_EXPECTED if "all-minilm-l12-v2" in args.model.lower() else {}
        default_output = "smoke_minilm_humaneval"
    else:
        raise ValueError("--task_groups is required for --suite custom")

    if not task_runs:
        raise ValueError("No tasks selected.")

    output_dir = Path(args.output_dir or default_output)
    return task_runs, aggregates, expected, output_dir


def main() -> None:
    args = parse_args()
    task_runs, aggregates, expected, output_dir = _build_run_plan(args)

    print(f"model={args.model}", flush=True)
    print(f"backend={args.backend}", flush=True)
    print(f"tasks={[run.label for run in task_runs]}", flush=True)
    validate_cache_safety(task_runs, args.overwrite_strategy)

    model = load_embedding_model(
        args.model,
        backend=args.backend,
        device=args.device,
        torch_dtype=args.torch_dtype,
    )
    config = EmbeddingRunConfig(
        model_name=args.model,
        output_dir=output_dir,
        batch_size=args.batch_size,
        overwrite_strategy=args.overwrite_strategy,
        show_progress_bar=not args.no_progress,
        continue_on_error=args.continue_on_error,
    )
    summary = run_embedding_suite(
        model=model,
        task_runs=task_runs,
        config=config,
        aggregates=aggregates,
        expected=expected,
    )

    print("\n=== aggregate ===", flush=True)
    print(json.dumps(summary["aggregate"], indent=2, ensure_ascii=False), flush=True)
    if summary["table_expected"]:
        print("=== table_expected ===", flush=True)
        print(json.dumps(summary["table_expected"], indent=2, ensure_ascii=False), flush=True)
        print("=== difference(run - table) ===", flush=True)
        print(json.dumps(summary["difference"], indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
