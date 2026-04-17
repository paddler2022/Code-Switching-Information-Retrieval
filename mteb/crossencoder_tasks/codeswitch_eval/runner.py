"""Reusable embedding evaluation runner."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import mteb

from .paths import ensure_output_dir
from .scoring import compare_scores, main_score_percent, mean_score, write_json
from .tasks import TaskRun, dataset_sizes, prepare_task


@dataclass(frozen=True)
class AggregateSpec:
    label: str
    source_labels: list[str]


@dataclass
class EmbeddingRunConfig:
    model_name: str
    output_dir: Path
    batch_size: int = 64
    overwrite_strategy: str = "always"
    show_progress_bar: bool = True
    continue_on_error: bool = False
    encode_kwargs: dict[str, Any] = field(default_factory=dict)


def run_embedding_task(task_run: TaskRun, model: Any, config: EmbeddingRunConfig) -> dict[str, Any]:
    print(f"\n=== {task_run.label}: {task_run.task.metadata.name} ===", flush=True)
    try:
        prepare_task(task_run.task)
        sizes = dataset_sizes(task_run.task)
        print(f"dataset_sizes={sizes}", flush=True)

        encode_kwargs = {
            "batch_size": config.batch_size,
            "show_progress_bar": config.show_progress_bar,
        }
        encode_kwargs.update(config.encode_kwargs)

        model_result = mteb.evaluate(
            model,
            task_run.task,
            encode_kwargs=encode_kwargs,
            overwrite_strategy=config.overwrite_strategy,
            show_progress_bar=config.show_progress_bar,
        )
        score = main_score_percent(model_result)
        print(f"main_score={score:.4f}", flush=True)
        return {
            "label": task_run.label,
            "group": task_run.group,
            "index": task_run.index,
            "task_name": task_run.task.metadata.name,
            "score": score,
            "dataset_sizes": sizes,
            "status": "ok",
        }
    except Exception as exc:
        if not config.continue_on_error:
            raise
        print(f"[ERROR] {task_run.label}: {exc}", flush=True)
        return {
            "label": task_run.label,
            "group": task_run.group,
            "index": task_run.index,
            "task_name": getattr(getattr(task_run.task, "metadata", None), "name", None),
            "score": None,
            "dataset_sizes": {},
            "status": "error",
            "error": repr(exc),
        }


def run_embedding_suite(
    *,
    model: Any,
    task_runs: list[TaskRun],
    config: EmbeddingRunConfig,
    aggregates: list[AggregateSpec] | None = None,
    expected: dict[str, float] | None = None,
) -> dict[str, Any]:
    output_dir = ensure_output_dir(config.output_dir)
    validate_cache_safety(task_runs, config.overwrite_strategy)
    runs = [run_embedding_task(task_run, model, config) for task_run in task_runs]

    aggregate_scores: dict[str, float | None] = {}
    for aggregate in aggregates or []:
        aggregate_scores[aggregate.label] = mean_score(runs, aggregate.source_labels)

    actual_scores: dict[str, float | None] = {
        run["label"]: run["score"]
        for run in runs
        if run.get("status") == "ok"
    }
    actual_scores.update(aggregate_scores)

    summary = {
        "model": config.model_name,
        "runs": runs,
        "aggregate": aggregate_scores,
        "table_expected": expected or {},
        "difference": compare_scores(actual_scores, expected or {}),
    }
    summary_path = write_json(output_dir / "results_summary.json", summary)
    print(f"\nwrote {summary_path}", flush=True)
    return summary


def validate_cache_safety(task_runs: list[TaskRun], overwrite_strategy: str) -> None:
    """Avoid stale MTEB cache reuse across variants that share a task name.

    Note: in our repo, CSR-L task ``__init__`` appends ``_{language}`` to
    ``metadata.name``, so Chinese and Japanese instances do NOT collide by
    default. This check is still useful if someone registers legacy
    ``*CodeSwitching`` tasks that skip the suffix mutation.
    """
    if overwrite_strategy == "always":
        return

    task_name_counts = Counter(run.task.metadata.name for run in task_runs)
    duplicate_names = sorted(name for name, count in task_name_counts.items() if count > 1)
    if not duplicate_names:
        return

    raise ValueError(
        "Unsafe MTEB cache strategy for duplicate task names. "
        f"Duplicate task names: {duplicate_names}. "
        "Use --overwrite_strategy always, or run one language variant at a time."
    )
