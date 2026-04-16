"""Score extraction and summary helpers."""

from __future__ import annotations

import json
from pathlib import Path
from statistics import mean
from typing import Any


def score_to_percent(score: float) -> float:
    return score * 100 if score <= 1.5 else score


def extract_task_results(model_result: Any) -> list[Any]:
    if hasattr(model_result, "task_results"):
        return list(model_result.task_results)
    if isinstance(model_result, list):
        return list(model_result)
    return [model_result]


def main_score_percent(model_result: Any) -> float:
    task_result = extract_task_results(model_result)[0]
    return score_to_percent(float(task_result.get_score()))


def mean_score(runs: list[dict[str, Any]], labels: list[str]) -> float | None:
    scores = [
        float(run["score"])
        for run in runs
        if run.get("status") == "ok" and run.get("label") in labels and run.get("score") is not None
    ]
    if not scores:
        return None
    return mean(scores)


def compare_scores(actual: dict[str, float | None], expected: dict[str, float]) -> dict[str, float | None]:
    comparison: dict[str, float | None] = {}
    for label, expected_score in expected.items():
        actual_score = actual.get(label)
        comparison[label] = None if actual_score is None else actual_score - expected_score
    return comparison


def write_json(path: str | Path, payload: dict[str, Any]) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return output_path
