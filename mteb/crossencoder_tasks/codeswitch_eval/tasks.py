"""Task registry and dataset preparation helpers.

Rewritten relative to the original package:

* The original package shipped 6 ``*CodeSwitching`` classes recovered from
  ``.pyc`` files; each took ``query_file=`` / ``instruction_file=`` kwargs and
  read local JSONL under ``CodeSwitching_Dataset_fixed/``.
* In our repo those 6 classes were renamed to ``*CSRL`` and migrated to
  HuggingFace loading via a ``language=`` kwarg (zh / ja). The 6 task-group
  labels from the original package are preserved so the legacy CLI / shims
  keep working:

    - ``Original``          → 4 non-CS tasks (Core17/News21/Robust04 IR + HumanEval)
    - ``OG_Retrieval``      → 2 non-CS retrieval tasks (TRECCOVID + Touche2020v3)
    - ``Fixed_Chinese``     → 4 CSR-L IR/code tasks with language="zh"
    - ``Fixed_Japanese``    → 4 CSR-L IR/code tasks with language="ja"
    - ``Fixed_R_Chinese``   → 2 CSR-L retrieval tasks with language="zh"
    - ``Fixed_R_Japanese``  → 2 CSR-L retrieval tasks with language="ja"

The ``Fixed_*`` groups all go through the same two tiny loaders parameterised
by language, instead of the six near-duplicate hard-coded loaders in the
original package.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from mteb.tasks.instruction_reranking.eng import (
    Core17InstructionRetrieval,
    Core17InstructionRetrievalCSRL,
    News21InstructionRetrieval,
    News21InstructionRetrievalCSRL,
    Robust04InstructionRetrieval,
    Robust04InstructionRetrievalCSRL,
)
from mteb.tasks.retrieval.code import (
    HumanEvalRetrieval,
    HumanEvalRetrievalCSRL,
)
from mteb.tasks.retrieval.eng import (
    TRECCOVID,
    TRECCOVIDCSRL,
    Touche2020v3Retrieval,
    Touche2020v3RetrievalCSRL,
)


TaskLoader = Callable[[], list[Any]]


@dataclass(frozen=True)
class TaskRun:
    label: str
    group: str
    index: int
    task: Any


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def _load_original() -> list[Any]:
    """4 non-CS tasks: the instruction-reranking trio + code retrieval."""
    return [
        Core17InstructionRetrieval(),
        News21InstructionRetrieval(),
        Robust04InstructionRetrieval(),
        HumanEvalRetrieval(),
    ]


def _load_og_retrieval() -> list[Any]:
    """2 non-CS retrieval tasks."""
    return [
        TRECCOVID(),
        Touche2020v3Retrieval(),
    ]


def _load_fixed_ir(language: str) -> list[Any]:
    """4 CSR-L IR/code tasks for the given language."""
    return [
        HumanEvalRetrievalCSRL(language=language),
        Core17InstructionRetrievalCSRL(language=language),
        News21InstructionRetrievalCSRL(language=language),
        Robust04InstructionRetrievalCSRL(language=language),
    ]


def _load_fixed_retrieval(language: str) -> list[Any]:
    """2 CSR-L retrieval tasks for the given language."""
    return [
        TRECCOVIDCSRL(language=language),
        Touche2020v3RetrievalCSRL(language=language),
    ]


TASK_REGISTRY: dict[str, TaskLoader] = {
    "Original": _load_original,
    "OG_Retrieval": _load_og_retrieval,
    "Fixed_Chinese": lambda: _load_fixed_ir("zh"),
    "Fixed_Japanese": lambda: _load_fixed_ir("ja"),
    "Fixed_R_Chinese": lambda: _load_fixed_retrieval("zh"),
    "Fixed_R_Japanese": lambda: _load_fixed_retrieval("ja"),
}


def available_task_groups() -> list[str]:
    return sorted(TASK_REGISTRY)


def resolve_tasks(task_names: list[str], *, strict: bool = False) -> list[Any]:
    """Expand group names into task instances."""
    resolved: list[Any] = []
    for name in task_names:
        loader = TASK_REGISTRY.get(name)
        if loader is None:
            message = f"Unknown task group: {name}. Available: {available_task_groups()}"
            if strict:
                raise KeyError(message)
            print(f"[WARN] {message}")
            continue
        resolved.extend(loader())
    return resolved


def resolve_task_runs(task_names: list[str], *, strict: bool = True) -> list[TaskRun]:
    """Resolve task groups and retain labels for summaries."""
    runs: list[TaskRun] = []
    for group in task_names:
        loader = TASK_REGISTRY.get(group)
        if loader is None:
            message = f"Unknown task group: {group}. Available: {available_task_groups()}"
            if strict:
                raise KeyError(message)
            print(f"[WARN] {message}")
            continue
        for index, task in enumerate(loader()):
            runs.append(
                TaskRun(
                    label=f"{group}:{task.metadata.name}",
                    group=group,
                    index=index,
                    task=task,
                )
            )
    return runs


def humaneval_smoke_task_runs() -> list[TaskRun]:
    """Return the smallest table smoke suite: HumanEval original + zh/ja CSR-L."""
    original = resolve_tasks(["Original"], strict=True)[3]       # HumanEvalRetrieval
    chinese = resolve_tasks(["Fixed_Chinese"], strict=True)[0]    # HumanEvalRetrievalCSRL(zh)
    japanese = resolve_tasks(["Fixed_Japanese"], strict=True)[0]  # HumanEvalRetrievalCSRL(ja)
    return [
        TaskRun("HumanEval Orig", "Original", 3, original),
        TaskRun("HumanEval Chinese CSR", "Fixed_Chinese", 0, chinese),
        TaskRun("HumanEval Japanese CSR", "Fixed_Japanese", 0, japanese),
    ]


def filter_task_runs(task_runs: list[TaskRun], name_substrings: list[str] | None) -> list[TaskRun]:
    if not name_substrings:
        return task_runs
    lowered = [item.lower() for item in name_substrings]
    return [
        run
        for run in task_runs
        if any(item in run.task.metadata.name.lower() for item in lowered)
    ]


def prepare_task(task: Any, *, clear_top_ranked: bool = False) -> Any:
    task.load_data()
    task.convert_v1_dataset_format_to_v2()
    if clear_top_ranked:
        for subset in task.dataset:
            for split in task.dataset[subset]:
                task.dataset[subset][split]["top_ranked"] = None
    return task


def _first_split_data(
    task: Any,
    preferred_subset: str = "default",
    preferred_split: str = "test",
) -> tuple[str, str, dict[str, Any] | None]:
    dataset = getattr(task, "dataset", None)
    if not dataset:
        return preferred_subset, preferred_split, None

    if preferred_subset in dataset and preferred_split in dataset[preferred_subset]:
        return preferred_subset, preferred_split, dataset[preferred_subset][preferred_split]

    for subset, subset_data in dataset.items():
        if preferred_split in subset_data:
            return subset, preferred_split, subset_data[preferred_split]
        for split, split_data in subset_data.items():
            return subset, split, split_data

    return preferred_subset, preferred_split, None


def dataset_sizes(task: Any) -> dict[str, int | str | None]:
    subset, split, split_data = _first_split_data(task)
    sizes: dict[str, int | str | None] = {
        "subset": subset,
        "split": split,
        "queries": None,
        "corpus": None,
        "relevant_docs": None,
    }
    if split_data is None:
        return sizes

    for key in ("queries", "corpus", "relevant_docs"):
        value = split_data.get(key)
        try:
            sizes[key] = len(value) if value is not None else None
        except TypeError:
            sizes[key] = None
    return sizes
