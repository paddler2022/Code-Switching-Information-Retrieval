#!/usr/bin/env python3
"""Build a compact Original-vs-CSR-L summary CSV.

Rows: one per model. Columns: nDCG@10 for retrieval tasks (Touche2020v3 /
HumanEval / TRECCOVID) and p-MRR for FollowIR tasks (News21 / Robust04 /
Core17), each split into Original / CSR-L-zh / CSR-L-ja, plus the FollowIR
macro-average p-MRR across the three IR tasks.

Adapted from the original package's
``build_original_vs_codeswitching_table.py``. Task-name mapping is updated
because our CS variants use ``*CSRL_{lang}`` (e.g. ``HumanEvalRetrievalCSRL_zh``)
rather than the legacy ``*CodeSwitching`` base names.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


# {display_name: (original_task_name, csrl_base_name)}; language suffix added below.
RETRIEVAL_TASKS: dict[str, tuple[str, str]] = {
    "Touche2020": ("Touche2020Retrieval.v3", "Touche2020Retrieval.v3CSRL"),
    "HumanEval": ("HumanEvalRetrieval", "HumanEvalRetrievalCSRL"),
    "TRECCOVID": ("TRECCOVID", "TRECCOVIDCSRL"),
}

FOLLOWIR_TASKS: dict[str, tuple[str, str]] = {
    "News21": ("News21InstructionRetrieval", "News21InstructionRetrievalCSRL"),
    "Robust04": ("Robust04InstructionRetrieval", "Robust04InstructionRetrievalCSRL"),
    "Core17": ("Core17InstructionRetrieval", "Core17InstructionRetrievalCSRL"),
}

LANGUAGES = ("zh", "ja")


def _fieldnames() -> list[str]:
    fields = ["model"]
    for name in RETRIEVAL_TASKS:
        fields.append(f"{name}_orig_nDCG@10")
        for lang in LANGUAGES:
            fields.append(f"{name}_cs_{lang}_nDCG@10")
    for name in FOLLOWIR_TASKS:
        fields.append(f"{name}_orig_p-MRR")
        for lang in LANGUAGES:
            fields.append(f"{name}_cs_{lang}_p-MRR")
    fields.extend([
        "FollowIR_orig_p-MRR_macro_avg_3tasks",
        *[f"FollowIR_cs_{lang}_p-MRR_macro_avg_3tasks" for lang in LANGUAGES],
        "FollowIR_orig_available_task_count",
        *[f"FollowIR_cs_{lang}_available_task_count" for lang in LANGUAGES],
    ])
    return fields


FIELDNAMES = _fieldnames()


def iter_result_jsons(root: Path):
    """Yield evaluation JSON files while skipping metadata and prediction inputs."""
    for path in sorted(root.rglob("*.json")):
        path_s = path.as_posix()
        if "/predictions/" in path_s:
            continue
        if path.name == "model_meta.json":
            continue
        if ".ipynb_checkpoints" in path_s or path.name.endswith("-checkpoint.json"):
            continue
        yield path


def load_results(root: Path) -> dict[str, dict[str, dict]]:
    """Return mapping: model -> task_name -> scores.test[0] dict."""
    results: dict[str, dict[str, dict]] = {}
    for path in iter_result_jsons(root):
        try:
            data = json.loads(path.read_text())
        except Exception:
            continue
        if not isinstance(data, dict):
            continue
        if "task_name" not in data or "scores" not in data:
            continue

        model = data.get("model") or path.parent.name
        task_name = data.get("task_name") or path.stem
        test_scores = (data.get("scores") or {}).get("test")
        if isinstance(test_scores, list) and test_scores:
            test_scores = test_scores[0]
        if not isinstance(test_scores, dict):
            continue

        results.setdefault(model, {})
        results[model][task_name] = test_scores

    return results


def build_rows(results: dict[str, dict[str, dict]]) -> list[dict]:
    rows: list[dict] = []

    for model in sorted(results.keys()):
        by_task = results[model]
        row: dict = {"model": model}

        for name, (orig_task, cs_base) in RETRIEVAL_TASKS.items():
            row[f"{name}_orig_nDCG@10"] = by_task.get(orig_task, {}).get("ndcg_at_10")
            for lang in LANGUAGES:
                runtime_name = f"{cs_base}_{lang}"
                row[f"{name}_cs_{lang}_nDCG@10"] = by_task.get(runtime_name, {}).get("ndcg_at_10")

        orig_pmrr_values: list[float] = []
        cs_pmrr_values: dict[str, list[float]] = {lang: [] for lang in LANGUAGES}

        for name, (orig_task, cs_base) in FOLLOWIR_TASKS.items():
            orig_val = by_task.get(orig_task, {}).get("p-MRR")
            row[f"{name}_orig_p-MRR"] = orig_val
            if orig_val is not None:
                orig_pmrr_values.append(orig_val)

            for lang in LANGUAGES:
                runtime_name = f"{cs_base}_{lang}"
                cs_val = by_task.get(runtime_name, {}).get("p-MRR")
                row[f"{name}_cs_{lang}_p-MRR"] = cs_val
                if cs_val is not None:
                    cs_pmrr_values[lang].append(cs_val)

        row["FollowIR_orig_p-MRR_macro_avg_3tasks"] = (
            sum(orig_pmrr_values) / 3.0 if len(orig_pmrr_values) == 3 else None
        )
        row["FollowIR_orig_available_task_count"] = len(orig_pmrr_values)
        for lang in LANGUAGES:
            values = cs_pmrr_values[lang]
            row[f"FollowIR_cs_{lang}_p-MRR_macro_avg_3tasks"] = (
                sum(values) / 3.0 if len(values) == 3 else None
            )
            row[f"FollowIR_cs_{lang}_available_task_count"] = len(values)

        rows.append(row)

    return rows


def write_csv(rows: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build Original-vs-CSR-L retrieval/followIR summary table."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Directory to scan for result JSONs.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent
        / "retrieval_and_followir_original_vs_csrl.csv",
        help="Output CSV path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = load_results(args.root)
    rows = build_rows(results)
    write_csv(rows, args.output)
    print(f"Wrote {args.output}")
    print(f"Rows: {len(rows)}")


if __name__ == "__main__":
    main()
