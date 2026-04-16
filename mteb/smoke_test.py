"""
Smoke test for playground.py.

Goal: fast correctness checks — no network, no GPU, no model downloads.
Covers the parts of playground.py that can go wrong without a full eval run:

  1. All task classes import + instantiate across every supported language.
  2. Each task's __init__ mutates metadata.name to f"{base}_{lang}".
  3. Every runtime task name has a matching prompt in QWEN3_PROMPTS.
  4. No QWEN3_PROMPTS key is orphaned (i.e. never produced by a loader).
  5. qwen3_instruction_template returns the right shape for query/document/empty.
  6. AbsEncoder.get_instruction (the actual lookup path used at encode-time)
     returns the correct prompt for every task when wired with QWEN3_PROMPTS.
  7. The __main__ dispatch logic (CS-MTEB_<lang>, CSR-L_<lang>) works and
     rejects unsupported languages.
  8. Data ingestion: a task's load_data can be overridden per-instance with
     tiny synthetic data, and the standard retrieval fields are populated.

Run:
    python smoke_test.py
"""
from __future__ import annotations

import sys
import traceback
import types
from typing import Callable

try:
    from playground import (
        QWEN3_PROMPTS,
        CS_MTEB_LANGS,
        CSR_L_LANGS,
        qwen3_instruction_template,
        load_cs_mteb_all_tasks,
        load_csr_l_all_tasks,
    )
    from mteb.types import PromptType
    from mteb.models.abs_encoder import AbsEncoder
    from mteb.tasks.retrieval.eng import TRECCOVIDCodeSwitching
except Exception as e:  # pragma: no cover
    print(f"[FATAL] import failed: {e}")
    traceback.print_exc()
    sys.exit(1)


# ------------------- cached task instantiations (fast; no network) -------------------

_CS_MTEB_CACHE: dict[str, list] = {}
_CSR_L_CACHE: dict[str, list] = {}


def _cs_mteb(lang: str):
    if lang not in _CS_MTEB_CACHE:
        _CS_MTEB_CACHE[lang] = load_cs_mteb_all_tasks(lang)
    return _CS_MTEB_CACHE[lang]


def _csr_l(lang: str):
    if lang not in _CSR_L_CACHE:
        _CSR_L_CACHE[lang] = load_csr_l_all_tasks(lang)
    return _CSR_L_CACHE[lang]


def _all_runtime_names() -> set[str]:
    names = set()
    for lang in CS_MTEB_LANGS:
        names.update(t.metadata.name for t in _cs_mteb(lang))
    for lang in CSR_L_LANGS:
        names.update(t.metadata.name for t in _csr_l(lang))
    return names


# ----------------------------------- checks -----------------------------------

def test_cs_mteb_instantiation():
    for lang in CS_MTEB_LANGS:
        tasks = _cs_mteb(lang)
        assert len(tasks) == 13, f"{lang}: expected 13 tasks, got {len(tasks)}"
        for t in tasks:
            assert t.metadata.name.endswith(f"_{lang}"), (
                f"{type(t).__name__} name={t.metadata.name!r} missing _{lang} suffix"
            )


def test_csr_l_instantiation():
    for lang in CSR_L_LANGS:
        tasks = _csr_l(lang)
        assert len(tasks) == 6, f"{lang}: expected 6 tasks, got {len(tasks)}"
        for t in tasks:
            assert t.metadata.name.endswith(f"_{lang}"), (
                f"{type(t).__name__} name={t.metadata.name!r} missing _{lang} suffix"
            )


def test_qwen3_prompts_size():
    expected = 13 * len(CS_MTEB_LANGS) + 6 * len(CSR_L_LANGS)
    assert len(QWEN3_PROMPTS) == expected, (
        f"QWEN3_PROMPTS has {len(QWEN3_PROMPTS)} keys, expected {expected}"
    )


def test_every_task_has_prompt():
    missing = [n for n in _all_runtime_names() if n not in QWEN3_PROMPTS]
    assert not missing, f"missing prompts for: {sorted(missing)}"


def test_no_orphan_prompts():
    orphans = set(QWEN3_PROMPTS) - _all_runtime_names()
    assert not orphans, f"unused QWEN3_PROMPTS keys: {sorted(orphans)}"


def test_instruction_template_query():
    out = qwen3_instruction_template("Find stuff", PromptType.query)
    assert out == "Instruct: Find stuff\nQuery:", out


def test_instruction_template_document():
    # documents intentionally get no instruction prefix (see playground.py)
    assert qwen3_instruction_template("ignored", PromptType.document) == ""


def test_instruction_template_empty():
    assert qwen3_instruction_template("", PromptType.query) == ""


class _StubEncoder(AbsEncoder):
    """Minimal AbsEncoder so we can exercise get_instruction without loading a model."""

    def __init__(self, prompts_dict):
        self.prompts_dict = prompts_dict

    def encode(self, inputs, *, task_metadata, hf_split, hf_subset,
               prompt_type=None, **kwargs):  # pragma: no cover - unused
        raise NotImplementedError


def test_get_instruction_e2e():
    """Mirrors the real encode-time path: AbsEncoder.get_instruction -> prompts_dict[task.metadata.name]."""
    enc = _StubEncoder(QWEN3_PROMPTS)
    for name in _all_runtime_names():
        # Find a task with this name (any one is fine)
        task = next(
            t for lang in CS_MTEB_LANGS for t in _cs_mteb(lang)
            if t.metadata.name == name
        ) if name in {t.metadata.name for lang in CS_MTEB_LANGS for t in _cs_mteb(lang)} else next(
            t for lang in CSR_L_LANGS for t in _csr_l(lang)
            if t.metadata.name == name
        )
        got = enc.get_instruction(task.metadata, PromptType.query)
        assert got == QWEN3_PROMPTS[name], (
            f"{name}: got {got!r}, expected {QWEN3_PROMPTS[name]!r}"
        )


def test_dispatch_valid_lang():
    assert len(load_cs_mteb_all_tasks("zh")) == 13
    assert len(load_csr_l_all_tasks("ja")) == 6


def test_dispatch_invalid_lang_raises():
    try:
        load_cs_mteb_all_tasks("xx")
    except ValueError:
        return
    raise AssertionError("expected ValueError for unsupported language 'xx'")


def test_monkey_patched_load_data():
    """Validate the data-ingestion contract without hitting the network and
    without modifying mteb/tasks: bind a fake load_data on a single instance."""
    task = TRECCOVIDCodeSwitching(language="zh")

    def fake_load_data(self, **kwargs):
        if getattr(self, "data_loaded", False):
            return
        self.queries = {"test": {"q0": "query zero", "q1": "query one"}}
        self.corpus = {"test": {
            "d0": {"title": "t0", "text": "doc zero"},
            "d1": {"title": "t1", "text": "doc one"},
        }}
        self.relevant_docs = {"test": {"q0": {"d0": 1}, "q1": {"d1": 1}}}
        self.data_loaded = True

    task.load_data = types.MethodType(fake_load_data, task)
    task.load_data()

    assert task.data_loaded is True
    assert len(task.queries["test"]) == 2
    assert len(task.corpus["test"]) == 2
    assert task.relevant_docs["test"]["q0"] == {"d0": 1}
    # class-level load_data must be untouched (no mteb/tasks changes)
    assert TRECCOVIDCodeSwitching.load_data is not task.load_data


# ----------------------------------- runner -----------------------------------

TESTS: list[tuple[str, Callable[[], None]]] = [
    ("cs-mteb: instantiation + name suffix", test_cs_mteb_instantiation),
    ("csr-l : instantiation + name suffix", test_csr_l_instantiation),
    ("QWEN3_PROMPTS: correct total size", test_qwen3_prompts_size),
    ("QWEN3_PROMPTS: every runtime task covered", test_every_task_has_prompt),
    ("QWEN3_PROMPTS: no orphan keys", test_no_orphan_prompts),
    ("template: query formatting", test_instruction_template_query),
    ("template: document -> empty", test_instruction_template_document),
    ("template: empty instruction -> empty", test_instruction_template_empty),
    ("AbsEncoder.get_instruction: all tasks hit prompt", test_get_instruction_e2e),
    ("dispatch: valid langs load expected task counts", test_dispatch_valid_lang),
    ("dispatch: unsupported lang raises ValueError", test_dispatch_invalid_lang_raises),
    ("monkey-patched load_data: ingestion shape ok", test_monkey_patched_load_data),
]


def main() -> int:
    print("Smoke test for playground.py")
    print("-" * 60)
    passed = failed = 0
    for name, fn in TESTS:
        try:
            fn()
            print(f"  PASS  {name}")
            passed += 1
        except AssertionError as e:
            print(f"  FAIL  {name}: {e}")
            failed += 1
        except Exception as e:
            print(f"  ERROR {name}: {type(e).__name__}: {e}")
            traceback.print_exc()
            failed += 1
    print("-" * 60)
    print(f"Summary: {passed} passed, {failed} failed")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
