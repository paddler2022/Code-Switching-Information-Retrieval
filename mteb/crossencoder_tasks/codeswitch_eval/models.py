"""Model loading, instruction templating, and prompt dictionary.

Key differences from the original package:

* ``QWEN3_PROMPTS`` is now populated with the 18 CSR-L + Original prompts that
  the reranker actually looks up (original package shipped an empty dict and
  duplicated a larger table inside ``run_crossencoder_reranking.py``).
* ``qwen3_instruction_template(instruction, query)`` keeps the two-argument
  concatenating signature used by the cross-encoder query-formatting path.
  This is deliberately different from ``playground.qwen3_instruction_template``
  which is used on the embedding-model side and follows the MTEB
  ``(instruction, prompt_type)`` protocol.
"""

from __future__ import annotations

import os
from typing import Any

import mteb
import torch
from sentence_transformers import SentenceTransformer


# --------------------------------------------------------------------------
# QWEN3_PROMPTS — per-task instructions, keyed by the runtime metadata.name.
#
# Each CSR-L task's ``__init__`` mutates ``metadata.name`` to
# ``f"{base}_{language}"``, so keys for the CSR-L variants are expanded
# accordingly. Original (non-CS) task names stay as-is.
# --------------------------------------------------------------------------

_ORIGINAL_PROMPTS: dict[str, str] = {
    "HumanEvalRetrieval": "Given a question about code problem, retrieval code that can solve user's problem",
    "TRECCOVID": "Given a query on COVID-19, retrieve documents that answer the query",
    "Touche2020Retrieval.v3": "Given a question, retrieve detailed and persuasive arguments that answer the question",
    "Core17InstructionRetrieval": "Retrieval the relevant passage for the given query",
    "News21InstructionRetrieval": "Retrieval the relevant passage for the given query",
    "Robust04InstructionRetrieval": "Retrieval the relevant passage for the given query",
}

_CSR_L_BASE_PROMPTS: dict[str, str] = {
    "HumanEvalRetrievalCSRL": "Given a question about code problem, retrieval code that can solve user's problem",
    "TRECCOVIDCSRL": "Given a query on COVID-19, retrieve documents that answer the query",
    "Touche2020Retrieval.v3CSRL": "Given a question, retrieve detailed and persuasive arguments that answer the question",
    "Core17InstructionRetrievalCSRL": "Retrieval the relevant passage for the given query",
    "News21InstructionRetrievalCSRL": "Retrieval the relevant passage for the given query",
    "Robust04InstructionRetrievalCSRL": "Retrieval the relevant passage for the given query",
}

CSR_L_LANGS: list[str] = ["zh", "ja"]

QWEN3_PROMPTS: dict[str, str] = dict(_ORIGINAL_PROMPTS)
QWEN3_PROMPTS.update(
    {
        f"{name}_{lang}": prompt
        for name, prompt in _CSR_L_BASE_PROMPTS.items()
        for lang in CSR_L_LANGS
    }
)

DEFAULT_INSTRUCTION = "Given a web search query, retrieve relevant passages that answer the query"


def qwen3_instruction_template(instruction: str, query: str) -> str:
    """Build the instruction-prefixed query used by Qwen-style cross-encoders."""
    return f"Instruct: {instruction}\nQuery: {query}"


def torch_dtype_from_name(name: str | torch.dtype | None) -> torch.dtype:
    if isinstance(name, torch.dtype):
        return name
    if name is None:
        return torch.float16

    normalized = str(name).lower()
    mapping = {
        "fp16": torch.float16,
        "float16": torch.float16,
        "half": torch.float16,
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }
    if normalized not in mapping:
        raise ValueError(f"Unsupported torch dtype: {name}")
    return mapping[normalized]


def qwen_model_kwargs(torch_dtype: str | torch.dtype | None = torch.float16) -> dict[str, Any]:
    """Return Qwen kwargs that are safe on commodity GPUs by default."""
    kwargs: dict[str, Any] = {"torch_dtype": torch_dtype_from_name(torch_dtype)}
    attn_implementation = os.environ.get("CODESWITCH_ATTN_IMPLEMENTATION", "sdpa").strip()
    if attn_implementation:
        kwargs["attn_implementation"] = attn_implementation
    return kwargs


def _auto_device(device: str) -> str:
    if device != "auto":
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


def smart_load_model(model_path: str):
    """Load an embedding model through MTEB with Qwen-specific safety kwargs."""
    if "qwen" in model_path.lower():
        return mteb.get_model(model_path, model_kwargs=qwen_model_kwargs())
    return mteb.get_model(model_path)


def load_model_ST(model_path: str, device: str = "auto") -> SentenceTransformer:
    """Load a SentenceTransformer model for compatibility with older scripts."""
    return load_embedding_model(model_path, backend="sentence-transformers", device=device)


def load_embedding_model(
    model_name: str,
    *,
    backend: str = "sentence-transformers",
    device: str = "auto",
    torch_dtype: str | torch.dtype | None = torch.float16,
):
    """Load an embedding model with a small explicit backend surface."""
    normalized_backend = backend.lower()
    if normalized_backend == "mteb":
        return smart_load_model(model_name)

    if normalized_backend not in {"sentence-transformers", "sentence_transformers", "st"}:
        raise ValueError(f"Unsupported embedding backend: {backend}")

    resolved_device = _auto_device(device)
    model_kwargs: dict[str, Any] = {}
    if "qwen" in model_name.lower():
        model_kwargs.update(qwen_model_kwargs(torch_dtype))
        model_kwargs["device_map"] = resolved_device

    if model_kwargs:
        model = SentenceTransformer(model_name, model_kwargs=model_kwargs, device=resolved_device)
    else:
        model = SentenceTransformer(model_name, device=resolved_device)

    tokenizer = getattr(model, "tokenizer", None)
    if tokenizer is not None and "qwen" in model_name.lower():
        tokenizer.padding_side = "left"
    return model
