import mteb
import os
import json
from mteb import MTEB
from mteb.types import PromptType
from mteb.models.sentence_transformer_wrapper import SentenceTransformerEncoderWrapper
from mteb.models.instruct_wrapper import InstructSentenceTransformerModel
from sentence_transformers import SentenceTransformer
from mteb.tasks.retrieval.eng import TRECCOVIDCodeSwitching, TRECCOVIDCSRL, ArguAna, ArguAnaCodeSwitching, \
    ClimateFEVERHardNegativesV2, ClimateFEVERHardNegativesV2CodeSwitching, TRECCOVID, Touche2020v3Retrieval, \
    Touche2020v3RetrievalCodeSwitching, Touche2020v3RetrievalCSRL
from mteb.tasks.instruction_reranking.eng import Core17InstructionRetrievalCodeSwitching, \
    Core17InstructionRetrievalCSRL, News21InstructionRetrievalCodeSwitching, News21InstructionRetrievalCSRL, \
    Robust04InstructionRetrievalCodeSwitching, Robust04InstructionRetrievalCSRL, Core17InstructionRetrieval, \
    News21InstructionRetrieval, Robust04InstructionRetrieval
from mteb.tasks.classification.eng import TweetSentimentExtractionClassificationV2, \
    TweetSentimentExtractionClassificationCodeSwitching
from mteb.tasks.reranking.eng import AskUbuntuDupQuestions, AskUbuntuDupQuestionsCodeSwitching
from mteb.tasks.sts.eng import STSBenchmarkSTS, STSBenchmarkCodeSwitching
from mteb.tasks.pair_classification.eng import TwitterSemEval2015PC, TwitterSemEval2015CodeSwitching
from mteb.tasks.clustering.eng import ArXivHierarchicalClusteringP2P, ArXivHierarchicalClusteringP2PCodeSwitching
from mteb.tasks.retrieval.code import HumanEvalRetrievalCodeSwitching, HumanEvalRetrievalCSRL, HumanEvalRetrieval
import torch
import argparse
from datetime import datetime

# E5 models' Instruct Prompt Template
E5_PROMPTS = {
    PromptType.query.value: "query: ",
    PromptType.document.value: "passage: ",
}

# Languages supported by each benchmark (also used at runtime to dispatch tasks)
CS_MTEB_LANGS = ["zh", "ja", "de", "es", "ko", "fr", "it", "pt", "nl"]
CSR_L_LANGS = ["zh", "ja"]

# Base prompts for each CS-MTEB task (base metadata name, before per-language suffix).
# Each task's __init__ rewrites self.metadata.name to f"{base}_{language}", so the
# prompts dict below must be expanded with the same suffix for lookup to hit.
_CS_MTEB_BASE_PROMPTS = {
    # Retrieval
    "ArguAnaCodeSwitching": "Given a claim, find documents that refute the claim",
    "ClimateFEVERHardNegatives.v2CodeSwitching": "Given a claim about climate change, retrieve documents that support or refute the claim",
    "TRECCOVIDCodeSwitching": "Given a query on COVID-19, retrieve documents that answer the query",
    "Touche2020Retrieval.v3CodeSwitching": "Given a question, retrieve detailed and persuasive arguments that answer the question",
    "HumanEvalRetrievalCodeSwitching": "Given a question about code problem, retrieval code that can solve user's problem",
    # Instruction Retrieval (FollowIR)
    "Core17InstructionRetrievalCodeSwitching": "Retrieval the relevant passage for the given query",
    "News21InstructionRetrievalCodeSwitching": "Retrieval the relevant passage for the given query",
    "Robust04InstructionRetrievalCodeSwitching": "Retrieval the relevant passage for the given query",
    # Classification / Clustering / PairClassification / STS / Reranking
    "TweetSentimentExtractionClassificationCodeSwitching": "Classify the sentiment of a given tweet as either positive, negative, or neutral",
    "ArXivHierarchicalClusteringP2PCodeSwitching": "Identify the main and secondary category of Arxiv papers based on the titles and abstracts",
    "TwitterSemEval2015CodeSwitching": "Retrieve tweets that are semantically similar to the given tweet",
    "STSBenchmarkCodeSwitching": "Retrieve semantically similar text",
    "AskUbuntuDupQuestionsCodeSwitching": "Retrieve duplicate questions from AskUbuntu forum",
}


# Base prompts for each CSR-L task (zh/ja only).
_CSR_L_BASE_PROMPTS = {
    # Retrieval
    "HumanEvalRetrievalCSRL": "Given a question about code problem, retrieval code that can solve user's problem",
    "TRECCOVIDCSRL": "Given a query on COVID-19, retrieve documents that answer the query",
    "Touche2020Retrieval.v3CSRL": "Given a question, retrieve detailed and persuasive arguments that answer the question",
    # Instruction Retrieval (FollowIR)
    "Core17InstructionRetrievalCSRL": "Retrieval the relevant passage for the given query",
    "News21InstructionRetrievalCSRL": "Retrieval the relevant passage for the given query",
    "Robust04InstructionRetrievalCSRL": "Retrieval the relevant passage for the given query",
}

# Prompts for the English Original counterparts of the CS/CSR-L tasks.
# Keys are the runtime metadata.name of each original task (not the Python
# class name) and mirror the semantics of the CS variants.
_ORIGINAL_BASE_PROMPTS = {
    # Retrieval
    "ArguAna": "Given a claim, find documents that refute the claim",
    "ClimateFEVERHardNegatives.v2": "Given a claim about climate change, retrieve documents that support or refute the claim",
    "TRECCOVID": "Given a query on COVID-19, retrieve documents that answer the query",
    "Touche2020Retrieval.v3": "Given a question, retrieve detailed and persuasive arguments that answer the question",
    "HumanEvalRetrieval": "Given a question about code problem, retrieval code that can solve user's problem",
    # Instruction Retrieval (FollowIR)
    "Core17InstructionRetrieval": "Retrieval the relevant passage for the given query",
    "News21InstructionRetrieval": "Retrieval the relevant passage for the given query",
    "Robust04InstructionRetrieval": "Retrieval the relevant passage for the given query",
    # Classification / Clustering / PairClassification / STS / Reranking
    "TweetSentimentExtractionClassification.v2": "Classify the sentiment of a given tweet as either positive, negative, or neutral",
    "ArXivHierarchicalClusteringP2P": "Identify the main and secondary category of Arxiv papers based on the titles and abstracts",
    "TwitterSemEval2015": "Retrieve tweets that are semantically similar to the given tweet",
    "STSBenchmark": "Retrieve semantically similar text",
    "AskUbuntuDupQuestions": "Retrieve duplicate questions from AskUbuntu forum",
}

# Qwen3 models' Instruct Prompt Template on different tasks, keyed by the
# runtime task name (i.e. f"{base_name}_{language}") to match the mutation
# performed in each CS/CSR-L task's __init__.
QWEN3_PROMPTS = {
    f"{name}_{lang}": prompt
    for name, prompt in _CS_MTEB_BASE_PROMPTS.items()
    for lang in CS_MTEB_LANGS
}
QWEN3_PROMPTS.update({
    f"{name}_{lang}": prompt
    for name, prompt in _CSR_L_BASE_PROMPTS.items()
    for lang in CSR_L_LANGS
})
QWEN3_PROMPTS.update(_ORIGINAL_BASE_PROMPTS)

def qwen3_instruction_template(instruction: str, prompt_type: PromptType | None = None) -> str:
    """Qwen3 Embedding models' Instruct templates (refer to qwen3_models.py)"""
    if not instruction or prompt_type == PromptType.document:
        return ""
    if isinstance(instruction, dict):
        if prompt_type is None:
            instruction = list(instruction.values())[0]
        else:
            instruction = instruction[prompt_type]
    return f"Instruct: {instruction}\nQuery:"


def get_model_kwargs(model_path):
    """selecting model_kwargs according to different models"""
    kwargs = {
        "torch_dtype": torch.bfloat16,
        "device_map": "cuda"
    }
    if "qwen" in model_path.lower() or "llama" in model_path.lower() or "SFR-Embedding-2_R" in model_path:
        kwargs["attn_implementation"] = "flash_attention_2"
        print("Using Flash Attention 2")
    else:
        print("Using default Attention")
    return kwargs


def load_model(model_path):
    "Loading models from hf path"
    if "SFR-Embedding-2_R" in model_path:
        model = mteb.get_model(
            "Salesforce/SFR-Embedding-2_R",
            attn_implementation="flash_attention_2",
            torch_dtype=torch.bfloat16
        )

    elif "llama" in model_path.lower():
        model = mteb.get_model(model_path)
    else:

        model_kwargs = get_model_kwargs(model_path)
        if "multilingual-e5" in model_path:
            _gritlm_unsafe = {'device_map', 'mode', 'instruction_template', 'instruction'}
            model = mteb.get_model(
                model_path,
            )
        else:
            model = mteb.get_model(
                model_path,
                model_kwargs=model_kwargs,
                trust_remote_code=True,
            )

    model_path_lower = model_path.lower()
    if "qwen" in model_path_lower and hasattr(model, "prompts_dict"):
        if model.prompts_dict is None:
            model.prompts_dict = {}
        model.prompts_dict.update(QWEN3_PROMPTS)
        print(f"[INFO] Injected {len(QWEN3_PROMPTS)} CS prompts into model.prompts_dict "
              f"(total {len(model.prompts_dict)} keys)")
    elif "e5" in model_path_lower and hasattr(model, "model_prompts"):
        if model.model_prompts is None:
            model.model_prompts = {}
        model.model_prompts.update(E5_PROMPTS)
        print(f"[INFO] Injected E5 prompts into model.model_prompts: {E5_PROMPTS}")
    return model


def load_model_ST(model_path):
    """Loading models from local path"""
    model_path_lower = model_path.lower()
    model_kwargs = get_model_kwargs(model_path)

    # E5 models
    if "e5" in model_path_lower:
        print(f"Using E5 models and loading E5 Instruct template")
        model = SentenceTransformerEncoderWrapper(
            model=model_path,
            revision=None,
            model_prompts=E5_PROMPTS,
            trust_remote_code=True
        )
        return model
    # Qwen3 Embedding models
    elif "qwen" in model_path_lower and "embedding" in model_path_lower:
        print(f"Using Qwen Embedding models and loading Qwen Instruct template")
        model = InstructSentenceTransformerModel(
            model_name=model_path,
            revision=None,
            instruction_template=qwen3_instruction_template,
            apply_instruction_to_passages=False,
            prompts_dict=QWEN3_PROMPTS,
            model_kwargs=model_kwargs,
            trust_remote_code=True,
        )
        if hasattr(model, 'model') and hasattr(model.model, 'tokenizer'):
            model.model.tokenizer.padding_side = 'left'
        return model
    # MiniLM models
    elif "minilm" in model_path_lower:
        print(f"Using MiniLM_models...")
        model = SentenceTransformer(
            model_path,
            device="cuda"
        )
        return model
    else:
        print(f"No special prompts. Loading models...")
        model = SentenceTransformer(
            model_path,
            model_kwargs=model_kwargs,
            device="cuda"
        )
        model.tokenizer.padding_side = 'left'
        return model


def load_cs_mteb_retrieval_tasks(language):
    return [
        ArguAnaCodeSwitching(language=language),
        ClimateFEVERHardNegativesV2CodeSwitching(language=language),
        TRECCOVIDCodeSwitching(language=language),
        Touche2020v3RetrievalCodeSwitching(language=language),
        HumanEvalRetrievalCodeSwitching(language=language),
    ]


def load_cs_mteb_ir_tasks(language):
    return [
        Core17InstructionRetrievalCodeSwitching(language=language),
        News21InstructionRetrievalCodeSwitching(language=language),
        Robust04InstructionRetrievalCodeSwitching(language=language),
    ]


def load_cs_mteb_other_tasks(language):
    return [
        TweetSentimentExtractionClassificationCodeSwitching(language=language),
        ArXivHierarchicalClusteringP2PCodeSwitching(language=language),
        TwitterSemEval2015CodeSwitching(language=language),
        STSBenchmarkCodeSwitching(language=language),
        AskUbuntuDupQuestionsCodeSwitching(language=language),
    ]


def load_cs_mteb_all_tasks(language):
    tasks = []
    tasks.extend(load_cs_mteb_retrieval_tasks(language))
    tasks.extend(load_cs_mteb_ir_tasks(language))
    tasks.extend(load_cs_mteb_other_tasks(language))
    return tasks


def load_csr_l_retrieval_tasks(language):
    return [
        HumanEvalRetrievalCSRL(language=language),
        TRECCOVIDCSRL(language=language),
        Touche2020v3RetrievalCSRL(language=language),
    ]


def load_csr_l_ir_tasks(language):
    return [
        Core17InstructionRetrievalCSRL(language=language),
        News21InstructionRetrievalCSRL(language=language),
        Robust04InstructionRetrievalCSRL(language=language),
    ]


def load_csr_l_all_tasks(language):
    tasks = []
    tasks.extend(load_csr_l_retrieval_tasks(language))
    tasks.extend(load_csr_l_ir_tasks(language))
    return tasks


def load_original_retrieval_tasks():
    return [
        ArguAna(),
        ClimateFEVERHardNegativesV2(),
        TRECCOVID(),
        Touche2020v3Retrieval(),
        HumanEvalRetrieval(),
    ]


def load_original_ir_tasks():
    return [
        Core17InstructionRetrieval(),
        News21InstructionRetrieval(),
        Robust04InstructionRetrieval(),
    ]


def load_original_other_tasks():
    return [
        TweetSentimentExtractionClassificationV2(),
        ArXivHierarchicalClusteringP2P(),
        TwitterSemEval2015PC(),
        STSBenchmarkSTS(),
        AskUbuntuDupQuestions(),
    ]


def load_original_all_tasks():
    tasks = []
    tasks.extend(load_original_retrieval_tasks())
    tasks.extend(load_original_ir_tasks())
    tasks.extend(load_original_other_tasks())
    return tasks


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="MTEB configuration selection"
    )
    parser.add_argument("--tasks", nargs='+', help="list of the tasks in types you need to run", type=str,
                        required=True)
    parser.add_argument("--model_path", required=True, help="Either hf path or local path", type=str)
    parser.add_argument("--batch_size", required=True, help="batch_size", type=int, default=32)
    parser.add_argument("--output_subfolder_name", help="subfolder name for output", type=str, default=None)
    parser.add_argument(
        "--evaluation_output_dir",
        required=True,
        help="Evaluation output directory",
        type=str,
    )

    args = parser.parse_args()

    current_time = datetime.now()
    date = current_time.strftime("%Y%m%d")

    if args.output_subfolder_name:
        output_dir = os.path.join(args.evaluation_output_dir, date, args.output_subfolder_name)
    else:
        output_dir = os.path.join(args.evaluation_output_dir, date)

    All_tasks_names = []

    for task in args.tasks:
        # ==================== CS-MTEB tasks (9 languages) ====================
        # Usage: --tasks CS-MTEB_zh CS-MTEB_ja CS-MTEB_de ...
        if task.startswith("CS-MTEB_"):
            lang = task.split("_", 1)[1]
            if lang not in CS_MTEB_LANGS:
                print(f"ERROR: CS-MTEB does not support language '{lang}'. Supported: {CS_MTEB_LANGS}")
                continue
            curr_tasks = load_cs_mteb_all_tasks(lang)

        # ==================== CSR-L tasks (zh/ja only) ====================
        # Usage: --tasks CSR-L_zh CSR-L_ja
        elif task.startswith("CSR-L_"):
            lang = task.split("_", 1)[1]
            if lang not in CSR_L_LANGS:
                print(f"ERROR: CSR-L does not support language '{lang}'. Supported: {CSR_L_LANGS}")
                continue
            curr_tasks = load_csr_l_all_tasks(lang)

        # ==================== Original (English) tasks ====================
        # The non-CodeSwitching counterparts of the CS-MTEB tasks.
        # Usage: --tasks Original | Original_retrieval | Original_ir | Original_other
        elif task == "Original":
            curr_tasks = load_original_all_tasks()
        elif task == "Original_retrieval":
            curr_tasks = load_original_retrieval_tasks()
        elif task == "Original_ir":
            curr_tasks = load_original_ir_tasks()
        elif task == "Original_other":
            curr_tasks = load_original_other_tasks()

        else:
            print(f"ERROR: Task '{task}' is not supported!")
            print(f"  Supported formats:")
            print(f"    CS-MTEB_{{lang}}  (lang: {CS_MTEB_LANGS})")
            print(f"    CSR-L_{{lang}}    (lang: {CSR_L_LANGS})")
            print(f"    Original | Original_retrieval | Original_ir | Original_other")
            continue
        All_tasks_names.extend(curr_tasks)

    # Load model
    if "aligned" in args.model_path.lower():
        print(f"[INFO] Detected aligned model, using load_model_ST (with prompt support)")
        model = load_model_ST(args.model_path)
    else:
        print(f"[INFO] Using load_model (MTEB built-in config)")
        model = load_model(args.model_path)

    evaluation = MTEB(tasks=All_tasks_names)

    model_name = os.path.basename(args.model_path)
    results = evaluation.run(
        model,
        encode_kwargs={"batch_size": args.batch_size, "show_progress_bar": True},
        output_folder=output_dir,
        model_name=model_name,
    )
    print(results)
