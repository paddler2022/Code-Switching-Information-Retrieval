from __future__ import annotations

from datasets import load_dataset
from tqdm import tqdm

from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata

CS_MTEB_REPO = "UTokyo-Yokoya-Lab/ClimateFEVER_hardnegatives_CS-MTEB"
SUPPORTED_LANGUAGES = ["zh", "ja", "de", "es", "ko", "fr", "it", "pt", "nl"]


class ClimateFEVERHardNegativesV2CodeSwitching(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="ClimateFEVERHardNegatives.v2CodeSwitching",
        description="ClimateFEVER HardNegatives V2 Code-Switching variant.",
        reference="https://www.sustainablefinance.uzh.ch/en/research/climate-fever.html",
        dataset={"path": CS_MTEB_REPO, "revision": "main"},
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=("2001-01-01", "2020-12-31"),
        domains=["Encyclopaedic", "Written"],
        task_subtypes=["Claim verification"],
        license="cc-by-sa-4.0",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
        adapted_from=["ClimateFEVER"],
        bibtex_citation=r"""
@misc{diggelmann2021climatefever,
  archiveprefix = {arXiv},
  author = {Thomas Diggelmann and Jordan Boyd-Graber and Jannis Bulian and Massimiliano Ciaramita and Markus Leippold},
  eprint = {2012.00614},
  primaryclass = {cs.CL},
  title = {CLIMATE-FEVER: A Dataset for Verification of Real-World Climate Claims},
  year = {2021},
}
""",
        prompt={"query": "Given a claim about climate change, retrieve documents that support or refute the claim"},
    )

    def __init__(self, language: str = "zh", **kwargs):
        super().__init__(**kwargs)
        if language not in SUPPORTED_LANGUAGES:
            raise ValueError(f"Unsupported language '{language}'. Supported: {SUPPORTED_LANGUAGES}")
        self.language = language
        self.metadata.name = f"ClimateFEVERHardNegatives.v2CodeSwitching_{language}"

    def load_data(self, **kwargs):
        if self.data_loaded:
            return

        repo = CS_MTEB_REPO
        query_config = f"queries_{self.language}_en"

        print(f"Loading queries from {repo} config={query_config}")
        query_ds = load_dataset(repo, query_config)
        query_split = list(query_ds.values())[0]

        print(f"Loading corpus from {repo}")
        corpus_ds = load_dataset(repo, "corpus")
        corpus_split = list(corpus_ds.values())[0]

        print(f"Loading qrels from {repo}")
        qrels_ds = load_dataset(repo, "default")
        qrels_split = list(qrels_ds.values())[0]

        self.queries = {"test": {}}
        self.corpus = {"test": {}}
        self.relevant_docs = {"test": {}}

        for item in tqdm(query_split, desc="Loading queries"):
            qid = str(item.get('_id') or item.get('id'))
            self.queries["test"][qid] = item['text']

        for item in tqdm(corpus_split, desc="Loading corpus"):
            doc_id = str(item.get('_id') or item.get('id'))
            self.corpus["test"][doc_id] = {"title": item.get('title', ''), "text": item.get('text', '')}

        for item in tqdm(qrels_split, desc="Loading qrels"):
            qid = str(item.get('query-id'))
            doc_id = str(item.get('corpus-id'))
            score = int(item.get('score'))
            if qid in self.queries["test"]:
                if qid not in self.relevant_docs["test"]:
                    self.relevant_docs["test"][qid] = {}
                self.relevant_docs["test"][qid][doc_id] = score

        print(f"Loaded {len(self.queries['test'])} queries, {len(self.corpus['test'])} documents, {len(self.relevant_docs['test'])} qrels")
        self.data_loaded = True
