from __future__ import annotations

from datasets import load_dataset
from tqdm import tqdm

from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata

CSR_L_REPO = "UTokyo-Yokoya-Lab/HumanEvalRetrieval-CSR-L"
SUPPORTED_LANGUAGES = ["zh", "ja"]


class HumanEvalRetrievalCSRL(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="HumanEvalRetrievalCSRL",
        description="HumanEval Retrieval CSR-L variant (Chinese/Japanese code-switching).",
        reference="https://huggingface.co/datasets/embedding-benchmark/HumanEval",
        dataset={"path": CSR_L_REPO, "revision": "main"},
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn", "python-Code"],
        main_score="ndcg_at_10",
        date=("2021-01-01", "2021-12-31"),
        domains=["Programming"],
        task_subtypes=["Code retrieval"],
        license="mit",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation="""@article{chen2021evaluating,
  archiveprefix = {arXiv},
  author = {Mark Chen and Jerry Tworek and Heewoo Jun and others},
  eprint = {2107.03374},
  primaryclass = {cs.LG},
  title = {Evaluating Large Language Models Trained on Code},
  year = {2021},
}""",
    )

    def __init__(self, language: str = "zh", **kwargs):
        super().__init__(**kwargs)
        if language not in SUPPORTED_LANGUAGES:
            raise ValueError(f"Unsupported language '{language}'. CSR-L only supports: {SUPPORTED_LANGUAGES}")
        self.language = language
        self.metadata.name = f"HumanEvalRetrievalCSRL_{language}"

    def load_data(self, **kwargs):
        if self.data_loaded:
            return

        repo = CSR_L_REPO
        query_config = f"queries_{self.language}_en"

        print(f"Loading queries from {repo} config={query_config}")
        query_split = list(load_dataset(repo, query_config).values())[0]

        print(f"Loading corpus from {repo}")
        corpus_split = list(load_dataset(repo, "corpus").values())[0]

        print(f"Loading qrels from {repo}")
        qrels_split = list(load_dataset(repo, "qrels").values())[0]

        self.queries = {"test": {}}
        self.corpus = {"test": {}}
        self.relevant_docs = {"test": {}}

        for item in tqdm(query_split, desc="Loading queries"):
            qid = str(item.get('id') or item.get('_id'))
            self.queries["test"][qid] = item['text']

        for item in tqdm(corpus_split, desc="Loading corpus"):
            doc_id = str(item.get('id') or item.get('_id'))
            self.corpus["test"][doc_id] = {"title": "", "text": item.get('text', '')}

        for item in tqdm(qrels_split, desc="Loading qrels"):
            qid = str(item.get('query-id'))
            doc_id = str(item.get('corpus-id'))
            score = int(item.get('score', 1))
            if qid in self.queries["test"]:
                if qid not in self.relevant_docs["test"]:
                    self.relevant_docs["test"][qid] = {}
                self.relevant_docs["test"][qid][doc_id] = score

        print(f"Loaded {len(self.queries['test'])} queries, {len(self.corpus['test'])} documents, {len(self.relevant_docs['test'])} qrels")
        self.data_loaded = True
