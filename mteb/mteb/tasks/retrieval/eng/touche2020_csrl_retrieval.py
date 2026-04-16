from __future__ import annotations

from datasets import load_dataset
from tqdm import tqdm

from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata

CSR_L_REPO = "UTokyo-Yokoya-Lab/webis-touche2020-v3-CSR-L"
SUPPORTED_LANGUAGES = ["zh", "ja"]


class Touche2020v3RetrievalCSRL(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="Touche2020Retrieval.v3CSRL",
        description="Touché 2020 Argument Retrieval CSR-L variant (Chinese/Japanese code-switching).",
        reference="https://github.com/castorini/touche-error-analysis",
        dataset={"path": CSR_L_REPO, "revision": "main"},
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=("2020-09-23", "2020-09-23"),
        domains=["Academic"],
        task_subtypes=["Question answering"],
        license="cc-by-sa-4.0",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@inproceedings{Thakur_etal_SIGIR2024,
  author = {Nandan Thakur and Luiz Bonifacio and Maik {Fr\"{o}be} and Alexander Bondarenko and Ehsan Kamalloo and Martin Potthast and Matthias Hagen and Jimmy Lin},
  booktitle = {Proceedings of the 47th International ACM SIGIR Conference on Research and Development in Information Retrieval},
  title = {Systematic Evaluation of Neural Retrieval Models on the {Touch\'{e}} 2020 Argument Retrieval Subset of {BEIR}},
  year = {2024},
}
""",
        adapted_from=["Touche2020"],
    )

    def __init__(self, language: str = "zh", **kwargs):
        super().__init__(**kwargs)
        if language not in SUPPORTED_LANGUAGES:
            raise ValueError(f"Unsupported language '{language}'. CSR-L only supports: {SUPPORTED_LANGUAGES}")
        self.language = language
        self.metadata.name = f"Touche2020Retrieval.v3CSRL_{language}"

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
        qrels_split = list(load_dataset(repo, "default").values())[0]

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
            score = int(item.get('score', 1))
            if qid in self.queries["test"]:
                if qid not in self.relevant_docs["test"]:
                    self.relevant_docs["test"][qid] = {}
                self.relevant_docs["test"][qid][doc_id] = score

        print(f"Loaded {len(self.queries['test'])} queries, {len(self.corpus['test'])} documents, {len(self.relevant_docs['test'])} qrels")
        self.data_loaded = True
