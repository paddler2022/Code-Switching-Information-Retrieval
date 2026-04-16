from __future__ import annotations

from datasets import load_dataset
from tqdm import tqdm

from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata

CS_MTEB_REPO = "UTokyo-Yokoya-Lab/trec-covid_CS-MTEB"
SUPPORTED_LANGUAGES = ["zh", "ja", "de", "es", "ko", "fr", "it", "pt", "nl"]


class TRECCOVIDCodeSwitching(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="TRECCOVIDCodeSwitching",
        description="TREC-COVID Code-Switching variant.",
        reference="https://ir.nist.gov/covidSubmit/index.html",
        dataset={"path": CS_MTEB_REPO, "revision": "main"},
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=None,
        domains=["Medical", "Academic", "Written"],
        task_subtypes=["Question answering"],
        license=None,
        annotations_creators=None,
        dialect=None,
        sample_creation=None,
        bibtex_citation=r"""
@misc{roberts2021searching,
  archiveprefix = {arXiv},
  author = {Kirk Roberts and Tasmeer Alam and Steven Bedrick and Dina Demner-Fushman and Kyle Lo and Ian Soboroff and Ellen Voorhees and Lucy Lu Wang and William R Hersh},
  eprint = {2104.09632},
  primaryclass = {cs.IR},
  title = {Searching for Scientific Evidence in a Pandemic: An Overview of TREC-COVID},
  year = {2021},
}
""",
        prompt={"query": "Given a query on COVID-19, retrieve documents that answer the query"},
    )

    def __init__(self, language: str = "zh", **kwargs):
        super().__init__(**kwargs)
        if language not in SUPPORTED_LANGUAGES:
            raise ValueError(f"Unsupported language '{language}'. Supported: {SUPPORTED_LANGUAGES}")
        self.language = language
        self.metadata.name = f"TRECCOVIDCodeSwitching_{language}"

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
            score = int(item.get('score', 1))
            if qid in self.queries["test"]:
                if qid not in self.relevant_docs["test"]:
                    self.relevant_docs["test"][qid] = {}
                self.relevant_docs["test"][qid][doc_id] = score

        print(f"Loaded {len(self.queries['test'])} queries, {len(self.corpus['test'])} documents, {len(self.relevant_docs['test'])} qrels")
        self.data_loaded = True
