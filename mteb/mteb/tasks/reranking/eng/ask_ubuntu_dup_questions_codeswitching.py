from __future__ import annotations

from datasets import load_dataset
from tqdm import tqdm

from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata

CS_MTEB_REPO = "UTokyo-Yokoya-Lab/AskUbuntuDupQuestions_CS-MTEB"
SUPPORTED_LANGUAGES = ["zh", "ja", "de", "es", "ko", "fr", "it", "pt", "nl"]


class AskUbuntuDupQuestionsCodeSwitching(AbsTaskRetrieval):
    """AskUbuntuDupQuestions Code-Switching variant. Queries are rewritten in
    {lang}-English code-switching style; corpus/qrels/top_ranked come from the
    same CS-MTEB repo."""

    metadata = TaskMetadata(
        name="AskUbuntuDupQuestionsCodeSwitching",
        description="AskUbuntuDupQuestions Code-Switching variant. Questions from AskUbuntu with manual annotations marking pairs of questions as similar or non-similar.",
        reference="https://github.com/taolei87/askubuntu",
        dataset={"path": CS_MTEB_REPO, "revision": "main"},
        type="Reranking",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="map_at_1000",
        date=None,
        domains=["Programming", "Web"],
        task_subtypes=None,
        license=None,
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@article{wang-2021-TSDAE,
  author = {Wang, Kexin and Reimers, Nils and  Gurevych, Iryna},
  journal = {arXiv preprint arXiv:2104.06979},
  month = {4},
  title = {TSDAE: Using Transformer-based Sequential Denoising Auto-Encoderfor Unsupervised Sentence Embedding Learning},
  url = {https://arxiv.org/abs/2104.06979},
  year = {2021},
}
""",
    )

    def __init__(self, language: str = "zh", **kwargs):
        super().__init__(**kwargs)
        if language not in SUPPORTED_LANGUAGES:
            raise ValueError(f"Unsupported language '{language}'. Supported: {SUPPORTED_LANGUAGES}")
        self.language = language
        self.metadata.name = f"AskUbuntuDupQuestionsCodeSwitching_{language}"

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

        print(f"Loading top_ranked from {repo}")
        top_ranked_ds = load_dataset(repo, "top_ranked")
        top_ranked_split = list(top_ranked_ds.values())[0]

        self.queries = {"test": {}}
        self.corpus = {"test": {}}
        self.relevant_docs = {"test": {}}
        self.top_ranked = {"test": {}}

        for item in tqdm(query_split, desc="Loading queries"):
            qid = str(item.get("_id") or item.get("id"))
            self.queries["test"][qid] = item["text"]

        for item in tqdm(corpus_split, desc="Loading corpus"):
            doc_id = str(item.get("_id") or item.get("id"))
            self.corpus["test"][doc_id] = {
                "title": item.get("title", ""),
                "text": item.get("text", ""),
            }

        for item in tqdm(qrels_split, desc="Loading qrels"):
            qid = str(item.get("query-id"))
            doc_id = str(item.get("corpus-id"))
            score = int(item.get("score", 1))
            if qid in self.queries["test"]:
                self.relevant_docs["test"].setdefault(qid, {})[doc_id] = score

        for item in tqdm(top_ranked_split, desc="Loading top_ranked"):
            qid = str(item.get("query-id"))
            doc_ids = [str(x) for x in item.get("corpus-ids", [])]
            if qid in self.queries["test"]:
                self.top_ranked["test"][qid] = doc_ids

        print(f"Loaded {len(self.queries['test'])} queries, "
              f"{len(self.corpus['test'])} documents, "
              f"{len(self.relevant_docs['test'])} qrels, "
              f"{len(self.top_ranked['test'])} top_ranked lists")
        self.data_loaded = True
