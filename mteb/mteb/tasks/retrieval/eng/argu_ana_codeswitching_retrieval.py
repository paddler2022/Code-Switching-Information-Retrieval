from __future__ import annotations

from datasets import load_dataset
from tqdm import tqdm

from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata

# CS-MTEB HuggingFace repo for this task
CS_MTEB_REPO = "UTokyo-Yokoya-Lab/arguana_CS-MTEB"

# Supported languages
SUPPORTED_LANGUAGES = ["zh", "ja", "de", "es", "ko", "fr", "it", "pt", "nl"]


class ArguAnaCodeSwitching(AbsTaskRetrieval):
    """
    ArguAna Code-Switching variant task.
    - queries: loaded from CS-MTEB HuggingFace repo (language-specific config)
    - corpus and qrels: loaded from the same repo (original data)

    Usage:
        task = ArguAnaCodeSwitching(language="ja")
    """

    ignore_identical_ids = True

    metadata = TaskMetadata(
        name="ArguAnaCodeSwitching",
        description="ArguAna Code-Switching variant. Retrieval of the Best Counterargument without Prior Topic Knowledge.",
        reference="http://argumentation.bplaced.net/arguana/data",
        dataset={
            "path": CS_MTEB_REPO,
            "revision": "main",
        },
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=["2018-01-01", "2018-07-01"],
        domains=["Social", "Web", "Written"],
        task_subtypes=["Discourse coherence"],
        license="cc-by-sa-4.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@inproceedings{wachsmuth2018retrieval,
  author = {Wachsmuth, Henning and Syed, Shahbaz and Stein, Benno},
  booktitle = {ACL},
  title = {Retrieval of the Best Counterargument without Prior Topic Knowledge},
  year = {2018},
}
""",
        prompt={"query": "Given a claim, find documents that refute the claim"},
    )

    def __init__(self, language: str = "zh", **kwargs):
        """
        Args:
            language: Language code for code-switching queries.
                      Supported: zh, ja, de, es, ko, fr, it, pt, nl
        """
        super().__init__(**kwargs)
        if language not in SUPPORTED_LANGUAGES:
            raise ValueError(f"Unsupported language '{language}'. Supported: {SUPPORTED_LANGUAGES}")
        self.language = language
        # Append language to task name so results save as ArguAnaCodeSwitching_{language}.json
        self.metadata.name = f"ArguAnaCodeSwitching_{language}"

    def load_data(self, **kwargs):
        if self.data_loaded:
            return

        repo = CS_MTEB_REPO
        query_config = f"queries_{self.language}_en"

        # Load queries from CS-MTEB repo
        print(f"Loading queries from {repo} config={query_config}")
        query_ds = load_dataset(repo, query_config)
        query_split = list(query_ds.values())[0]  # get the first (only) split

        # Load corpus and qrels from the same repo
        print(f"Loading corpus from {repo}")
        corpus_ds = load_dataset(repo, "corpus")
        corpus_split = list(corpus_ds.values())[0]

        print(f"Loading qrels from {repo}")
        qrels_ds = load_dataset(repo, "default")
        qrels_split = list(qrels_ds.values())[0]

        # Build data structures
        self.queries = {"test": {}}
        self.corpus = {"test": {}}
        self.relevant_docs = {"test": {}}

        for item in tqdm(query_split, desc="Loading queries"):
            qid = str(item.get('_id') or item.get('id'))
            self.queries["test"][qid] = item['text']

        for item in tqdm(corpus_split, desc="Loading corpus"):
            doc_id = str(item.get('_id') or item.get('id'))
            self.corpus["test"][doc_id] = {
                "title": item.get('title', ''),
                "text": item.get('text', '')
            }

        for item in tqdm(qrels_split, desc="Loading qrels"):
            qid = str(item.get('query-id'))
            doc_id = str(item.get('corpus-id'))
            score = int(item.get('score'))
            if qid in self.queries["test"]:
                if qid not in self.relevant_docs["test"]:
                    self.relevant_docs["test"][qid] = {}
                self.relevant_docs["test"][qid][doc_id] = score

        print(f"Loaded {len(self.queries['test'])} queries")
        print(f"Loaded {len(self.corpus['test'])} documents")
        print(f"Loaded {len(self.relevant_docs['test'])} query-document relevance pairs")

        self.data_loaded = True
