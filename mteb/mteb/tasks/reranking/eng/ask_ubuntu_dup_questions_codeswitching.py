from __future__ import annotations

import json
import os
from datasets import load_dataset
from tqdm import tqdm

from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


def load_jsonl(filepath):

    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


class AskUbuntuDupQuestionsCodeSwitching(AbsTaskRetrieval):
    """
    AskUbuntuDupQuestions Code-Switching Task
    """

    metadata = TaskMetadata(
        name="AskUbuntuDupQuestionsCodeSwitching",
        description="AskUbuntuDupQuestions Code-Switching variant. Questions from AskUbuntu with manual annotations marking pairs of questions as similar or non-similar. Corpus and qrels are loaded from the official dataset.",
        reference="https://github.com/taolei87/askubuntu",
        dataset={
            "path": "mteb/AskUbuntuDupQuestions",
            "revision": "c5691e3c48741d5f83b5cc8e630653d7a8cfc048",
        },
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
        prompt="Retrieve duplicate questions from AskUbuntu forum",
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

    def __init__(self, query_file: str = None, **kwargs):

        super().__init__(**kwargs)
        self.query_file = query_file or os.getenv("ASKUBUNTU_QUERY_FILE")

    def load_data(self, **kwargs):
        """
        Load data
        - queries: Load from local file
        - corpus and qrels: load from huggingface
        """
        if self.data_loaded:
            return

        if not self.query_file:
            raise ValueError(
                "Query file path not provided. "
                "Please pass query_file parameter or set ASKUBUNTU_QUERY_FILE environment variable."
            )

        if not os.path.exists(self.query_file):
            raise FileNotFoundError(f"Query file not found: {self.query_file}")

        print(f"Loading queries from local file: {self.query_file}")
        query_lines = load_jsonl(self.query_file)

        dataset_path = self.metadata.dataset["path"]
        revision = self.metadata.dataset["revision"]

        print(f"Loading corpus from HuggingFace: {dataset_path}")
        corpus_dataset = load_dataset(dataset_path, "corpus", revision=revision)
        corpus_lines = list(corpus_dataset['test'])

        print(f"Loading qrels from HuggingFace: {dataset_path}")
        qrels_dataset = load_dataset(dataset_path, "default", revision=revision)
        qrels_lines = list(qrels_dataset['test'])

        self.queries = {"test": {}}
        self.corpus = {"test": {}}
        self.relevant_docs = {"test": {}}

        for idx, item in enumerate(tqdm(query_lines, desc="Loading queries")):
            try:
                qid = str(item.get('_id') or item['id'])
                text = item['text']
                self.queries["test"][qid] = text
            except KeyError as e:
                raise KeyError(f"Missing key {e} in query item {idx}: {item}")

        for item in tqdm(corpus_lines, desc="Loading corpus"):
            doc_id = str(item.get('_id') or item.get('id'))
            self.corpus["test"][doc_id] = {
                "title": item.get('title', ''),
                "text": item.get('text', '')
            }

        for item in tqdm(qrels_lines, desc="Loading qrels"):
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
