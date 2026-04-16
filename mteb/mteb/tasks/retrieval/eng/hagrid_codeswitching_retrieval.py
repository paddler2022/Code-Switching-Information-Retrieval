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


class HagridRetrievalCodeSwitching(AbsTaskRetrieval):

    metadata = TaskMetadata(
        name="HagridRetrievalCodeSwitching",
        dataset={
            "path": "mteb/HagridRetrieval",
            "revision": "ae4f8bebcb82af2028863b778e1eebf4f5f23628",
        },
        reference="https://github.com/project-miracl/hagrid",
        description=(
            "HAGRID Code-Switching variant with custom queries. "
            "HAGRID (Human-in-the-loop Attributable Generative Retrieval for Information-seeking Dataset) "
            "is a dataset for generative information-seeking scenarios. "
            "Corpus and qrels are loaded from the official dataset."
        ),
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["dev"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=("2019-02-01", "2022-10-18"),
        domains=["Encyclopaedic", "Written"],
        task_subtypes=[],
        license="apache-2.0",
        annotations_creators="expert-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@article{hagrid,
  author = {Ehsan Kamalloo and Aref Jafari and Xinyu Zhang and Nandan Thakur and Jimmy Lin},
  journal = {arXiv:2307.16883},
  title = {{HAGRID}: A Human-LLM Collaborative Dataset for Generative Information-Seeking with Attribution},
  year = {2023},
}
""",
    )

    def __init__(self, query_file: str = None, **kwargs):

        super().__init__(**kwargs)
        self.query_file = query_file or os.getenv("HAGRID_QUERY_FILE")

    def load_data(self, **kwargs):

        if self.data_loaded:
            return

        if not self.query_file:
            raise ValueError(
                "Query file path not provided. "
                "Please pass query_file parameter or set HAGRID_QUERY_FILE environment variable."
            )

        if not os.path.exists(self.query_file):
            raise FileNotFoundError(f"Query file not found: {self.query_file}")

        print(f"Loading queries from local file: {self.query_file}")
        query_lines = load_jsonl(self.query_file)

        dataset_path = self.metadata.dataset["path"]
        revision = self.metadata.dataset["revision"]

        print(f"Loading corpus from HuggingFace: {dataset_path}")
        corpus_dataset = load_dataset(dataset_path, "corpus", revision=revision)
        corpus_lines = list(corpus_dataset['dev'])

        print(f"Loading qrels from HuggingFace: {dataset_path}")
        qrels_dataset = load_dataset(dataset_path, "qrels", revision=revision)
        qrels_lines = list(qrels_dataset['dev'])

        self.queries = {"dev": {}}
        self.corpus = {"dev": {}}
        self.relevant_docs = {"dev": {}}

        for item in tqdm(query_lines, desc="Loading queries"):
            qid = str(item.get('id') or item.get('_id'))
            text = item.get('text', '')
            self.queries["dev"][qid] = text

        for item in tqdm(corpus_lines, desc="Loading corpus"):
            doc_id = str(item.get('_id') or item.get('id'))
            self.corpus["dev"][doc_id] = {
                "title": item.get('title', ''),
                "text": item.get('text', '')
            }

        for item in tqdm(qrels_lines, desc="Loading qrels"):
            qid = str(item.get('query-id'))
            doc_id = str(item.get('corpus-id'))
            score = item.get('score', 1)

            if qid in self.queries["dev"]:
                if qid not in self.relevant_docs["dev"]:
                    self.relevant_docs["dev"][qid] = {}
                self.relevant_docs["dev"][qid][doc_id] = score

        print(f"Loaded {len(self.queries['dev'])} queries")
        print(f"Loaded {len(self.corpus['dev'])} documents")
        print(f"Loaded {len(self.relevant_docs['dev'])} query-document relevance pairs")

        self.data_loaded = True
