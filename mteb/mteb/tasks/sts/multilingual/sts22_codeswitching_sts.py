from __future__ import annotations

import json
import os
from datasets import Dataset

from mteb.abstasks.sts import AbsTaskSTS
from mteb.abstasks.task_metadata import TaskMetadata


def load_jsonl(filepath):
    """Load JSONL file"""
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


_LANGUAGES = {
    "en": ["eng-Latn"],
}


class STS22CodeSwitching(AbsTaskSTS):
    """
    STS22 Code-Switching variant task
    - fully loaded from local jsonl file (sentence1, sentence2, score)

    Local data format requirements:
    {"sentence1": "...", "sentence2": "...", "score": 3.2}
    """

    fast_loading = True

    metadata = TaskMetadata(
        name="STS22CodeSwitching",
        description="STS22 Code-Switching variant. SemEval 2022 Task 8: Multilingual News Article Similarity. Sentence pairs are loaded from local file, scores from official dataset.",
        reference="https://competitions.codalab.org/competitions/33835",
        dataset={
            "path": "mteb/sts22-crosslingual-sts",
            "revision": "de9d86b3b84231dc21f76c7b7af1f28e2f57f6e3",
        },
        type="STS",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=_LANGUAGES,
        main_score="cosine_spearman",
        date=("2020-01-01", "2020-06-11"),
        domains=["News", "Written"],
        task_subtypes=[],
        license="not specified",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@inproceedings{chen-etal-2022-semeval,
  address = {Seattle, United States},
  author = {Chen, Xi  and
Zeynali, Ali  and
Camargo, Chico  and
Fl{\"o}ck, Fabian  and
Gaffney, Devin  and
Grabowicz, Przemyslaw  and
Hale, Scott  and
Jurgens, David  and
Samory, Mattia},
  booktitle = {Proceedings of the 16th International Workshop on Semantic Evaluation (SemEval-2022)},
  doi = {10.18653/v1/2022.semeval-1.155},
  editor = {Emerson, Guy  and
Schluter, Natalie  and
Stanovsky, Gabriel  and
Kumar, Ritesh  and
Palmer, Alexis  and
Schneider, Nathan  and
Singh, Siddharth  and
Ratan, Shyam},
  month = jul,
  pages = {1094--1106},
  publisher = {Association for Computational Linguistics},
  title = {{S}em{E}val-2022 Task 8: Multilingual news article similarity},
  url = {https://aclanthology.org/2022.semeval-1.155},
  year = {2022},
}
""",
    )

    min_score = 1
    max_score = 4

    def __init__(self, data_file: str = None, **kwargs):
        """
        Initialize task

        Args:
            data_file: Local data jsonl file path. If None, reads from environment variable STS22_DATA_FILE
        """
        super().__init__(**kwargs)
        self.data_file = data_file or os.getenv("STS22_DATA_FILE")

    def load_data(self, **kwargs):
        """
        Load data: fully loaded from local jsonl file (sentence1, sentence2, score)
        """
        if self.data_loaded:
            return

        # ========== 1. Validate data file path ==========
        if not self.data_file:
            raise ValueError(
                "Data file path not provided. "
                "Please pass data_file parameter or set STS22_DATA_FILE environment variable."
            )

        if not os.path.exists(self.data_file):
            raise FileNotFoundError(f"Data file not found: {self.data_file}")

        # ========== 2. Load data from local file ==========
        print(f"Loading data from local file: {self.data_file}")
        local_data = load_jsonl(self.data_file)

        # ========== 3. Build dataset ==========
        sentences1 = []
        sentences2 = []
        scores = []

        for item in local_data:
            sentences1.append(item['sentence1'])
            sentences2.append(item['sentence2'])
            scores.append(item['score'])

        # Build {"en": {"test": Dataset}} structure
        self.dataset = {
            "en": {
                "test": Dataset.from_dict({
                    "sentence1": sentences1,
                    "sentence2": sentences2,
                    "score": scores,
                })
            }
        }

        print(f"Loaded {len(sentences1)} sentence pairs for test split")
        self.data_loaded = True
