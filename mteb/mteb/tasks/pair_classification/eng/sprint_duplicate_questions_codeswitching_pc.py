from __future__ import annotations

import json
import os

from datasets import Dataset, DatasetDict

from mteb.abstasks.pair_classification import AbsTaskPairClassification
from mteb.abstasks.task_metadata import TaskMetadata


def load_jsonl(filepath):

    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


class SprintDuplicateQuestionsPCCodeSwitching(AbsTaskPairClassification):
    """
    SprintDuplicateQuestionsPC Code-Switching Task
    """

    metadata = TaskMetadata(
        name="SprintDuplicateQuestionsCodeSwitching",
        description="SprintDuplicateQuestions Code-Switching variant. Duplicate questions from the Sprint community. All data loaded from local file.",
        reference="https://www.aclweb.org/anthology/D18-1131/",
        dataset={
            "path": "mteb/sprintduplicatequestions-pairclassification",
            "revision": "d66bd1f72af766a5cc4b0ca5e00c162f89e8cc46",
        },
        type="PairClassification",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="max_ap",
        date=(
            "2018-10-01",
            "2018-12-30",
        ),
        domains=["Programming", "Written"],
        task_subtypes=["Duplicate Detection"],
        license="not specified",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        prompt="Retrieve duplicate questions from Sprint forum",
        bibtex_citation=r"""
@inproceedings{shah-etal-2018-adversarial,
  address = {Brussels, Belgium},
  author = {Shah, Darsh  and
Lei, Tao  and
Moschitti, Alessandro  and
Romeo, Salvatore  and
Nakov, Preslav},
  booktitle = {Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing},
  doi = {10.18653/v1/D18-1131},
  editor = {Riloff, Ellen  and
Chiang, David  and
Hockenmaier, Julia  and
Tsujii, Jun{'}ichi},
  month = oct # {-} # nov,
  pages = {1056--1063},
  publisher = {Association for Computational Linguistics},
  title = {Adversarial Domain Adaptation for Duplicate Question Detection},
  url = {https://aclanthology.org/D18-1131},
  year = {2018},
}
""",
    )

    def __init__(self, test_file: str = None, **kwargs):

        super().__init__(**kwargs)
        self.test_file = test_file or os.getenv("SPRINT_DUP_TEST_FILE")

    def load_data(self, **kwargs):
        """
        Load data from local file
        """
        if self.data_loaded:
            return

        if not self.test_file:
            raise ValueError(
                "Data file path not provided. "
                "Please pass test_file parameter or set SPRINT_DUP_TEST_FILE environment variable."
            )

        if not os.path.exists(self.test_file):
            raise FileNotFoundError(f"Data file not found: {self.test_file}")
        print(f"Loading data from local file: {self.test_file}")
        local_data = load_jsonl(self.test_file)
        sentence1 = []
        sentence2 = []
        labels = []
        for item in local_data:
            sentence1.append(item.get('sentence1', item.get('sent1', '')))
            sentence2.append(item.get('sentence2', item.get('sent2', '')))
            labels.append(item.get('labels', item.get('label', 0)))

        self.dataset = DatasetDict({
            "test": Dataset.from_dict({
                "sentence1": sentence1,
                "sentence2": sentence2,
                "labels": labels,
            })
        })

        print(f"Loaded {len(sentence1)} samples for test split")
        self.data_loaded = True
