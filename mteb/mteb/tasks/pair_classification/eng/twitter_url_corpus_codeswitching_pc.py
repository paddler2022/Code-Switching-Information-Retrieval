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


class TwitterURLCorpusCodeSwitching(AbsTaskPairClassification):
    """
    TwitterURLCorpus Code-Switching Task
    """

    metadata = TaskMetadata(
        name="TwitterURLCorpusCodeSwitching",
        dataset={
            "path": "mteb/twitterurlcorpus-pairclassification",
            "revision": "8b6510b0b1fa4e4c4f879467980e9be563ec1cdf",
        },
        description="TwitterURLCorpus Code-Switching variant. Paraphrase-Pairs of Tweets. All data loaded from local file.",
        reference="https://languagenet.github.io/",
        category="t2t",
        modalities=["text"],
        type="PairClassification",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="max_ap",
        date=None,
        domains=["Social", "Written"],
        task_subtypes=[],
        license="not specified",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@inproceedings{lan-etal-2017-continuously,
  address = {Copenhagen, Denmark},
  author = {Lan, Wuwei  and
Qiu, Siyu  and
He, Hua  and
Xu, Wei},
  booktitle = {Proceedings of the 2017 Conference on Empirical Methods in Natural Language Processing},
  doi = {10.18653/v1/D17-1126},
  editor = {Palmer, Martha  and
Hwa, Rebecca  and
Riedel, Sebastian},
  month = sep,
  pages = {1224--1234},
  publisher = {Association for Computational Linguistics},
  title = {A Continuously Growing Dataset of Sentential Paraphrases},
  url = {https://aclanthology.org/D17-1126},
  year = {2017},
}
""",
        prompt="Retrieve tweets that are semantically similar to the given tweet",
    )

    def __init__(self, test_file: str = None, **kwargs):

        super().__init__(**kwargs)
        self.test_file = test_file or os.getenv("TWITTER_URL_TEST_FILE")

    def load_data(self, **kwargs):
        """
        Load data from local file
        """
        if self.data_loaded:
            return

        if not self.test_file:
            raise ValueError(
                "Data file path not provided. "
                "Please pass test_file parameter or set TWITTER_URL_TEST_FILE environment variable."
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
