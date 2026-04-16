from __future__ import annotations

import json
import os

from datasets import Dataset, DatasetDict

from mteb.abstasks.task_metadata import TaskMetadata
from mteb.abstasks.text.bitext_mining import AbsTaskBitextMining

_SPLITS = ["test"]


def load_jsonl(filepath):

    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


class BUCCBitextMiningCodeSwitching(AbsTaskBitextMining):
    """
    BUCC BitextMining Code-Switching Task
    """

    fast_loading = True
    metadata = TaskMetadata(
        name="BUCCBitextMiningCodeSwitching",
        dataset={
            "path": "mteb/bucc-bitext-mining",
            "revision": "1739dc11ffe9b7bfccd7f3d585aeb4c544fc6677",
        },
        description="BUCC bitext mining Code-Switching variant (based on BUCC.v2). All data loaded from local file. zh-en only.",
        reference="https://comparable.limsi.fr/bucc2018/bucc2018-task.html",
        type="BitextMining",
        category="t2t",
        modalities=["text"],
        eval_splits=_SPLITS,
        eval_langs={"zh-en": ["cmn-Hans", "eng-Latn"]},
        main_score="f1",
        date=("2017-01-01", "2018-12-31"),
        domains=["Written"],
        task_subtypes=[],
        license="not specified",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="human-translated",
        bibtex_citation=r"""
@inproceedings{zweigenbaum-etal-2017-overview,
  address = {Vancouver, Canada},
  author = {Zweigenbaum, Pierre  and
Sharoff, Serge  and
Rapp, Reinhard},
  booktitle = {Proceedings of the 10th Workshop on Building and Using Comparable Corpora},
  doi = {10.18653/v1/W17-2512},
  editor = {Sharoff, Serge  and
Zweigenbaum, Pierre  and
Rapp, Reinhard},
  month = aug,
  pages = {60--67},
  publisher = {Association for Computational Linguistics},
  title = {Overview of the Second {BUCC} Shared Task: Spotting Parallel Sentences in Comparable Corpora},
  url = {https://aclanthology.org/W17-2512},
  year = {2017},
}
""",
    )

    def __init__(self, test_file: str = None, **kwargs):

        super().__init__(**kwargs)
        self.test_file = test_file or os.getenv("BUCC_TEST_FILE")

    def load_data(self, **kwargs):

        if self.data_loaded:
            return

        if not self.test_file:
            raise ValueError(
                "Data file path not provided. "
                "Please pass test_file parameter or set BUCC_TEST_FILE environment variable."
            )

        if not os.path.exists(self.test_file):
            raise FileNotFoundError(f"Data file not found: {self.test_file}")

        print(f"Loading data from local file: {self.test_file}")
        local_data = load_jsonl(self.test_file)

        sentence1 = []
        sentence2 = []
        for item in local_data:
            sentence1.append(item.get('sentence1', ''))
            sentence2.append(item.get('sentence2', ''))

        self.dataset = DatasetDict({
            "test": Dataset.from_dict({
                "sentence1": sentence1,
                "sentence2": sentence2,
            })
        })

        print(f"Loaded {len(sentence1)} samples for test split")
        self.data_loaded = True
