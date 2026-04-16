from __future__ import annotations

import json
import os

from datasets import Dataset, DatasetDict

from mteb.abstasks.task_metadata import TaskMetadata
from mteb.abstasks.text.bitext_mining import AbsTaskBitextMining

_SPLIT = ["train"]


def load_jsonl(filepath):
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


class BibleNLPBitextMiningCodeSwitching(AbsTaskBitextMining):

    fast_loading = True
    metadata = TaskMetadata(
        name="BibleNLPBitextMiningCodeSwitching",
        dataset={
            "path": "davidstap/biblenlp-corpus-mmteb",
            "revision": "264a18480c529d9e922483839b4b9758e690b762",
        },
        description="BibleNLP bitext mining Code-Switching variant. All data loaded from local file. eng-cmn only.",
        reference="https://arxiv.org/abs/2304.09919",
        type="BitextMining",
        category="t2t",
        modalities=["text"],
        eval_splits=_SPLIT,
        eval_langs={"eng-cmn": ["eng-Latn", "cmn-Hans"]},
        main_score="f1",
        date=("1997-01-01", "2020-12-31"),
        domains=["Religious", "Written"],
        task_subtypes=[],
        license="cc-by-sa-4.0",
        annotations_creators="expert-annotated",
        dialect=[],
        sample_creation="created",
        bibtex_citation=r"""
@article{akerman2023ebible,
  author = {Akerman, Vesa and Baines, David and Daspit, Damien and Hermjakob, Ulf and Jang, Taeho and Leong, Colin and Martin, Michael and Mathew, Joel and Robie, Jonathan and Schwarting, Marcus},
  journal = {arXiv preprint arXiv:2304.09919},
  title = {The eBible Corpus: Data and Model Benchmarks for Bible Translation for Low-Resource Languages},
  year = {2023},
}
""",
    )

    def __init__(self, train_file: str = None, **kwargs):

        super().__init__(**kwargs)
        self.train_file = train_file or os.getenv("BIBLENLP_TRAIN_FILE")

    def load_data(self, **kwargs):

        if self.data_loaded:
            return

        if not self.train_file:
            raise ValueError(
                "Data file path not provided. "
                "Please pass train_file parameter or set BIBLENLP_TRAIN_FILE environment variable."
            )

        if not os.path.exists(self.train_file):
            raise FileNotFoundError(f"Data file not found: {self.train_file}")

        print(f"Loading data from local file: {self.train_file}")
        local_data = load_jsonl(self.train_file)

        sentence1 = []
        sentence2 = []
        for item in local_data:
            sentence1.append(item.get('sentence1', ''))
            sentence2.append(item.get('sentence2', ''))

        self.dataset = DatasetDict({
            "train": Dataset.from_dict({
                "sentence1": sentence1,
                "sentence2": sentence2,
            })
        })

        print(f"Loaded {len(sentence1)} samples for train split")
        self.data_loaded = True
