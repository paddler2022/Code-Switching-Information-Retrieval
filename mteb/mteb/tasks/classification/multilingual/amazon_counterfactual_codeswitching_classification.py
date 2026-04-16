from __future__ import annotations

import json
import os
from datasets import Dataset, DatasetDict

from mteb.abstasks.classification import AbsTaskClassification
from mteb.abstasks.task_metadata import TaskMetadata


def load_jsonl(filepath):

    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


class AmazonCounterfactualClassificationCodeSwitching(AbsTaskClassification):
    """
    AmazonCounterfactualClassification Code-Switching Task
    - Load completely from local jsonl files（text and label）
    - only supports English (en)
    """

    metadata = TaskMetadata(
        name="AmazonCounterfactualClassificationCodeSwitching",
        dataset={
            "path": "mteb/amazon_counterfactual",
            "revision": "1f7e6a9d6fa6e64c53d146e428565640410c0df1",
        },
        description=(
            "AmazonCounterfactualClassification Code-Switching variant. "
            "All data loaded from local file. English only."
        ),
        reference="https://arxiv.org/abs/2104.06893",
        category="t2c",
        modalities=["text"],
        type="Classification",
        eval_splits=["validation", "test"],
        eval_langs={"en": ["eng-Latn"]},
        main_score="accuracy",
        date=(
            "2018-01-01",
            "2021-12-31",
        ),
        domains=["Reviews", "Written"],
        task_subtypes=["Counterfactual Detection"],
        license="cc-by-4.0",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@inproceedings{oneill-etal-2021-wish,
  address = {Online and Punta Cana, Dominican Republic},
  author = {O{'}Neill, James  and
Rozenshtein, Polina  and
Kiryo, Ryuichi  and
Kubota, Motoko  and
Bollegala, Danushka},
  booktitle = {Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing},
  doi = {10.18653/v1/2021.emnlp-main.568},
  editor = {Moens, Marie-Francine  and
Huang, Xuanjing  and
Specia, Lucia  and
Yih, Scott Wen-tau},
  month = nov,
  pages = {7092--7108},
  publisher = {Association for Computational Linguistics},
  title = {{I} Wish {I} Would Have Loved This One, But {I} Didn{'}t {--} A Multilingual Dataset for Counterfactual Detection in Product Review},
  url = {https://aclanthology.org/2021.emnlp-main.568},
  year = {2021},
}
""",
        prompt="Classify a given Amazon customer review text as either counterfactual or not-counterfactual",
    )

    samples_per_label = 32

    def __init__(
        self,
        validation_file: str = None,
        test_file: str = None,
        **kwargs
    ):
        """
        Args:
            validation_file: validation split local jsonl file path
            test_file: test split local jsonl file path
        """
        super().__init__(**kwargs)
        self.validation_file = validation_file or os.getenv("AMAZON_COUNTERFACTUAL_VALIDATION_FILE")
        self.test_file = test_file or os.getenv("AMAZON_COUNTERFACTUAL_TEST_FILE")

    def load_data(self, **kwargs):

        if self.data_loaded:
            return

        split_files = {
            "validation": self.validation_file,
            "test": self.test_file,
        }

        dataset_dict = {}

        # Load train split from HuggingFace
        print("Loading train data from HuggingFace...")
        from datasets import load_dataset
        hf_dataset = load_dataset(
            self.metadata.dataset["path"],
            "en",
            revision=self.metadata.dataset["revision"],
            trust_remote_code=True,
        )
        dataset_dict["train"] = hf_dataset["train"]
        print(f"Loaded {len(dataset_dict['train'])} samples for train split from HuggingFace")

        # Load validation and test from local files
        for split, filepath in split_files.items():
            if not filepath:
                print(f"Warning: No file provided for {split} split, skipping.")
                continue

            if not os.path.exists(filepath):
                raise FileNotFoundError(f"Data file not found for {split}: {filepath}")

            print(f"Loading {split} data from: {filepath}")
            local_data = load_jsonl(filepath)

            texts = [item.get('text', '') for item in local_data]
            labels = [item.get('label', 0) for item in local_data]

            dataset_dict[split] = Dataset.from_dict({
                "text": texts,
                "label": labels,
            })
            print(f"Loaded {len(texts)} samples for {split} split")

        if not dataset_dict:
            raise ValueError(
                "No data files provided. "
                "Please pass validation_file/test_file parameters or set "
                "AMAZON_COUNTERFACTUAL_VALIDATION_FILE/AMAZON_COUNTERFACTUAL_TEST_FILE environment variables."
            )

        # Wrap dataset under "en" key to match eval_langs={"en": ["eng-Latn"]} structure
        self.dataset = {"en": DatasetDict(dataset_dict)}
        self.data_loaded = True
