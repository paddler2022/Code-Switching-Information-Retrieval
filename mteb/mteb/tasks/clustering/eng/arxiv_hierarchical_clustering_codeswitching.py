from __future__ import annotations

from datasets import Dataset, DatasetDict, load_dataset

from mteb.abstasks.clustering import AbsTaskClustering
from mteb.abstasks.task_metadata import TaskMetadata

CS_MTEB_REPO = "UTokyo-Yokoya-Lab/arxiv-clustering-p2p_CS-MTEB"
SUPPORTED_LANGUAGES = ["zh", "ja", "de", "es", "ko", "fr", "it", "pt", "nl"]

N_SAMPLES = 2048


def split_labels(record: dict) -> dict:
    record["labels"] = record["labels"].split(".")
    return record


class ArXivHierarchicalClusteringP2PCodeSwitching(AbsTaskClustering):
    metadata = TaskMetadata(
        name="ArXivHierarchicalClusteringP2PCodeSwitching",
        description="ArXivHierarchicalClusteringP2P Code-Switching variant. Clustering of titles+abstract from arxiv.",
        reference="https://www.kaggle.com/Cornell-University/arxiv",
        dataset={"path": CS_MTEB_REPO, "revision": "main"},
        type="Clustering",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="v_measure",
        date=("1991-01-01", "2021-01-01"),
        domains=["Academic", "Written"],
        task_subtypes=[],
        license="cc0-1.0",
        annotations_creators="derived",
        dialect=["Thematic clustering"],
        sample_creation="found",
        bibtex_citation="",
    )

    def __init__(self, language: str = "zh", **kwargs):
        super().__init__(**kwargs)
        if language not in SUPPORTED_LANGUAGES:
            raise ValueError(f"Unsupported language '{language}'. Supported: {SUPPORTED_LANGUAGES}")
        self.language = language
        self.metadata.name = f"ArXivHierarchicalClusteringP2PCodeSwitching_{language}"

    def load_data(self, **kwargs):
        if self.data_loaded:
            return

        repo = CS_MTEB_REPO
        test_config = f"test_{self.language}_en"

        print(f"Loading data from {repo} config={test_config}")
        test_ds = load_dataset(repo, test_config)
        test_data = list(test_ds.values())[0]

        sentences = [item['sentences'] for item in test_data]
        labels = [item['labels'] for item in test_data]

        self.dataset = DatasetDict({
            "test": Dataset.from_dict({"sentences": sentences, "labels": labels})
        })

        self.dataset = self.dataset.map(split_labels)

        if len(self.dataset["test"]) > N_SAMPLES:
            self.dataset["test"] = self.dataset["test"].train_test_split(
                test_size=N_SAMPLES, seed=self.seed
            )["test"]

        print(f"Loaded {len(self.dataset['test'])} samples for test split")
        self.data_loaded = True

