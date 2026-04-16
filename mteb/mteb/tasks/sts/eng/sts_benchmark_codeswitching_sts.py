from __future__ import annotations

from datasets import load_dataset

from mteb.abstasks.sts import AbsTaskSTS
from mteb.abstasks.task_metadata import TaskMetadata

CS_MTEB_REPO = "UTokyo-Yokoya-Lab/stsbenchmark-sts_CS-MTEB"
SUPPORTED_LANGUAGES = ["zh", "ja", "de", "es", "ko", "fr", "it", "pt", "nl"]


class STSBenchmarkCodeSwitching(AbsTaskSTS):
    metadata = TaskMetadata(
        name="STSBenchmarkCodeSwitching",
        description="STSBenchmark Code-Switching variant.",
        reference="https://github.com/PhilipMay/stsb-multi-mt/",
        dataset={"path": CS_MTEB_REPO, "revision": "main"},
        type="STS",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="cosine_spearman",
        date=None,
        domains=["Blog", "News", "Written"],
        task_subtypes=[],
        license="not specified",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="machine-translated and verified",
        bibtex_citation=r"""
@inproceedings{huggingface:dataset:stsb_multi_mt,
  author = {Philip May},
  title = {Machine translated multilingual STS benchmark dataset.},
  url = {https://github.com/PhilipMay/stsb-multi-mt},
  year = {2021},
}
""",
    )

    min_score = 0
    max_score = 5

    def __init__(self, language: str = "zh", **kwargs):
        super().__init__(**kwargs)
        if language not in SUPPORTED_LANGUAGES:
            raise ValueError(f"Unsupported language '{language}'. Supported: {SUPPORTED_LANGUAGES}")
        self.language = language
        self.metadata.name = f"STSBenchmarkCodeSwitching_{language}"

    def load_data(self, **kwargs):
        if self.data_loaded:
            return

        repo = CS_MTEB_REPO
        test_config = f"test_{self.language}_en"

        print(f"Loading test data from {repo} config={test_config}")
        test_ds = load_dataset(repo, test_config)
        test_data = list(test_ds.values())[0]

        self.dataset = {"test": test_data}

        print(f"Loaded {len(test_data)} sentence pairs for test split")
        self.data_loaded = True
