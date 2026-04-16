from __future__ import annotations

from datasets import DatasetDict, load_dataset

from mteb.abstasks.pair_classification import AbsTaskPairClassification
from mteb.abstasks.task_metadata import TaskMetadata

CS_MTEB_REPO = "UTokyo-Yokoya-Lab/twittersemeval2015-pairclassification_CS-MTEB"
SUPPORTED_LANGUAGES = ["zh", "ja", "de", "es", "ko", "fr", "it", "pt", "nl"]


class TwitterSemEval2015CodeSwitching(AbsTaskPairClassification):
    metadata = TaskMetadata(
        name="TwitterSemEval2015CodeSwitching",
        dataset={"path": CS_MTEB_REPO, "revision": "main"},
        description="TwitterSemEval2015 Code-Switching variant.",
        reference="https://alt.qcri.org/semeval2015/task1/",
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
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@inproceedings{xu-etal-2015-semeval,
  address = {Denver, Colorado},
  author = {Xu, Wei and Callison-Burch, Chris and Dolan, Bill},
  booktitle = {Proceedings of the 9th International Workshop on Semantic Evaluation ({S}em{E}val 2015)},
  doi = {10.18653/v1/S15-2001},
  pages = {1--11},
  title = {{S}em{E}val-2015 Task 1: Paraphrase and Semantic Similarity in {T}witter ({PIT})},
  year = {2015},
}
""",
        prompt="Retrieve tweets that are semantically similar to the given tweet",
    )

    def __init__(self, language: str = "zh", **kwargs):
        super().__init__(**kwargs)
        if language not in SUPPORTED_LANGUAGES:
            raise ValueError(f"Unsupported language '{language}'. Supported: {SUPPORTED_LANGUAGES}")
        self.language = language
        self.metadata.name = f"TwitterSemEval2015CodeSwitching_{language}"

    def load_data(self, **kwargs):
        if self.data_loaded:
            return

        repo = CS_MTEB_REPO
        test_config = f"test_{self.language}_en"

        print(f"Loading test data from {repo} config={test_config}")
        test_ds = load_dataset(repo, test_config)
        test_data = list(test_ds.values())[0]

        self.dataset = DatasetDict({"test": test_data})

        print(f"Loaded {len(test_data)} samples for test split")
        self.data_loaded = True
