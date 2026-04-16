from __future__ import annotations

from datasets import Dataset, DatasetDict, load_dataset

from mteb.abstasks.classification import AbsTaskClassification
from mteb.abstasks.task_metadata import TaskMetadata

CS_MTEB_REPO = "UTokyo-Yokoya-Lab/tweet_sentiment_extraction_CS-MTEB"
SUPPORTED_LANGUAGES = ["zh", "ja", "de", "es", "ko", "fr", "it", "pt", "nl"]


class TweetSentimentExtractionClassificationCodeSwitching(AbsTaskClassification):
    metadata = TaskMetadata(
        name="TweetSentimentExtractionClassificationCodeSwitching",
        description="TweetSentimentExtractionClassification Code-Switching variant.",
        reference="https://www.kaggle.com/competitions/tweet-sentiment-extraction/overview",
        dataset={"path": CS_MTEB_REPO, "revision": "main"},
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2020-01-01", "2020-12-31"),
        domains=["Social", "Written"],
        task_subtypes=["Sentiment/Hate speech"],
        license="not specified",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@misc{tweet-sentiment-extraction,
  author = {Maggie, Phil Culliton, Wei Chen},
  publisher = {Kaggle},
  title = {Tweet Sentiment Extraction},
  url = {https://kaggle.com/competitions/tweet-sentiment-extraction},
  year = {2020},
}
""",
        prompt="Classify the sentiment of a given tweet as either positive, negative, or neutral",
    )

    samples_per_label = 32

    def __init__(self, language: str = "zh", **kwargs):
        super().__init__(**kwargs)
        if language not in SUPPORTED_LANGUAGES:
            raise ValueError(f"Unsupported language '{language}'. Supported: {SUPPORTED_LANGUAGES}")
        self.language = language
        self.metadata.name = f"TweetSentimentExtractionClassificationCodeSwitching_{language}"

    def load_data(self, **kwargs):
        if self.data_loaded:
            return

        repo = CS_MTEB_REPO
        test_config = f"test_{self.language}_en"

        # Load train from default config
        print(f"Loading train data from {repo} config=default")
        train_data = load_dataset(repo, "default")["train"]

        # Load code-switching test
        print(f"Loading test data from {repo} config={test_config}")
        test_ds = load_dataset(repo, test_config)
        test_data = list(test_ds.values())[0]

        self.dataset = DatasetDict({
            "train": train_data,
            "test": test_data,
        })

        print(f"Loaded {len(train_data)} train, {len(test_data)} test samples")
        self.data_loaded = True
