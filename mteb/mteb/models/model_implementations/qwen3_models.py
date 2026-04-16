# from mteb.models.instruct_wrapper import InstructSentenceTransformerModel
# from mteb.models.model_meta import ModelMeta
# from mteb.models.models_protocols import EncoderProtocol, PromptType
#
#
# def instruction_template(
#     instruction: str, prompt_type: PromptType | None = None
# ) -> str:
#     if not instruction or prompt_type == PromptType.document:
#         return ""
#     if isinstance(instruction, dict):
#         if prompt_type is None:
#             instruction = next(iter(instruction.values()))  # TODO
#         else:
#             instruction = instruction[prompt_type]
#     return f"Instruct: {instruction}\nQuery:"
#
#
# multilingual_langs = [
#     "afr-Latn",
#     "ara-Arab",
#     "aze-Latn",
#     "bel-Cyrl",
#     "bul-Cyrl",
#     "ben-Beng",
#     "cat-Latn",
#     "ceb-Latn",
#     "ces-Latn",
#     "cym-Latn",
#     "dan-Latn",
#     "deu-Latn",
#     "ell-Grek",
#     "eng-Latn",
#     "spa-Latn",
#     "est-Latn",
#     "eus-Latn",
#     "fas-Arab",
#     "fin-Latn",
#     "fra-Latn",
#     "glg-Latn",
#     "guj-Gujr",
#     "heb-Hebr",
#     "hin-Deva",
#     "hrv-Latn",
#     "hat-Latn",
#     "hun-Latn",
#     "hye-Armn",
#     "ind-Latn",
#     "isl-Latn",
#     "ita-Latn",
#     "jpn-Jpan",
#     "jav-Latn",
#     "kat-Geor",
#     "kaz-Cyrl",
#     "khm-Khmr",
#     "kan-Knda",
#     "kor-Hang",
#     "kir-Cyrl",
#     "lao-Laoo",
#     "lit-Latn",
#     "lav-Latn",
#     "mkd-Cyrl",
#     "mal-Mlym",
#     "mon-Cyrl",
#     "mar-Deva",
#     "msa-Latn",
#     "mya-Mymr",
#     "nep-Deva",
#     "nld-Latn",
#     "nor-Latn",
#     "nob-Latn",
#     "nno-Latn",
#     "pan-Guru",
#     "pol-Latn",
#     "por-Latn",
#     "que-Latn",
#     "ron-Latn",
#     "rus-Cyrl",
#     "sin-Sinh",
#     "slk-Latn",
#     "slv-Latn",
#     "swa-Latn",
#     "tam-Taml",
#     "tel-Telu",
#     "tha-Thai",
#     "tgl-Latn",
#     "tur-Latn",
#     "ukr-Cyrl",
#     "urd-Arab",
#     "vie-Latn",
#     "yor-Latn",
#     "zho-Hans",
# ]
#
# QWEN3_CITATION = """@article{qwen3embedding,
#   title={Qwen3 Embedding: Advancing Text Embedding and Reranking Through Foundation Models},
#   author={Zhang, Yanzhao and Li, Mingxin and Long, Dingkun and Zhang, Xin and Lin, Huan and Yang, Baosong and Xie, Pengjun and Yang, An and Liu, Dayiheng and Lin, Junyang and Huang, Fei and Zhou, Jingren},
#   journal={arXiv preprint arXiv:2506.05176},
#   year={2025}
# }"""
#
# training_data = {
#     "T2Retrieval",
#     "DuRetrieval",
#     "MMarcoReranking",
#     "CMedQAv2-reranking",
#     "NQ",
#     "MSMARCO",
#     "HotpotQA",
#     "FEVER",
#     "MrTidyRetrieval",
#     "MIRACLRetrieval",
#     "CodeSearchNet",
# }
#
#
# def q3e_instruct_loader(
#     model_name_or_path: str, revision: str, **kwargs
# ) -> EncoderProtocol:
#     model = InstructSentenceTransformerModel(
#         model_name_or_path,
#         revision=revision,
#         instruction_template=instruction_template,
#         apply_instruction_to_passages=False,
#         **kwargs,
#     )
#     encoder = model.model._first_module()
#     if encoder.auto_model.config._attn_implementation == "flash_attention_2":
#         # The Qwen3 code only use left padding in flash_attention_2 mode.
#         encoder.tokenizer.padding_side = "left"
#     return model
#
#
# Qwen3_Embedding_0B6 = ModelMeta(
#     loader=q3e_instruct_loader,
#     name="Qwen/Qwen3-Embedding-0.6B",
#     languages=multilingual_langs,
#     open_weights=True,
#     revision="b22da495047858cce924d27d76261e96be6febc0",  # Commit of @tomaarsen
#     release_date="2025-06-05",
#     n_parameters=595776512,
#     memory_usage_mb=1136,
#     embed_dim=1024,
#     max_tokens=32768,
#     license="apache-2.0",
#     reference="https://huggingface.co/Qwen/Qwen3-Embedding-0.6B",
#     similarity_fn_name="cosine",
#     framework=["Sentence Transformers", "PyTorch"],
#     use_instructions=True,
#     public_training_code=None,
#     public_training_data=None,
#     training_datasets=training_data,
#     citation=QWEN3_CITATION,
# )
#
# Qwen3_Embedding_4B = ModelMeta(
#     loader=q3e_instruct_loader,
#     name="Qwen/Qwen3-Embedding-4B",
#     languages=multilingual_langs,
#     open_weights=True,
#     revision="636cd9bf47d976946cdbb2b0c3ca0cb2f8eea5ff",  # Commit of @tomaarsen
#     release_date="2025-06-05",
#     n_parameters=4021774336,
#     memory_usage_mb=7671,
#     embed_dim=2560,
#     max_tokens=32768,
#     license="apache-2.0",
#     reference="https://huggingface.co/Qwen/Qwen3-Embedding-4B",
#     similarity_fn_name="cosine",
#     framework=["Sentence Transformers", "PyTorch"],
#     use_instructions=True,
#     public_training_code=None,
#     public_training_data=None,
#     training_datasets=training_data,
#     citation=QWEN3_CITATION,
# )
#
# Qwen3_Embedding_8B = ModelMeta(
#     loader=q3e_instruct_loader,
#     name="Qwen/Qwen3-Embedding-8B",
#     languages=multilingual_langs,
#     open_weights=True,
#     revision="4e423935c619ae4df87b646a3ce949610c66241c",  # Commit of @tomaarsen
#     release_date="2025-06-05",
#     n_parameters=7567295488,
#     memory_usage_mb=14433,
#     embed_dim=4096,
#     max_tokens=32768,
#     license="apache-2.0",
#     reference="https://huggingface.co/Qwen/Qwen3-Embedding-8B",
#     similarity_fn_name="cosine",
#     framework=["Sentence Transformers", "PyTorch"],
#     use_instructions=True,
#     public_training_code=None,
#     public_training_data=None,
#     training_datasets=training_data,
#     citation=QWEN3_CITATION,
# )
from mteb.models.instruct_wrapper import InstructSentenceTransformerModel
from mteb.models.model_meta import ModelMeta
from mteb.models.models_protocols import EncoderProtocol, PromptType

# Qwen3 official prompts for MTEB tasks
# Source: https://huggingface.co/Qwen/Qwen3-Embedding-0.6B
QWEN3_PROMPTS = {
    # Classification tasks
    "AmazonCounterfactualClassification": "Classify a given Amazon customer review text as either counterfactual or not-counterfactual",
    "AmazonPolarityClassification": "Classify Amazon reviews into positive or negative sentiment",
    "AmazonReviewsClassification": "Classify the given Amazon review into its appropriate rating category",
    "Banking77Classification": "Given a online banking query, find the corresponding intents",
    "EmotionClassification": "Classify the emotion expressed in the given Twitter message into one of the six emotions: anger, fear, joy, love, sadness, and surprise",
    "ImdbClassification": "Classify the sentiment expressed in the given movie review text from the IMDB dataset",
    "MassiveIntentClassification": "Given a user utterance as query, find the user intents",
    "MassiveScenarioClassification": "Given a user utterance as query, find the user scenarios",
    "MTOPDomainClassification": "Classify the intent domain of the given utterance in task-oriented conversation",
    "MTOPIntentClassification": "Classify the intent of the given utterance in task-oriented conversation",
    "ToxicConversationsClassification": "Classify the given comments as either toxic or not toxic",
    "TweetSentimentExtractionClassification": "Classify the sentiment of a given tweet as either positive, negative, or neutral",
    "TNews": "Classify the fine-grained category of the given news title",
    "IFlyTek": "Given an App description text, find the appropriate fine-grained category",
    "MultilingualSentiment": "Classify sentiment of the customer review into positive, neutral, or negative",
    "JDReview": "Classify the customer review for iPhone on e-commerce platform into positive or negative",
    "OnlineShopping": "Classify the customer review for online shopping into positive or negative",
    "Waimai": "Classify the customer review from a food takeaway platform into positive or negative",
    # Clustering tasks
    "ArxivClusteringP2P": "Identify the main and secondary category of Arxiv papers based on the titles and abstracts",
    "ArxivClusteringS2S": "Identify the main and secondary category of Arxiv papers based on the titles",
    "BiorxivClusteringP2P": "Identify the main category of Biorxiv papers based on the titles and abstracts",
    "BiorxivClusteringS2S": "Identify the main category of Biorxiv papers based on the titles",
    "MedrxivClusteringP2P": "Identify the main category of Medrxiv papers based on the titles and abstracts",
    "MedrxivClusteringS2S": "Identify the main category of Medrxiv papers based on the titles",
    "RedditClustering": "Identify the topic or theme of Reddit posts based on the titles",
    "RedditClusteringP2P": "Identify the topic or theme of Reddit posts based on the titles and posts",
    "StackExchangeClustering": "Identify the topic or theme of StackExchange posts based on the titles",
    "StackExchangeClusteringP2P": "Identify the topic or theme of StackExchange posts based on the given paragraphs",
    "TwentyNewsgroupsClustering": "Identify the topic or theme of the given news articles",
    "CLSClusteringS2S": "Identify the main category of scholar papers based on the titles",
    "CLSClusteringP2P": "Identify the main category of scholar papers based on the titles and abstracts",
    "ThuNewsClusteringS2S": "Identify the topic or theme of the given news articles based on the titles",
    "ThuNewsClusteringP2P": "Identify the topic or theme of the given news articles based on the titles and contents",
    # Reranking tasks
    "AskUbuntuDupQuestions": "Retrieve duplicate questions from AskUbuntu forum",
    "MindSmallReranking": "Retrieve relevant news articles based on user browsing history",
    "SciDocsRR": "Given a title of a scientific paper, retrieve the titles of other relevant papers",
    "StackOverflowDupQuestions": "Retrieve duplicate questions from StackOverflow forum",
    "SprintDuplicateQuestions": "Retrieve duplicate questions from Sprint forum",
    "TwitterSemEval2015": "Retrieve tweets that are semantically similar to the given tweet",
    "TwitterURLCorpus": "Retrieve tweets that are semantically similar to the given tweet",
    "T2Reranking": "Given a Chinese search query, retrieve web passages that answer the question",
    "MmarcoReranking": "Given a Chinese search query, retrieve web passages that answer the question",
    "CMedQAv1": "Given a Chinese community medical question, retrieve replies that best answer the question",
    "CMedQAv2": "Given a Chinese community medical question, retrieve replies that best answer the question",
    "Ocnli": "Retrieve semantically similar text.",
    "Cmnli": "Retrieve semantically similar text.",
    # Retrieval tasks
    "ArguAna": "Given a claim, find documents that refute the claim",
    "ClimateFEVER": "Given a claim about climate change, retrieve documents that support or refute the claim",
    "ClimateFEVERHardNegatives": "Given a claim about climate change, retrieve documents that support or refute the claim",
    "DBPedia": "Given a query, retrieve relevant entity descriptions from DBPedia",
    "FEVER": "Given a claim, retrieve documents that support or refute the claim",
    "FEVERHardNegatives": "Given a claim, retrieve documents that support or refute the claim",
    "FiQA2018": "Given a financial question, retrieve user replies that best answer the question",
    "HotpotQA": "Given a multi-hop question, retrieve documents that can help answer the question",
    "HotpotQAHardNegatives": "Given a multi-hop question, retrieve documents that can help answer the question",
    "MSMARCO": "Given a web search query, retrieve relevant passages that answer the query",
    "NFCorpus": "Given a question, retrieve relevant documents that best answer the question",
    "NQ": "Given a question, retrieve Wikipedia passages that answer the question",
    "QuoraRetrieval": "Given a question, retrieve questions that are semantically equivalent to the given question",
    "SCIDOCS": "Given a scientific paper title, retrieve paper abstracts that are cited by the given paper",
    "SciFact": "Given a scientific claim, retrieve documents that support or refute the claim",
    "Touche2020": "Given a question, retrieve detailed and persuasive arguments that answer the question",
    "Touche2020Retrieval.v3": "Given a question, retrieve detailed and persuasive arguments that answer the question",
    "Touche2020Retrieval.v3CodeSwitching": "Given a question, retrieve detailed and persuasive arguments that answer the question",
    "TRECCOVID": "Given a query on COVID-19, retrieve documents that answer the query",
    "TRECCOVIDCodeSwitching": "Given a query on COVID-19, retrieve documents that answer the query",
    "T2Retrieval": "Given a Chinese search query, retrieve web passages that answer the question",
    "MMarcoRetrieval": "Given a web search query, retrieve relevant passages that answer the query",
    "DuRetrieval": "Given a Chinese search query, retrieve web passages that answer the question",
    "CovidRetrieval": "Given a question on COVID-19, retrieve news articles that answer the question",
    "CmedqaRetrieval": "Given a Chinese community medical question, retrieve replies that best answer the question",
    "EcomRetrieval": "Given a user query from an e-commerce website, retrieve description sentences of relevant products",
    "MedicalRetrieval": "Given a medical question, retrieve user replies that best answer the question",
    "VideoRetrieval": "Given a video search query, retrieve the titles of relevant videos",
    # Instruction Retrieval tasks (FollowIR)
    "Core17InstructionRetrieval": "Retrieval the relevant passage for the given query",
    "News21InstructionRetrieval": "Retrieval the relevant passage for the given query",
    "Robust04InstructionRetrieval": "Retrieval the relevant passage for the given query",
    "Core17InstructionRetrievalCodeSwitching": "Retrieval the relevant passage for the given query",
    "News21InstructionRetrievalCodeSwitching": "Retrieval the relevant passage for the given query",
    "Robust04InstructionRetrievalCodeSwitching": "Retrieval the relevant passage for the given query",
    # STS tasks
    "STSBenchmarkMultilingualSTS": "Retrieve semantically similar text",
    "SICKFr": "Retrieve semantically similar text",
    "SummEvalFr": "Given a news summary, retrieve other semantically similar summaries",
    "STS12": "Retrieve semantically similar text",
    "STS13": "Retrieve semantically similar text",
    "STS14": "Retrieve semantically similar text",
    "STS15": "Retrieve semantically similar text",
    "STS16": "Retrieve semantically similar text",
    "STS17": "Retrieve semantically similar text",
    "STS22": "Retrieve semantically similar text",
    "STSBenchmark": "Retrieve semantically similar text",
    "BIOSSES": "Retrieve semantically similar text",
    "SICK-R": "Retrieve semantically similar text",
    "SummEval": "Retrieve semantically similar text",
    # Pair Classification tasks
    "PawsX": "Retrieve semantically similar text",
    "XNLI": "Retrieve semantically similar text",
    # CQADupstack tasks
    "CQADupstackRetrieval": "Given a question, retrieve detailed question descriptions from Stackexchange that are duplicates to the given question",
    "CQADupstackGamingRetrieval": "Given a question, retrieve detailed question descriptions from Stackexchange that are duplicates to the given question",
    "CQADupstackUnixRetrieval": "Given a question, retrieve detailed question descriptions from Stackexchange that are duplicates to the given question",
    # Code retrieval tasks
    "AppsRetrieval": "Given a question about code problem, retrieval code that can solve user's problem",
    "COIRCodeSearchNetRetrieval": "Given a code snippet, retrieve the comment corresponding to that code.",
    "CodeEditSearchRetrieval": "Given a piece of code, retrieval code that in the ",
    "CodeFeedbackMT": "Given a question about coding, retrieval code or passage that can solve user's question",
    "CodeFeedbackST": "Given a question about coding, retrieval code or passage that can solve user's question",
    "CodeSearchNetCCRetrieval": "Given a code comment, retrieve the code snippet corresponding to that comment.",
    "CodeSearchNetRetrieval": "Given a code snippet, retrieve the comment corresponding to that code.",
    "CodeTransOceanContest": "Given a piece for code, retrieval semantically similar code",
    "CodeTransOceanDL": "Given a piece for code, retrieval semantically similar code",
    "CosQA": "Given a question about coding, retrieval code or passage that can solve user's question",
    "StackOverflowQA": "Given a question about coding, retrieval code or passage that can solve user's question",
    "SyntheticText2SQL": "Given a user's question, retrieve SQL queries that are appropriate responses to the question",
    # Bitext Mining tasks
    "BibleNLPBitextMining": "Retrieve parallel sentences",
    "BUCC.v2": "Retrieve parallel sentences",
    "DiaBlaBitextMining": "Retrieve parallel sentences",
    "FloresBitextMining": "Retrieve parallel sentences",
    "Tatoeba": "Retrieve parallel sentences",
    # Other retrieval tasks
    "HagridRetrieval": "Retrieval the relevant passage for the given query",
    "LegalBenchCorporateLobbying": "Retrieval the relevant passage for the given query",
    "LEMBPasskeyRetrieval": "Retrieval the relevant passage for the given query",
    "BelebeleRetrieval": "Retrieval the relevant passage for the given query",
    "MLQARetrieval": "Retrieval the relevant passage for the given query",
    "StatcanDialogueDatasetRetrieval": "Retrieval the relevant passage for the given query",
    "WikipediaRetrievalMultilingual": "Retrieval the relevant passage for the given query",
    "WebLINXCandidatesReranking": "Retrieval the relevant passage for the given query",
    "WikipediaRerankingMultilingual": "Retrieval the relevant passage for the given query",
    "MIRACLRetrievalHardNegatives": "Retrieval relevant passage for the given query",
}


def instruction_template(
    instruction: str, prompt_type: PromptType | None = None
) -> str:
    if not instruction or prompt_type == PromptType.document:
        return ""
    if isinstance(instruction, dict):
        if prompt_type is None:
            instruction = next(iter(instruction.values()))  # TODO
        else:
            instruction = instruction[prompt_type]
    return f"Instruct: {instruction}\nQuery:"


multilingual_langs = [
    "afr-Latn",
    "ara-Arab",
    "aze-Latn",
    "bel-Cyrl",
    "bul-Cyrl",
    "ben-Beng",
    "cat-Latn",
    "ceb-Latn",
    "ces-Latn",
    "cym-Latn",
    "dan-Latn",
    "deu-Latn",
    "ell-Grek",
    "eng-Latn",
    "spa-Latn",
    "est-Latn",
    "eus-Latn",
    "fas-Arab",
    "fin-Latn",
    "fra-Latn",
    "glg-Latn",
    "guj-Gujr",
    "heb-Hebr",
    "hin-Deva",
    "hrv-Latn",
    "hat-Latn",
    "hun-Latn",
    "hye-Armn",
    "ind-Latn",
    "isl-Latn",
    "ita-Latn",
    "jpn-Jpan",
    "jav-Latn",
    "kat-Geor",
    "kaz-Cyrl",
    "khm-Khmr",
    "kan-Knda",
    "kor-Hang",
    "kir-Cyrl",
    "lao-Laoo",
    "lit-Latn",
    "lav-Latn",
    "mkd-Cyrl",
    "mal-Mlym",
    "mon-Cyrl",
    "mar-Deva",
    "msa-Latn",
    "mya-Mymr",
    "nep-Deva",
    "nld-Latn",
    "nor-Latn",
    "nob-Latn",
    "nno-Latn",
    "pan-Guru",
    "pol-Latn",
    "por-Latn",
    "que-Latn",
    "ron-Latn",
    "rus-Cyrl",
    "sin-Sinh",
    "slk-Latn",
    "slv-Latn",
    "swa-Latn",
    "tam-Taml",
    "tel-Telu",
    "tha-Thai",
    "tgl-Latn",
    "tur-Latn",
    "ukr-Cyrl",
    "urd-Arab",
    "vie-Latn",
    "yor-Latn",
    "zho-Hans",
]

QWEN3_CITATION = """@article{qwen3embedding,
  title={Qwen3 Embedding: Advancing Text Embedding and Reranking Through Foundation Models},
  author={Zhang, Yanzhao and Li, Mingxin and Long, Dingkun and Zhang, Xin and Lin, Huan and Yang, Baosong and Xie, Pengjun and Yang, An and Liu, Dayiheng and Lin, Junyang and Huang, Fei and Zhou, Jingren},
  journal={arXiv preprint arXiv:2506.05176},
  year={2025}
}"""

training_data = {
    "T2Retrieval",
    "DuRetrieval",
    "MMarcoReranking",
    "CMedQAv2-reranking",
    "NQ",
    "MSMARCO",
    "HotpotQA",
    "FEVER",
    "MrTidyRetrieval",
    "MIRACLRetrieval",
    "CodeSearchNet",
}


def q3e_instruct_loader(
    model_name_or_path: str, revision: str, **kwargs
) -> EncoderProtocol:
    model = InstructSentenceTransformerModel(
        model_name_or_path,
        revision=revision,
        instruction_template=instruction_template,
        apply_instruction_to_passages=False,
        prompts_dict=QWEN3_PROMPTS,
        **kwargs,
    )
    encoder = model.model._first_module()
    if encoder.auto_model.config._attn_implementation == "flash_attention_2":
        # The Qwen3 code only use left padding in flash_attention_2 mode.
        encoder.tokenizer.padding_side = "left"
    return model


Qwen3_Embedding_0B6 = ModelMeta(
    loader=q3e_instruct_loader,
    name="Qwen/Qwen3-Embedding-0.6B",
    languages=multilingual_langs,
    open_weights=True,
    revision="b22da495047858cce924d27d76261e96be6febc0",  # Commit of @tomaarsen
    release_date="2025-06-05",
    n_parameters=595776512,
    memory_usage_mb=2272,
    embed_dim=1024,
    max_tokens=32768,
    license="apache-2.0",
    reference="https://huggingface.co/Qwen/Qwen3-Embedding-0.6B",
    similarity_fn_name="cosine",
    framework=["Sentence Transformers", "PyTorch"],
    use_instructions=True,
    public_training_code=None,
    public_training_data=None,
    training_datasets=training_data,
    citation=QWEN3_CITATION,
)

Qwen3_Embedding_4B = ModelMeta(
    loader=q3e_instruct_loader,
    name="Qwen/Qwen3-Embedding-4B",
    languages=multilingual_langs,
    open_weights=True,
    revision="636cd9bf47d976946cdbb2b0c3ca0cb2f8eea5ff",  # Commit of @tomaarsen
    release_date="2025-06-05",
    n_parameters=4021774336,
    memory_usage_mb=15341,
    embed_dim=2560,
    max_tokens=32768,
    license="apache-2.0",
    reference="https://huggingface.co/Qwen/Qwen3-Embedding-4B",
    similarity_fn_name="cosine",
    framework=["Sentence Transformers", "PyTorch"],
    use_instructions=True,
    public_training_code=None,
    public_training_data=None,
    training_datasets=training_data,
    citation=QWEN3_CITATION,
)

Qwen3_Embedding_8B = ModelMeta(
    loader=q3e_instruct_loader,
    name="Qwen/Qwen3-Embedding-8B",
    languages=multilingual_langs,
    open_weights=True,
    revision="4e423935c619ae4df87b646a3ce949610c66241c",  # Commit of @tomaarsen
    release_date="2025-06-05",
    n_parameters=7567295488,
    memory_usage_mb=28866,
    embed_dim=4096,
    max_tokens=32768,
    license="apache-2.0",
    reference="https://huggingface.co/Qwen/Qwen3-Embedding-8B",
    similarity_fn_name="cosine",
    framework=["Sentence Transformers", "PyTorch"],
    use_instructions=True,
    public_training_code=None,
    public_training_data=None,
    training_datasets=training_data,
    citation=QWEN3_CITATION,
)
