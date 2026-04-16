from __future__ import annotations

from datasets import load_dataset
from tqdm import tqdm

from mteb._evaluators.retrieval_metrics import evaluate_p_mrr_change
from mteb.abstasks import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata

CSR_L_REPO = "UTokyo-Yokoya-Lab/robust04-instructions-mteb-CSR-L"
SUPPORTED_LANGUAGES = ["zh", "ja"]


class Robust04InstructionRetrievalCSRL(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="Robust04InstructionRetrievalCSRL",
        description="Robust04 Instruction Retrieval CSR-L variant (Chinese/Japanese code-switching).",
        reference="https://arxiv.org/abs/2403.15246",
        dataset={"path": CSR_L_REPO, "revision": "main"},
        type="InstructionReranking",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="p-MRR",
        date=("2023-08-01", "2024-04-01"),
        domains=["News", "Written"],
        task_subtypes=[],
        license="mit",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@misc{weller2024followir,
  archiveprefix = {arXiv},
  author = {Orion Weller and Benjamin Chang and Sean MacAvaney and Kyle Lo and Arman Cohan and Benjamin Van Durme and Dawn Lawrie and Luca Soldaini},
  eprint = {2403.15246},
  primaryclass = {cs.IR},
  title = {FollowIR: Evaluating and Teaching Information Retrieval Models to Follow Instructions},
  year = {2024},
}
""",
    )

    def __init__(self, language: str = "zh", **kwargs):
        super().__init__(**kwargs)
        if language not in SUPPORTED_LANGUAGES:
            raise ValueError(f"Unsupported language '{language}'. CSR-L only supports: {SUPPORTED_LANGUAGES}")
        self.language = language
        self.metadata.name = f"Robust04InstructionRetrievalCSRL_{language}"

    def load_data(self, **kwargs):
        if self.data_loaded:
            return

        repo = CSR_L_REPO
        q_cfg = f"queries_{self.language}_en"
        i_cfg = f"instructions_{self.language}_en"

        print(f"Loading queries from {repo} config={q_cfg}")
        query_split = list(load_dataset(repo, q_cfg).values())[0]

        print(f"Loading instructions from {repo} config={i_cfg}")
        instr_split = list(load_dataset(repo, i_cfg).values())[0]

        print(f"Loading corpus from {repo}")
        corpus_split = list(load_dataset(repo, "corpus").values())[0]

        print(f"Loading qrels from {repo}")
        qrels_split = list(load_dataset(repo, "default").values())[0]

        print(f"Loading top_ranked from {repo}")
        top_ranked_split = list(load_dataset(repo, "top_ranked").values())[0]

        self.queries = {"test": {}}
        self.instructions = {"test": {}}
        self.corpus = {"test": {}}
        self.relevant_docs = {"test": {}}
        self.top_ranked = {"test": {}}

        for item in tqdm(query_split, desc="Loading queries"):
            self.queries["test"][str(item.get('_id'))] = item.get('text')

        for item in tqdm(instr_split, desc="Loading instructions"):
            self.instructions["test"][str(item.get('query-id'))] = item.get('instruction')

        for item in tqdm(corpus_split, desc="Loading corpus"):
            doc_id = str(item.get('_id') or item.get('id'))
            self.corpus["test"][doc_id] = {"title": item.get('title'), "text": item.get('text')}

        for item in tqdm(qrels_split, desc="Loading qrels"):
            qid = str(item.get('query-id'))
            doc_id = str(item.get('corpus-id'))
            score = int(item.get('score', 1))
            if qid in self.queries["test"]:
                if qid not in self.relevant_docs["test"]:
                    self.relevant_docs["test"][qid] = {}
                self.relevant_docs["test"][qid][doc_id] = score

        for item in tqdm(top_ranked_split, desc="Loading top_ranked"):
            qid = str(item.get('query-id'))
            if qid in self.queries["test"]:
                self.top_ranked["test"][qid] = item.get('corpus-ids', [])

        print(f"Loaded {len(self.queries['test'])} queries, {len(self.instructions['test'])} instructions, {len(self.corpus['test'])} documents")
        self.data_loaded = True

    def task_specific_scores(self, scores, qrels, results, hf_split, hf_subset):
        qrel_diff_ds = load_dataset(CSR_L_REPO, "qrel_diff", split="qrel_diff")
        changed_qrels = {item["query-id"]: item["corpus-ids"] for item in qrel_diff_ds}
        return evaluate_p_mrr_change(qrels, results, changed_qrels, self.k_values)
