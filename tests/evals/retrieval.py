import json
import time
from pathlib import Path
from uuid import uuid4

import pandas as pd
from pydantic import BaseModel
from utils.retrieval_metrics import (
    calc_map,
    calc_mrr,
    calc_ndcg,
    calc_precision,
    calc_recall,
)

from lib.models.embedding import EmbeddingModel
from lib.models.rerank import RerankerModel
from lib.types import Chunk, Document
from lib.vectordb import KnowledgeBase


# Configuration ----------------------------------------------------------------

DATA_PATH = Path(__file__).parent / "data"
DOCS_JSON = DATA_PATH / "wiki_docs.jsonl"
QA_JSON = DATA_PATH / "wiki_qa.jsonl"
RESULTS_JSON = DATA_PATH / "results.json"
CUTOFFS = [1, 5, 10]
MAX_K = max(CUTOFFS)
embedding_model = EmbeddingModel()
reranker_model = RerankerModel()


# Types ------------------------------------------------------------------------


# TODO: combine the rag-eval representations with the local representations
# TODO: change ids to be int by default
class DatasetChunk(BaseModel):
    heading: str
    level: int
    content: str


class DatasetDocument(BaseModel):
    title: str
    source: str
    language: str
    chunks: list[DatasetChunk]


class DatasetQA(BaseModel):
    type: str
    language: str
    article_title: str
    chunks: list[int]
    question: str
    answer: str


# Evals ------------------------------------------------------------------------


def load_docs_qa() -> tuple[list[DatasetDocument], list[DatasetQA]]:
    docs_df = pd.read_json(DOCS_JSON, lines=True)
    docs = [
        DatasetDocument.model_validate(row.to_dict())
        for _, row in docs_df.iterrows()
    ]
    qa_df = pd.read_json(QA_JSON, lines=True)
    qa = [
        DatasetQA.model_validate(row.to_dict()) for _, row in qa_df.iterrows()
    ]
    return docs, qa


def main() -> None:
    docs, qas = load_docs_qa()

    eval_results = []
    for doc_id, doc in enumerate(docs):
        # Insert document and is chunks
        document = Document(
            id=str(doc_id),
            source=doc.source,
            metadata={"created_at": f"{time.time()}"},
        )
        chunks = [
            Chunk(
                id=str(chunk_id),
                doc_id=str(doc_id),
                page=-1,
                text=chunk.content,
            )
            for chunk_id, chunk in enumerate(doc.chunks)
        ]

        db = KnowledgeBase(name=doc.title, test=True)
        db.insert(
            docs=[document], chunks=chunks, embedding_model=embedding_model
        )

        # Get QAs
        doc_qa_pairs = list(
            filter(
                lambda qa: qa.article_title == doc.title
                and qa.language == doc.language,
                qas,
            )
        )

        if len(doc_qa_pairs) == 0:
            continue

        # Evaluate
        ground_truth: list[list[int]] = []
        retrieved_ids: list[list[int]] = []

        for qa in doc_qa_pairs:
            # Expected
            ground_truth.append(qa.chunks)

            retrieved_chunks = db.search(
                query=qa.question,
                top_k=MAX_K,
                top_r=20,
                embedding_model=embedding_model,
                reranker_model=reranker_model,
                threshold=0.6,
            )

            # Actual
            retrieved_ids.append([
                int(chunk.chunk.id) for chunk in retrieved_chunks
            ])

        PRECISIONs = calc_precision(retrieved_ids, ground_truth, CUTOFFS)
        RECALLs = calc_recall(retrieved_ids, ground_truth, CUTOFFS)
        MRRs = calc_mrr(retrieved_ids, ground_truth, CUTOFFS)
        NDCGs = calc_ndcg(retrieved_ids, ground_truth, CUTOFFS)
        MAPs = calc_map(retrieved_ids, ground_truth, CUTOFFS)

        for i, c in enumerate(CUTOFFS):
            print(f"precision@{c}: {PRECISIONs[i]}")
            print(f"recall@{c}: {RECALLs[i]}")
            print(f"mrr@{c}: {MRRs[i]}")
            print(f"ndcg@{c}: {NDCGs[i]}")
            print(f"map@{c}: {MAPs[i]}")

        # Save
        eval_results.append({
            "doc_id": doc_id,
            "title": doc.title,
            "metrics": {
                **{
                    f"precision@{c}": PRECISIONs[i]
                    for i, c in enumerate(CUTOFFS)
                },
                **{f"recall@{c}": RECALLs[i] for i, c in enumerate(CUTOFFS)},
                **{f"mrr@{c}": MRRs[i] for i, c in enumerate(CUTOFFS)},
                **{f"ndcg@{c}": NDCGs[i] for i, c in enumerate(CUTOFFS)},
                **{f"map@{c}": MAPs[i] for i, c in enumerate(CUTOFFS)},
            },
        })

    with open(RESULTS_JSON, "w") as f:
        json.dump(eval_results, f, indent=2)
    print(f"Saved evaluation results to {RESULTS_JSON}")


if __name__ == "__main__":
    main()
