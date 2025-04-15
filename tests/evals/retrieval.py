import json
import time
from pathlib import Path

from pydantic import BaseModel
from utils.retrieval_metrics import (
    calc_map,
    calc_mrr,
    calc_ndcg,
    calc_precision,
    calc_recall,
)

from rag.models.embedding import EmbeddingModel
from rag.models.rerank import RerankerModel
from rag.rag.vectordb import KnowledgeBase
from rag.utils.types import Chunk, Document

# --- Configuration ---
DATA_PATH = Path(__file__).parent / "data"
DOCS_JSON = DATA_PATH / "wiki_docs.json"
QA_JSON = DATA_PATH / "wiki_qa.json"
RESULTS_JSON = DATA_PATH / "results.json"
CUTOFFS = [1, 5, 10]
MAX_K = max(CUTOFFS)


# --- Types ---
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
    id: int
    type: str
    language: str
    article_title: str
    chunks: list[int]
    question: str
    answer: str


# --- Dataset ---
with open(DOCS_JSON, "r") as f:
    data = json.load(f)
    docs_data = [DatasetDocument.model_validate(item) for item in data]

with open(QA_JSON, "r") as f:
    data = json.load(f)
    qa_data = [DatasetQA.model_validate(item) for item in data]


# --- Models ---
embedding_model = EmbeddingModel()
reranker_model = RerankerModel()


def main() -> None:
    eval_results = []
    for doc_id, doc in enumerate(docs_data):
        # create a vectordb instance for the document
        db = KnowledgeBase(name=doc.title)

        # create the document, chunks objects to insert
        document = Document(
            id=str(doc_id),
            source=doc.source,
            metadata={"created_at": f"{time.time()}"},
        )
        chunks = [
            Chunk(
                id=str(chunk_id),
                doc_id=str(doc_id),
                page=-1,  # not important
                text=chunk.content,
            )
            for chunk_id, chunk in enumerate(doc.chunks)
        ]

        if not chunks:
            continue

        # insert the document and chunks in the vectordb
        db.insert(
            docs=[document], chunks=chunks, embedding_model=embedding_model
        )

        ground_truth: list[list[int]] = []
        retrieved_ids: list[list[int]] = []

        # compute the qa pairs
        doc_qa_pairs = list(
            filter(lambda qa: qa.article_title == doc.title, qa_data)
        )

        if len(doc_qa_pairs) == 0:
            continue

        for qa in doc_qa_pairs:
            query = qa.question
            ground_truth.append(qa.chunks)

            # perform search
            retrieved_chunks = db.search(
                query=query,
                top_k=MAX_K,
                top_r=20,
                embedding_model=embedding_model,
                reranker_model=reranker_model,
                threshold=1,
            )

            # obtain the actual retrieved ids of the chunks
            retrieved_ids.append([
                int(chunk.chunk.id) for chunk in retrieved_chunks
            ])

        # Calculate metrics
        PRECISIONs = calc_precision(retrieved_ids, ground_truth, CUTOFFS)
        RECALLs = calc_recall(retrieved_ids, ground_truth, CUTOFFS)
        MRRs = calc_mrr(retrieved_ids, ground_truth, CUTOFFS)
        NDCGs = calc_ndcg(retrieved_ids, ground_truth, CUTOFFS)
        MAPs = calc_map(retrieved_ids, ground_truth, CUTOFFS)

        # print results
        for i, c in enumerate(CUTOFFS):
            print(f"precision@{c}: {PRECISIONs[i]}")
            print(f"recall@{c}: {RECALLs[i]}")
            print(f"mrr@{c}: {MRRs[i]}")
            print(f"ndcg@{c}: {NDCGs[i]}")
            print(f"map@{c}: {MAPs[i]}")

        # save results
        result = {
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
        }

        eval_results.append(result)

    with open(RESULTS_JSON, "w") as f:
        json.dump(eval_results, f, indent=2)
    print(f"Saved evaluation results to {RESULTS_JSON}")


if __name__ == "__main__":
    main()
