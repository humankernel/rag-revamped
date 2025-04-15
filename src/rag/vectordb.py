import logging
import pickle
import time
from pathlib import Path

import numpy as np
import torch

from models.embedding import (
    ColbertEmbeddings,
    DenseEmbeddings,
    EmbeddingModel,
    SparseEmbeddings,
    calculate_hybrid_scores,
    colbert_similarity,
    dense_similarity,
    sparse_similarity,
)
from models.rerank import RerankerVLLMModel
from utils.types import Chunk, Document, RetrievedChunk

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class KnowledgeBase:
    def __init__(
        self,
        name: str,
    ) -> None:
        self.name = name
        self.db_path = (
            Path(__file__).resolve().parent.parent.parent.parent
            / "data"
            / name
            / "vectordb.pkl"
        )
        # data
        self.documents = []
        self.chunks = []
        self.dense_embeddings = []
        self.sparse_embeddings = []
        self.colbert_embeddings = []

        # load
        self.load()

    @property
    def is_empty(self) -> bool:
        return len(self.chunks) == 0

    def insert(
        self,
        docs: list[Document],
        chunks: list[Chunk],
        embedding_model: EmbeddingModel,
        batch_size: int = 32,
    ) -> None:
        assert docs and chunks, "Docs and Chunks shouldn't be empty"
        assert all(isinstance(d, Document) for d in docs), (
            "Docs should be of type `Document`"
        )
        assert all(isinstance(c, Chunk) for c in chunks), (
            "Docs should be of type `Chunk`"
        )

        texts = list(chunk.text for chunk in chunks)

        logger.debug(f"Embedding {len(docs)} chunks.")
        start = time.time()
        dense, sparse, colbert = self._get_embedding(texts, embedding_model)
        logger.debug(
            f"Finished Embedding {len(docs)} chunks. Time {time.time() - start}"
        )

        if len(self.documents) == 0:
            self.dense_embeddings = dense
            self.sparse_embeddings = sparse
            self.colbert_embeddings = colbert
        else:
            self.dense_embeddings = np.vstack([self.dense_embeddings, dense])
            self.sparse_embeddings.extend(sparse)
            # can't do vstack because colbert are unhomougenious
            self.colbert_embeddings.extend(colbert)

        self.documents.extend(docs)
        self.chunks.extend(chunks)
        self.save()

    def search(
        self,
        query: str,
        embedding_model: EmbeddingModel,
        reranker_model: RerankerVLLMModel,
        top_k: int = 20,
        top_r: int = 10,
        threshold: float = 1,
    ) -> list[RetrievedChunk]:
        """Hybrid Search with Reranking"""
        assert not self.is_empty, "No data loaded."
        assert 0 < threshold <= 1, "Threshold should be > 0 and <= 1"

        # TODO: rethink this cache
        # q_dense, q_sparse, q_colbert = self._get_embedding_cached(query)
        q_dense, q_sparse, q_colbert = self._get_embedding(
            [query], embedding_model
        )
        dense_scores = dense_similarity(q_dense, self.dense_embeddings)
        sparse_scores = sparse_similarity(q_sparse, self.sparse_embeddings)
        colbert_scores = colbert_similarity(q_colbert, self.colbert_embeddings)
        hybrid_scores = calculate_hybrid_scores(
            scores=(dense_scores, sparse_scores, colbert_scores),
            weights=(0.4, 0.2, 0.4),
        )
        values, top_indices = torch.topk(hybrid_scores[0], k=top_k)
        # top_indices = torch.argsort(hybrid_scores)[-top_k:][::-1]

        if reranker_model:
            top_texts = list(
                self.chunks[idx].text for idx in top_indices.tolist()
            )
            pairs = list((query, text) for text in top_texts)
            rerank_scores = reranker_model.compute_score(pairs)
            top_indices = np.argsort(rerank_scores)[-top_r:][::-1]
            # values, top_indices = np.topk(rerank_scores, k=top_r)

        return [
            RetrievedChunk(
                chunk=self.chunks[idx],
                scores={
                    "dense_score": dense_scores[0][idx],
                    "sparse_score": sparse_scores[0][idx],
                    "colbert_score": colbert_scores[0][idx],
                    "hybrid_score": hybrid_scores[0][idx],
                    "rerank_score": rerank_scores[idx]
                    if rerank_scores
                    else None,
                },
            )
            for idx in top_indices[top_indices > threshold].tolist()
        ]

    def _get_embedding(
        self, texts: list[str], embedding_model: EmbeddingModel
    ) -> tuple[DenseEmbeddings, SparseEmbeddings, ColbertEmbeddings]:
        result = embedding_model.encode(
            texts, return_dense=True, return_sparse=True, return_colbert=True
        )
        return result["dense"], result["sparse"], result["colbert"]

    # @lru_cache(maxsize=1000)
    # def _get_embedding_cached(self, text: str) -> tuple[np.ndarray, ...]:
    #     return self._get_embedding([text])[0]

    def save(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "documents": self.documents,
            "chunks": self.chunks,
            "dense_embeddings": self.dense_embeddings,
            "sparse_embeddings": self.sparse_embeddings,
            "colbert_embeddings": self.colbert_embeddings,
        }
        with open(self.db_path, "wb") as f:
            pickle.dump(data, f)
        logger.debug(f"Saved KnowledgeBase state to {self.db_path}")

    def load(self) -> None:
        if self.db_path.exists():
            with open(self.db_path, "rb") as f:
                data = pickle.load(f)
            self.documents = data.get("documents", [])
            self.chunks = data.get("chunks", [])
            self.dense_embeddings = data.get("dense_embeddings", [])
            self.sparse_embeddings = data.get("sparse_embeddings", [])
            self.colbert_embeddings = data.get("colbert_embeddings", [])
            logger.debug(f"Loaded KnowledgeBase state from {self.db_path}")
        else:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            logger.debug(f"No saved state found at {self.db_path}")
