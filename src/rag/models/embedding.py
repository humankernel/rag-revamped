from itertools import batched
from threading import Lock
from typing import Literal, Optional

import numpy as np
from FlagEmbedding import BGEM3FlagModel
import torch

from rag.settings import settings
from rag.utils.helpers import SingletonMeta

DenseEmbeddings = np.ndarray[np.float16]
SparseEmbeddings = dict[str, float]
ColbertEmbeddings = np.ndarray[np.float16]


class EmbeddingModel(metaclass=SingletonMeta):
    _model: BGEM3FlagModel | None = None
    _lock: Lock = Lock()

    def __init__(
        self, model_name_or_path: str = settings.EMBEDDING_MODEL, **kwargs
    ):
        """Initialize the EmbeddingModel with a specified model name and configuration.

        Args:
            model_name (str, optional): Name of the embedding model to load. Defaults to settings.EMBEDDING_MODEL.
            **kwargs: Additional keyword arguments passed to BGEM3FlagModel.

        Notes:
            The model is configured with normalize_embeddings=True, use_fp16=True, and devices from settings.
        """
        self._model = BGEM3FlagModel(
            model_name_or_path,
            normalize_embeddings=True,
            use_fp16=True,
            devices=settings.DEVICE,
            **kwargs,
        )

    # FIX: Sometimes it returns no sparse embeddings or different ones between runs
    # [{}, {}, {}]
    def encode(
        self,
        sentences: list[str],
        return_dense: bool = True,
        return_sparse: bool = False,
        return_colbert: bool = False,
        batch_size: int = 32,
        **kwargs,
    ) -> dict[
        Literal["dense", "sparse", "colbert"],
        Optional[
            np.ndarray[DenseEmbeddings]
            | list[SparseEmbeddings]
            | list[ColbertEmbeddings]
        ],
    ]:
        """Encode a list of sentences into dense, sparse, and/or Colbert embeddings.

        Args:
            sentences (list[str]): List of sentences to encode.
            return_dense (bool, optional): Whether to return dense embeddings. Defaults to True.
            return_sparse (bool, optional): Whether to return sparse embeddings. Defaults to False.
            return_colbert (bool, optional): Whether to return Colbert embeddings. Defaults to False.
            batch_size (int, optional): Number of sentences to process per batch. Defaults to 32.
            **kwargs: Additional arguments passed to the BGEM3FlagModel.encode method.

        Returns:
            dict[Literal["dense", "sparse", "colbert"], Optional[...]]:
                A dictionary with keys "dense", "sparse", and "colbert", mapping to:
                - "dense": np.ndarray[DenseEmbeddings] of shape (total_size, embedding_dim) if return_dense is True, else None.
                - "sparse": list[SparseEmbeddings] if return_sparse is True, else None.
                - "colbert": list[ColbertEmbeddings] if return_colbert is True, else None.

        Raises:
            AssertionError: If the model is not initialized or no embedding type is requested.
            ValueError: If batch_size is not positive.

        Notes:
            - Sparse embeddings may occasionally return empty dictionaries ({}) or vary between runs due to model behavior.
            - Dense embeddings are preallocated with a fixed embedding_dim of 1024.
        """
        assert self._model, "LLM model not initialized"
        assert return_dense or return_sparse or return_colbert, (
            "At least one of return_dense | return_sparse | return_colbert must be True"
        )
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

        # preallocate
        # TODO: conv to list again is incase sentences is a generator
        total_size = len(list(sentences))
        embedding_dim = 1024

        dense_embeddings = np.zeros(
            (total_size, embedding_dim), dtype=np.float16
        )
        sparse_embeddings: list[SparseEmbeddings] = []
        colbert_embeddings: list[ColbertEmbeddings] = []

        with self._lock:
            idx = 0
            for batch in batched(sentences, batch_size):
                result = self._model.encode(
                    batch,
                    batch_size=batch_size,
                    return_dense=return_dense,
                    return_sparse=return_sparse,
                    return_colbert_vecs=return_colbert,
                    max_length=settings.EMBEDDING_TOKEN_LIMIT,
                    **kwargs,
                )
                if return_dense:
                    dense_embeddings[idx : idx + batch_size] = result[
                        "dense_vecs"
                    ]
                if return_sparse:
                    sparse_embeddings.extend(result["lexical_weights"])
                if return_colbert:
                    colbert_embeddings.extend(result["colbert_vecs"])
                idx += batch_size

        results: dict[
            Literal["dense", "sparse", "colbert"],
            Optional[
                np.ndarray[DenseEmbeddings]
                | list[SparseEmbeddings]
                | list[ColbertEmbeddings]
            ],
        ] = {
            "dense": dense_embeddings if return_dense else None,
            "sparse": sparse_embeddings if return_sparse else None,
            "colbert": colbert_embeddings if return_colbert else None,
        }
        return results


def dense_similarity(
    embeddings_1: np.ndarray[DenseEmbeddings],
    embeddings_2: np.ndarray[DenseEmbeddings],
) -> torch.tensor:
    """Compute dense similarity scores between two sets of embeddings using dot product.

    Args:
        embeddings_1 (list[DenseEmbeddings]): First set of dense embeddings (queries).
        embeddings_2 (list[DenseEmbeddings]): Second set of dense embeddings (documents).

    Returns:
        np.ndarray[np.float16]: Similarity scores matrix of shape (len(embeddings_1), len(embeddings_2)).
            Since embeddings are normalized, dot product approximates cosine similarity.

    Raises:
        AssertionError: If embeddings_1 or embeddings_2 are not NumPy arrays.
    """
    assert isinstance(embeddings_1, np.ndarray) and isinstance(
        embeddings_2, np.ndarray
    ), "Both dense embeddings should be of type `DenseEmbeddings`"

    # Convert to PyTorch tensors
    e1 = torch.from_numpy(embeddings_1)  # Shape: (n_queries, dim)
    e2 = torch.from_numpy(embeddings_2)  # Shape: (n_documents, dim)

    # Compute dot product (since the embeddings are normalized)
    scores = e1 @ e2.T  # Shape: (n_queries, n_documents)
    return scores


def sparse_similarity(
    lexical_weights_1: list[SparseEmbeddings],
    lexical_weights_2: list[SparseEmbeddings],
) -> torch.tensor:
    """Compute sparse similarity scores between two sets of lexical weights using cosine normalization.

    Args:
        lexical_weights_1 (list[SparseEmbeddings]): First set of sparse embeddings (queries).
        lexical_weights_2 (list[SparseEmbeddings]): Second set of sparse embeddings (documents).

    Returns:
        list[float]: Similarity scores matrix of shape (len(lexical_weights_1), len(lexical_weights_2)),
            normalized to [0, 1] using cosine similarity.

    Raises:
        AssertionError: If lexical_weights_1 or lexical_weights_2 are not lists.
    """
    assert isinstance(lexical_weights_1, list) and isinstance(
        lexical_weights_2, list
    ), "Both sparse embeddings should be of type list"

    def compute_single(lw1: SparseEmbeddings, lw2: SparseEmbeddings):
        """Returns normalized [0, 1] weights for sparse score"""
        dot_product = sum(
            weight * lw2.get(token, 0) for token, weight in lw1.items()
        )
        # cosine normalize by L2 norms of sparse vectors
        norm1 = np.sqrt(sum(w**2 for w in lw1.values()))
        norm2 = np.sqrt(sum(w**2 for w in lw2.values()))
        return dot_product / (norm1 * norm2) if norm1 * norm2 > 0 else 0

    scores_array = [
        [compute_single(lw1, lw2) for lw2 in lexical_weights_2]
        for lw1 in lexical_weights_1
    ]
    return torch.tensor(scores_array, dtype=torch.float16)


def colbert_similarity(
    embeddings_1: list[ColbertEmbeddings],
    embeddings_2: list[ColbertEmbeddings],
) -> torch.Tensor:
    """Compute colbert scores of input queries and passages.

    Args:
        embeddings_1 (list[np.ndarray]): List of Multi-vector embeddings for queries.
        embeddings_2 (list[np.ndarray]): List of Multi-vector embeddings for passages/corpus.

    Returns:
        torch.Tensor: Tensor of shape (n_queries, n_passages) containing Colbert scores.
    """
    # Pad embeddings to max token length
    max_q_tokens = max(e.shape[0] for e in embeddings_1)
    max_p_tokens = max(e.shape[0] for e in embeddings_2)

    q_padded = [
        np.pad(e, ((0, max_q_tokens - e.shape[0]), (0, 0)), mode="constant")
        for e in embeddings_1
    ]
    p_padded = [
        np.pad(e, ((0, max_p_tokens - e.shape[0]), (0, 0)), mode="constant")
        for e in embeddings_2
    ]

    # Convert all embeddings to tensors at once
    q_reps = torch.from_numpy(
        np.stack(q_padded, dtype=np.float16)
    )  # (n_queries, max_q_tokens, dim)
    p_reps = torch.from_numpy(
        np.stack(p_padded, dtype=np.float16)
    )  # (n_passages, max_p_tokens, dim)

    token_scores = torch.einsum("qid,pjd->qpij", q_reps, p_reps)
    scores = (
        token_scores.max(dim=-1)[0].sum(dim=-1) / max_q_tokens
    )  # (n_queries, n_passages)
    return scores


def calculate_hybrid_scores(
    scores: tuple[torch.Tensor, ...],
    weights: tuple[float, ...],
) -> torch.tensor:
    """Calculate hybrid retrieval scores by combining multiple similarity scores with weights.

    Args:
        scores_list (Tuple[torch.Tensor, ...]): A tuple of similarity score tensors, each of dtype torch.float16.
            Expected shapes: (n_documents, 1) for a single query or (n_queries, n_documents) for multiple queries.
            Examples: dense_scores, sparse_scores, colbert_scores.
        weights (List[float]): List of weights corresponding to each score tensor in scores_list.
            Must match the length of scores_list and sum to a non-zero value.

    Returns:
        torch.Tensor: Hybrid scores tensor of shape (n_documents,) or (n_queries, n_documents),
            computed as a weighted sum of the input scores, with dtype torch.float16.

    Raises:
        ValueError: If scores_list and weights have mismatched lengths, weights sum to zero,
            or score shapes are incompatible.
        TypeError: If any score in scores_list is not a torch.Tensor or not float16 dtype.

    Examples:
        >>> dense_scores = torch.tensor([[0.6177], [0.3333], [0.5869]], dtype=torch.float16)
        >>> sparse_scores = torch.tensor([[0.2620], [0.0000], [0.2878]], dtype=torch.float16)
        >>> colbert_scores = torch.tensor([[0.8054], [0.5704], [0.4001]], dtype=torch.float16)
        >>> scores_list = (dense_scores, sparse_scores, colbert_scores)
        >>> weights = [0.6, 0.2, 0.2]
        >>> EmbeddingModel.calculate_hybrid_scores(scores_list, weights)
        tensor([[0.5835],
                [0.3140],
                [0.4895]], dtype=torch.float16)
    """
    assert len(scores) == len(weights), (
        f"Number of score sets ({len(scores)}) must match number of weights ({len(weights)})"
    )
    assert scores, "scores_list cannot be empty"
    assert sum(weights) == 1, "Sum of weights must be one"
    assert all(isinstance(score, torch.Tensor) for score in scores), (
        "Expected torch.tensor for all scores"
    )
    assert all(score.dtype == torch.float16 for score in scores), (
        "Expected torch.float16 for all scores"
    )

    scores = torch.stack(scores)
    weights = torch.tensor(weights, dtype=torch.float16).view(-1, 1, 1)
    hybrid_scores = (scores * weights).sum(dim=0)
    return hybrid_scores
