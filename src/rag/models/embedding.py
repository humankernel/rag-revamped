from threading import Lock
from typing import Literal, Optional, TypeVar

import numpy as np
import torch
from FlagEmbedding import BGEM3FlagModel

from rag.settings import settings
from rag.utils.helpers import SingletonMeta


N = TypeVar("N", bound="int")

DenseEmbeddings = np.ndarray[np.float16, tuple[N, Literal[1024]]]
SparseEmbeddings = list[dict[str, float]]
ColbertEmbeddings = list[np.ndarray[np.float16, tuple[N, Literal[1024]]]]


class EmbeddingModel(metaclass=SingletonMeta):
    _model: Optional[BGEM3FlagModel] = None
    _lock: Lock = Lock()

    def __init__(
        self, model_name_or_path: str = settings.EMBEDDING_MODEL, **kwargs
    ):
        self._model = BGEM3FlagModel(
            model_name_or_path,
            normalize_embeddings=True,
            use_fp16=True,
            devices=settings.DEVICE,
            **kwargs,
        )

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
        Optional[DenseEmbeddings | SparseEmbeddings | ColbertEmbeddings],
    ]:
        with self._lock:
            result = self._model.encode(
                sentences,
                batch_size=batch_size,
                return_dense=return_dense,
                return_sparse=return_sparse,
                return_colbert_vecs=return_colbert,
                max_length=settings.EMBEDDING_TOKEN_LIMIT,
                convert_to_numpy=False,
                **kwargs,
            )

        results = {
            "dense": result.get("dense_vecs", None),
            "sparse": result.get("lexical_weights", None),
            "colbert": result.get("colbert_vecs", None),
        }
        return results


def dense_similarity(
    embeddings_1: DenseEmbeddings,
    embeddings_2: DenseEmbeddings,
) -> torch.tensor:
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
    lexical_weights_1: SparseEmbeddings,
    lexical_weights_2: SparseEmbeddings,
) -> torch.tensor:
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
    embeddings_1: ColbertEmbeddings,
    embeddings_2: ColbertEmbeddings,
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
        >>> dense_scores = torch.tensor(
        ...     [[0.6177], [0.3333], [0.5869]], dtype=torch.float16
        ... )
        >>> sparse_scores = torch.tensor(
        ...     [[0.2620], [0.0000], [0.2878]], dtype=torch.float16
        ... )
        >>> colbert_scores = torch.tensor(
        ...     [[0.8054], [0.5704], [0.4001]], dtype=torch.float16
        ... )
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
