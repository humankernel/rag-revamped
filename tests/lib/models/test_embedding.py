from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from lib.models.embedding import (
    EmbeddingModel,
    calculate_hybrid_scores,
    colbert_similarity,
    dense_similarity,
    sparse_similarity,
)

# Fixtures --------------------------------------------------------------------


@pytest.fixture
def mock_embedding_model():
    """Fixture to initialize EmbeddingModel."""
    with patch("lib.models.embedding.BGEM3FlagModel") as mock_model:
        model = MagicMock()
        mock_model.return_value = model

        def mock_encode(batch, *args, **kwargs):
            current_batch_size = len(batch)
            return {
                "dense_vecs": np.random.rand(current_batch_size, 1024).astype(
                    np.float16
                ),
                "lexical_weights": [
                    {"token": 0.5} for _ in range(current_batch_size)
                ],
                "colbert_vecs": [
                    np.random.rand(10, 128).astype(np.float16)
                    for _ in range(current_batch_size)
                ],
            }

        model.encode.side_effect = mock_encode
        yield model


@pytest.fixture
def sample_embeddings():
    dense = np.random.rand(2, 1024).astype(np.float16)
    sparse = [{"a": 0.5, "b": 0.5}, {"c": 1.0}]
    colbert = [
        np.random.rand(5, 128).astype(np.float16),
        np.random.rand(7, 128).astype(np.float16),
    ]
    return dense, sparse, colbert


# Test Cases -------------------------------------------------------------------


def test_encode_basic(mock_embedding_model):
    model = EmbeddingModel()
    result = model.encode(
        ["text1", "text2", "text3"],
        return_dense=True,
        return_sparse=True,
        return_colbert=True,
    )
    assert "dense" in result
    assert result["dense"].shape == (3, 1024)
    assert len(result["sparse"]) == 3
    assert len(result["colbert"]) == 3


@pytest.mark.parametrize(
    ["return_dense", "return_sparse", "return_colbert"],
    (
        (True, False, False),
        (True, False, True),
        (True, True, False),
        (True, True, True),
        (False, False, True),
        (False, True, False),
    ),
)
def test_encode_flags(
    mock_embedding_model, return_dense, return_sparse, return_colbert
):
    model = EmbeddingModel()
    result = model.encode(
        ["text", "text2", "text3"],
        return_dense=return_dense,
        return_sparse=return_sparse,
        return_colbert=return_colbert,
    )

    if return_dense:
        assert result["dense"] is not None
        assert len(result["dense"]) == 3
    if return_sparse:
        assert result["sparse"] is not None
        assert len(result["sparse"]) == 3
    if return_colbert:
        assert result["colbert"] is not None
        assert len(result["colbert"]) == 3


# --- Dense Similarity ---------------------------------------------------------


def test_dense_similarity_shape(sample_embeddings):
    dense, _, _ = sample_embeddings
    scores = dense_similarity(dense[:1], dense)
    assert scores.shape == (1, 2)


def test_identical_dense_vectors():
    """Identical normalized vectors should return similarity of 1.0"""
    vec = np.array([[0.6, 0.8]], dtype=np.float16)
    scores = dense_similarity(vec, vec)
    assert torch.allclose(
        scores, torch.tensor([1.0], dtype=torch.float16), atol=1e-3
    )


def test_orthogonal_vectors():
    """Orthogonal vectors should return similarity of 0.0"""
    vec1 = np.array([[1.0, 0.0]], dtype=np.float16)
    vec2 = np.array([[0.0, 1.0]], dtype=np.float16)
    scores = dense_similarity(vec1, vec2)
    assert torch.allclose(
        scores, torch.tensor([0.0], dtype=torch.float16), atol=1e-3
    )


def test_negative_similarity():
    """Vectors in opposite directions should return -1.0"""
    vec1 = np.array([[0.6, 0.8]], dtype=np.float16)
    vec2 = np.array([[-0.6, -0.8]], dtype=np.float16)
    scores = dense_similarity(vec1, vec2)
    assert torch.allclose(
        scores, torch.tensor([-1.0], dtype=torch.float16), atol=1e-3
    )


def test_non_normalized_inputs():
    """Should handle non-normalized inputs correctly"""
    vec1 = np.array([[3.0, 4.0]], dtype=np.float16)  # L2 norm = 5
    vec2 = np.array([[1.0, 1.0]], dtype=np.float16)  # L2 norm = √2
    scores = dense_similarity(vec1, vec2)
    expected = 0.6 * 0.7071 + 0.8 * 0.7071  # Normalized vectors' dot product
    assert torch.allclose(
        scores, torch.tensor([expected], dtype=torch.float16), atol=1e-3
    )


def test_precision_limits():
    """Should handle float16 precision limitations gracefully"""
    vec1 = np.array([[0.1, 0.2, 0.3, 0.4]], dtype=np.float16)
    vec2 = np.array([[0.4, 0.3, 0.2, 0.1]], dtype=np.float16)
    scores = dense_similarity(vec1, vec2)

    # Calculate expected value in float16
    vec1_norm = vec1 / np.linalg.norm(vec1, axis=1, keepdims=True)
    vec2_norm = vec2 / np.linalg.norm(vec2, axis=1, keepdims=True)
    expected = (vec1_norm @ vec2_norm.T).item()

    assert abs(scores.item() - expected) < 0.001  # Allow 0.1% error


# --- Sparse Similarity --------------------------------------------------------


def test_identical_sparse_vectors():
    """Identical vectors should return similarity 1.0"""
    vec = [{"apple": 1.0, "banana": 1.0}]
    scores = sparse_similarity(vec, vec)
    assert torch.allclose(scores, torch.tensor([[1.0]], dtype=torch.float16)), (
        "Identical vectors should score 1.0"
    )


def test_no_overlap():
    """Vectors with no overlapping tokens should return 0.0"""
    vec1 = [{"apple": 1.0}]
    vec2 = [{"orange": 1.0}]
    scores = sparse_similarity(vec1, vec2)
    assert torch.allclose(scores, torch.tensor([[0.0]], dtype=torch.float16)), (
        "No overlap should score 0.0"
    )


def test_partial_sparse_overlap():
    """Vectors with partial token overlap"""
    vec1 = [{"apple": 3.0, "banana": 4.0}]  # L2 norm = 5
    vec2 = [{"apple": 3.0, "kiwi": 4.0}]  # L2 norm = 5
    expected = (3 * 3) / (5 * 5)  # 9/25 = 0.36
    scores = sparse_similarity(vec1, vec2)
    assert torch.allclose(
        scores, torch.tensor([[expected]], dtype=torch.float16)
    )


def test_empty_vector():
    """Handle empty vectors (dicts) in input lists"""
    vec1 = [{}]  # Zero vector
    vec2 = [{"apple": 1.0}]
    scores = sparse_similarity(vec1, vec2)
    assert torch.allclose(scores, torch.tensor([[0.0]], dtype=torch.float16)), (
        "Empty vector should score 0.0"
    )


def test_zero_weights():
    """Vectors with zero-weight tokens"""
    vec1 = [{"apple": 0.0, "banana": 0.0}]
    vec2 = [{"apple": 1.0}]
    scores = sparse_similarity(vec1, vec2)
    assert torch.allclose(scores, torch.tensor([[0.0]], dtype=torch.float16)), (
        "Zero weights should score 0.0"
    )


def test_output_shape():
    """Verify correct matrix dimensions"""
    vec1 = [{"a": 1.0}, {"b": 1.0}]  # 2 vectors
    vec2 = [{"c": 1.0}, {"d": 1.0}, {"e": 1.0}]  # 3 vectors
    scores = sparse_similarity(vec1, vec2)
    assert scores.shape == (2, 3), f"Expected (2,3) shape, got {scores.shape}"


def test_float16_dtype():
    """Verify output tensor dtype"""
    vec1 = [{"a": 1.0}]
    vec2 = [{"b": 1.0}]
    scores = sparse_similarity(vec1, vec2)
    assert scores.dtype == torch.float16, (
        f"Expected float16, got {scores.dtype}"
    )


def test_empty_input_lists():
    """Should raise assertion for empty input lists"""
    with pytest.raises(AssertionError):
        sparse_similarity([], [{"a": 1.0}])

    with pytest.raises(AssertionError):
        sparse_similarity([{"a": 1.0}], [])


# --- Colbert Similarity -------------------------------------------------------


def test_colbert_similarity_basic():
    """Test core ColBERT similarity with one simple case"""

    # Setup - 1 query with 2 tokens, 1 passage with 3 tokens
    query = [
        np.array(
            [
                [1.0, 0.0],  # Token 1
                [0.0, 1.0],
            ],  # Token 2
            dtype=np.float16,
        )
    ]

    passage = [
        np.array(
            [
                [1.0, 0.0],  # Matches query token 1 (score=1)
                [0.0, 0.9],  # Best match for query token 2 (score=0.9)
                [0.5, 0.5],
            ],  # Not the best match for either
            dtype=np.float16,
        )
    ]

    # Compute scores
    scores = colbert_similarity(query, passage)

    # Verify
    expected = torch.tensor([[0.95]])  # (1*1 + 1*0.9)/2 = 0.95

    assert scores.shape == (1, 1), "Output shape mismatch"
    assert torch.allclose(scores, expected, atol=0.01), (
        "Score calculation incorrect"
    )


def test_perfect_match():
    """Identical query and passage should return 1.0"""
    emb = np.array([[1.0, 0.0, 0.0]], dtype=np.float16)
    scores = colbert_similarity([emb], [emb])
    assert torch.allclose(scores, torch.tensor([[1.0]]))


def test_partial_match():
    """Test partial token overlap"""
    query = [
        np.array(
            [
                [1.0, 0.0],  # Will match first passage token
                [0.0, 1.0],  # Will match second passage token
            ],
            dtype=np.float16,
        )
    ]

    passage = [
        np.array(
            [
                [1.0, 0.0],  # Perfect match for query token 1
                [0.0, 0.8],  # Partial match for query token 2
                [0.5, 0.5],  # Irrelevant token
            ],
            dtype=np.float16,
        )
    ]

    scores = colbert_similarity(query, passage)
    expected = torch.tensor([[(1.0 + 0.8) / 2]])  # Average of best matches
    assert torch.allclose(scores, expected, atol=0.01)


def test_orthogonal_colbert_vectors():
    """Completely dissimilar vectors should return 0.0"""
    query = [np.array([[1.0, 0.0]], dtype=np.float16)]
    passage = [np.array([[0.0, 1.0]], dtype=np.float16)]
    scores = colbert_similarity(query, passage)
    assert torch.allclose(scores, torch.tensor([[0.0]]))


def test_variable_length():
    """Test handling of different length sequences"""
    query = [np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float16)]

    passage = [
        np.array(
            [
                [1.0, 0.0],  # Only token
            ],
            dtype=np.float16,
        )
    ]

    scores = colbert_similarity(query, passage)
    # Should average scores for both query tokens
    # Even though passage only has 1 token
    expected = torch.tensor([[(1.0 + 0.0) / 2]])  # (match + no-match)/2
    assert torch.allclose(scores, expected)


# --- Hybrid Similarity --------------------------------------------------------


def test_basic_weighted_combination():
    """Test basic weighted score combination"""
    score1 = torch.tensor([[0.8]], dtype=torch.float16)
    score2 = torch.tensor([[0.4]], dtype=torch.float16)
    hybrid = calculate_hybrid_scores(
        scores=(score1, score2), weights=(0.7, 0.3)
    )
    expected = torch.tensor([[0.8 * 0.7 + 0.4 * 0.3]], dtype=torch.float16)
    assert torch.allclose(hybrid, expected, atol=1e-3)


def test_multiple_queries():
    """Test batch processing with multiple queries"""
    score1 = torch.tensor([[0.9], [0.3]], dtype=torch.float16)  # 2 queries
    score2 = torch.tensor([[0.6], [0.8]], dtype=torch.float16)
    hybrid = calculate_hybrid_scores(
        scores=(score1, score2), weights=(0.5, 0.5)
    )
    expected = torch.tensor(
        [[0.75], [0.55]], dtype=torch.float16
    )  # [(0.9+0.6)/2, (0.3+0.8)/2]
    assert torch.allclose(hybrid, expected, atol=1e-3)


def test_three_components():
    """Test combination of three different scores"""
    scores = (
        torch.tensor([[0.6]], dtype=torch.float16),
        torch.tensor([[0.3]], dtype=torch.float16),
        torch.tensor([[0.9]], dtype=torch.float16),
    )
    weights = (0.5, 0.3, 0.2)
    hybrid = calculate_hybrid_scores(scores, weights)
    expected = torch.tensor(
        [[0.6 * 0.5 + 0.3 * 0.3 + 0.9 * 0.2]], dtype=torch.float16
    )
    assert torch.allclose(hybrid, expected, atol=1e-3)


def test_input_validation():
    """Test input validation checks"""
    # Test weights sum != 1
    with pytest.raises(AssertionError):
        calculate_hybrid_scores(
            scores=(torch.tensor([[1.0]], dtype=torch.float16),),
            weights=(0.8,),  # Doesn't sum to 1
        )

    # Test mismatched score/weight lengths
    with pytest.raises(AssertionError):
        calculate_hybrid_scores(
            scores=(torch.tensor([[1.0]], dtype=torch.float16),),
            weights=(0.5, 0.5),
        )

    # Test wrong dtype
    with pytest.raises(AssertionError):
        calculate_hybrid_scores(
            scores=(torch.tensor([[1.0]], dtype=torch.float16),), weights=(1.0,)
        )


def test_precision_handling():
    """Test float16 precision handling"""
    # Small values that could underflow in float16
    score1 = torch.tensor([[0.001]], dtype=torch.float16)
    score2 = torch.tensor([[0.002]], dtype=torch.float16)
    hybrid = calculate_hybrid_scores(
        scores=(score1, score2), weights=(0.5, 0.5)
    )
    expected = torch.tensor([[0.0015]], dtype=torch.float16)
    assert torch.allclose(hybrid, expected, atol=1e-4)
