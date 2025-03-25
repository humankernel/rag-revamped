from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from rag.models.embedding import (
    EmbeddingModel,
    calculate_hybrid_scores,
    colbert_similarity,
    dense_similarity,
    sparse_similarity,
)


@pytest.fixture
def mock_embedding_model():
    """Fixture to initialize EmbeddingModel."""
    with patch("rag.models.embedding.BGEM3FlagModel") as mock_model:
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


def test_singleton(mock_embedding_model):
    model1 = EmbeddingModel()
    model2 = EmbeddingModel()
    assert model1 is model2


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


@pytest.fixture
def sample_embeddings():
    dense = np.random.rand(2, 1024).astype(np.float16)
    sparse = [{"a": 0.5, "b": 0.5}, {"c": 1.0}]
    colbert = [
        np.random.rand(5, 128).astype(np.float16),
        np.random.rand(7, 128).astype(np.float16),
    ]
    return dense, sparse, colbert


def test_dense_similarity_shape(sample_embeddings):
    dense, _, _ = sample_embeddings
    scores = dense_similarity(dense[:1], dense)
    assert scores.shape == (1, 2)


def test_sparse_similarity_calculation():
    """Test basic cosine similarity calculation"""
    emb1 = [{"a": 3.0, "b": 4.0}]  # L2 norm = 5
    emb2 = [{"a": 3.0, "b": 4.0}]  # L2 norm = 5
    result = sparse_similarity(emb1, emb2)
    expected = torch.tensor([[1.0]], dtype=torch.float16)
    assert torch.allclose(result, expected, atol=1e-4)


def test_partial_overlap():
    """Test embeddings with partial token overlap"""
    emb1 = [{"a": 3.0, "b": 4.0}]  # Norm = 5
    emb2 = [{"a": 6.0, "c": 8.0}]  # Norm = 10
    result = sparse_similarity(emb1, emb2)
    expected_value = (3 * 6) / (5 * 10)  # 18/50 = 0.36
    expected = torch.tensor([[expected_value]], dtype=torch.float16)
    assert torch.allclose(result, expected, atol=1e-4)


def test_empty_embeddings():
    """Test handling of empty embeddings"""
    emb1 = [{}]
    emb2 = [{"a": 1.0}]
    result = sparse_similarity(emb1, emb2)
    expected = torch.tensor([[0.0]], dtype=torch.float16)
    assert torch.allclose(result, expected, atol=1e-4)


def test_colbert_padding(sample_embeddings):
    _, _, colbert = sample_embeddings
    scores = colbert_similarity(colbert, colbert)
    assert scores.shape == (2, 2)
    assert not torch.isnan(scores).any()


def test_colbert_score_range():
    # Test with identical embeddings
    emb = [np.random.rand(5, 128).astype(np.float16)]
    scores = colbert_similarity(emb, emb)
    assert scores[0, 0] > 0.9  # Should be high similarity


def test_basic_combination():
    scores = (
        torch.tensor([[0.5]], dtype=torch.float16),
        torch.tensor([[0.5]], dtype=torch.float16),
    )
    weights = (0.5, 0.5)
    result = calculate_hybrid_scores(scores, weights)
    assert torch.allclose(result, torch.tensor([[0.5]], dtype=torch.float16))


def test_multiple_queries():
    scores = (
        torch.rand(3, 10, dtype=torch.float16),
        torch.rand(3, 10, dtype=torch.float16),
    )
    weights = (0.7, 0.3)
    result = calculate_hybrid_scores(scores, weights)
    assert result.shape == (3, 10)


def test_score_error_handling():
    with pytest.raises(AssertionError):
        calculate_hybrid_scores((torch.tensor([0.5]),), (0.5, 0.5))

    with pytest.raises(AssertionError):
        calculate_hybrid_scores((), (0.5,))

    with pytest.raises(AssertionError):
        calculate_hybrid_scores((torch.tensor([0.5]),), (1.1,))


def test_mixed_precision():
    scores = (
        torch.tensor([[0.5]], dtype=torch.float16),
        torch.tensor([[0.5]], dtype=torch.float32),  # Invalid dtype
    )
    with pytest.raises(AssertionError):
        calculate_hybrid_scores(scores, (0.5, 0.5))
