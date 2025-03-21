import numpy as np
import pytest
import torch

from rag.models.embedding import EmbeddingModel


@pytest.fixture
def embedding_model():
    """Fixture to initialize EmbeddingModel."""
    return EmbeddingModel()


@pytest.fixture
def sample_sentences():
    """Fixture providing sample sentences."""
    return ["This is a test sentence.", "Another sentence for testing embeddings."]


def test_dense_only(embedding_model, sample_sentences):
    result = embedding_model.encode(sample_sentences, return_dense=True, return_sparse=False, return_colbert=False)
    assert isinstance(result["dense"], np.ndarray), "Dense embeddings should be a numpy array"
    assert result["dense"].shape[0] == len(sample_sentences), "Number of dense embeddings should match sentences"
    assert result["sparse"] is None, "Sparse embeddings should be None"
    assert result["colbert"] is None, "Colbert embeddings should be None"


def test_sparse_only(embedding_model, sample_sentences):
    result = embedding_model.encode(sample_sentences, return_dense=False, return_sparse=True, return_colbert=False)
    assert result["dense"] is None, "Dense embeddings should be None"
    assert isinstance(result["sparse"], list), "Sparse embeddings should be a list"
    assert len(result["sparse"]) == len(sample_sentences), "Number of sparse embeddings should match sentences"
    assert result["colbert"] is None, "Colbert embeddings should be None"


def test_colbert_only(embedding_model, sample_sentences):
    result = embedding_model.encode(sample_sentences, return_dense=False, return_sparse=False, return_colbert=True)
    assert result["dense"] is None, "Dense embeddings should be None"
    assert result["sparse"] is None, "Sparse embeddings should be None"
    assert len(result["colbert"]) == len(sample_sentences), "Number of colbert embeddings should match sentences"


def test_all_embeddings(embedding_model, sample_sentences):
    result = embedding_model.encode(sample_sentences, return_dense=True, return_sparse=True, return_colbert=True)
    assert isinstance(result["dense"], np.ndarray), "Dense embeddings should be a numpy array"
    assert result["dense"].shape[0] == len(sample_sentences), "Number of dense embeddings should match sentences"
    assert isinstance(result["sparse"], list), "Sparse embeddings should be a list"
    assert len(result["sparse"]) == len(sample_sentences), "Number of sparse embeddings should match sentences"
    assert isinstance(result["colbert"], list), "Colbert embeddings should be a list"
    assert len(result["colbert"]) == len(sample_sentences), "Number of colbert embeddings should match sentences"


def test_invalid_batch_size(embedding_model, sample_sentences):
    with pytest.raises(ValueError, match="batch_size must be positive"):
        embedding_model.encode(sample_sentences, batch_size=0)


@pytest.fixture
def dense_embeddings():
    dense_embeddings_1 = np.array([[0.267, 0.534, 0.802], [0.4558, 0.57, 0.684]], dtype=np.float16)
    dense_embeddings_2 = np.array([[0.503, 0.574, 0.646], [0.5234, 0.5757, 0.6284]], dtype=np.float16)
    return dense_embeddings_1, dense_embeddings_2


@pytest.fixture
def sparse_embeddings():
    sparse_embeddings_1 = [{"a": 0.1, "b": 0.2}, {"c": 0.3, "d": 0.4}]
    sparse_embeddings_2 = [{"a": 0.5, "e": 0.6}, {"b": 0.7, "d": 0.8}]
    return sparse_embeddings_1, sparse_embeddings_2


@pytest.fixture
def colbert_embeddings():
    colbert_embeddings_1 = [np.array([[0.1, 0.2], [0.3, 0.4]]), np.array([[0.5, 0.6], [0.7, 0.8]])]
    colbert_embeddings_2 = [np.array([[0.9, 1.0], [1.1, 1.2]]), np.array([[1.3, 1.4], [1.5, 1.6]])]
    return colbert_embeddings_1, colbert_embeddings_2


@pytest.fixture
def hybrid_scores_data():
    dense_scores = torch.tensor([[0.6177, 0.3333], [0.5869, 0.4001]], dtype=torch.float16)
    sparse_scores = torch.tensor([[0.2620, 0.0000], [0.2878, 0.5704]], dtype=torch.float16)
    colbert_scores = torch.tensor([[0.8054, 0.5704], [0.4001, 0.8054]], dtype=torch.float16)
    weights = [0.6, 0.2, 0.2]
    return (dense_scores, sparse_scores, colbert_scores), weights


def test_dense_similarity(dense_embeddings):
    dense_1, dense_2 = dense_embeddings
    scores = EmbeddingModel.dense_similarity(dense_1, dense_2)
    expected_scores = torch.tensor([[0.9590, 0.9512], [0.9985, 0.9966]], dtype=torch.float16)
    assert torch.allclose(scores, expected_scores, atol=1e-3), "Dense similarity scores do not match expected values"
    assert scores.dtype == torch.float16, "Expected float16 dtype for dense similarity scores"
    assert scores.shape == (2, 2), "Expected shape (2, 2) for dense similarity scores"


def test_sparse_similarity(sparse_embeddings):
    sparse_1, sparse_2 = sparse_embeddings
    scores = EmbeddingModel.sparse_similarity(sparse_1, sparse_2)
    expected_scores = torch.tensor([[0.2864, 0.5889], [0.0000, 0.6021]], dtype=torch.float16)
    assert torch.allclose(scores, expected_scores, atol=1e-3), "Sparse similarity scores do not match expected values"
    assert scores.dtype == torch.float16, "Expected float16 dtype for sparse similarity scores"
    assert scores.shape == (2, 2), "Expected shape (2, 2) for sparse similarity scores"


def test_colbert_similarity(colbert_embeddings):
    colbert_1, colbert_2 = colbert_embeddings
    scores = EmbeddingModel.colbert_similarity(colbert_1, colbert_2)
    expected_scores = torch.tensor([[0.5800, 0.7800], [1.5000, 2.0200]], dtype=torch.float16)
    assert torch.allclose(scores, expected_scores, atol=1e-2), "Colbert similarity scores do not match expected values"
    assert scores.dtype == torch.float16, "Expected float16 dtype for colbert similarity scores"
    assert scores.shape == (2, 2), "Expected shape (2, 2) for colbert similarity scores"


def test_calculate_hybrid_scores(hybrid_scores_data):
    scores_list, weights = hybrid_scores_data
    hybrid_scores = EmbeddingModel.calculate_hybrid_scores(scores_list, weights)
    expected_hybrid_scores = torch.tensor([[0.5840, 0.3140], [0.4897, 0.5152]], dtype=torch.float16)
    assert torch.allclose(hybrid_scores, expected_hybrid_scores, atol=1e-3), (
        "Hybrid scores do not match expected values"
    )
    assert hybrid_scores.dtype == torch.float16, "Expected float16 dtype for hybrid scores"
    assert hybrid_scores.shape == (2, 2), "Expected shape (2, 2) for hybrid scores"
