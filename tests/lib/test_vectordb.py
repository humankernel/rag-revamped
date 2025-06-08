from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from lib.models.embedding import EmbeddingModel
from lib.models.rerank import RerankerModel
from lib.schemas import Chunk, Document
from lib.vectordb import VectorDB

EMBEDDING_OUTPUT_SIZE = 768

# Fixtures --------------------------------------------------------------------


@pytest.fixture
def mock_embedding_model():
    model = Mock(spec=EmbeddingModel)
    model.encode.return_value = {
        "dense": np.random.rand(1, 768).astype(np.float16),
        "sparse": [{"test": 0.5}],
        "colbert": [np.random.rand(32, 128).astype(np.float16)],
    }
    return model


@pytest.fixture
def mock_reranker_model():
    model = Mock(spec=RerankerModel)
    model.compute_score.return_value = torch.tensor(
        [[0.9]], dtype=torch.float16
    )
    return model


@pytest.fixture
def sample_documents():
    return [
        Document(id="1", source="test", metadata={"created_at": "2023-01-01"})
    ]


@pytest.fixture
def sample_chunks():
    return [
        Chunk(
            id="1",
            doc_id="1",
            text="Test chunk text",
            original_text="Test original chunk text",
        )
    ]


# Test Cases ------------------------------------------------------------------


def test_insert_documents_and_chunks(
    mock_embedding_model, sample_documents, sample_chunks
):
    kb = VectorDB()
    kb.insert(
        docs=sample_documents,
        chunks=sample_chunks,
        embedding_model=mock_embedding_model,
    )

    documents_count = len(sample_documents)
    chunks_count = len(sample_chunks)

    assert len(kb.documents) == documents_count
    assert len(kb.chunks) == chunks_count
    assert kb.dense_embeddings.shape == (documents_count, EMBEDDING_OUTPUT_SIZE)
    assert len(kb.colbert_embeddings) == documents_count


def test_save_and_load(mock_embedding_model, sample_documents, sample_chunks):
    db_path = Path("testdb")

    # Insert and save
    kb = VectorDB(db_path=db_path)
    kb.insert(
        docs=sample_documents,
        chunks=sample_chunks,
        embedding_model=mock_embedding_model,
    )

    # Load into new instance
    kb_loaded = VectorDB(db_path=db_path)

    assert not kb_loaded.is_empty
    assert len(kb_loaded.chunks) == len(sample_chunks)
    assert np.array_equal(kb.dense_embeddings, kb_loaded.dense_embeddings)


def test_search_with_reranking(
    mock_embedding_model, mock_reranker_model, sample_documents, sample_chunks
):
    kb = VectorDB(Path("testdb"))
    kb.insert(
        docs=sample_documents,
        chunks=sample_chunks,
        embedding_model=mock_embedding_model,
    )
    results = kb.search(
        query="test query",
        embedding_model=mock_embedding_model,
        reranker_model=mock_reranker_model,
        top_k=1,
        top_r=1,
        threshold=0.5,
    )

    assert len(results) == len(sample_chunks)
    assert results[0].chunk.text == "Test chunk text"
    assert results[0].scores["rerank_score"] > 0.9
