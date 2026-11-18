
from unittest.mock import patch

from langchain_core.documents import Document

from app.context_compressor.compressor import (
    cosine_similarity,
    compress_documents,
)


def test_cosine_similarity_identical_vectors():
    vector = [1.0, 2.0, 3.0]

    score = cosine_similarity(vector, vector)

    assert abs(score - 1.0) < 0.0001


def test_cosine_similarity_orthogonal_vectors():
    score = cosine_similarity(
        [1.0, 0.0],
        [0.0, 1.0]
    )

    assert score == 0.0


def test_cosine_similarity_zero_vector():
    score = cosine_similarity(
        [0.0, 0.0],
        [1.0, 2.0]
    )

    assert score == 0.0


def test_compress_documents_handles_empty_input():
    result = compress_documents(
        question="What is Python?",
        documents=[]
    )

    assert result == []


@patch("app.context_compressor.compressor.model")
def test_compress_documents_ranks_relevant_documents(mock_model):

    documents = [
        Document(
            page_content="Unrelated content",
            metadata={"filename": "unrelated.pdf"}
        ),
        Document(
            page_content="Python programming language",
            metadata={"filename": "python.pdf"}
        ),
    ]

    mock_model.encode.side_effect = [
        [1.0, 0.0],
        [
            [0.0, 1.0],
            [1.0, 0.0],
        ],
    ]

    results = compress_documents(
        question="Python programming",
        documents=documents,
        max_documents=2,
        min_score=0.35
    )

    assert len(results) == 1
    assert results[0].metadata["filename"] == "python.pdf"
    assert results[0].metadata["compression_score"] == 1.0


@patch("app.context_compressor.compressor.model")
def test_compress_documents_fallback_when_no_score_passes(mock_model):

    documents = [
        Document(
            page_content="Document one",
            metadata={"filename": "one.pdf"}
        ),
        Document(
            page_content="Document two",
            metadata={"filename": "two.pdf"}
        ),
    ]

    mock_model.encode.side_effect = [
        [1.0, 0.0],
        [
            [0.0, 1.0],
            [0.0, 1.0],
        ],
    ]

    results = compress_documents(
        question="Test question",
        documents=documents,
        max_documents=1,
        min_score=0.35
    )

    assert len(results) == 1
    assert results[0].metadata["compression_score"] == 0.0
