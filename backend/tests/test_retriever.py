
from unittest.mock import MagicMock

from langchain_core.documents import Document

from app.retriever.multi_query_retriever import (
    generate_queries,
    normalize_content,
    multi_query_search,
)


def test_generate_queries_returns_four_queries():
    question = "What is machine learning?"

    queries = generate_queries(question)

    assert len(queries) == 4
    assert queries[0] == question


def test_generate_queries_handles_empty_question():
    assert generate_queries("") == []
    assert generate_queries("   ") == []


def test_normalize_content_removes_case_and_whitespace_differences():
    first = "Machine Learning   is powerful"
    second = "machine learning is powerful"

    assert normalize_content(first) == normalize_content(second)


def test_multi_query_search_removes_duplicate_documents(monkeypatch):
    from app.retriever import multi_query_retriever

    document_one = Document(
        page_content="Python is a programming language.",
        metadata={"source": "resume1.pdf", "page": 0}
    )

    document_duplicate = Document(
        page_content="Python is a programming language.",
        metadata={"source": "resume2.pdf", "page": 1}
    )

    mock_bm25 = MagicMock()
    monkeypatch.setattr(
        multi_query_retriever,
        "BM25Retriever",
        lambda: mock_bm25
    )

    monkeypatch.setattr(
        multi_query_retriever,
        "hybrid_search",
        lambda **kwargs: [document_one, document_duplicate]
    )

    results = multi_query_search(
        question="What is Python?",
        k=5
    )

    assert len(results) == 1
    assert results[0].page_content == document_one.page_content


def test_multi_query_search_handles_empty_question(monkeypatch):
    from app.retriever import multi_query_retriever

    def fail_if_called():
        raise AssertionError("BM25 should not be initialized")

    monkeypatch.setattr(
        multi_query_retriever,
        "BM25Retriever",
        fail_if_called
    )

    results = multi_query_search(
        question="   ",
        k=5
    )

    assert results == []
