
from pathlib import Path
from collections import defaultdict

from app.retriever.bm25_retriever import BM25Retriever
from app.vectordb.chroma_db import get_vector_db


DENSE_WEIGHT = 0.7
BM25_WEIGHT = 0.3
RRF_K = 60


def dense_search(query: str, k: int = 5):
    """
    Retrieve documents using dense vector search.
    """

    db = get_vector_db()

    return db.similarity_search(query, k=k)


def reciprocal_rank_fusion(
    dense_documents,
    bm25_documents,
    k: int = 5
):
    """
    Combine dense and BM25 rankings using Reciprocal Rank Fusion.
    """

    scores = defaultdict(float)
    document_map = {}

    # Dense retrieval ranking
    for rank, document in enumerate(dense_documents, start=1):

        key = (
            document.metadata.get("source", ""),
            document.metadata.get("page", 0),
            document.page_content
        )

        scores[key] += DENSE_WEIGHT / (RRF_K + rank)
        document_map[key] = document

    # BM25 retrieval ranking
    for rank, document in enumerate(bm25_documents, start=1):

        key = (
            document.metadata.get("source", ""),
            document.metadata.get("page", 0),
            document.page_content
        )

        scores[key] += BM25_WEIGHT / (RRF_K + rank)
        document_map[key] = document

    ranked = sorted(
        scores.items(),
        key=lambda item: item[1],
        reverse=True
    )

    results = []

    for key, score in ranked[:k]:

        document = document_map[key]

        if document.metadata is None:
            document.metadata = {}

        document.metadata["hybrid_score"] = round(
            float(score),
            6
        )

        results.append(document)

    return results


def hybrid_search(
    query: str,
    k: int = 5,
    bm25_retriever=None
):
    """
    Weighted Hybrid Search.

    Combines:
        Dense Vector Search = 70%
        BM25 Keyword Search = 30%

    Reciprocal Rank Fusion combines the two rankings.
    """

    if not query.strip():
        return []

    dense_documents = dense_search(query, k)

    if bm25_retriever is None:
        bm25_retriever = BM25Retriever()

    bm25_documents = bm25_retriever.search(query, k)

    return reciprocal_rank_fusion(
        dense_documents=dense_documents,
        bm25_documents=bm25_documents,
        k=k
    )


def main():

    print("=" * 60)
    print("Weighted Hybrid Retrieval")
    print("=" * 60)

    query = input("\nEnter query: ").strip()

    if not query:
        print("Query cannot be empty.")
        return

    results = hybrid_search(query, k=5)

    print(f"\nResults found: {len(results)}")

    for index, document in enumerate(results, start=1):

        metadata = document.metadata or {}

        filename = metadata.get(
            "filename",
            Path(metadata.get("source", "Unknown")).name
        )

        print(f"\nResult {index}")
        print("-" * 40)
        print("File:", filename)
        print("Page:", metadata.get("page", "Unknown"))
        print("Hybrid Score:", metadata.get("hybrid_score", 0))
        print("\nContent:")
        print(document.page_content[:300])


if __name__ == "__main__":
    main()
