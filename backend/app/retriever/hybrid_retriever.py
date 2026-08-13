from pathlib import Path
from collections import defaultdict

from app.retriever.bm25_retriever import BM25Retriever
from app.vectordb.chroma_db import get_vector_db


DENSE_WEIGHT = 0.7
BM25_WEIGHT = 0.3


def dense_search_with_scores(query: str, k: int = 5):
    """
    Dense vector search with normalized scores.
    """

    db = get_vector_db()

    results = db.similarity_search_with_relevance_scores(query, k=k)

    return results


def bm25_search_with_scores(query: str, k: int = 5):
    """
    BM25 keyword search with normalized scores.
    """

    retriever = BM25Retriever()

    tokenized_query = query.split()

    scores = retriever.bm25.get_scores(tokenized_query)

    if len(scores) == 0:
        return []

    max_score = max(scores)

    ranked = sorted(
        zip(retriever.documents, scores),
        key=lambda x: x[1],
        reverse=True
    )

    results = []

    for doc, score in ranked[:k]:

        normalized = score / max_score if max_score > 0 else 0

        results.append((doc, normalized))

    return results


def hybrid_search(query: str, k: int = 5):
    """
    Weighted Hybrid Search (Dense + BM25).
    """

    dense_results = dense_search_with_scores(query, k)

    bm25_results = bm25_search_with_scores(query, k)

    document_scores = defaultdict(float)
    document_map = {}

    for doc, score in dense_results:

        key = (
            doc.metadata.get("source", ""),
            doc.metadata.get("page", 0),
            doc.page_content
        )

        document_scores[key] += score * DENSE_WEIGHT

        document_map[key] = doc

    for doc, score in bm25_results:

        key = (
            doc.metadata.get("source", ""),
            doc.metadata.get("page", 0),
            doc.page_content
        )

        document_scores[key] += score * BM25_WEIGHT

        document_map[key] = doc

    ranked = sorted(
        document_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )

    return [
        document_map[key]
        for key, _ in ranked[:k]
    ]


def main():

    query = input("Enter query: ")

    results = hybrid_search(query)

    print("\n" + "=" * 60)
    print("Weighted Hybrid Search Results")
    print("=" * 60)

    for i, doc in enumerate(results, start=1):

        filename = Path(
            doc.metadata.get("source", "Unknown")
        ).name

        page = doc.metadata.get("page", 0) + 1

        print(f"\nResult {i}")
        print("-" * 40)

        print(f"File : {filename}")
        print(f"Page : {page}")

        print(doc.page_content[:300])


if __name__ == "__main__":
    main()