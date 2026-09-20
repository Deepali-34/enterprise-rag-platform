from pathlib import Path
from collections import defaultdict

from app.retriever.bm25_retriever import BM25Retriever
from app.vectordb.chroma_db import get_vector_db


DENSE_WEIGHT = 0.7
BM25_WEIGHT = 0.3


def normalize_scores(scores):
    """
    Normalize a list of scores to the range 0-1.

    Uses min-max normalization so that even negative
    dense similarity scores are converted into valid
    normalized scores.
    """

    if not scores:
        return []

    min_score = min(scores)
    max_score = max(scores)

    if max_score == min_score:
        return [1.0 for _ in scores]

    return [
        (score - min_score) / (max_score - min_score)
        for score in scores
    ]


def dense_search_with_scores(query: str, k: int = 5):
    """
    Dense vector search with normalized scores.

    Chroma's raw relevance values can sometimes fall outside
    the expected 0-1 range depending on the distance metric.

    We therefore retrieve documents with similarity_search()
    and normalize the returned distance/similarity values
    ourselves.
    """

    db = get_vector_db()

    documents = db.similarity_search(query, k=k)

    if not documents:
        return []

    # Retrieve the same documents with their underlying
    # similarity/distance information when available.
    collection = getattr(db, "_collection", None)

    if collection is None:
        return [
            (document, 1.0 / (index + 1))
            for index, document in enumerate(documents)
        ]

    try:

        query_embedding = db._embedding_function.embed_query(query)

        raw_results = collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            include=["documents", "metadatas", "distances"]
        )

        distances = raw_results.get("distances", [[]])[0]

        if not distances:
            return [
                (document, 1.0 / (index + 1))
                for index, document in enumerate(documents)
            ]

        # Chroma distances are lower for more similar documents.
        # Convert distances into similarity values.
        similarity_scores = [
            1.0 / (1.0 + float(distance))
            for distance in distances
        ]

        normalized_scores = normalize_scores(similarity_scores)

        # Match returned documents with normalized scores.
        results = []

        for document, score in zip(
            documents,
            normalized_scores
        ):
            results.append((document, score))

        return results

    except Exception:

        # Safe fallback if direct collection access is
        # unavailable for the installed Chroma version.
        return [
            (document, 1.0 / (index + 1))
            for index, document in enumerate(documents)
        ]


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

        normalized = (
            score / max_score
            if max_score > 0
            else 0
        )

        results.append((doc, normalized))

    return results


def hybrid_search(query: str, k: int = 5):
    """
    Weighted Hybrid Search.

    Combines:

        Dense Vector Search = 70%
        BM25 Keyword Search = 30%
    """

    dense_results = dense_search_with_scores(
        query,
        k
    )

    bm25_results = bm25_search_with_scores(
        query,
        k
    )

    document_scores = defaultdict(float)
    document_map = {}

    # ----------------------------------------
    # Dense Search
    # ----------------------------------------

    for doc, score in dense_results:

        key = (
            doc.metadata.get("source", ""),
            doc.metadata.get("page", 0),
            doc.page_content
        )

        document_scores[key] += (
            score * DENSE_WEIGHT
        )

        document_map[key] = doc

    # ----------------------------------------
    # BM25 Search
    # ----------------------------------------

    for doc, score in bm25_results:

        key = (
            doc.metadata.get("source", ""),
            doc.metadata.get("page", 0),
            doc.page_content
        )

        document_scores[key] += (
            score * BM25_WEIGHT
        )

        document_map[key] = doc

    # ----------------------------------------
    # Final Ranking
    # ----------------------------------------

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

    query = input("Enter query: ").strip()

    if not query:
        print("Query cannot be empty.")
        return

    results = hybrid_search(query)

    print("\n" + "=" * 60)
    print("Weighted Hybrid Search Results")
    print("=" * 60)

    for i, doc in enumerate(results, start=1):

        filename = Path(
            doc.metadata.get(
                "source",
                "Unknown"
            )
        ).name

        page = doc.metadata.get(
            "page",
            0
        ) + 1

        print(f"\nResult {i}")
        print("-" * 40)

        print(f"File : {filename}")
        print(f"Page : {page}")

        print(doc.page_content[:300])


if __name__ == "__main__":
    main()