
from typing import List
from collections import defaultdict
import re

from app.retriever.hybrid_retriever import hybrid_search
from app.retriever.bm25_retriever import BM25Retriever


NUMBER_OF_QUERIES = 4
DEFAULT_TOP_K = 5


def generate_queries(question: str) -> List[str]:
    """
    Generate alternative search queries locally.
    No LLM API calls are required.
    """

    question = " ".join(question.split()).strip()

    if not question:
        return []

    queries = [
        question,
        f"Explain {question}",
        f"Key concepts related to {question}",
        f"Technical details and important points about {question}",
    ]

    unique_queries = []

    for query in queries:
        if query not in unique_queries:
            unique_queries.append(query)

    return unique_queries[:NUMBER_OF_QUERIES]


def normalize_content(content: str) -> str:
    """
    Normalize document content for duplicate detection.
    """

    return re.sub(
        r"\s+",
        " ",
        content.lower()
    ).strip()


def multi_query_search(
    question: str,
    k: int = DEFAULT_TOP_K
):
    """
    Multi-Query Hybrid Retrieval.

    Flow:
        User Question
            ↓
        Local Query Expansion
            ↓
        Dense + BM25 Hybrid Search
            ↓
        Duplicate Removal
            ↓
        Reciprocal Rank Fusion
            ↓
        Final Top-K Documents
    """

    queries = generate_queries(question)

    if not queries:
        return []

    # Build the BM25 index only once for this search.
    bm25_retriever = BM25Retriever()

    document_scores = defaultdict(float)
    document_objects = {}

    for query in queries:

        documents = hybrid_search(
            query=query,
            k=k,
            bm25_retriever=bm25_retriever
        )

        for rank, document in enumerate(documents, start=1):

            # Merge exact duplicate content even when it appears
            # in different PDF files.
            key = normalize_content(document.page_content)

            if not key:
                continue

            rank_score = 1.0 / (60 + rank)

            document_scores[key] += rank_score

            # Preserve the first document's source metadata.
            if key not in document_objects:
                document_objects[key] = document

    ranked_documents = sorted(
        document_scores.items(),
        key=lambda item: item[1],
        reverse=True
    )

    results = []

    for key, score in ranked_documents:

        document = document_objects[key]

        if document.metadata is None:
            document.metadata = {}

        document.metadata["multi_query_score"] = round(
            float(score),
            6
        )

        results.append(document)

        if len(results) >= k:
            break

    return results


def main():

    print("=" * 70)
    print("Enterprise RAG Platform")
    print("Advanced RAG - Multi-Query Hybrid Retrieval")
    print("=" * 70)

    question = input("\nEnter your question: ").strip()

    if not question:
        print("Question cannot be empty.")
        return

    print("\nGenerating alternative queries...")

    queries = generate_queries(question)

    for index, query in enumerate(queries, start=1):
        print(f"{index}. {query}")

    print("\nRunning Multi-Query Hybrid Search...")

    documents = multi_query_search(
        question=question,
        k=DEFAULT_TOP_K
    )

    print("\n" + "=" * 70)
    print("Final Ranked Documents")
    print("=" * 70)

    print(f"Total unique chunks: {len(documents)}")

    for index, document in enumerate(documents, start=1):

        metadata = document.metadata or {}

        print(f"\nResult {index}")
        print("-" * 50)

        print("Source:", metadata.get("source", "Unknown"))
        print("Page:", metadata.get("page", "Unknown"))
        print(
            "Score:",
            metadata.get("multi_query_score", 0)
        )

        print("\nContent:")
        print(document.page_content[:500])


if __name__ == "__main__":
    main()
