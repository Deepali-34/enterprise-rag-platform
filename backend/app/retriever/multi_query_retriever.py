from typing import List
from collections import defaultdict

from app.retriever.hybrid_retriever import hybrid_search


NUMBER_OF_QUERIES = 4
DEFAULT_TOP_K = 5


def generate_queries(question: str) -> List[str]:
    """
    Generate local alternative search queries without using an LLM.

    This avoids consuming Gemini API quota during retrieval.
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

    # Remove duplicates while preserving order
    unique_queries = []

    for query in queries:
        query = query.strip()

        if query and query not in unique_queries:
            unique_queries.append(query)

    return unique_queries[:NUMBER_OF_QUERIES]


def document_key(document):
    """
    Create a unique identifier for a retrieved document chunk.
    """

    metadata = document.metadata or {}

    source = metadata.get("source", "")
    page = metadata.get("page", "")

    content = document.page_content.strip()

    return (
        str(source),
        str(page),
        content
    )


def multi_query_search(
    question: str,
    k: int = DEFAULT_TOP_K
):
    """
    Perform Multi-Query Retrieval using local query expansion.

    Flow:

    User Question
        ↓
    Local Query Expansion
        ↓
    Hybrid Search
        ↓
    Deduplication
        ↓
    Rank Fusion
        ↓
    Top K Documents
    """

    queries = generate_queries(question)

    if not queries:
        return []

    document_scores = defaultdict(float)
    document_objects = {}

    for query in queries:

        documents = hybrid_search(
            query=query,
            k=k
        )

        for rank, document in enumerate(documents):

            key = document_key(document)

            # Higher score for higher-ranked documents
            rank_score = k - rank

            document_scores[key] += rank_score

            document_objects[key] = document

    # Sort documents by accumulated score
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

        document.metadata["multi_query_score"] = score

        results.append(document)

        if len(results) >= k:
            break

    return results


def main():

    print("=" * 70)
    print("Enterprise RAG Platform")
    print("Advanced RAG - Multi-Query Retrieval")
    print("=" * 70)

    question = input("\nEnter your question: ").strip()

    if not question:
        print("Question cannot be empty.")
        return

    # ---------------------------------------------------------
    # Generate local queries
    # ---------------------------------------------------------

    print("\nGenerating local alternative queries...")

    queries = generate_queries(question)

    print("\nGenerated Queries")
    print("-" * 70)

    for index, query in enumerate(queries, start=1):
        print(f"{index}. {query}")

    # ---------------------------------------------------------
    # Hybrid Search
    # ---------------------------------------------------------

    print("\nRunning Hybrid Search for each query...")

    documents = multi_query_search(
        question=question,
        k=DEFAULT_TOP_K
    )

    # ---------------------------------------------------------
    # Display results
    # ---------------------------------------------------------

    print("\n" + "=" * 70)
    print("Final Ranked Documents")
    print("=" * 70)

    print(f"Total unique chunks: {len(documents)}")

    for index, document in enumerate(documents, start=1):

        metadata = document.metadata or {}

        print(f"\nResult {index}")
        print("-" * 50)

        print(
            f"Source : "
            f"{metadata.get('source', 'Unknown')}"
        )

        print(
            f"Page   : "
            f"{metadata.get('page', 'Unknown')}"
        )

        print(
            f"Score  : "
            f"{metadata.get('multi_query_score', 0)}"
        )

        print("\nContent:")
        print(document.page_content[:500])


if __name__ == "__main__":
    main()