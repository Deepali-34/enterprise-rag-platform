from app.retriever.multi_query_retriever import multi_query_search


def search_documents(query: str):
    """
    Search the knowledge base using Multi-Query Retrieval.

    Multi-Query Retrieval generates multiple variations of the
    user's query and sends each query through the Hybrid Search
    pipeline.
    """

    return multi_query_search(
        question=query,
        k=5
    )


def main():

    query = input("Enter your question: ").strip()

    if not query:
        print("Question cannot be empty.")
        return

    results = search_documents(query)

    print("\n" + "=" * 60)
    print("Multi-Query Retrieved Chunks")
    print("=" * 60)

    print(f"\nTotal Results: {len(results)}")

    for i, doc in enumerate(results, start=1):

        metadata = doc.metadata or {}

        print(f"\nResult {i}")
        print("-" * 40)

        print(
            f"Source : "
            f"{metadata.get('source', 'Unknown')}"
        )

        print(
            f"Page   : "
            f"{metadata.get('page', 'Unknown')}"
        )

        print("\nContent:\n")
        print(doc.page_content[:500])


if __name__ == "__main__":
    main()