from pathlib import Path

from app.retriever.hybrid_retriever import hybrid_search


def search_documents(query: str):
    """
    Perform Hybrid Search (Dense + BM25).
    """

    return hybrid_search(query)


def main():

    query = input("Enter your question: ")

    results = search_documents(query)

    print("\n" + "=" * 60)
    print("Hybrid Search Results")
    print("=" * 60)

    for i, doc in enumerate(results, start=1):

        print(f"\nResult {i}")
        print("-" * 40)

        source = doc.metadata.get("source", "Unknown")
        filename = Path(source).name

        page = doc.metadata.get(
            "page",
            doc.metadata.get("page_number", 0)
        ) + 1

        print(f"File : {filename}")
        print(f"Page : {page}")

        print("\nContent:\n")
        print(doc.page_content[:500])


if __name__ == "__main__":
    main()