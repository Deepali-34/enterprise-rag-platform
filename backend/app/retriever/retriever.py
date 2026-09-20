from app.retriever.multi_query_retriever import multi_query_search
from app.context_compressor.compressor import compress_documents
from app.reranker.reranker import rerank_documents


# Retrieval configuration
INITIAL_TOP_K = 5
COMPRESSED_TOP_K = 3
FINAL_TOP_K = 3
COMPRESSION_MIN_SCORE = 0.35


def search_documents(query: str):
    """
    Complete Advanced RAG retrieval pipeline.

    Pipeline:
        Multi-Query Retrieval
            ↓
        Hybrid Search
            ↓
        Context Compression
            ↓
        Cross-Encoder Re-ranking
            ↓
        Final Documents
    """

    # ---------------------------------------------------------
    # Stage 1: Multi-Query + Hybrid Search
    # ---------------------------------------------------------

    documents = multi_query_search(
        question=query,
        k=INITIAL_TOP_K
    )

    # ---------------------------------------------------------
    # Stage 2: Context Compression
    # ---------------------------------------------------------

    compressed_documents = compress_documents(
        question=query,
        documents=documents,
        max_documents=COMPRESSED_TOP_K,
        min_score=COMPRESSION_MIN_SCORE
    )

    # ---------------------------------------------------------
    # Stage 3: Cross-Encoder Re-ranking
    # ---------------------------------------------------------

    reranked_documents = rerank_documents(
        question=query,
        documents=compressed_documents,
        top_k=FINAL_TOP_K
    )

    return reranked_documents


def main():

    query = input("Enter your question: ").strip()

    if not query:
        print("Question cannot be empty.")
        return

    print("\nRunning Advanced RAG retrieval...")

    results = search_documents(query)

    print("\n" + "=" * 70)
    print("Advanced RAG Retrieved Chunks")
    print("=" * 70)

    print(f"\nFinal Results: {len(results)}")

    for i, doc in enumerate(results, start=1):

        metadata = doc.metadata or {}

        print(f"\nResult {i}")
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
            f"Multi-Query Score : "
            f"{metadata.get('multi_query_score', 'N/A')}"
        )

        print(
            f"Compression Score : "
            f"{metadata.get('compression_score', 'N/A')}"
        )

        print(
            f"Re-rank Score : "
            f"{metadata.get('rerank_score', 'N/A')}"
        )

        print("\nContent:")
        print(doc.page_content[:500])


if __name__ == "__main__":
    main()