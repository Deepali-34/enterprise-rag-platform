from typing import List

from langchain_core.documents import Document
from sentence_transformers import CrossEncoder


# Local cross-encoder model
MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

DEFAULT_TOP_K = 3


# Load model once when the module starts
reranker_model = CrossEncoder(MODEL_NAME)


def rerank_documents(
    question: str,
    documents: List[Document],
    top_k: int = DEFAULT_TOP_K
):
    """
    Re-rank retrieved documents using a Cross-Encoder.

    The Cross-Encoder directly evaluates:

        (question, document)

    and produces a relevance score.
    """

    if not documents:
        return []

    # Create question-document pairs
    pairs = [
        [question, document.page_content]
        for document in documents
    ]

    # Calculate relevance scores
    scores = reranker_model.predict(pairs)

    scored_documents = []

    for document, score in zip(documents, scores):

        if document.metadata is None:
            document.metadata = {}

        document.metadata["rerank_score"] = round(
            float(score),
            4
        )

        scored_documents.append(
            (document, float(score))
        )

    # Highest relevance first
    scored_documents.sort(
        key=lambda item: item[1],
        reverse=True
    )

    # Return top documents
    reranked_documents = [
        document
        for document, score in scored_documents[:top_k]
    ]

    return reranked_documents


def main():

    print("=" * 70)
    print("Enterprise RAG Platform")
    print("Advanced RAG - Re-ranking")
    print("=" * 70)

    question = input("\nEnter your question: ").strip()

    if not question:
        print("Question cannot be empty.")
        return

    # ---------------------------------------------------------
    # Context Compression
    # ---------------------------------------------------------

    from app.context_compressor.compressor import compress_documents
    from app.retriever.multi_query_retriever import multi_query_search

    print("\nRunning Multi-Query + Hybrid Retrieval...")

    documents = multi_query_search(
        question=question,
        k=5
    )

    print(
        f"Retrieved documents: {len(documents)}"
    )

    print("\nRunning Context Compression...")

    compressed_documents = compress_documents(
        question=question,
        documents=documents,
        max_documents=3,
        min_score=0.35
    )

    print(
        f"Documents after compression: "
        f"{len(compressed_documents)}"
    )

    # ---------------------------------------------------------
    # Re-ranking
    # ---------------------------------------------------------

    print("\nRunning Cross-Encoder Re-ranking...")

    reranked_documents = rerank_documents(
        question=question,
        documents=compressed_documents,
        top_k=3
    )

    # ---------------------------------------------------------
    # Display results
    # ---------------------------------------------------------

    print("\n" + "=" * 70)
    print("Final Re-ranked Documents")
    print("=" * 70)

    print(
        f"Final documents: "
        f"{len(reranked_documents)}"
    )

    for index, document in enumerate(
        reranked_documents,
        start=1
    ):

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
            f"Compression Score : "
            f"{metadata.get('compression_score', 0)}"
        )

        print(
            f"Re-rank Score : "
            f"{metadata.get('rerank_score', 0)}"
        )

        print("\nContent:")
        print(document.page_content[:500])


if __name__ == "__main__":
    main()