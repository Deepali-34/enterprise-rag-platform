from typing import List

from langchain_core.documents import Document

from app.embeddings.embedding_generator import model


# Maximum number of documents allowed after compression
DEFAULT_MAX_DOCUMENTS = 3

# Minimum semantic similarity required
DEFAULT_MIN_SCORE = 0.35


def cosine_similarity(vector_a, vector_b):
    """
    Calculate cosine similarity between two embedding vectors.
    """

    dot_product = sum(
        a * b for a, b in zip(vector_a, vector_b)
    )

    magnitude_a = sum(
        a * a for a in vector_a
    ) ** 0.5

    magnitude_b = sum(
        b * b for b in vector_b
    ) ** 0.5

    if magnitude_a == 0 or magnitude_b == 0:
        return 0.0

    return dot_product / (magnitude_a * magnitude_b)


def compress_documents(
    question: str,
    documents: List[Document],
    max_documents: int = DEFAULT_MAX_DOCUMENTS,
    min_score: float = DEFAULT_MIN_SCORE
):
    """
    Compress retrieved documents using semantic similarity.

    Process:
        1. Generate an embedding for the question.
        2. Generate embeddings for retrieved chunks.
        3. Calculate cosine similarity.
        4. Remove low-relevance chunks.
        5. Keep the highest-scoring chunks.
    """

    if not documents:
        return []

    # ---------------------------------------------------------
    # Generate question embedding
    # ---------------------------------------------------------

    question_embedding = model.encode(
        question
    )

    # ---------------------------------------------------------
    # Generate document embeddings
    # ---------------------------------------------------------

    document_texts = [
        document.page_content
        for document in documents
    ]

    document_embeddings = model.encode(
        document_texts
    )

    # ---------------------------------------------------------
    # Calculate similarity scores
    # ---------------------------------------------------------

    scored_documents = []

    for document, document_embedding in zip(
        documents,
        document_embeddings
    ):

        score = cosine_similarity(
            question_embedding,
            document_embedding
        )

        if document.metadata is None:
            document.metadata = {}

        document.metadata["compression_score"] = round(
            float(score),
            4
        )

        scored_documents.append(
            (document, score)
        )

    # ---------------------------------------------------------
    # Rank by semantic relevance
    # ---------------------------------------------------------

    scored_documents.sort(
        key=lambda item: item[1],
        reverse=True
    )

    # ---------------------------------------------------------
    # Keep only relevant documents
    # ---------------------------------------------------------

    compressed_documents = []

    for document, score in scored_documents:

        if score < min_score:
            continue

        compressed_documents.append(document)

        if len(compressed_documents) >= max_documents:
            break

    return compressed_documents


def main():

    print("=" * 70)
    print("Enterprise RAG Platform")
    print("Advanced RAG - Context Compression")
    print("=" * 70)

    question = input("\nEnter your question: ").strip()

    if not question:
        print("Question cannot be empty.")
        return

    # Import here to keep the retrieval pipeline separate
    from app.retriever.multi_query_retriever import multi_query_search

    # ---------------------------------------------------------
    # Multi-Query + Hybrid Retrieval
    # ---------------------------------------------------------

    print("\nRunning Multi-Query + Hybrid Retrieval...")

    documents = multi_query_search(
        question=question,
        k=5
    )

    print(
        f"\nRetrieved documents before compression: "
        f"{len(documents)}"
    )

    # ---------------------------------------------------------
    # Context Compression
    # ---------------------------------------------------------

    print("\nRunning Context Compression...")

    compressed_documents = compress_documents(
        question=question,
        documents=documents,
        max_documents=3,
        min_score=DEFAULT_MIN_SCORE
    )

    # ---------------------------------------------------------
    # Display results
    # ---------------------------------------------------------

    print("\n" + "=" * 70)
    print("Compressed Context")
    print("=" * 70)

    print(
        f"Documents after compression: "
        f"{len(compressed_documents)}"
    )

    for index, document in enumerate(
        compressed_documents,
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

        print("\nContent:")
        print(document.page_content[:500])


if __name__ == "__main__":
    main()