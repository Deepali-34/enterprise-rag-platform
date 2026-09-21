from typing import List

from langchain_core.documents import Document
from app.embeddings.embedding_generator import model


DEFAULT_MAX_DOCUMENTS = 3
DEFAULT_MIN_SCORE = 0.35


def cosine_similarity(vector_a, vector_b):
    """
    Calculate cosine similarity between two vectors.
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

    return dot_product / (
        magnitude_a * magnitude_b
    )


def compress_documents(
    question: str,
    documents: List[Document],
    max_documents: int = DEFAULT_MAX_DOCUMENTS,
    min_score: float = DEFAULT_MIN_SCORE
):
    """
    Compress retrieved documents by semantic similarity.

    The documents are ranked using cosine similarity between
    the question embedding and each document embedding.

    Important:
    If no document reaches the minimum similarity threshold,
    the highest-scoring documents are still retained so that
    the RAG pipeline does not incorrectly return an empty result.
    """

    if not documents:
        return []

    # ---------------------------------------------------------
    # Generate question embedding
    # ---------------------------------------------------------

    question_embedding = model.encode(question)

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
            (document, float(score))
        )

    # ---------------------------------------------------------
    # Sort by similarity
    # ---------------------------------------------------------

    scored_documents.sort(
        key=lambda item: item[1],
        reverse=True
    )

    # ---------------------------------------------------------
    # First try the minimum similarity threshold
    # ---------------------------------------------------------

    filtered_documents = [
        (document, score)
        for document, score in scored_documents
        if score >= min_score
    ]

    # ---------------------------------------------------------
    # Robust fallback
    #
    # If every document falls below the threshold, keep the
    # highest-scoring documents instead of returning zero.
    #
    # The cross-encoder re-ranker will perform another,
    # stronger relevance evaluation afterwards.
    # ---------------------------------------------------------

    if not filtered_documents:
        print(
            "[COMPRESSION] No documents reached "
            f"minimum score {min_score}. "
            "Keeping highest-scoring candidates."
        )

        filtered_documents = scored_documents[
            :max_documents
        ]

    # ---------------------------------------------------------
    # Keep only the configured number of documents
    # ---------------------------------------------------------

    compressed_documents = [
        document
        for document, score in filtered_documents[
            :max_documents
        ]
    ]

    # ---------------------------------------------------------
    # Debug information
    # ---------------------------------------------------------

    print(
        f"[COMPRESSION] Input documents: "
        f"{len(documents)}"
    )

    print(
        f"[COMPRESSION] Output documents: "
        f"{len(compressed_documents)}"
    )

    for index, document in enumerate(
        compressed_documents,
        start=1
    ):

        metadata = document.metadata or {}

        print(
            f"[COMPRESSION] Document {index}: "
            f"{metadata.get('filename', 'Unknown')} | "
            f"score="
            f"{metadata.get('compression_score', 'N/A')}"
        )

    return compressed_documents