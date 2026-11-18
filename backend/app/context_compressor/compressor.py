
from pathlib import Path
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

    return dot_product / (magnitude_a * magnitude_b)


def compress_documents(
    question: str,
    documents: List[Document],
    max_documents: int = DEFAULT_MAX_DOCUMENTS,
    min_score: float = DEFAULT_MIN_SCORE
):
    """
    Rank retrieved documents using semantic similarity.

    Retain the highest-scoring documents if none reaches
    the configured minimum similarity threshold.
    """

    if not documents:
        return []

    question_embedding = model.encode(question)

    document_texts = [
        document.page_content
        for document in documents
    ]

    document_embeddings = model.encode(document_texts)

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

    scored_documents.sort(
        key=lambda item: item[1],
        reverse=True
    )

    filtered_documents = [
        (document, score)
        for document, score in scored_documents
        if score >= min_score
    ]

    if not filtered_documents:
        print(
            f"[COMPRESSION] No documents reached "
            f"minimum score {min_score}. "
            "Keeping highest-scoring candidates."
        )

        filtered_documents = scored_documents[:max_documents]

    compressed_documents = [
        document
        for document, score in filtered_documents[:max_documents]
    ]

    print(
        f"[COMPRESSION] Input documents: {len(documents)}"
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

        filename = metadata.get("filename")

        if not filename:
            filename = Path(
                metadata.get("source", "Unknown")
            ).name

        print(
            f"[COMPRESSION] Document {index}: "
            f"{filename} | "
            f"score={metadata.get('compression_score', 'N/A')}"
        )

    return compressed_documents
