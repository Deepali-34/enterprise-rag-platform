
import re

from rank_bm25 import BM25Okapi

from app.vectordb.chroma_db import get_all_documents


def tokenize(text: str):
    """Convert text into lowercase word tokens."""
    return re.findall(r"\b\w+\b", text.lower())


class BM25Retriever:
    """
    Keyword-based BM25 retriever.
    """

    def __init__(self):

        self.documents = get_all_documents()

        self.corpus = [
            tokenize(doc.page_content)
            for doc in self.documents
        ]

        self.bm25 = (
            BM25Okapi(self.corpus)
            if self.corpus
            else None
        )

    def search_with_scores(self, query: str, k: int = 5):
        """
        Return top-k documents with normalized BM25 scores.
        """

        if not self.documents or self.bm25 is None:
            return []

        tokenized_query = tokenize(query)

        if not tokenized_query:
            return []

        scores = self.bm25.get_scores(tokenized_query)

        max_score = max(scores) if len(scores) else 0

        ranked = sorted(
            zip(self.documents, scores),
            key=lambda item: item[1],
            reverse=True
        )

        results = []

        for document, score in ranked[:k]:

            normalized_score = (
                float(score) / float(max_score)
                if max_score > 0
                else 0.0
            )

            results.append(
                (document, normalized_score)
            )

        return results

    def search(self, query: str, k: int = 5):
        """Return only the top-k documents."""

        return [
            document
            for document, score in self.search_with_scores(query, k)
        ]


def bm25_search(query: str, k: int = 5):
    """Convenience function for BM25 search."""

    retriever = BM25Retriever()

    return retriever.search(query, k)


def main():

    print("=" * 60)
    print("BM25 Keyword Retrieval")
    print("=" * 60)

    query = input("\nEnter query: ").strip()

    if not query:
        print("Query cannot be empty.")
        return

    retriever = BM25Retriever()

    results = retriever.search_with_scores(query, k=5)

    print(f"\nResults found: {len(results)}")

    for index, (document, score) in enumerate(results, start=1):

        metadata = document.metadata or {}

        print(f"\nResult {index}")
        print("-" * 40)

        print(
            "File:",
            metadata.get("filename", metadata.get("source", "Unknown"))
        )

        print("BM25 Score:", round(score, 4))

        print("\nContent:")
        print(document.page_content[:300])


if __name__ == "__main__":
    main()
