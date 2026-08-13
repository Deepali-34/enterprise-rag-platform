from rank_bm25 import BM25Okapi

from app.vectordb.chroma_db import get_all_documents


class BM25Retriever:
    """
    Keyword-based BM25 retriever.
    """

    def __init__(self):

        self.documents = get_all_documents()

        self.corpus = [
            doc.page_content.split()
            for doc in self.documents
        ]

        self.bm25 = BM25Okapi(self.corpus)

    def search(self, query, k=3):
        """
        Return top-k BM25 documents.
        """

        tokenized_query = query.split()

        scores = self.bm25.get_scores(tokenized_query)

        ranked = sorted(
            zip(self.documents, scores),
            key=lambda x: x[1],
            reverse=True
        )

        return [
            doc
            for doc, score in ranked[:k]
        ]


def bm25_search(query, k=3):
    """
    Convenience function.
    """

    retriever = BM25Retriever()

    return retriever.search(query, k)


def main():

    query = input("Enter query: ")

    results = bm25_search(query)

    print("\n" + "=" * 60)
    print("BM25 Results")
    print("=" * 60)

    for i, doc in enumerate(results, start=1):

        print(f"\nResult {i}")
        print("-" * 40)

        print(
            doc.metadata.get(
                "filename",
                doc.metadata.get("source", "Unknown")
            )
        )

        print(doc.page_content[:300])


if __name__ == "__main__":
    main()