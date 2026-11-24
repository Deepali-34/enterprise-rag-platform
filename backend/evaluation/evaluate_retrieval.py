
import sys
import json
import time
from pathlib import Path

# Add backend directory to Python import path.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.retriever.retriever import search_documents

TOP_K = 3

# These two PDFs are byte-for-byte identical.
# Treat them as one logical source during evaluation.
DUPLICATE_SOURCE_ALIASES = {
    "API_Test_RAG_Guide.pdf": "RAG_LLM_Architecture_Guide.pdf",
}

TEST_CASES = [
    {
        "query": "What are the basic features of Java and what are the differences between JDK, JRE and JVM?",
        "expected_sources": ["JAVA SYLLABUS.pdf"],
    },
    {
        "query": "What Java programming exercises involve functions, loops, conditionals and calculating averages?",
        "expected_sources": ["Java Exercise 1.pdf"],
    },
    {
        "query": "How are one-dimensional arrays created, initialized and traversed in Java?",
        "expected_sources": ["Java Lecture 10.pdf"],
    },
    {
        "query": "How can a two-dimensional array or matrix be created, populated and printed in Java?",
        "expected_sources": ["Java Lecture 11.pdf"],
    },
    {
        "query": "Explain the RAG architecture and how embeddings, vector search, BM25 and reranking support retrieval.",
        "expected_sources": ["RAG_LLM_Architecture_Guide.pdf"],
    },
    {
        "query": "What instructions are given for the Internetworking Essentials practical examination?",
        "expected_sources": ["big data.pdf"],
    },
]


def get_source(document):
    """Extract and normalize the source filename."""
    source = document.metadata.get("source", "Unknown")
    filename = Path(str(source)).name

    return DUPLICATE_SOURCE_ALIASES.get(filename, filename)


def evaluate_case(test_case):
    """Evaluate the production retrieval pipeline for one query."""

    query = test_case["query"]
    expected = set(test_case["expected_sources"])

    start_time = time.perf_counter()
    documents = search_documents(query)
    latency = time.perf_counter() - start_time

    top_documents = documents[:TOP_K]

    retrieved_chunks = [
        get_source(document) for document in top_documents
    ]

    # Source-label Precision@K:
    # Relevant source-attributed chunks / K.
    relevant_chunks = sum(
        1 for source in retrieved_chunks if source in expected
    )

    precision = relevant_chunks / TOP_K

    # Document/source Recall@K:
    # Expected sources found / total expected sources.
    retrieved_source_set = set(retrieved_chunks)

    relevant_sources = expected.intersection(retrieved_source_set)

    recall = len(relevant_sources) / len(expected)

    # Reciprocal rank of the first relevant chunk.
    reciprocal_rank = 0.0

    for rank, source in enumerate(retrieved_chunks, start=1):
        if source in expected:
            reciprocal_rank = 1 / rank
            break

    return {
        "query": query,
        "expected_sources": sorted(expected),
        "retrieved_chunks": retrieved_chunks,
        "relevant_chunks": relevant_chunks,
        "precision_at_3": round(precision, 4),
        "source_recall_at_3": round(recall, 4),
        "reciprocal_rank": round(reciprocal_rank, 4),
        "latency_seconds": round(latency, 3),
    }


def main():
    results = []

    print("\n========== MULTI-DOCUMENT RETRIEVAL EVALUATION ==========")
    print("Total test cases:", len(TEST_CASES))
    print("Top K:", TOP_K)

    for index, test_case in enumerate(TEST_CASES, start=1):
        print(f"\nTest {index}: {test_case['query']}")

        try:
            result = evaluate_case(test_case)
            results.append(result)

            print("Expected:", result["expected_sources"])
            print("Retrieved:", result["retrieved_chunks"])
            print("Relevant chunks:", result["relevant_chunks"])
            print("Precision@3:", result["precision_at_3"])
            print("Source Recall@3:", result["source_recall_at_3"])
            print("Reciprocal Rank:", result["reciprocal_rank"])
            print("Latency:", result["latency_seconds"], "seconds")

        except Exception as error:
            print("Evaluation failed:", str(error))

    if not results:
        print("\nNo test cases completed successfully.")
        return

    successful_count = len(results)

    summary = {
        "total_test_cases": len(TEST_CASES),
        "successful_test_cases": successful_count,
        "average_precision_at_3": round(
            sum(r["precision_at_3"] for r in results) / successful_count,
            4,
        ),
        "average_source_recall_at_3": round(
            sum(r["source_recall_at_3"] for r in results) / successful_count,
            4,
        ),
        "mean_reciprocal_rank": round(
            sum(r["reciprocal_rank"] for r in results) / successful_count,
            4,
        ),
        "average_latency_seconds": round(
            sum(r["latency_seconds"] for r in results) / successful_count,
            3,
        ),
    }

    report = {
        "evaluation_type": "Multi-document retrieval evaluation",
        "top_k": TOP_K,
        "metric_notes": {
            "precision_at_3": (
                "Fraction of top-three chunks attributed to an expected source. "
                "This is a source-label proxy, not human-judged chunk relevance."
            ),
            "source_recall_at_3": (
                "Fraction of expected source documents represented in the top three."
            ),
            "mean_reciprocal_rank": (
                "Average reciprocal rank of the first chunk from an expected source."
            ),
            "duplicate_handling": (
                "API_Test_RAG_Guide.pdf is normalized to "
                "RAG_LLM_Architecture_Guide.pdf because the files are identical."
            ),
        },
        "summary": summary,
        "results": results,
    }

    output_file = (
        Path(__file__).resolve().parent
        / "retrieval_evaluation_report.json"
    )

    output_file.write_text(
        json.dumps(report, indent=4),
        encoding="utf-8",
    )

    print("\n========== FINAL SUMMARY ==========")
    print(json.dumps(summary, indent=4))
    print("\nReport saved to:", output_file)


if __name__ == "__main__":
    main()
