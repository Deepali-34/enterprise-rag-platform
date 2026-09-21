from app.retriever.multi_query_retriever import multi_query_search
from app.context_compressor.compressor import compress_documents
from app.reranker.reranker import rerank_documents


INITIAL_TOP_K = 5
COMPRESSED_TOP_K = 3
FINAL_TOP_K = 3
COMPRESSION_MIN_SCORE = 0.35


def search_documents(query: str):
    """
    Execute the complete advanced retrieval pipeline.

    Pipeline:
        Multi-Query + Hybrid Search
            ↓
        Context Compression
            ↓
        Cross-Encoder Re-ranking
            ↓
        Final Documents
    """

    # ---------------------------------------------------------
    # Stage 1: Multi-Query + Hybrid Retrieval
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

    # ---------------------------------------------------------
    # Final results
    # ---------------------------------------------------------

    return reranked_documents