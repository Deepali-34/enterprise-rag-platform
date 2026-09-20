from typing import List, Dict, Any

from app.retriever.retriever import search_documents
from app.llm.gemini_llm import llm


FALLBACK_MESSAGE = (
    "I could not find that information in the provided documents."
)


def build_context(documents) -> str:
    """
    Convert retrieved LangChain documents into structured context
    for the final LLM.
    """

    if not documents:
        return ""

    context_parts = []

    for index, doc in enumerate(documents, start=1):

        metadata = doc.metadata or {}

        source = metadata.get("source", "Unknown")
        page = metadata.get("page", "Unknown")

        content = doc.page_content.strip()

        if not content:
            continue

        context_parts.append(
            f"""
--- SOURCE {index} ---
Filename: {source}
Page: {page}

Content:
{content}
"""
        )

    return "\n".join(context_parts).strip()


def create_prompt(context: str, question: str) -> str:
    """
    Create the final source-grounded prompt.
    """

    return f"""
You are the final answer generator for an Enterprise RAG Platform.

Your task is to answer the user's question using ONLY the
retrieved document context provided below.

IMPORTANT RULES:

1. Carefully read ALL retrieved sources.

2. Use information directly supported by the retrieved documents.

3. You may combine information from multiple retrieved sources
   when they discuss the same topic.

4. Do NOT require the exact wording of the user's question
   to appear in the documents.

5. Do NOT use outside knowledge.

6. If the documents provide PART of the answer, explain the
   supported information clearly and honestly.

7. If the documents mention a topic but do NOT explain it in
   sufficient detail, say what the documents actually establish
   and clearly state that the detailed explanation is not provided
   in the documents.

8. NEVER invent definitions, mechanisms, examples, code,
   explanations, or facts that are not supported by the documents.

9. Do NOT mention embeddings, retrieval, vector databases,
   compression, re-ranking, Multi-Query, Hybrid Search,
   or other internal RAG implementation details.

10. Do NOT mention "the retrieved documents" repeatedly.
    Answer naturally as if you are answering from the provided
    source material.

11. Keep the answer concise and directly relevant to the question.

12. Only use the fallback message below when the documents contain
    NO relevant information about the user's question:

    "{FALLBACK_MESSAGE}"

DOCUMENT CONTEXT:

{context}

USER QUESTION:

{question}

FINAL ANSWER:
"""


def extract_answer(response) -> str:
    """
    Extract text from the Gemini response.
    """

    answer = getattr(response, "text", None)

    if answer is None:
        answer = str(response)

    return str(answer).strip()


def extract_sources(documents) -> List[Dict[str, Any]]:
    """
    Extract unique source filename and page information.
    """

    sources = []

    for doc in documents:

        metadata = doc.metadata or {}

        source = metadata.get("source", "Unknown")
        page = metadata.get("page", 0)

        source_info = {
            "filename": str(source),
            "page": page
        }

        if source_info not in sources:
            sources.append(source_info)

    return sources


def ask_rag(question: str):
    """
    Complete Advanced RAG pipeline:

    Question
        ↓
    Multi-Query Retrieval
        ↓
    Hybrid Search
        ↓
    Context Compression
        ↓
    Cross-Encoder Re-ranking
        ↓
    Context Construction
        ↓
    Gemini
        ↓
    Answer + Sources
    """

    question = question.strip()

    if not question:
        raise ValueError("Question cannot be empty.")

    # ----------------------------------------
    # 1. Advanced Retrieval
    # ----------------------------------------

    documents = search_documents(question)

    if not documents:
        return {
            "answer": FALLBACK_MESSAGE,
            "sources": []
        }

    # ----------------------------------------
    # 2. Build Context
    # ----------------------------------------

    context = build_context(documents)

    if not context:
        return {
            "answer": FALLBACK_MESSAGE,
            "sources": extract_sources(documents)
        }

    # ----------------------------------------
    # 3. Create Prompt
    # ----------------------------------------

    prompt = create_prompt(
        context=context,
        question=question
    )

    # ----------------------------------------
    # 4. Generate Answer
    # ----------------------------------------

    response = llm.invoke(prompt)

    answer = extract_answer(response)

    # ----------------------------------------
    # 5. Extract Sources
    # ----------------------------------------

    sources = extract_sources(documents)

    return {
        "answer": answer,
        "sources": sources
    }


def main():

    print("=" * 70)
    print("Enterprise RAG Platform")
    print("Advanced RAG Pipeline")
    print("=" * 70)

    question = input("\nAsk a question: ").strip()

    if not question:
        print("Question cannot be empty.")
        return

    try:

        result = ask_rag(question)

        print("\n" + "=" * 70)
        print("Answer")
        print("=" * 70)

        print(result["answer"])

        print("\n" + "=" * 70)
        print("Sources")
        print("=" * 70)

        if not result["sources"]:

            print("No sources found.")

        else:

            for source in result["sources"]:

                print(
                    f"- {source['filename']} "
                    f"(Page {source['page']})"
                )

    except Exception as e:

        print("\n" + "=" * 70)
        print("RAG Error")
        print("=" * 70)

        print(str(e))


if __name__ == "__main__":
    main()