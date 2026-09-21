from typing import List, Dict, Any

from app.retriever.retriever import search_documents
from app.llm.gemini_llm import llm
from app.memory.conversation_memory import session_memory_manager


FALLBACK_MESSAGE = (
    "I could not find that information in the provided documents."
)


def build_context(documents) -> str:
    """
    Build the document context that will be provided to the LLM.
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


def create_prompt(
    context: str,
    question: str,
    conversation_history: str = ""
) -> str:
    """
    Create the document-grounded prompt used by Gemini.
    """

    if conversation_history:
        history_section = f"""
PREVIOUS CONVERSATION:

{conversation_history}
"""
    else:
        history_section = """
PREVIOUS CONVERSATION:

No previous conversation.
"""

    return f"""
You are the final answer generator for an Enterprise RAG Platform.

Your task is to answer the user's CURRENT question using the
provided document context and previous conversation only.

IMPORTANT RULES:

1. Carefully read ALL document sources.
2. Use information directly supported by the documents.
3. You may use previous conversation to understand references
   such as "it", "this", "that", "they", or follow-up questions.
4. Previous conversation provides conversational context only.
   It must NOT be treated as additional factual evidence when
   answering questions about the documents.
5. The document context is the authoritative source for
   document-grounded factual answers.
6. You may combine information from multiple document sources.
7. Do NOT require the exact wording of the user's question
   to appear in the documents.
8. If the documents provide only PART of the answer, explain
   the supported information clearly and honestly.
9. If the documents mention the topic but do not explain it
   in sufficient detail, state what the documents establish
   and clearly say that the detailed explanation is not provided.
10. NEVER invent facts, definitions, mechanisms, examples,
    code, or explanations that are not supported by the documents.
11. Do NOT use outside knowledge.
12. Do NOT mention embeddings, retrieval, vector databases,
    compression, re-ranking, Multi-Query, Hybrid Search,
    or other internal RAG implementation details.
13. Keep the answer concise and directly relevant.
14. Only use the fallback message when the documents contain
    NO relevant information about the current question.

Fallback message:
"{FALLBACK_MESSAGE}"

{history_section}

CURRENT DOCUMENT CONTEXT:
{context}

CURRENT USER QUESTION:
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
    Extract unique source filename/page information.
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


def ask_rag(
    question: str,
    session_id: str = "default"
):
    """
    Standard non-streaming RAG pipeline.

    Returns:
        {
            "answer": str,
            "sources": list
        }
    """

    question = question.strip()
    session_id = session_id.strip()

    if not question:
        raise ValueError("Question cannot be empty.")

    if not session_id:
        session_id = "default"

    # ---------------------------------------------------------
    # Session memory
    # ---------------------------------------------------------

    memory = session_memory_manager.get_memory(session_id)

    # ---------------------------------------------------------
    # Advanced document retrieval
    # ---------------------------------------------------------

    documents = search_documents(question)

    # ---------------------------------------------------------
    # No relevant documents
    # ---------------------------------------------------------

    if not documents:
        answer = FALLBACK_MESSAGE

        memory.add_turn(
            question=question,
            answer=answer
        )

        return {
            "answer": answer,
            "sources": []
        }

    # ---------------------------------------------------------
    # Build document context
    # ---------------------------------------------------------

    context = build_context(documents)

    if not context:
        answer = FALLBACK_MESSAGE

        memory.add_turn(
            question=question,
            answer=answer
        )

        return {
            "answer": answer,
            "sources": extract_sources(documents)
        }

    # ---------------------------------------------------------
    # Previous conversation
    # ---------------------------------------------------------

    conversation_history = memory.get_history_text()

    # ---------------------------------------------------------
    # Create grounded prompt
    # ---------------------------------------------------------

    prompt = create_prompt(
        context=context,
        question=question,
        conversation_history=conversation_history
    )

    # ---------------------------------------------------------
    # Generate complete Gemini response
    # ---------------------------------------------------------

    response = llm.invoke(prompt)

    answer = extract_answer(response)

    # ---------------------------------------------------------
    # Extract sources
    # ---------------------------------------------------------

    sources = extract_sources(documents)

    # ---------------------------------------------------------
    # Save conversation
    # ---------------------------------------------------------

    memory.add_turn(
        question=question,
        answer=answer
    )

    return {
        "answer": answer,
        "sources": sources
    }


def stream_rag(
    question: str,
    session_id: str = "default"
):
    """
    Streaming RAG pipeline.

    The response is yielded incrementally while the complete
    answer is saved to the session memory after streaming ends.
    """

    question = question.strip()
    session_id = session_id.strip()

    if not question:
        raise ValueError("Question cannot be empty.")

    if not session_id:
        session_id = "default"

    # ---------------------------------------------------------
    # Session memory
    # ---------------------------------------------------------

    memory = session_memory_manager.get_memory(session_id)

    # ---------------------------------------------------------
    # Advanced document retrieval
    # ---------------------------------------------------------

    documents = search_documents(question)

    # ---------------------------------------------------------
    # No relevant documents
    # ---------------------------------------------------------

    if not documents:
        answer = FALLBACK_MESSAGE

        memory.add_turn(
            question=question,
            answer=answer
        )

        yield answer
        return

    # ---------------------------------------------------------
    # Build document context
    # ---------------------------------------------------------

    context = build_context(documents)

    if not context:
        answer = FALLBACK_MESSAGE

        memory.add_turn(
            question=question,
            answer=answer
        )

        yield answer
        return

    # ---------------------------------------------------------
    # Previous conversation
    # ---------------------------------------------------------

    conversation_history = memory.get_history_text()

    # ---------------------------------------------------------
    # Create grounded prompt
    # ---------------------------------------------------------

    prompt = create_prompt(
        context=context,
        question=question,
        conversation_history=conversation_history
    )

    # ---------------------------------------------------------
    # Stream Gemini response
    # ---------------------------------------------------------

    answer_parts = []

    for chunk in llm.stream(prompt):
        text = getattr(chunk, "text", None)

        if not text:
            continue

        answer_parts.append(text)

        yield text

    # ---------------------------------------------------------
    # Save complete streamed response
    # ---------------------------------------------------------

    complete_answer = "".join(answer_parts).strip()

    memory.add_turn(
        question=question,
        answer=complete_answer
    )


# -------------------------------------------------------------
# CLI TEST
# -------------------------------------------------------------

if __name__ == "__main__":

    print("=" * 60)
    print("Enterprise RAG - Conversational Test")
    print("=" * 60)

    session_id = input("\nEnter session ID: ").strip()

    if not session_id:
        session_id = "default"

    print("\nSession:", session_id)
    print("\nType 'exit' to stop.\n")

    while True:

        question = input("You: ").strip()

        if question.lower() == "exit":
            break

        if not question:
            continue

        try:

            result = ask_rag(
                question=question,
                session_id=session_id
            )

            print("\nAssistant:")
            print(result["answer"])

            print("\nSources:")

            for source in result["sources"]:
                print(
                    f"- {source['filename']} "
                    f"(Page {source['page']})"
                )

            print()

        except Exception as e:

            print("\nError:", e)