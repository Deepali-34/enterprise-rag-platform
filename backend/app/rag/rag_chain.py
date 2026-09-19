from app.retriever.retriever import search_documents
from app.llm.gemini_llm import llm


def build_context(documents):
    """
    Build structured context from retrieved documents.
    """

    context_parts = []

    for index, doc in enumerate(documents, start=1):

        metadata = doc.metadata or {}

        source = metadata.get("source", "Unknown")
        page = metadata.get("page", "Unknown")

        context_parts.append(
            f"""
SOURCE {index}
Filename: {source}
Page: {page}

Content:
{doc.page_content}
"""
        )

    return "\n".join(context_parts)


def create_prompt(context, question):
    """
    Create the grounded RAG prompt.
    """

    prompt = f"""
You are an AI assistant for an Enterprise RAG Platform.

Answer the user's question ONLY using the retrieved document context.

Rules:
1. Use only the provided context.
2. If the answer is present in the context, answer clearly.
3. Do not use outside knowledge.
4. Do not claim information is unavailable if the context contains
   relevant information.
5. If the information genuinely cannot be found, say:
   "I could not find that information in the provided documents."
6. Give a concise and useful answer.

Retrieved Document Context:
{context}

User Question:
{question}

Answer:
"""

    return prompt


def ask_rag(question):
    """
    Complete RAG pipeline.

    Question
        ↓
    Multi-Query Retrieval
        ↓
    Hybrid Search
        ↓
    Context Construction
        ↓
    Gemini
        ↓
    Answer + Sources
    """

    documents = search_documents(question)

    context = build_context(documents)

    prompt = create_prompt(
        context=context,
        question=question
    )

    response = llm.invoke(prompt)

    # Current LangChain API
    answer = response.text

    sources = []

    for doc in documents:

        metadata = doc.metadata or {}

        source = metadata.get("source", "Unknown")
        page = metadata.get("page", 0)

        source_info = {
            "filename": source,
            "page": page
        }

        if source_info not in sources:
            sources.append(source_info)

    return {
        "answer": answer,
        "sources": sources
    }


def main():

    print("=" * 60)
    print("Enterprise RAG Platform")
    print("=" * 60)

    question = input("\nAsk a question: ").strip()

    if not question:
        print("Question cannot be empty.")
        return

    result = ask_rag(question)

    print("\n" + "=" * 60)
    print("Answer")
    print("=" * 60)

    print(result["answer"])

    print("\n" + "=" * 60)
    print("Sources")
    print("=" * 60)

    for source in result["sources"]:

        print(
            f"- {source['filename']} "
            f"(Page {source['page']})"
        )


if __name__ == "__main__":
    main()