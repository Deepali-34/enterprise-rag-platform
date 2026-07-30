from pathlib import Path

from app.retriever.retriever import search_documents
from app.llm.gemini_llm import llm


def build_context(documents):
    """
    Build context from retrieved documents.
    """

    context = ""

    for doc in documents:
        context += doc.page_content
        context += "\n\n"

    return context


def create_prompt(context, question):
    """
    Create prompt for Gemini.
    """

    prompt = f"""
You are an AI assistant.

Answer the question ONLY using the context below.

If the answer is not present in the context, say:

"I could not find that information in the provided documents."

Context:
{context}

Question:
{question}

Answer:
"""

    return prompt


def extract_sources(documents):
    """
    Extract unique document sources.
    """

    unique_sources = {}

    for doc in documents:

        metadata = doc.metadata

        # Use enriched filename if available
        source = metadata.get(
            "filename",
            metadata.get("source", "Unknown")
        )

        filename = Path(source).name

        # Convert page number to human-readable format
        page = metadata.get(
            "page_number",
            metadata.get("page", 0)
        ) + 1

        key = (filename, page)

        if key not in unique_sources:
            unique_sources[key] = {
                "filename": filename,
                "page": page
            }

    return list(unique_sources.values())


def ask_rag(question):
    """
    Enterprise RAG Pipeline
    """

    documents = search_documents(question)

    context = build_context(documents)

    prompt = create_prompt(context, question)

    response = llm.invoke(prompt)

    sources = extract_sources(documents)

    return {
        "answer": response.text(),
        "sources": sources
    }


def main():

    print("=" * 60)
    print("Enterprise RAG Platform")
    print("=" * 60)

    question = input("\nAsk a question: ")

    result = ask_rag(question)

    print("\n" + "=" * 60)
    print("Answer")
    print("=" * 60)

    print(result["answer"])

    print("\n" + "=" * 60)
    print("Sources")
    print("=" * 60)

    for source in result["sources"]:
        print(f"{source['filename']} (Page {source['page']})")


if __name__ == "__main__":
    main()